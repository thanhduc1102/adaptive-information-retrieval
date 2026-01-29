"""
SMOKE TEST Training Script
--------------------------
Use this script to verify the pipeline (Train -> Eval -> Save) works 
without waiting for hours. It limits the dataset size significantly.
"""

import logging
import os
import sys
import yaml
import argparse
import torch
import torch.multiprocessing as mp
from pathlib import Path
import time
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, ConfigManager, set_seed
from src.utils.data_loader import DatasetFactory
from src.pipeline import AdaptiveIRPipeline
from src.training.train_rl_quickly import OptimizedRLTrainingLoop
from torch.utils.data import Subset

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def print_gpu_info():
    """Print detailed GPU information."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    print("\n" + "=" * 60)
    print("🖥️  GPU Information")
    print("=" * 60)
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
        
        # Current memory usage
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
    
    print("=" * 60 + "\n")
    return True

def setup_multi_gpu():
    """Setup for multi-GPU training."""
    if torch.cuda.device_count() > 1:
        print(f"🚀 Multi-GPU mode: {torch.cuda.device_count()} GPUs available")
        torch.cuda.set_device(0)
        return True
    return False

def check_and_setup_java():
    """Ensure Java is available for Pyserini."""
    if os.environ.get('JAVA_HOME'):
        return

    import glob
    candidates = []
    candidates.extend(sorted(glob.glob('/usr/lib/jvm/java-21-openjdk-*'), reverse=True))
    candidates.extend(sorted(glob.glob('/usr/lib/jvm/java-17-openjdk-*'), reverse=True))
    candidates.extend(sorted(glob.glob('/usr/lib/jvm/java-11-openjdk-*'), reverse=True))
    
    if candidates:
        best_java = candidates[0]
        print(f"Auto-configuring JAVA_HOME: {best_java}")
        os.environ['JAVA_HOME'] = best_java

def setup_search_engine(config: dict, dataset_adapter=None):
    """Initialize search engine."""
    dataset_type = config['data'].get('dataset_type', 'msmarco')
    
    if dataset_type in ['msa_LEGACY_MODE_DISABLED', 'trec-car', 'jeopardy', 'legacy', 'hdf5']:
        if dataset_adapter is None:
            raise ValueError("Legacy datasets require dataset_adapter")
        from src.utils.simple_searcher import SimpleBM25Searcher
        searcher = SimpleBM25Searcher(
            dataset_adapter,
            k1=config['retrieval'].get('bm25_k1', 0.9),
            b=config['retrieval'].get('bm25_b', 0.4)
        )
        return searcher
    else:
        from pyserini.search.lucene import LuceneSearcher
        index_path = config['data']['index_path']
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index not found at {index_path}")
        
        searcher = LuceneSearcher(index_path)
        searcher.set_bm25(
            config['retrieval'].get('bm25_k1', 0.9),
            config['retrieval'].get('bm25_b', 0.4)
        )
        return searcher

def slice_dataset(dataset, limit, logger, name="dataset"):
    """Helper function to slice different types of datasets."""
    if limit is None or limit <= 0:
        return dataset

    # --- ƯU TIÊN 1: Xử lý LegacyDatasetAdapter (MSA) ---
    # Class này không có __len__, nên phải check list 'queries' trực tiếp
    if hasattr(dataset, 'queries') and isinstance(dataset.queries, list):
        original_len = len(dataset.queries) # Lấy len từ list, không gọi len(dataset)
        
        # Cắt list queries
        dataset.queries = dataset.queries[:limit]
        
        logger.warning(f"✂️ SLICED {name}: {original_len} -> {len(dataset.queries)} samples (Modified .queries list)")
        return dataset
    
    # --- ƯU TIÊN 2: Xử lý Standard PyTorch Dataset ---
    # Các dataset này có __len__
    try:
        original_len = len(dataset)
        logger.warning(f"✂️ SLICED {name}: {original_len} -> {limit} samples (using Subset)")
        from torch.utils.data import Subset
        return Subset(dataset, range(min(original_len, limit)))
    except TypeError:
        # Trường hợp xấu nhất: Không đo được độ dài, không cắt được
        logger.warning(f"⚠️ WARNING: Cannot slice {name}. Object has no len() and no 'queries' list. Using full dataset.")
        return dataset

def main(args):
    """Main training function."""
    check_and_setup_java()
    
    if not print_gpu_info():
        print("Warning: Running on CPU, this will be slow!")
    
    multi_gpu = setup_multi_gpu()
    
    config = load_config(args.config)
    config_manager = ConfigManager(config)
    
    # --- OVERRIDES ---
    if args.device: config_manager.set('system.device', args.device)
    if args.seed: config_manager.set('system.seed', args.seed)
    if args.epochs: config_manager.set('training.num_epochs', args.epochs)
    if args.batch_size: config_manager.set('training.batch_size', args.batch_size)
    if args.max_samples: config_manager.set('data.max_samples', args.max_samples)
    
    # Setup logging
    log_dir = Path(config['training'].get('log_dir', './logs_smoke_test'))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file=str(log_dir / 'train_smoke.log'))
    
    logger.info("=" * 80)
    logger.info("🔥 SMOKE TEST MODE: Adaptive IR System")
    logger.info("=" * 80)
    
    set_seed(config['system'].get('seed', 42))
    
    # Device
    device = config['system'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # Load datasets
    logger.info("\n📚 Loading datasets...")
    dataset_factory = DatasetFactory(config['data'])
    
    train_dataset = dataset_factory.create_dataset('train')
    val_dataset = dataset_factory.create_dataset('dev')
    test_dataset = dataset_factory.create_dataset('test') if args.test else None
    
    # --- DATA SLICING LOGIC (CORE CHANGE) ---
    max_samples = config['data'].get('max_samples', None)
    
    if max_samples:
        logger.info(f"\n🔪 Applying Smoke Test Limits (Max: {max_samples})...")
        train_dataset = slice_dataset(train_dataset, max_samples, logger, "Train")
        # Với Validation, ta cắt nhỏ hơn nữa để Eval chạy nhanh cực đại
        val_limit = min(max_samples, 100) 
        val_dataset = slice_dataset(val_dataset, val_limit, logger, "Validation")
    # ----------------------------------------

    # Log final counts
    try:
        # Try retrieving len depending on dataset type
        t_len = len(train_dataset.queries) if hasattr(train_dataset, 'queries') else len(train_dataset)
        v_len = len(val_dataset.queries) if hasattr(val_dataset, 'queries') else len(val_dataset)
        logger.info(f"Final Train Size: {t_len}")
        logger.info(f"Final Val Size: {v_len}")
    except:
        pass

    # Setup search engine
    logger.info("\n🔍 Initializing search engine...")
    dataset_type = config['data'].get('dataset_type', 'msmarco')
    if dataset_type in ['msa_LEGACY_MODE_DISABLED', 'trec-car', 'jeopardy', 'legacy', 'hdf5']:
        search_engine = setup_search_engine(config, dataset_adapter=train_dataset)
    else:
        search_engine = setup_search_engine(config)
    
    # Load embedding model
    logger.info("\n📊 Loading embedding model...")
    embedding_model = None
    if config['rl_agent'].get('use_pretrained_embeddings', True):
        embedding_type = config.get('embeddings', {}).get('type', 'sentence-transformers')
        if embedding_type == 'legacy':
            from src.utils.legacy_embeddings import LegacyEmbeddingAdapter
            embeddings_path = config['embeddings']['path']
            embedding_model = LegacyEmbeddingAdapter(embeddings_path)
        else:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(
                config['rl_agent'].get('embedding_model', 'all-MiniLM-L6-v2')
            )
    
    # Initialize pipeline
    logger.info("\n🏗️ Building pipeline...")
    pipeline = AdaptiveIRPipeline(config=config, search_engine=search_engine, embedding_model=embedding_model)
    
    if args.checkpoint:
        pipeline.load_rl_checkpoint(args.checkpoint)
    
    # Initialize optimized training loop
    logger.info("\n⚡ Initializing optimized training loop...")
    training_loop = OptimizedRLTrainingLoop(
        config=config,
        pipeline=pipeline,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )
    
    logger.info("\n🎯 Starting Smoke Test Training...")
    try:
        training_loop.train()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrupted")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        raise
    
    logger.info("\n✅ Smoke Test Completed!")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    parser = argparse.ArgumentParser(description='Smoke Test for Adaptive IR')
    
    parser.add_argument('--config', type=str, default='./configs/msa_smoke_config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--max-samples', type=int, default=None, help='Override max samples for smoke test')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--test', action='store_true')
    
    args = parser.parse_args()
    main(args)