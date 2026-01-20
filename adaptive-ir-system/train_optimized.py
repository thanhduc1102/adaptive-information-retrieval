"""
Optimized Training Script for Multi-GPU

Train the Adaptive IR system with GPU optimizations:
- Multi-GPU support (DataParallel)
- Mixed precision training (FP16)
- Batched episode collection
- Pre-computed embeddings cache
- Efficient memory management
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, ConfigManager, set_seed
from src.utils.data_loader import DatasetFactory
from src.pipeline import AdaptiveIRPipeline
from src.training.train_rl_optimized import OptimizedRLTrainingLoop


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
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi-processor count: {props.multi_processor_count}")
        
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
        # Set default GPU
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
    candidates.extend(sorted(glob.glob('/usr/lib/jvm/java-*-openjdk-*'), reverse=True))
    
    final_candidates = []
    seen = set()
    for c in candidates:
        if c not in seen:
            final_candidates.append(c)
            seen.add(c)

    if final_candidates:
        best_java = final_candidates[0]
        print(f"Auto-configuring JAVA_HOME: {best_java}")
        os.environ['JAVA_HOME'] = best_java


def setup_search_engine(config: dict, dataset_adapter=None):
    """Initialize search engine."""
    dataset_type = config['data'].get('dataset_type', 'msmarco')
    
    if dataset_type in ['msa', 'trec-car', 'jeopardy', 'legacy', 'hdf5']:
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


def main(args):
    """Main training function."""
    
    # Setup Java
    check_and_setup_java()
    
    # Check GPU
    if not print_gpu_info():
        print("Warning: Running on CPU, this will be slow!")
    
    # Setup multi-GPU
    multi_gpu = setup_multi_gpu()
    
    # Load config
    config = load_config(args.config)
    config_manager = ConfigManager(config)
    
    # Override with command line args
    if args.device:
        config_manager.set('system.device', args.device)
    if args.seed:
        config_manager.set('system.seed', args.seed)
    if args.epochs:
        config_manager.set('training.num_epochs', args.epochs)
    if args.batch_size:
        config_manager.set('training.batch_size', args.batch_size)
    if args.no_amp:
        config_manager.set('training.use_amp', False)
    
    # Setup logging
    log_dir = Path(config['training'].get('log_dir', './logs_msa_optimized'))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file=str(log_dir / 'train.log'))
    
    logger.info("=" * 80)
    logger.info("🚀 Adaptive IR System - Optimized GPU Training")
    logger.info("=" * 80)
    
    # Set seed
    seed = config['system'].get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Device
    device = config['system'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
        config_manager.set('system.device', device)
    
    logger.info(f"Device: {device}")
    logger.info(f"Multi-GPU: {multi_gpu}")
    logger.info(f"Mixed Precision (AMP): {config['training'].get('use_amp', True)}")
    
    # Load datasets
    logger.info("\n📚 Loading datasets...")
    start_time = time.time()
    
    dataset_factory = DatasetFactory(config['data'])
    
    train_dataset = dataset_factory.create_dataset('train')
    val_dataset = dataset_factory.create_dataset('dev')
    test_dataset = dataset_factory.create_dataset('test') if args.test else None
    
    train_queries = train_dataset.load_queries()
    val_queries = val_dataset.load_queries()
    
    logger.info(f"Train queries: {len(train_queries)}")
    logger.info(f"Val queries: {len(val_queries)}")
    if test_dataset:
        logger.info(f"Test queries: {len(test_dataset.load_queries())}")
    logger.info(f"Dataset loading time: {time.time() - start_time:.1f}s")
    
    # Setup search engine
    logger.info("\n🔍 Initializing search engine...")
    
    dataset_type = config['data'].get('dataset_type', 'msmarco')
    if dataset_type in ['msa', 'trec-car', 'jeopardy', 'legacy', 'hdf5']:
        search_engine = setup_search_engine(config, dataset_adapter=train_dataset)
        logger.info("Search engine: SimpleBM25 (legacy dataset)")
    else:
        search_engine = setup_search_engine(config)
        logger.info(f"Index: {config['data']['index_path']}")
    
    # Load embedding model
    logger.info("\n📊 Loading embedding model...")
    
    embedding_model = None
    if config['rl_agent'].get('use_pretrained_embeddings', True):
        embedding_type = config.get('embeddings', {}).get('type', 'sentence-transformers')
        
        if embedding_type == 'legacy':
            from src.utils.legacy_embeddings import LegacyEmbeddingAdapter
            embeddings_path = config['embeddings']['path']
            embedding_model = LegacyEmbeddingAdapter(embeddings_path)
            logger.info(f"Loaded legacy Word2Vec embeddings ({embedding_model.embedding_dim}-dim)")
        else:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(
                config['rl_agent'].get('embedding_model', 'all-MiniLM-L6-v2')
            )
            logger.info(f"Loaded embedding model: {config['rl_agent']['embedding_model']}")
    
    # Initialize pipeline
    logger.info("\n🏗️ Building pipeline...")
    
    pipeline = AdaptiveIRPipeline(
        config=config,
        search_engine=search_engine,
        embedding_model=embedding_model
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
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
    
    # Print training config
    logger.info("\n" + "=" * 60)
    logger.info("Training Configuration:")
    logger.info(f"  Epochs: {config['training']['num_epochs']}")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Collect batch size: {config['training'].get('collect_batch_size', 16)}")
    logger.info(f"  Episodes per update: {config['training']['episodes_per_update']}")
    logger.info(f"  PPO epochs: {config['training'].get('ppo_epochs', 4)}")
    logger.info(f"  Buffer size: {config['training'].get('buffer_size', 10000)}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info("=" * 60)
    
    # Start training
    logger.info("\n🎯 Starting training...")
    total_start = time.time()
    
    try:
        training_loop.train()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        raise
    
    total_time = time.time() - total_start
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Training completed!")
    logger.info(f"Total time: {total_time / 60:.1f} minutes")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Enable TF32 for faster matrix multiplication on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cudnn benchmark for faster convolutions
    torch.backends.cudnn.benchmark = True
    
    parser = argparse.ArgumentParser(description='Optimized GPU Training for Adaptive IR')
    
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/msa_optimized_gpu.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        dest='batch_size',
        help='Batch size'
    )
    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='Disable mixed precision training'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test evaluation after training'
    )
    
    args = parser.parse_args()
    
    main(args)
