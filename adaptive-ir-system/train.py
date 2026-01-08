"""
Main Training Script

Train the full Adaptive IR system on MS MARCO.
"""

import logging
import os
import sys
import yaml
import argparse
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.utils import setup_logging, ConfigManager, set_seed
from src.utils.data_loader import DatasetFactory
from src.pipeline import AdaptiveIRPipeline
from src.training import RLTrainingLoop


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def check_and_setup_java():
    """
    Ensure Java is available for Pyserini.
    Auto-configures JAVA_HOME if found in standard Linux locations.
    """
    if os.environ.get('JAVA_HOME'):
        return

    # Try to detect JAVA_HOME on Linux/WSL (Ubuntu/Debian paths)
    import glob
    
    # Priority: Java 21 > Java 17 > Java 11
    # Pyserini recent versions often need 21 for vector search features
    candidates = []
    candidates.extend(sorted(glob.glob('/usr/lib/jvm/java-21-openjdk-*'), reverse=True))
    candidates.extend(sorted(glob.glob('/usr/lib/jvm/java-17-openjdk-*'), reverse=True))
    candidates.extend(sorted(glob.glob('/usr/lib/jvm/java-11-openjdk-*'), reverse=True))
    # Fallback to any other
    candidates.extend(sorted(glob.glob('/usr/lib/jvm/java-*-openjdk-*'), reverse=True))
    
    # Filter duplicates while preserving order
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
        
        if "java-11" in best_java:
             print("\n\033[93m⚠ WARNING: Using Java 11. Recent Pyserini versions require Java 21+.")
             print("If you encounter 'Module jdk.incubator.vector not found', please install Java 21:")
             print("  apt-get update && apt-get install -y openjdk-21-jdk\033[0m\n")
    else:
        # Warn user if Java likely missing
        print("\n⚠ WARNING: JAVA_HOME not set and no OpenJDK found in /usr/lib/jvm/.")
        print("  Pyserini requires Java 11+. If the script fails, install JDK: sudo apt install openjdk-21-jdk\n")


def setup_search_engine(config: dict, dataset_adapter=None):
    """
    Initialize search engine.
    
    For legacy datasets with corpus, use SimpleBM25Searcher.
    For MS MARCO, use Pyserini LuceneSearcher.
    
    Args:
        config: Configuration dict
        dataset_adapter: Optional LegacyDatasetAdapter (for legacy datasets)
    
    Returns:
        Search engine instance
    """
    dataset_type = config['data'].get('dataset_type', 'msmarco')
    
    # For legacy datasets with corpus, use simple BM25
    if dataset_type in ['msa', 'trec-car', 'jeopardy', 'legacy', 'hdf5']:
        if dataset_adapter is None:
            raise ValueError("Legacy datasets require dataset_adapter")
        
        logger = logging.getLogger(__name__)
        logger.info(f"Using SimpleBM25Searcher for {dataset_type} dataset")
        
        try:
            from src.utils.simple_searcher import SimpleBM25Searcher
        except ImportError as e:
            print(f"\n\033[91mCRITICAL ERROR: Failed to import SimpleBM25Searcher.\033[0m")
            print(f"Cause: {e}")
            print("Legacy datasets require 'rank_bm25'. Please install it:")
            print("  pip install rank_bm25\n")
            sys.exit(1)
        
        searcher = SimpleBM25Searcher(
            dataset_adapter,
            k1=config['retrieval'].get('bm25_k1', 0.9),
            b=config['retrieval'].get('bm25_b', 0.4)
        )
        
        return searcher
    
    # For MS MARCO, use Pyserini/Lucene
    else:
        # Import here to allow JAVA_HOME setup before Pyserini initializes JVM
        try:
            from pyserini.search.lucene import LuceneSearcher
        except Exception as e:
            print(f"\n\033[91mCRITICAL ERROR: Failed to import Pyserini.\033[0m")
            print(f"Cause: {e}")
            print("Please ensure Java 11+ is installed (e.g., 'sudo apt install openjdk-21-jdk')")
            print("and JAVA_HOME is set correctly.\n")
            sys.exit(1)

        index_path = config['data']['index_path']
        
        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"Index not found at {index_path}. "
                f"Please run: python scripts/build_index.py"
            )
        
        searcher = LuceneSearcher(index_path)
        searcher.set_bm25(
            config['retrieval'].get('bm25_k1', 0.9),
            config['retrieval'].get('bm25_b', 0.4)
        )
        
        return searcher


def main(args):
    """Main training function."""
    
    # Setup Java environment before any Pyserini imports
    check_and_setup_java()
    
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
    
    # Setup logging
    log_dir = Path(config['training'].get('log_dir', './logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file=str(log_dir / 'train.log'))
    
    logger.info("=" * 80)
    logger.info("Adaptive IR System - Training")
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
    
    # Load datasets
    logger.info("Loading datasets...")
    
    dataset_factory = DatasetFactory(config['data'])
    
    train_dataset = dataset_factory.create_dataset('train')
    val_dataset = dataset_factory.create_dataset('dev')
    test_dataset = dataset_factory.create_dataset('test') if args.test else None
    
    logger.info(f"Train queries: {len(train_dataset.load_queries())}")
    logger.info(f"Val queries: {len(val_dataset.load_queries())}")
    if test_dataset:
        logger.info(f"Test queries: {len(test_dataset.load_queries())}")
    
    # Setup search engine
    logger.info("Initializing search engine...")
    
    # For legacy datasets, pass the train_dataset adapter
    dataset_type = config['data'].get('dataset_type', 'msmarco')
    if dataset_type in ['msa', 'trec-car', 'jeopardy', 'legacy', 'hdf5']:
        search_engine = setup_search_engine(config, dataset_adapter=train_dataset)
        logger.info(f"Search engine: SimpleBM25 (legacy dataset)")
    else:
        search_engine = setup_search_engine(config)
        logger.info(f"Index: {config['data']['index_path']}")
    
    # Initialize pipeline
    logger.info("Building pipeline...")
    
    # Load embedding model if needed
    embedding_model = None
    if config['rl_agent'].get('use_pretrained_embeddings', True):
        embedding_type = config.get('embeddings', {}).get('type', 'sentence-transformers')
        
        if embedding_type == 'legacy':
            # Use legacy Word2Vec embeddings
            from src.utils.legacy_embeddings import LegacyEmbeddingAdapter
            embeddings_path = config['embeddings']['path']
            embedding_model = LegacyEmbeddingAdapter(embeddings_path)
            logger.info(f"Loaded legacy Word2Vec embeddings from {embeddings_path}")
        else:
            # Use sentence-transformers
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(
                config['rl_agent'].get('embedding_model', 'all-MiniLM-L6-v2')
            )
            logger.info(f"Loaded embedding model: {config['rl_agent']['embedding_model']}")
    
    pipeline = AdaptiveIRPipeline(
        config=config,
        search_engine=search_engine,
        embedding_model=embedding_model
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        pipeline.load_rl_checkpoint(args.checkpoint)
    
    # Initialize training loop
    logger.info("Initializing training loop...")
    training_loop = RLTrainingLoop(
        config=config,
        pipeline=pipeline,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )
    
    # Start training
    logger.info("Starting training...")
    logger.info(f"Epochs: {config['training']['num_epochs']}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Episodes per update: {config['training']['episodes_per_update']}")
    logger.info("-" * 80)
    
    try:
        training_loop.train()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Adaptive IR system')
    
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/default_config.yaml',
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
        '--test',
        action='store_true',
        help='Run test evaluation after training'
    )
    
    args = parser.parse_args()
    
    main(args)
