"""
Quick Start with Legacy Datasets

This script demonstrates how to use the adaptive IR system with
legacy HDF5 datasets from the original dl4ir-query-reformulator.
"""

import sys
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, set_seed
from src.utils.legacy_loader import LegacyDatasetAdapter
from src.utils.legacy_embeddings import LegacyEmbeddingAdapter


def example_1_load_dataset():
    """Example 1: Load and explore legacy dataset."""
    print("\n" + "="*60)
    print("Example 1: Loading Legacy Dataset")
    print("="*60)
    
    # Paths
    dataset_path = "../Query Reformulator/trec_car_dataset.hdf5"
    
    # Load dataset
    adapter = LegacyDatasetAdapter(dataset_path, dataset_path, split='train')
    
    # Load queries
    queries = adapter.load_queries()
    print(f"\nLoaded {len(queries):,} queries")
    
    # Load qrels
    qrels = adapter.load_qrels()
    print(f"Loaded qrels for {len(qrels):,} queries")
    
    # Show example
    qid = list(queries.keys())[0]
    query = queries[qid]
    relevant_docs = qrels[qid]
    
    print(f"\nExample query (QID={qid}):")
    print(f"  Query: {query}")
    print(f"  Relevant docs: {list(relevant_docs.keys())[:5]}")
    
    adapter.close()


def example_2_load_embeddings():
    """Example 2: Load and use Word2Vec embeddings."""
    print("\n" + "="*60)
    print("Example 2: Loading Word2Vec Embeddings")
    print("="*60)
    
    # Load embeddings
    embeddings_path = "../Query Reformulator/D_cbow_pdw_8B.pkl"
    adapter = LegacyEmbeddingAdapter(embeddings_path)
    
    print(f"\nEmbedding dimension: {adapter.embedding_dim}")
    
    # Embed text
    query = "machine learning algorithms for information retrieval"
    embedding = adapter.encode(query, convert_to_tensor=True)
    
    print(f"\nQuery: {query}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {embedding.norm().item():.4f}")


def example_3_config_for_legacy():
    """Example 3: Show how to configure for legacy datasets."""
    print("\n" + "="*60)
    print("Example 3: Configuration for Legacy Datasets")
    print("="*60)
    
    config = {
        'data': {
            'dataset_type': 'trec-car',  # or 'jeopardy', 'msa'
            'data_dir': '../Query Reformulator'
        },
        'embeddings': {
            'type': 'legacy',
            'path': '../Query Reformulator/D_cbow_pdw_8B.pkl',
            'embedding_dim': 500,
            'freeze': False
        },
        'rl_agent': {
            'embedding_dim': 500,  # Must match embeddings
            'embedding_type': 'legacy'
        }
    }
    
    print("\nExample configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    print("\nTo train with this config:")
    print("  1. Save config to configs/my_legacy_config.yaml")
    print("  2. Run: python train.py --config configs/my_legacy_config.yaml")


def example_4_dataset_comparison():
    """Example 4: Compare available datasets."""
    print("\n" + "="*60)
    print("Example 4: Dataset Comparison")
    print("="*60)
    
    datasets_info = {
        'TREC-CAR': {
            'file': 'trec_car_dataset.hdf5',
            'description': 'Wikipedia sections as queries',
            'metric': 'Recall@40',
            'expected_performance': '47.6% (baseline)',
            'size': '~2M paragraphs, ~500K queries'
        },
        'Jeopardy': {
            'file': 'jeopardy_dataset.hdf5',
            'description': 'TV show questions',
            'metric': 'Accuracy',
            'expected_performance': 'Varies',
            'size': '~5.9M Wikipedia articles'
        },
        'MS Academic': {
            'file': 'msa_dataset.hdf5',
            'description': 'Paper titles as queries',
            'metric': 'Citation recall',
            'expected_performance': 'Varies',
            'size': 'Academic papers corpus'
        }
    }
    
    print("\nAvailable Legacy Datasets:")
    print("-" * 60)
    
    data_dir = Path("../Query Reformulator")
    
    for name, info in datasets_info.items():
        file_path = data_dir / info['file']
        exists = "✓" if file_path.exists() else "✗"
        
        print(f"\n{exists} {name}")
        print(f"  File: {info['file']}")
        print(f"  Description: {info['description']}")
        print(f"  Primary metric: {info['metric']}")
        print(f"  Expected: {info['expected_performance']}")
        print(f"  Size: {info['size']}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("LEGACY DATASET QUICK START")
    print("="*60)
    
    try:
        example_1_load_dataset()
    except Exception as e:
        print(f"\n⚠ Example 1 failed: {e}")
    
    try:
        example_2_load_embeddings()
    except Exception as e:
        print(f"\n⚠ Example 2 failed: {e}")
    
    example_3_config_for_legacy()
    example_4_dataset_comparison()
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Test data loading: python scripts/test_legacy_data.py")
    print("2. Train on legacy data: python train.py --config configs/legacy_config.yaml")
    print("3. Compare with baseline: Check original paper metrics")
    print("\nFor MS MARCO dataset, use configs/default_config.yaml instead.")
    print("="*60)


if __name__ == "__main__":
    main()
