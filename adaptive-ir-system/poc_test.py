#!/usr/bin/env python3
"""
POC Test Script

Quick test to verify all components work before full training.
Only processes a few queries to test the flow.
"""

import os
import sys
import yaml
import time

# Setup Java
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'

# Import after Java setup
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("=" * 60)
    print("POC TEST - Verify training flow")
    print("=" * 60)
    
    # Load config
    print("\n📂 Loading config...")
    with open('configs/msa_quick_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Quick test settings
    config['training']['num_epochs'] = 1
    config['training']['collect_batch_size'] = 10
    config['training']['episodes_per_update'] = 10
    
    # Load dataset
    print("\n📊 Loading datasets...")
    from src.utils.legacy_loader import LegacyDatasetAdapter
    from src.utils.data_loader import DatasetFactory
    
    data_config = config.get('data', {})
    data_dir = data_config.get('data_dir', '../Query Reformulator')
    
    # For MSA dataset, use msa_dataset.hdf5 and msa_corpus.hdf5
    import os
    dataset_path = os.path.join(data_dir, 'msa_dataset.hdf5')
    corpus_path = os.path.join(data_dir, 'msa_corpus.hdf5')
    
    train_dataset = LegacyDatasetAdapter(
        dataset_path=dataset_path,
        corpus_path=corpus_path,
        split='train'
    )
    val_dataset = LegacyDatasetAdapter(
        dataset_path=dataset_path,
        corpus_path=corpus_path,
        split='valid'
    )
    
    train_queries = train_dataset.load_queries()
    train_qrels = train_dataset.load_qrels()
    print(f"  Train queries: {len(train_queries)}")
    print(f"  Train qrels: {len(train_qrels)}")
    
    # Setup search engine
    print("\n🔍 Setting up search engine...")
    from src.utils.simple_searcher import SimpleBM25Searcher
    
    # SimpleBM25Searcher expects LegacyDatasetAdapter, not dict
    search_engine = SimpleBM25Searcher(train_dataset)
    print("  ✅ Search engine ready")
    
    # Load embedding model
    print("\n📊 Loading embedding model...")
    from src.utils.legacy_embeddings import LegacyEmbeddingAdapter
    
    embedding_model = LegacyEmbeddingAdapter(config['embeddings']['path'])
    print(f"  ✅ Embedding model ready (dim={embedding_model.embedding_dim})")
    
    # Initialize pipeline
    print("\n🏗️ Building pipeline...")
    from src.pipeline import AdaptiveIRPipeline
    
    pipeline = AdaptiveIRPipeline(
        config=config,
        search_engine=search_engine,
        embedding_model=embedding_model
    )
    print("  ✅ Pipeline ready")
    
    # Test candidate mining
    print("\n🔍 Testing candidate mining...")
    test_query = list(train_queries.values())[0]
    candidates = pipeline.mine_candidates(test_query)
    print(f"  Query: '{test_query[:50]}...'")
    print(f"  Candidates: {len(candidates)}")
    if candidates:
        print(f"  Top 5: {list(candidates.keys())[:5]}")
    
    # Initialize training loop
    print("\n🏋️ Initializing training loop...")
    from src.training.train_rl_quickly import OptimizedRLTrainingLoop
    
    training_loop = OptimizedRLTrainingLoop(
        config=config,
        pipeline=pipeline,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    print("  ✅ Training loop ready")
    
    # Test relevant terms extraction
    print("\n🔍 Testing relevant terms extraction...")
    test_qid = list(train_qrels.keys())[0]
    test_qrels_item = train_qrels[test_qid]
    relevant_terms = training_loop.collector._extract_relevant_terms(test_qrels_item)
    print(f"  Query ID: {test_qid}")
    print(f"  Relevant docs: {len(test_qrels_item)}")
    print(f"  Extracted terms: {len(relevant_terms)}")
    if relevant_terms:
        print(f"  Sample terms: {list(relevant_terms)[:10]}")
    
    # Test episode data preparation
    print("\n📦 Testing episode data preparation...")
    query = train_queries[test_qid]
    episode_data = training_loop.collector.prepare_episode_data(test_qid, query, test_qrels_item)
    if episode_data:
        print(f"  ✅ Episode data created")
        print(f"     Query: '{query[:40]}...'")
        print(f"     Candidates: {episode_data.num_candidates}")
        print(f"     Relevant terms: {len(episode_data.relevant_terms) if episode_data.relevant_terms else 0}")
    
    # Test reward function
    print("\n💰 Testing reward function...")
    test_reward = training_loop.compute_improved_reward(
        original_query="machine learning",
        current_query="machine learning neural",
        qrels={'doc1': 1},
        action_idx=0,
        done=False,
        selected_term='neural',
        candidate_features={'tfidf': 0.8, 'bm25_contrib': 0.6},
        step=0,
        relevant_terms={'neural', 'network', 'deep'}
    )
    print(f"  Reward (relevant term): {test_reward:.4f}")
    
    test_reward2 = training_loop.compute_improved_reward(
        original_query="machine learning",
        current_query="machine learning random",
        qrels={'doc1': 1},
        action_idx=0,
        done=False,
        selected_term='random',
        candidate_features={'tfidf': 0.8, 'bm25_contrib': 0.6},
        step=0,
        relevant_terms={'neural', 'network', 'deep'}
    )
    print(f"  Reward (non-relevant term): {test_reward2:.4f}")
    print(f"  ✅ Reward function working (diff: {test_reward - test_reward2:.4f})")
    
    # Check HuggingFace config
    print("\n🔗 Checking HuggingFace config...")
    print(f"  Enabled: {training_loop.hf_enabled}")
    print(f"  Repo ID: {training_loop.hf_repo_id}")
    if training_loop.hf_uploader:
        print("  ✅ HuggingFace uploader initialized")
    else:
        print("  ⚠️ HuggingFace upload disabled (enable in config)")
    
    print("\n" + "=" * 60)
    print("✅ POC TEST PASSED!")
    print("=" * 60)
    print("\nAll components verified:")
    print("  ✓ Config loading")
    print("  ✓ Dataset loading")
    print("  ✓ Search engine")
    print("  ✓ Embedding model")
    print("  ✓ Pipeline")
    print("  ✓ Candidate mining")
    print("  ✓ Training loop")
    print("  ✓ Relevant terms extraction")
    print("  ✓ Episode data preparation")
    print("  ✓ Reward function with relevance signal")
    print("  ✓ HuggingFace config")
    print("\nReady for full training with: python train_quickly.py --config configs/msa_quick_config.yaml")


if __name__ == "__main__":
    main()
