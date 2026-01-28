#!/usr/bin/env python3
"""
Quick Test Script for Optimized Training
Tests GPU utilization with batched processing.
"""

import torch
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_gpu():
    """Test GPU operations."""
    print("=" * 60)
    print("GPU TEST")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    # Basic test
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    
    # Matrix multiply test
    print("\nMatrix multiply benchmark:")
    sizes = [1000, 2000, 4000]
    
    for size in sizes:
        x = torch.randn(size, size, device='cuda')
        y = torch.randn(size, size, device='cuda')
        
        # Warm up
        _ = torch.mm(x, y)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            z = torch.mm(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"  {size}x{size}: {elapsed*100:.1f}ms per op")
    
    print(f"\nGPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print("✅ GPU test passed!")
    return True


def test_rl_agent():
    """Test RL agent with batched input."""
    print("\n" + "=" * 60)
    print("RL AGENT TEST")
    print("=" * 60)
    
    from src.rl_agent import QueryReformulatorAgent
    
    config = {
        'embedding_dim': 500,
        'hidden_dim': 256,
        'num_attention_heads': 4,
        'num_encoder_layers': 2,
        'dropout': 0.1
    }
    
    agent = QueryReformulatorAgent(config).cuda()
    agent.eval()
    
    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")
    
    # Test different batch sizes
    batch_sizes = [1, 8, 16, 32, 64]
    num_cands = 50
    
    print("\nBatch size benchmark:")
    
    for batch_size in batch_sizes:
        query_emb = torch.randn(batch_size, 500, device='cuda')
        current_emb = torch.randn(batch_size, 500, device='cuda')
        cand_embs = torch.randn(batch_size, num_cands, 500, device='cuda')
        cand_feats = torch.randn(batch_size, num_cands, 3, device='cuda')
        
        # Warm up
        with torch.no_grad():
            _ = agent.select_action(query_emb, current_emb, cand_embs, cand_feats)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                action, log_prob, value = agent.select_action(
                    query_emb, current_emb, cand_embs, cand_feats
                )
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        throughput = batch_size * 100 / elapsed
        print(f"  Batch {batch_size:3d}: {elapsed*10:.1f}ms/100 ops, {throughput:.0f} samples/sec")
    
    print(f"\nGPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print("✅ RL Agent test passed!")
    return True


def test_data_loading():
    """Test data loading speed."""
    print("\n" + "=" * 60)
    print("DATA LOADING TEST")
    print("=" * 60)
    
    import yaml
    from src.utils.data_loader import DatasetFactory
    
    # Load config
    with open('configs/msa_optimized_gpu.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Loading dataset...")
    start = time.time()
    
    factory = DatasetFactory(config['data'])
    train_dataset = factory.create_dataset('train')
    
    # Load queries
    queries = train_dataset.load_queries()
    qrels = train_dataset.load_qrels()
    
    print(f"Loaded {len(queries)} queries in {time.time() - start:.1f}s")
    print(f"Queries with qrels: {len([q for q in queries if q in qrels])}")
    
    # Sample query
    sample_qid = list(queries.keys())[0]
    print(f"\nSample query: '{queries[sample_qid][:50]}...'")
    print(f"Relevant docs: {len(qrels.get(sample_qid, {}))}")
    
    print("✅ Data loading test passed!")
    return True


def test_embedding_cache():
    """Test embedding cache."""
    print("\n" + "=" * 60)
    print("EMBEDDING CACHE TEST")
    print("=" * 60)
    
    import yaml
    from src.training.train_rl_optimized import EmbeddingCache
    from src.utils.legacy_embeddings import LegacyEmbeddingAdapter
    
    # Load config
    with open('configs/msa_optimized_gpu.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load embedding model
    print("Loading embedding model...")
    embedding_model = LegacyEmbeddingAdapter(config['embeddings']['path'])
    
    # Create cache
    cache = EmbeddingCache(embedding_model, device='cuda', max_size=10000)
    
    # Test single embedding
    text = "machine learning neural network"
    emb = cache.get(text)
    print(f"Single embedding shape: {emb.shape}")
    
    # Test batch embedding
    texts = [
        "deep learning",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "transformer architecture"
    ]
    
    start = time.time()
    embs = cache.get_batch(texts)
    elapsed = time.time() - start
    
    print(f"Batch embedding shape: {embs.shape}")
    print(f"Batch time: {elapsed*1000:.1f}ms")
    
    # Test cache hit
    embs2 = cache.get_batch(texts)
    stats = cache.stats()
    print(f"Cache stats: {stats}")
    
    print("✅ Embedding cache test passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "🚀 " * 20)
    print("OPTIMIZED TRAINING QUICK TEST")
    print("🚀 " * 20 + "\n")
    
    tests = [
        ("GPU Operations", test_gpu),
        ("RL Agent", test_rl_agent),
        ("Data Loading", test_data_loading),
        ("Embedding Cache", test_embedding_cache),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} FAILED: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("✅ All tests passed!" if all_passed else "❌ Some tests failed!"))
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
