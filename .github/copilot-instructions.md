# Adaptive Information Retrieval - AI Agent Instructions

## Architecture Overview

**4-Stage Adaptive IR Pipeline** solving the bounded recall problem in cascade ranking:

```
Query → Stage 0: Candidate Mining (BM25 top-k0)
      → Stage 1: RL Query Reformulation (Actor-Critic Transformer)
      → Stage 2: Multi-Query Retrieval + RRF Fusion
      → Stage 3: BERT Cross-Encoder Re-ranking → Results
```

**Two codebases exist:**
- `adaptive-ir-system/` — **Active development** (PyTorch, Python 3.10+)
- `dl4ir-query-reformulator/` — **Legacy reference only** (Theano, Python 2.7)

## Detailed System Flow

### Training Pipeline Flow
```
1. DataManager.load_all()
   ├── Load Word2Vec embeddings (374K vectors, 500-dim)
   ├── Load corpus (480K documents from HDF5)
   ├── Load dataset (queries + qrels for train/valid/test)
   └── Pre-compute query embeddings (once, stored in tensor)

2. BM25Engine.build_index()
   └── Build BM25Okapi index from corpus

3. Training Loop (PPO):
   ├── AsyncBatchPrefetcher prepares next batch (parallel)
   │   ├── Sample batch_qids from train set
   │   ├── Get query embeddings (pre-computed)
   │   ├── BM25 search for candidates (lazy cached)
   │   └── Compute candidate features
   │
   ├── Agent.select_action() → choose term to expand
   │
   ├── Compute reward:
   │   ├── Expand query with selected term
   │   ├── BM25 search with expanded query (lazy cached)
   │   └── reward = Δ Recall@10 × 5.0
   │
   └── PPO Update (every 512 samples):
       ├── Normalize rewards
       ├── Compute advantages
       └── Update policy with clipped objective

4. Evaluation:
   ├── Baseline: BM25 only
   └── RL+RRF: Agent selects expansion → Multi-query → RRF fusion
```

### Key Optimization Points
| Issue | Solution in `train_optimized_v2.py` |
|-------|-------------------------------------|
| Pre-compute ALL queries wastes time | **Lazy caching** - cache on demand |
| Sequential batch preparation | **AsyncBatchPrefetcher** - parallel |
| Slow embedding computation | **Pre-compute once** for all queries |
| GPU underutilization | **Mixed precision (AMP)** training |
| Cache grows unbounded | **LRU-style eviction** when full |

## Key Source Files

| Purpose | File |
|---------|------|
| **Optimized training** | `train_optimized_v2.py` (NEW) |
| Pipeline orchestrator | `src/pipeline/adaptive_pipeline.py` |
| RL Agent (Actor-Critic) | `src/rl_agent/agent.py` |
| GPU training loop | `src/training/train_rl_optimized.py` |
| Metrics (Recall, MRR, nDCG) | `src/evaluation/metrics.py` |
| HDF5 dataset loader | `src/utils/legacy_loader.py` |
| BM25 search (no index) | `src/utils/simple_searcher.py` |
| RRF fusion | `src/fusion/rrf.py` |

## Quick Commands

```bash
cd adaptive-ir-system

# NEW: Optimized training (recommended)
python train_optimized_v2.py --mode quick   # ~3 min test
python train_optimized_v2.py --mode medium  # ~15 min
python train_optimized_v2.py --mode full    # Full training

# Original benchmark
python benchmark_pipeline.py

# Verify setup
python train_quick_test.py
python scripts/test_legacy_data.py
```

## Configuration Pattern

All parameters in YAML configs (`configs/`) or dataclass. CLI overrides supported:

```yaml
# Key settings
data:
  dataset_type: 'msa'          # msa | trec-car | jeopardy | msmarco
embeddings:
  type: 'legacy'               # 'legacy' → 500-dim Word2Vec
training:
  collect_batch_size: 32       # Parallel episode collection
  use_amp: true                # FP16 mixed precision
```

## Code Patterns

### Search Engine Selection (automatic by dataset type)
```python
# HDF5 datasets (msa, jeopardy, trec-car) → SimpleBM25Searcher
from src.utils.simple_searcher import SimpleBM25Searcher
searcher = SimpleBM25Searcher(corpus_adapter)

# MS MARCO → LuceneSearcher (requires pre-built index)
from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher(index_path)
```

### RL Agent Forward Pass
```python
# Returns: (action_logits, value_estimate)
logits, value = agent(query_emb, cand_embs, cand_features, mask)

# Select action
action, log_prob, value = agent.select_action(q_emb, c_embs, c_feats, mask)
```

### Async Batch Prefetching (NEW)
```python
prefetcher = AsyncBatchPrefetcher(data, search, config, train_qids)
prefetcher.start()
batch = prefetcher.get_batch()  # Non-blocking, pre-prepared
prefetcher.stop()
```

### Metrics Always Require K
```python
from src.evaluation.metrics import IRMetrics
IRMetrics.recall_at_k(retrieved, relevant, k=100)
IRMetrics.reciprocal_rank(retrieved, relevant)  # MRR
```

## Reward Function

```python
# In training loop
if action == STOP:
    reward = base_recall * 0.5
else:
    expanded_query = query + " " + doc_text[:50]
    new_recall = recall@10(search(expanded_query))
    reward = (new_recall - base_recall) * 5.0  # Scale for learning
```

## Critical Conventions

1. **Device handling**: Always use `config['system']['device']`, never hardcode `'cuda'`
2. **Java for Pyserini**: Entry points must call `check_and_setup_java()` from `train_optimized.py`
3. **Legacy embeddings**: `D_cbow_pdw_8B.pkl` is 500-dim; sentence-transformers are 384-dim
4. **Logging**: Use `setup_logging()` from `src/utils/` — includes emoji prefixes

## Data Files

| Dataset | Required Files | Location |
|---------|---------------|----------|
| MS Academic | `msa_dataset.hdf5`, `msa_corpus.hdf5`, `D_cbow_pdw_8B.pkl` | `Query Reformulator/` |
| MS MARCO | Download via `scripts/download_msmarco.py` | `data/msmarco/` |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA OOM | Reduce `batch_size`, enable `use_amp: true` |
| Java not found | Set `JAVA_HOME` or auto-detect handles `/usr/lib/jvm/java-*` |
| HDF5 key errors | Run `scripts/check_hdf5_structure.py` |
| Slow training | Use `train_optimized_v2.py` (async prefetch + lazy cache) |
| Pre-computing too slow | Already fixed in V2 - no upfront pre-compute |

## Proposal vs Implementation Status

| Proposal Stage | Status | Notes |
|---------------|--------|-------|
| Stage 0: Candidate Mining | ✅ Done | BM25 top-k0 |
| Stage 1: RL Reformulation | ✅ Done | Actor-Critic + PPO |
| Stage 2: Multi-Query + RRF | ✅ Done | In evaluation |
| Stage 3: BERT Re-rank | ⏳ TODO | Planned but not implemented |

## Legacy Reference (read-only)

`dl4ir-query-reformulator/` uses Theano 0.9 + Python 2.7:
```bash
THEANO_FLAGS='floatX=float32,device=gpu0' python run.py
```
Config in `parameters.py`. Do not modify—reference implementation only.
