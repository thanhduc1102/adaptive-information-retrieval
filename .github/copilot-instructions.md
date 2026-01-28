# Adaptive Information Retrieval - AI Agent Instructions

## Project Overview

**4-Stage Adaptive IR Pipeline** for query reformulation using reinforcement learning:

```
Query → BM25 Candidates → RL Reformulation → Multi-Query + RRF → [BERT Re-rank] → Results
```

- **Active codebase**: `adaptive-ir-system/` (PyTorch, Python 3.10+)
- **Legacy reference**: `dl4ir-query-reformulator/` (Theano - read-only)

## Quick Start Commands

```bash
cd adaptive-ir-system

# Training (use V2 - has async prefetch + lazy caching)
python train_optimized_v2.py --mode quick   # ~3 min test
python train_optimized_v2.py --mode full    # Full training

# Evaluation
python evaluate_baseline.py --split valid   # BM25 baseline
python benchmark_pipeline.py                # Full pipeline

# Verify data loads correctly
python scripts/test_legacy_data.py
```

## Key Files & Architecture

| Component | File | Purpose |
|-----------|------|---------|
| Training entry | `train_optimized_v2.py` | Main training with `OptimizedConfig` dataclass |
| RL Agent | `src/rl_agent/agent.py` | `QueryReformulatorAgent` (Actor-Critic Transformer) |
| BM25 Search | `src/utils/simple_searcher.py` | `SimpleBM25Searcher` for HDF5 datasets |
| Data Loading | `src/utils/legacy_loader.py` | `LegacyDatasetAdapter` for HDF5 datasets |
| Metrics | `src/evaluation/metrics.py` | `IRMetrics.recall_at_k()`, `reciprocal_rank()` |
| RRF Fusion | `src/fusion/rrf.py` | `RecipRankFusion.fuse(ranked_lists)` |
| Pipeline | `src/pipeline/adaptive_pipeline.py` | `AdaptiveIRPipeline` orchestrates all stages |

## Critical Code Patterns

### 1. Search Engine Selection (by dataset type)
```python
# HDF5 datasets (msa, trec-car, jeopardy) → SimpleBM25Searcher
if dataset_type in ['msa', 'trec-car', 'jeopardy']:
    from src.utils.simple_searcher import SimpleBM25Searcher
    searcher = SimpleBM25Searcher(corpus_adapter, k1=0.9, b=0.4)

# MS MARCO → requires pre-built Lucene index
else:
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(index_path)
```

### 2. Data Loading (always use adapters)
```python
from src.utils import LegacyDatasetAdapter
adapter = LegacyDatasetAdapter(dataset_path, corpus_path, split='train')
queries = adapter.load_queries()  # Dict[qid, query_text]
qrels = adapter.load_qrels()      # Dict[qid, Set[doc_ids]]
```

### 3. Embeddings (500-dim legacy Word2Vec)
```python
# Load once, cache forever - embeddings are in D_cbow_pdw_8B.pkl
word2vec = pickle.load(open(path, 'rb'), encoding='latin1')
# Embed text: mean of word vectors, fallback to UNK embedding
```

### 4. Device Handling
```python
# Always from config, never hardcode 'cuda'
device = config.get('system', {}).get('device', 'cuda')
# Or use helper: from src.utils import get_device
```

## Configuration

YAML configs in `configs/` with dataclass overrides. Key settings:
```yaml
data:
  dataset_type: 'msa'           # msa | trec-car | jeopardy | msmarco
embeddings:
  embedding_dim: 500            # Must match D_cbow_pdw_8B.pkl
training:
  use_amp: true                 # FP16 mixed precision
  batch_size: 64
```

## Data Files Location

| Dataset | Files | Path |
|---------|-------|------|
| MS Academic | `msa_dataset.hdf5`, `msa_corpus.hdf5`, `D_cbow_pdw_8B.pkl` | `Query Reformulator/` |
| MS MARCO | Download via `scripts/download_msmarco.py` | `data/msmarco/` |

## Conventions

1. **Imports**: Add `sys.path.insert(0, str(Path(__file__).parent / 'src'))` in scripts
2. **Logging**: Use `setup_logging()` from `src/utils/helpers.py`
3. **Java for Pyserini**: Call `check_and_setup_java()` at script start (auto-detects JAVA_HOME)
4. **Metrics always need k**: `IRMetrics.recall_at_k(retrieved, relevant, k=100)`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `batch_size`, enable `use_amp: true` |
| HDF5 key errors | Run `scripts/check_hdf5_structure.py` |
| Java not found | Auto-detect in `/usr/lib/jvm/java-*` or set `JAVA_HOME` |
| Import errors | Ensure `rank_bm25` installed for legacy datasets |

## Implementation Status

- ✅ Stage 0: Candidate Mining (BM25 top-k)
- ✅ Stage 1: RL Query Reformulation (Actor-Critic + PPO)
- ✅ Stage 2: Multi-Query Retrieval + RRF Fusion
- ⏳ Stage 3: BERT Cross-Encoder Re-ranking (scaffolded in `src/reranker/`)
