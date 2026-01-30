# Adaptive Information Retrieval - AI Agent Instructions

## Project Overview

**Adaptive Retrieve-Fuse-Re-rank Search Engine** using Deep RL query reformulation. Solves the **bounded recall problem**—documents not retrieved in stage 1 cannot be recovered by re-ranking.

**Two codebases:**
- **`adaptive-ir-system/`**: Active PyTorch implementation (focus here)
- **`dl4ir-query-reformulator/`**: Legacy Theano reference (read-only)

## Architecture: 4-Stage Pipeline

```
Query → [Stage 0] CandidateTermMiner → [Stage 1] QueryReformulatorAgent (RL)
      → [Stage 2] BM25 + RRF Fusion → [Stage 3] BERT Re-rank → Results
```

**Key insight**: Agent learns to select candidate terms that expand query, improving recall before re-ranking.

### Source File Map
| Component | File | Key Classes/Functions |
|-----------|------|----------------------|
| Pipeline | `src/pipeline/adaptive_pipeline.py` | `AdaptiveIRPipeline` - orchestrates all stages |
| RL Agent | `src/rl_agent/agent.py` | `QueryReformulatorAgent(nn.Module)` - Actor-Critic Transformer |
| Training | `src/training/train_rl_quickly.py` | `OptimizedRLTrainingLoop`, `EmbeddingCache`, `EpisodeData` |
| Candidate Mining | `src/candidate_mining/term_miner.py` | `CandidateTermMiner` - TF-IDF, BM25 contrib |
| RRF Fusion | `src/fusion/rrf.py` | `RecipRankFusion` - merge multi-query results |
| BERT Re-ranker | `src/reranker/bert_reranker.py` | `BERTReranker` - cross-encoder |
| Data | `src/utils/legacy_loader.py` | `LegacyDatasetAdapter`, `LegacyDatasetHDF5` |
| Search | `src/utils/simple_searcher.py` | `SimpleBM25Searcher` (in-memory, no Lucene) |
| Metrics | `src/evaluation/metrics.py` | `IRMetrics.recall_at_k()`, `.reciprocal_rank()`, `.ndcg_at_k()` |

## Critical Workflows

### 1. Always Verify Setup First
```bash
cd adaptive-ir-system
python poc_test.py                     # Tests all components
```

### 2. Training Commands
```bash
# Recommended: Legacy Word2Vec (~4.5h/epoch on 2x T4)
python train_quickly.py --config ./configs/msa_quick_config.yaml --epochs 10
```

### 3. Demo Full Pipeline
```bash
# Demo với sample queries
python demo_full_pipeline.py --sample 3 --no-bert

# So sánh với BM25 baseline
python demo_full_pipeline.py --query "your query" --compare

# Interactive mode
python demo_full_pipeline.py
```

### 4. Full Evaluation
```bash
# Compare all methods: baseline, RM3, RL+RRF, Full pipeline
python evaluate_full.py --split valid --num-queries 500 --no-bert

# With BERT re-ranking
python evaluate_full.py --split valid --num-queries 200
```

### 5. Fast Evaluation
```bash
python eval_checkpoint_optimized.py --checkpoint checkpoints_msa_optimized/best_model.pt --split valid
```

## Code Patterns (Follow These)

### Script Imports - ALWAYS add sys.path first
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))  # Required for src.* imports
```

### Java Setup - Copy this pattern for new scripts
```python
import os
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-21-openjdk-amd64'  # Before Pyserini imports
```

### Dataset Loading
```python
from src.utils.legacy_loader import LegacyDatasetAdapter
adapter = LegacyDatasetAdapter(
    dataset_path='../Query Reformulator/msa_dataset.hdf5',
    corpus_path='../Query Reformulator/msa_corpus.hdf5',
    split='train'  # or 'valid', 'test'
)
queries = adapter.load_queries()   # Dict[str, str]
qrels = adapter.load_qrels()       # Dict[str, Dict[str, int]]
```

### Search Engine
```python
from src.utils.simple_searcher import SimpleBM25Searcher
searcher = SimpleBM25Searcher(adapter, k1=0.9, b=0.4)
results = searcher.search("query text", k=100)  # List[Dict] with 'doc_id', 'score'
```

### Candidate Mining
```python
from src.candidate_mining import CandidateTermMiner
miner = CandidateTermMiner({'max_candidates': 50, 'methods': ['tfidf', 'bm25_contrib']})
candidates = miner.extract_candidates(query, documents, doc_scores)
```

### RRF Fusion
```python
from src.fusion import RecipRankFusion
rrf = RecipRankFusion(k=60)
fused = rrf.fuse([list1, list2, list3])  # List[Tuple[doc_id, score]]
```

### Metrics
```python
from src.evaluation.metrics import IRMetrics
recall = IRMetrics.recall_at_k(retrieved_ids, relevant_set, k=100)
mrr = IRMetrics.reciprocal_rank(retrieved_ids[:10], relevant_set)
ndcg = IRMetrics.ndcg_at_k(retrieved_ids, relevant_dict, k=10)  # dict for grades
```

## Configuration (`configs/msa_quick_config.yaml`)

**Critical settings:**
```yaml
training:
  collect_batch_size: 128    # Reduce if OOM
  use_amp: true              # FP16 - always enable
embeddings:
  type: 'legacy'             # 500-dim Word2Vec
rl_agent:
  num_query_variants: 4      # Number of reformulated queries
  max_steps_per_episode: 5   # Max terms to add
```

## Data Files

HDF5 datasets in `../Query Reformulator/`:
- `msa_dataset.hdf5` - queries + qrels
- `msa_corpus.hdf5` - document texts
- `D_cbow_pdw_8B.pkl` - Word2Vec embeddings

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA OOM` | Reduce `collect_batch_size`, enable `use_amp: true` |
| `JAVA_HOME not set` | `os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-21-openjdk-amd64'` |
| `ModuleNotFoundError: src.*` | Add `sys.path.insert(0, ...)` at script top |
| Agent outputs zeros | Normal early training; wait 50+ episodes |

## Key Scripts

| Script | Purpose |
|--------|---------|
| `train_quickly.py` | Main training script |
| `demo_full_pipeline.py` | Full 4-stage pipeline demo |
| `evaluate_full.py` | Compare all methods |
| `eval_checkpoint_optimized.py` | Fast BM25 evaluation |
| `inference.py` | Production inference |
| `poc_test.py` | Setup verification |
