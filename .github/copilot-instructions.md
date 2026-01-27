# Adaptive Information Retrieval - AI Agent Instructions

## Project Overview

This is an **Adaptive Retrieve-Fuse-Re-rank Search Engine** combining Deep RL query reformulation with modern IR techniques. The system solves the **bounded recall problem** in cascade ranking—documents not found in stage 1 retrieval cannot be recovered by later re-ranking.

**Two implementations coexist:**
1. **`dl4ir-query-reformulator/`**: Legacy Theano/Python 2.7 reference (read-only)
2. **`adaptive-ir-system/`**: Modern PyTorch implementation (active development)

## Architecture: 4-Stage Pipeline

```
Query → [Stage 0] Candidate Mining → [Stage 1] RL Reformulation
      → [Stage 2] Multi-Query Retrieval + RRF → [Stage 3] BERT Re-rank → Results
```

### Key Source Files
| Component | File | Purpose |
|-----------|------|---------|
| Pipeline orchestrator | `src/pipeline/adaptive_pipeline.py` | End-to-end processing |
| RL Agent | `src/rl_agent/agent.py` | Actor-Critic Transformer for term selection |
| Training loop | `src/training/train_rl_optimized.py` | GPU-optimized PPO with batched episodes |
| RRF Fusion | `src/fusion/rrf.py` | Merge multi-query results |
| Metrics | `src/evaluation/metrics.py` | Recall, MRR, nDCG, MAP computation |
| Data loading | `src/utils/data_loader.py` | MS MARCO + Legacy HDF5 datasets |

## Configuration-Driven Development

All training parameters live in YAML configs (`configs/`). Key patterns:

```yaml
# configs/msa_optimized_gpu.yaml - Reference GPU config
data:
  dataset_type: 'msa'           # Options: msa, trec-car, jeopardy, msmarco
  data_dir: '../Query Reformulator'

embeddings:
  type: 'legacy'                # 'legacy' for Word2Vec, 'sentence-transformers' otherwise
  path: '../Query Reformulator/D_cbow_pdw_8B.pkl'

training:
  collect_batch_size: 32        # Episodes collected in parallel
  episodes_per_update: 256      # Episodes before PPO update
  use_amp: true                 # Mixed precision (FP16)
```

**Override via CLI**: `python train_optimized.py --epochs 20 --batch-size 64 --no-amp`

## Developer Workflows

### Quick Test (Verify Setup)
```bash
cd adaptive-ir-system
python train_quick_test.py                    # Runs 1 epoch, small batch
python scripts/test_legacy_data.py            # Verify HDF5 data loading
```

### Full Training
```bash
python train_optimized.py --config configs/msa_optimized_gpu.yaml --test
```
- Logs to `logs_msa/train.log`
- Checkpoints saved automatically

### Inference
```bash
python inference.py --checkpoint checkpoints/best.pt --query "machine learning basics"
```

## Code Patterns

### Dataset Adapters
Use `DatasetFactory` to handle both modern and legacy formats:
```python
# src/utils/data_loader.py
factory = DatasetFactory(config['data'])
dataset = factory.create_dataset('train')  # Returns appropriate adapter
queries = dataset.load_queries()           # Unified interface
```

### Search Engine Abstraction
Two search backends with same interface:
- **`SimpleBM25Searcher`**: In-memory BM25 for HDF5 datasets (no index required)
- **`LuceneSearcher`**: Pyserini for MS MARCO (requires pre-built index)

Selection is automatic based on `dataset_type` in config.

### RL Agent Interface
```python
# Forward pass returns: (action_logits, value_estimate, stop_logit)
logits, value, stop = agent(query_emb, current_emb, candidate_embs, features)
```

### Embedding Cache Pattern
`EmbeddingCache` in training loop pre-computes embeddings to avoid redundant computation:
```python
cache = EmbeddingCache(embedding_model, device='cuda')
emb = cache.get("query text")        # Single text
embs = cache.get_batch(["t1", "t2"]) # Batched
```

## Important Conventions

1. **Device handling**: Always respect `config['system']['device']`. Check CUDA availability:
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   ```

2. **Java for Pyserini**: Auto-configured in `train_optimized.py::check_and_setup_java()`. If adding new entry points, include similar logic.

3. **Legacy vs Modern embeddings**: Set `embeddings.type: 'legacy'` for 500-dim Word2Vec, otherwise sentence-transformers (384-dim).

4. **Metrics at K**: Always specify cutoff: `recall_at_k(retrieved, relevant, k=100)`

5. **Logging**: Use `setup_logging()` from utils; logs include emoji prefixes for visual scanning.

## Data Requirements

| Dataset | Files Needed | Size |
|---------|--------------|------|
| MS Academic | `msa_dataset.hdf5`, `msa_corpus.hdf5`, `D_cbow_pdw_8B.pkl` | ~1.7GB |
| TREC-CAR | `trec_car_dataset.hdf5` + embeddings | Similar |
| MS MARCO | Download via `scripts/download_msmarco.py`, build index | ~3GB+ |

## Common Issues

| Problem | Solution |
|---------|----------|
| CUDA OOM | Reduce `collect_batch_size`, enable `use_amp: true` |
| Java not found | Set `JAVA_HOME` or install openjdk-11+ |
| HDF5 key errors | Check `scripts/check_hdf5_structure.py` for expected keys |
| Slow training | Use `train_optimized.py` (batched), not `train.py` |
| Agent outputs zeros | Normal early in training; check after 50+ episodes |

## Documentation

- **Vietnamese docs**: `docs/TONG_QUAN_HE_THONG.md` (architecture), `QUICK_START_MSA.md`
- **Diagrams**: `MERMAID_DIAGRAMS.md` for visual architecture
- **GPU optimization**: `docs/GPU_OPTIMIZATION_ANALYSIS.md`

---

## Legacy Reference (`dl4ir-query-reformulator/`)

The original Theano implementation is preserved for reference:
- **Python 2.7 + Theano 0.9** (not 1.0—causes NullTypeGradError)
- Training: `THEANO_FLAGS='floatX=float32,device=gpu0' python run.py`
- Config in `parameters.py`, not YAML
- Uses PyLucene (requires `lucene.initVM()` before imports)
