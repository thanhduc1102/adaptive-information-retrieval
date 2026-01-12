# Query Reformulator - AI Agent Instructions

## Project Overview

This is a **Deep Reinforcement Learning Query Reformulation Framework** implementing the paper "Task-Oriented Query Reformulation with Reinforcement Learning" (EMNLP 2017). The system uses an RL agent with actor-critic architecture to iteratively reformulate queries, aiming to improve information retrieval metrics (Recall, MAP, F1) by solving the bounded recall problem in cascade ranking systems.

## Architecture & Key Components

### Data Pipeline (HDF5-based)
- **Dataset**: Queries + ground-truth documents stored in HDF5 format
  - `dataset_hdf5.py`: Wrapper for accessing queries and document IDs via `get_queries()`, `get_doc_ids()`
  - `corpus_hdf5.py`: Document corpus access via `get_article_text()`, `get_article_title()`
- **Datasets**: MS Academic, Jeopardy, TREC-CAR (queries, corpus pairs)
- **Pre-trained embeddings**: 374K Word2Vec embeddings in `D_cbow_pdw_8B_norm.pkl`

### RL Agent Architecture
- **Actor-Critic** with CNNs for feature extraction
  - Query encoder: CNN over word embeddings (`filters_query`, `window_query`)
  - Candidate encoder: CNN over feedback document words (`filters_cand`, `window_cand`)
  - Actor: Selects terms via softmax policy (`n_hidden_actor` layers)
  - Critic: Estimates baseline value (`n_hidden_critic` layers)
- **MDP formulation**: State = current query + feedback docs, Action = binary term selection, Reward = IR metrics (RECALL/MAP/F1)
- **Policy**: Stochastic during training (`trng.multinomial`), greedy during evaluation

### Search Engine Integration
- **Lucene-based** (`lucene_search.py`): Uses PyLucene for document retrieval
  - Creates indexes at `index_folder` if missing
  - Multi-threaded query execution (`n_threads` parallel processes)
  - Caching support (`use_cache`) for query-document pairs
- **Custom Theano Op** (`op_search.py`): `Search` class wraps search engine calls in Theano computation graph
  - Retrieves `max_candidates` documents per query
  - Returns metrics (RECALL, PRECISION, F1, MAP, LOG-GMAP) and feedback documents

### Training Loop (`run.py`)
- **Iterative reformulation**: `n_iterations` rounds of query expansion
  - Iteration 0: Original query → retrieve feedback docs
  - Iterations 1+: Agent selects terms from feedback docs → reformulate → retrieve
- **Frozen phases**: `frozen_until` iterations act greedily without learning; `q_0_fixed_until` keeps original query fixed
- **REINFORCE with baseline**: Policy gradient loss = `(reward - baseline) * -log(p(action))`
- **Regularization**: Entropy regularization (`erate`), L2 regularization (`l2reg`)

## Critical Configuration (`parameters.py`)

```python
# Model hyperparameters
dim_proj = 500          # LSTM/hidden units
dim_emb = 500           # Word embedding dimension
batch_size_train = 64   # Training batch size
n_iterations = 2        # Query reformulation iterations
max_candidates = 40     # Max retrieved documents
max_feedback_docs = 7   # Max feedback docs for term mining
frozen_until = 1        # Greedy iterations before learning
reward = 'RECALL'       # Optimization metric: RECALL/F1/MAP/gMAP

# Search engine
engine = 'lucene'       # Must be 'lucene'
n_threads = 20          # Parallel search threads
index_folder = data_folder + '/index/'
```

## Development Workflows

### Setup & Prerequisites
- **Python 2.7** (legacy codebase)
- **Theano 0.9** (not 1.0 - NullTypeGradError issue)
- **PyLucene 6.2+** (Java-based, requires JVM initialization)
- **32GB RAM recommended**, 6GB+ GPU for training

### Training from Scratch
```bash
# CPU training
THEANO_FLAGS='floatX=float32' python run.py

# GPU training (K80 recommended)
THEANO_FLAGS='floatX=float32,device=gpu0' python run.py
```
- **Training time**: 800K iterations (~7-10 days on K80) to reach 47.6% Recall@40 on TREC-CAR
- **Cold start**: Model starts selecting terms after ~50K iterations
- **Checkpointing**: Model saved every `saveFreq` iterations to `model.npz`

### Using Pre-trained Models
Set `reload_model='model.npz'` in `parameters.py` to resume training or evaluate

### Evaluation
- Validation runs every `validFreq` iterations
- Metrics computed: RECALL, PRECISION, F1, MAP, LOG-GMAP (see `average_precision.py`)
- Test sizes configurable via `train_size`, `valid_size`, `test_size` (use `-1` for full dataset)

## Code Patterns & Conventions

### Theano Computation Graph
- **Shared variables**: `init_tparams()` converts NumPy params → Theano shared variables
- **Custom ops**: Search engine calls integrated via `theano.Op` subclass with `perform()` method
- **Gradient handling**: Use `theano.gradient.disconnected_grad()` for REINFORCE baseline, `grad_scale()` for critic

### Data Preprocessing (`utils.py`)
- **Text cleaning**: `clean()` removes Wikipedia/AQUAINT markup
- **Tokenization**: NLTK's `wordpunct_tokenize` throughout
- **Index conversion**: `text2idx()` converts text → vocabulary indices, handles UNK with `-1`, padding with `-2`
- **BoW**: Normalized bag-of-words via `BOW()`, `BOW2()`

### Vocabulary & Embeddings
- **Fixed vocabulary**: 374K words loaded from pickle
- **Embedding fine-tuning**: Set `fixed_wemb=False` to learn embeddings (default: frozen)
- **PCA compression**: If `dim_emb < embedding_dim`, apply PCA to reduce dimensionality

### Loss Computation
- **Policy gradient**: `-log(p(action)) * (reward - baseline)` for selected terms only (after `q_0_fixed_until`)
- **Baseline loss**: MSE of critic prediction `(reward - baseline)^2`
- **Entropy bonus**: `-erate * Σ p*log(p)` to encourage exploration

## Typical Modification Points

### Changing Datasets
1. Update `dataset_path`, `docs_path` in `parameters.py`
2. Ensure HDF5 files have keys: `queries_train/valid/test`, `doc_ids_train/valid/test`, `text`, `title`
3. Re-run to create new Lucene indexes

### Tuning RL Agent
- **Exploration**: Increase `erate` (entropy regularization)
- **Learning rate**: Adjust `lrate` for SGD or optimizer defaults
- **Freezing**: Set `frozen_until` to delay learning, `q_0_fixed_until` to preserve original query longer
- **Feedback quality**: Change `max_feedback_docs`, `max_candidates` to control term mining pool

### Custom Metrics
- Add metrics to `metrics_map` in `parameters.py` and compute in `op_search.py`'s `perform()` method
- Update `reward` parameter to optimize new metric

### Alternative Search Engines
- Implement interface matching `LuceneSearch.get_candidates()` signature
- Set `engine='<name>'` in `parameters.py` and import in `train()`

## Common Issues & Debugging

- **"NullTypeGradError"**: Downgrade to Theano 0.9 from 1.0
- **Index creation hangs**: Lucene index building is one-time but slow; check `index_folder` for completion
- **PyLucene import errors**: Requires `lucene.initVM()` before use (see `LuceneSearch.__init__`)
- **Memory issues**: Reduce `batch_size_train`, disable `use_cache`, or increase RAM
- **Agent not selecting terms**: Normal before ~50K iterations; check `frozen_until` isn't too high
- **Ground-truth doc not in index**: Warning printed but training continues; check corpus-dataset alignment

## Project Extensions (Proposal Context)

The Vietnamese proposal (`proposal_project_only.txt`) outlines extending this codebase to:
- Add **Reciprocal Rank Fusion (RRF)** for multi-query result merging
- Integrate **BERT Cross-Encoder** re-ranking after retrieval
- Evaluate on **MS MARCO** and **BEIR** (OOD generalization)
- Compare against baselines: RM3, dense retrievers, original RL reformulator

When implementing these, maintain the existing RL loop structure but insert RRF fusion before re-ranking, and replace Lucene metrics with BERT-scored metrics in the reward signal.
