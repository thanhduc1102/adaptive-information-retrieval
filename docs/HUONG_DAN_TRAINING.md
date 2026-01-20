# HƯỚNG DẪN TRAINING HỆ THỐNG ADAPTIVE IR

Tài liệu này hướng dẫn chi tiết từng bước để training RL Agent cho hệ thống Adaptive Information Retrieval.

---

## 📋 MỤC LỤC

1. [Chuẩn bị Môi trường](#1-chuẩn-bị-môi-trường)
2. [Cài đặt Dependencies](#2-cài-đặt-dependencies)
3. [Tải và Chuẩn bị Dữ liệu](#3-tải-và-chuẩn-bị-dữ-liệu)
4. [Cấu hình Training](#4-cấu-hình-training)
5. [Chạy Training](#5-chạy-training)
6. [Giám sát Quá trình Training](#6-giám-sát-quá-trình-training)
7. [Đánh giá Model](#7-đánh-giá-model)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. CHUẨN BỊ MÔI TRƯỜNG

### 1.1. Yêu cầu Hệ thống

**Hardware tối thiểu**:
- CPU: 4 cores
- RAM: 16GB
- Disk: 50GB trống
- GPU: NVIDIA GPU với 8GB+ VRAM (khuyến nghị)

**Hardware khuyến nghị**:
- CPU: 8+ cores
- RAM: 32GB+
- Disk: 100GB+ SSD
- GPU: NVIDIA GPU với 16GB+ VRAM (RTX 3090, A100)

**Software**:
- Python: 3.8, 3.9, hoặc 3.10
- CUDA: 11.7+ (nếu dùng GPU)
- Java: 11+ (cho Pyserini)

### 1.2. Kiểm tra GPU (nếu có)

```bash
# Kiểm tra NVIDIA GPU
nvidia-smi

# Kiểm tra CUDA
nvcc --version

# Kiểm tra PyTorch có nhận GPU không (sau khi cài)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### 1.3. Tạo Virtual Environment

```bash
# Tạo environment
python3.9 -m venv venv

# Kích hoạt
source venv/bin/activate  # Linux/Mac
# HOẶC
venv\Scripts\activate     # Windows

# Nâng cấp pip
pip install --upgrade pip setuptools wheel
```

---

## 2. CÀI ĐẶT DEPENDENCIES

### 2.1. Di chuyển vào thư mục dự án

```bash
cd adaptive-information-retrieval/adaptive-ir-system
```

### 2.2. Cài đặt Java (cho Pyserini)

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install -y openjdk-21-jdk

# Kiểm tra
java -version
```

**macOS**:
```bash
brew install openjdk@21

# Set JAVA_HOME
export JAVA_HOME=$(/usr/libexec/java_home -v21)
```

**Windows**:
- Download Java 21 từ: https://adoptium.net/
- Cài đặt và set JAVA_HOME trong Environment Variables

### 2.3. Cài đặt Python packages

```bash
# Cài đặt tất cả dependencies
pip install -r requirements.txt

# Hoặc cài từng nhóm:

# Core ML/DL
pip install torch>=2.0.0 transformers>=4.30.0 sentence-transformers>=2.2.0

# Information Retrieval
pip install pyserini>=0.21.0 rank-bm25>=0.2.2 pytrec-eval>=0.5

# Data Processing
pip install h5py>=3.8.0 pandas>=2.0.0 numpy>=1.24.0 nltk>=3.8 scikit-learn>=1.2.0

# Utilities
pip install pyyaml>=6.0 tqdm>=4.65.0 tensorboard>=2.13.0 wandb>=0.15.0

# Testing
pip install pytest>=7.3.0
```

### 2.4. Download NLTK data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2.5. Xác minh cài đặt

```bash
# Kiểm tra các package chính
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from pyserini.search.lucene import LuceneSearcher; print('Pyserini: OK')"
```

---

## 3. TẢI VÀ CHUẨN BỊ DỮ LIỆU

Có 2 lựa chọn dataset:
- **MS MARCO** (khuyến nghị - dataset chính thức, lớn)
- **Legacy datasets** (nhỏ hơn, cho testing nhanh)

### OPTION A: MS MARCO (Khuyến nghị)

#### 3.1. Tạo thư mục data

```bash
mkdir -p data/msmarco
cd data/msmarco
```

#### 3.2. Download MS MARCO dataset

**Tự động** (khuyến nghị):
```bash
cd ../../  # Quay về adaptive-ir-system
python scripts/download_msmarco.py \
  --data_dir ./data/msmarco \
  --subsets collection queries_train queries_dev qrels_train qrels_dev
```

**Thủ công**:
```bash
cd data/msmarco

# Collection (8.8M passages) - ~1GB
wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
tar -xzf collection.tar.gz

# Training queries & qrels
wget https://msmarco.blob.core.windows.net/msmarcoranking/queries.train.tsv
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv

# Dev queries & qrels
wget https://msmarco.blob.core.windows.net/msmarcoranking/queries.dev.tsv
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv

cd ../../
```

#### 3.3. Build BM25 Index

```bash
# Build index với Pyserini
python scripts/build_index.py \
  --collection ./data/msmarco/collection.tsv \
  --index ./data/msmarco/index \
  --threads 8

# Quá trình này mất 20-40 phút
# Yêu cầu ~10GB disk space cho index
```

**Xác minh index**:
```bash
# Test search
python -c "
from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher('./data/msmarco/index')
hits = searcher.search('what is covid', k=10)
print(f'Found {len(hits)} results')
print(f'Top result: {hits[0].raw}')
"
```

#### 3.4. Cấu trúc thư mục sau khi setup

```
data/msmarco/
├── collection.tsv           # 8.8M passages
├── queries.train.tsv        # ~500K training queries
├── queries.dev.tsv          # ~6,900 dev queries
├── qrels.train.tsv          # Training relevance judgments
├── qrels.dev.tsv            # Dev relevance judgments
└── index/                   # BM25 index (~10GB)
    ├── segments_1
    ├── _0.cfs
    └── ...
```

### OPTION B: Legacy Dataset (Nhanh hơn cho testing)

Nếu bạn có legacy datasets (MSA, TREC-CAR, Jeopardy) trong thư mục `Query Reformulator/`:

```bash
# Kiểm tra dữ liệu
python scripts/test_legacy_data.py --data_dir "../Query Reformulator"

# Nếu có msa_dataset.hdf5 và msa_corpus.hdf5
# Bạn có thể dùng config msa_config.yaml để training
```

---

## 4. CẤU HÌNH TRAINING

### 4.1. Tạo file config

Tạo file `configs/my_config.yaml`:

```yaml
# Cấu hình hệ thống
system:
  device: 'cuda'        # 'cuda' hoặc 'cpu'
  seed: 42
  num_workers: 4

# Cấu hình dữ liệu
data:
  dataset_type: 'msmarco'
  data_dir: './data/msmarco'
  index_path: './data/msmarco/index'

  # File paths (tự động tìm nếu không chỉ định)
  train_queries: './data/msmarco/queries.train.tsv'
  train_qrels: './data/msmarco/qrels.train.tsv'
  dev_queries: './data/msmarco/queries.dev.tsv'
  dev_qrels: './data/msmarco/qrels.dev.tsv'

# Embeddings (cho RL agent)
embeddings:
  type: 'sentence-transformers'
  model: 'all-MiniLM-L6-v2'  # 384-dim, nhanh

# Candidate Mining (Giai đoạn 0)
candidate_mining:
  enabled: true
  max_candidates: 100        # Số từ ứng viên tối đa
  min_score: 0.1
  methods:
    - 'tfidf'
    - 'bm25'
  top_k_per_method: 50
  top_k_docs: 10             # Số docs để mine candidates

# RL Agent (Giai đoạn 1)
rl_agent:
  enabled: true
  embedding_dim: 384         # Match embedding model
  hidden_dim: 256
  num_attention_heads: 4
  num_encoder_layers: 2
  dropout: 0.1
  max_steps_per_episode: 5   # Tối đa 5 từ được thêm vào
  num_query_variants: 4      # Tạo 4 query variants

  use_pretrained_embeddings: true
  embedding_model: 'all-MiniLM-L6-v2'

# RRF Fusion (Giai đoạn 2)
rrf_fusion:
  enabled: true
  k_constant: 60
  method: 'rrf'

# BERT Re-ranker (Giai đoạn 3)
bert_reranker:
  enabled: true
  model_name: 'cross-encoder/ms-marco-MiniLM-L-12-v2'
  batch_size: 128
  max_length: 512
  top_k_rerank: 100
  use_fp16: true             # FP16 nhanh hơn 2x

# Retrieval settings
retrieval:
  top_k: 100
  bm25_k1: 0.9
  bm25_b: 0.4

# Training hyperparameters
training:
  num_epochs: 50
  batch_size: 32
  learning_rate: 0.0003
  episodes_per_update: 128   # Update policy sau 128 episodes
  ppo_epochs: 4              # 4 PPO updates mỗi lần

  # PPO parameters
  gamma: 0.99                # Discount factor
  gae_lambda: 0.95           # GAE lambda
  clip_epsilon: 0.2          # PPO clip epsilon
  value_loss_coef: 0.5
  entropy_coef: 0.01         # Exploration bonus
  max_grad_norm: 0.5

  # Reward shaping
  reward_weights:
    recall: 0.7              # 70% weight cho Recall@100
    mrr: 0.3                 # 30% weight cho MRR@10

  # Checkpointing
  checkpoint_dir: './checkpoints'
  save_freq: 5               # Save mỗi 5 epochs

  # Early stopping
  early_stopping_patience: 10

  # Logging
  log_dir: './logs'

  # Replay buffer
  buffer_size: 10000

# Evaluation metrics
evaluation:
  metrics:
    - 'recall@10'
    - 'recall@50'
    - 'recall@100'
    - 'mrr@10'
    - 'ndcg@10'
    - 'map'
    - 'precision@10'
```

### 4.2. Config cho máy yếu (CPU hoặc GPU nhỏ)

```yaml
# configs/low_resource_config.yaml
system:
  device: 'cpu'              # Dùng CPU

rl_agent:
  embedding_dim: 128         # Giảm dimension
  hidden_dim: 128
  num_attention_heads: 2
  num_encoder_layers: 1

bert_reranker:
  enabled: false             # Tắt BERT re-ranking để nhanh hơn

training:
  batch_size: 16             # Giảm batch size
  episodes_per_update: 64
```

---

## 5. CHẠY TRAINING

### 5.1. Training cơ bản

```bash
# Training với config mặc định
python train.py --config configs/my_config.yaml

# Chỉ định device
python train.py \
  --config configs/my_config.yaml \
  --device cuda

# Chỉ định số epochs
python train.py \
  --config configs/my_config.yaml \
  --epochs 100

# Resume từ checkpoint
python train.py \
  --config configs/my_config.yaml \
  --checkpoint checkpoints/checkpoint_epoch_25.pt
```

### 5.2. Training với custom settings

```bash
# Training với custom seed
python train.py \
  --config configs/my_config.yaml \
  --seed 123 \
  --device cuda \
  --epochs 50
```

### 5.3. Chạy training trong background (Linux/Mac)

```bash
# Chạy trong background và log ra file
nohup python train.py \
  --config configs/my_config.yaml \
  --device cuda \
  > training.log 2>&1 &

# Xem log realtime
tail -f training.log

# Kiểm tra process
ps aux | grep train.py
```

### 5.4. Quy trình training chi tiết

Khi bạn chạy `train.py`, hệ thống thực hiện:

```
1. Load config
2. Setup logging → logs/train.log
3. Initialize search engine (BM25 index)
4. Load embedding model
5. Initialize pipeline:
   - Candidate Miner
   - RL Agent (Actor-Critic)
   - RRF Fusion
   - BERT Re-ranker

For each epoch (1 to num_epochs):

  For each query in training set:

    1. Mine candidates (Giai đoạn 0)
       - BM25 search → top-k docs
       - TF-IDF analysis
       - Extract 50-100 candidate terms

    2. Collect episode (Giai đoạn 1):
       a. Evaluate original query → metrics_before
       b. RL Agent selects terms iteratively:
          - Step 1: Select term_1 → query' = query + term_1
          - Step 2: Select term_2 → query'' = query' + term_2
          - ...
          - Step N: Select STOP
       c. Evaluate reformulated query → metrics_after
       d. Compute reward = w1*ΔRecall + w2*ΔMRR
       e. Store (state, action, reward) to replay buffer

    3. Update policy every 128 episodes:
       - Sample batch from replay buffer
       - Compute advantages
       - PPO update (4 epochs)
       - Compute loss

  Validation every 5 epochs:
    - Evaluate on dev set
    - Compute metrics (Recall@100, MRR@10, nDCG@10)
    - Save checkpoint if best
    - Check early stopping

  Save checkpoint:
    - checkpoints/checkpoint_epoch_X.pt
    - checkpoints/best_model.pt (best validation)

Final test evaluation:
  - Load best model
  - Evaluate on test set
  - Save results → checkpoints/test_results.json
```

---

## 6. GIÁM SÁT QUÁ TRÌNH TRAINING

### 6.1. Theo dõi qua logs

```bash
# Xem log realtime
tail -f logs/train.log

# Tìm lỗi
grep "ERROR" logs/train.log

# Xem validation metrics
grep "Validation" logs/train.log
```

**Log output mẫu**:
```
================================================================================
Adaptive IR System - Training
================================================================================
Random seed: 42
Device: cuda
Loading datasets...
Train queries: 502939
Val queries: 6980
Initializing search engine...
Index: ./data/msmarco/index
Building pipeline...
Loaded embedding model: all-MiniLM-L6-v2
Initialized Candidate Term Miner
Initialized RL Agent with 1,245,632 parameters
Initialized RRF Fusion (k=60)
Initialized BERT Re-ranker
Pipeline initialized successfully
Initializing training loop...
Starting training...
Epochs: 50
Batch size: 32
Episodes per update: 128
--------------------------------------------------------------------------------
Epoch 1/50: 100%|████████| 502939/502939 [2:15:32<00:00, 61.8it/s, reward=0.0234, episodes=1024]
Epoch 1/50 | Reward: 0.0234 | Loss: 0.1234
Epoch 5/50: 100%|████████| 502939/502939 [2:12:18<00:00, 63.2it/s, reward=0.0456, episodes=1024]
Epoch 5/50 | Reward: 0.0456 | Loss: 0.0987
Validation | Recall@100: 0.8123 | MRR@10: 0.3456
Saved best model with MRR@10: 0.3456
...
```

### 6.2. TensorBoard monitoring

```bash
# Khởi động TensorBoard (nếu enabled trong code)
tensorboard --logdir logs/ --port 6006

# Mở browser: http://localhost:6006
```

### 6.3. Weights & Biases (W&B) monitoring

```bash
# Login W&B (one-time)
wandb login

# Training sẽ tự động log lên W&B
# Xem tại: https://wandb.ai/your-username/adaptive-ir
```

### 6.4. Kiểm tra GPU usage

```bash
# Monitor GPU realtime
watch -n 1 nvidia-smi

# Hoặc dùng gpustat
pip install gpustat
gpustat -i 1
```

### 6.5. Ước tính thời gian

**MS MARCO (~500K training queries)**:
- **GPU (RTX 3090)**: ~2-3 giờ/epoch → 50 epochs = 100-150 giờ (4-6 ngày)
- **GPU (V100)**: ~1.5-2 giờ/epoch → 50 epochs = 75-100 giờ (3-4 ngày)
- **CPU**: ~10-15 giờ/epoch → Không khuyến nghị

**Mẹo tăng tốc**:
1. Giảm số queries training (sample 10% → nhanh hơn 10x)
2. Giảm `episodes_per_update` (128 → 64)
3. Tắt BERT re-ranker trong training
4. Dùng FP16 mixed precision
5. Tăng batch size nếu GPU đủ RAM

---

## 7. ĐÁNH GIÁ MODEL

### 7.1. Kiểm tra checkpoints

```bash
# List checkpoints
ls -lh checkpoints/

# Output:
# checkpoint_epoch_5.pt    (120MB)
# checkpoint_epoch_10.pt   (120MB)
# checkpoint_epoch_15.pt   (120MB)
# best_model.pt            (120MB)
```

### 7.2. Load và test checkpoint

```python
# test_checkpoint.py
import torch
from src.pipeline import AdaptiveIRPipeline
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer

# Load config
import yaml
with open('configs/my_config.yaml') as f:
    config = yaml.safe_load(f)

# Setup components
searcher = LuceneSearcher('./data/msmarco/index')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize pipeline
pipeline = AdaptiveIRPipeline(
    config=config,
    search_engine=searcher,
    embedding_model=embedding_model
)

# Load best model
pipeline.load_rl_checkpoint('checkpoints/best_model.pt')

# Test query
result = pipeline.search("what is covid-19", top_k=10, measure_latency=True)

print(f"Query: {result['query']}")
print(f"Query variants: {result['query_variants']}")
print(f"\nTop 10 results:")
for i, (doc_id, score) in enumerate(result['results'][:10], 1):
    doc = searcher.doc(doc_id)
    print(f"{i}. [{score:.4f}] {doc.raw()[:100]}...")

print(f"\nLatency:")
for stage, latency in result['latency'].items():
    print(f"  {stage}: {latency:.2f}ms")
```

```bash
python test_checkpoint.py
```

### 7.3. Evaluation script

```bash
# Evaluate trên dev set
python scripts/final_test.py \
  --config configs/my_config.yaml \
  --checkpoint checkpoints/best_model.pt \
  --split dev

# Evaluate trên test set
python scripts/final_test.py \
  --config configs/my_config.yaml \
  --checkpoint checkpoints/best_model.pt \
  --split test \
  --output results/test_results.json
```

### 7.4. So sánh với baseline

```python
# compare_baseline.py
from src.evaluation import IRMetricsAggregator

# Baseline (BM25 only)
baseline_metrics = {
    'recall@100': 0.75,
    'mrr@10': 0.28,
    'ndcg@10': 0.32
}

# Your model
your_metrics = {
    'recall@100': 0.86,
    'mrr@10': 0.41,
    'ndcg@10': 0.45
}

print("Improvement:")
for metric in baseline_metrics:
    baseline = baseline_metrics[metric]
    yours = your_metrics[metric]
    improvement = (yours - baseline) / baseline * 100
    print(f"{metric}: {baseline:.4f} → {yours:.4f} ({improvement:+.1f}%)")
```

---

## 8. TROUBLESHOOTING

### 8.1. Out of Memory (OOM)

**Triệu chứng**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB
```

**Giải pháp**:
```yaml
# Giảm batch_size trong config
training:
  batch_size: 16  # Từ 32 → 16

# Giảm max_candidates
candidate_mining:
  max_candidates: 50  # Từ 100 → 50

# Giảm embedding_dim
rl_agent:
  embedding_dim: 256  # Từ 512 → 256
  hidden_dim: 128     # Từ 256 → 128

# Hoặc dùng gradient accumulation
training:
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch = 32
```

### 8.2. Java/Pyserini errors

**Lỗi**: `Module jdk.incubator.vector not found`

**Giải pháp**:
```bash
# Cài Java 21
sudo apt install openjdk-21-jdk

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
```

**Lỗi**: `JVM cannot be started`

**Giải pháp**:
```bash
# Restart Python interpreter
# Pyserini chỉ khởi động JVM một lần, không thể restart
# Phải restart Python process
```

### 8.3. Training không converge

**Triệu chứng**: Reward không tăng sau nhiều epochs

**Giải pháp**:
```yaml
# 1. Giảm learning rate
training:
  learning_rate: 0.0001  # Từ 0.0003

# 2. Tăng exploration
training:
  entropy_coef: 0.02  # Từ 0.01

# 3. Thay đổi reward weights
training:
  reward_weights:
    recall: 0.8  # Tăng recall weight
    mrr: 0.2

# 4. Tăng episodes_per_update
training:
  episodes_per_update: 256  # Từ 128
```

### 8.4. Slow training

**Giải pháp**:
```yaml
# 1. Tắt BERT re-ranker trong training
bert_reranker:
  enabled: false

# 2. Giảm số query variants
rl_agent:
  num_query_variants: 2  # Từ 4

# 3. Giảm số docs để mine
candidate_mining:
  top_k_docs: 5  # Từ 10

# 4. Sample subset của training data
# Trong train.py, thêm:
# query_ids = query_ids[:50000]  # Chỉ lấy 50K queries
```

### 8.5. Index not found

**Lỗi**: `FileNotFoundError: Index not found at ./data/msmarco/index`

**Giải pháp**:
```bash
# Build index
python scripts/build_index.py \
  --collection ./data/msmarco/collection.tsv \
  --index ./data/msmarco/index

# Hoặc update path trong config
data:
  index_path: '/absolute/path/to/index'
```

### 8.6. Queries file not found

**Giải pháp**:
```bash
# Download lại
python scripts/download_msmarco.py \
  --data_dir ./data/msmarco \
  --subsets queries_train queries_dev qrels_train qrels_dev
```

### 8.7. Reward luôn = 0

**Nguyên nhân**: Không có qrels cho queries

**Giải pháp**:
```python
# Kiểm tra qrels
import pandas as pd
qrels = pd.read_csv('data/msmarco/qrels.train.tsv', sep='\t', header=None)
print(f"Number of qrels: {len(qrels)}")
print(qrels.head())

# Đảm bảo query_id trong queries có trong qrels
```

---

## 9. TIPS & BEST PRACTICES

### 9.1. Development workflow

```bash
# 1. Test trên subset nhỏ trước
# Sửa train.py để chỉ lấy 1000 queries:
query_ids = query_ids[:1000]

# 2. Training nhanh (5 epochs) để verify code
python train.py --config configs/my_config.yaml --epochs 5

# 3. Nếu OK, chạy full training
python train.py --config configs/my_config.yaml --epochs 50
```

### 9.2. Experiment tracking

```bash
# Tạo folder riêng cho mỗi experiment
mkdir -p experiments/exp_001_baseline
mkdir -p experiments/exp_002_higher_lr

# Copy config
cp configs/my_config.yaml experiments/exp_001_baseline/config.yaml

# Training
python train.py --config experiments/exp_001_baseline/config.yaml

# Log results
echo "Exp 001: Recall@100=0.86, MRR@10=0.41" >> experiments/results.txt
```

### 9.3. Hyperparameter tuning

**Thứ tự ưu tiên**:
1. `learning_rate`: [0.0001, 0.0003, 0.001]
2. `reward_weights`: Thử nhiều tỷ lệ recall/mrr
3. `num_query_variants`: [2, 3, 4, 5]
4. `max_candidates`: [50, 100, 150]
5. `hidden_dim`: [128, 256, 512]

### 9.4. Debugging

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Print intermediate results
# Trong collect_episode(), thêm:
print(f"Original query: {query}")
print(f"Candidates: {list(candidates.keys())[:10]}")
print(f"Selected terms: {selected_terms}")
print(f"Reformulated query: {current_query}")
print(f"Reward: {reward}")
```

---

## 10. CHECKLIST TRƯỚC KHI TRAINING

- [ ] Java 11+ đã cài và JAVA_HOME đã set
- [ ] Python 3.8+ đã cài
- [ ] Virtual environment đã tạo và activate
- [ ] Dependencies đã cài (`pip install -r requirements.txt`)
- [ ] NLTK data đã download
- [ ] Dataset đã download (MS MARCO hoặc legacy)
- [ ] BM25 index đã build
- [ ] Config file đã tạo và review
- [ ] Disk space đủ (ít nhất 50GB)
- [ ] GPU driver và CUDA đã cài (nếu dùng GPU)
- [ ] Test search với index thành công

---

## 11. LỆNH TRAINING HOÀN CHỈNH

```bash
# 1. Setup môi trường
cd adaptive-information-retrieval/adaptive-ir-system
source venv/bin/activate

# 2. Download data (nếu chưa có)
python scripts/download_msmarco.py --data_dir ./data/msmarco

# 3. Build index (nếu chưa có)
python scripts/build_index.py \
  --collection ./data/msmarco/collection.tsv \
  --index ./data/msmarco/index

# 4. Verify setup
python -c "from pyserini.search.lucene import LuceneSearcher; \
  searcher = LuceneSearcher('./data/msmarco/index'); \
  print(f'Index OK: {searcher.num_docs} docs')"

# 5. Training
python train.py \
  --config configs/my_config.yaml \
  --device cuda \
  --epochs 50 \
  2>&1 | tee training.log

# 6. Monitor (terminal khác)
tail -f logs/train.log
watch -n 5 nvidia-smi
```

---

## 12. KẾT QUẢ MẪU

Sau khi training xong, bạn sẽ có:

```
checkpoints/
├── checkpoint_epoch_5.pt
├── checkpoint_epoch_10.pt
├── checkpoint_epoch_15.pt
├── ...
├── checkpoint_epoch_50.pt
├── best_model.pt              # Model tốt nhất
└── test_results.json          # Kết quả test

logs/
└── train.log                  # Training logs

tensorboard/                   # TensorBoard logs (nếu enabled)
```

**test_results.json**:
```json
{
  "recall@10": 0.4523,
  "recall@50": 0.7234,
  "recall@100": 0.8612,
  "mrr@10": 0.4123,
  "ndcg@10": 0.4567,
  "map": 0.3987,
  "precision@10": 0.3456
}
```

---

**Chúc bạn training thành công!** 🚀

Nếu gặp vấn đề, hãy:
1. Kiểm tra logs: `logs/train.log`
2. Xem lại phần Troubleshooting
3. Giảm config xuống để test trên subset nhỏ
4. Mở issue trên GitHub repo
