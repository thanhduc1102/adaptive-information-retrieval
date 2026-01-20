# HƯỚNG DẪN TRAIN NHANH - TEST VỚI SUBSET NHỎ

Tài liệu này hướng dẫn train nhanh với subset nhỏ để test hệ thống (10-30 phút thay vì hàng chục giờ).

---

## 🎯 Mục tiêu

- ✅ Verify code hoạt động đúng
- ✅ Xem kết quả ban đầu như thế nào
- ✅ Test trên laptop/máy yếu
- ✅ Debug nhanh
- ⏱️ Thời gian: **10-30 phút** (thay vì 25-37 giờ)

---

## 🚀 OPTION 1: TRAIN CỰC NHANH (10-15 phút)

### Cài đặt dependencies (nếu chưa có)

```bash
cd /Users/vanhkhongpeo/Documents/Github/Adaptive_information_retrival/adaptive-information-retrieval/adaptive-ir-system

# Cài packages cơ bản
pip install torch numpy scikit-learn h5py rank-bm25 nltk tqdm pyyaml

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Train với 500 queries, 5 epochs

```bash
# Training với 500 queries, 5 epochs (~10-15 phút)
python train_quick_test.py \
  --config configs/msa_quick_test.yaml \
  --num_samples 500 \
  --epochs 5 \
  --device cuda

# Hoặc CPU (chậm hơn ~3x)
python train_quick_test.py \
  --config configs/msa_quick_test.yaml \
  --num_samples 500 \
  --epochs 5 \
  --device cpu
```

**Thời gian ước tính**:
- GPU: ~10-15 phút
- CPU: ~30-45 phút

---

## 🚀 OPTION 2: TRAIN VỪA PHẢI (20-30 phút)

### Train với 1000 queries, 10 epochs

```bash
# Training với 1000 queries, 10 epochs (~20-30 phút)
python train_quick_test.py \
  --config configs/msa_quick_test.yaml \
  --num_samples 1000 \
  --epochs 10 \
  --device cuda
```

**Thời gian ước tính**:
- GPU: ~20-30 phút
- CPU: ~60-90 phút

---

## 🚀 OPTION 3: TRAIN ĐẦY ĐỦ HƠN (1-2 giờ)

### Train với 5000 queries, 10 epochs

```bash
# Training với 5000 queries, 10 epochs (~1-2 giờ)
python train_quick_test.py \
  --config configs/msa_quick_test.yaml \
  --num_samples 5000 \
  --epochs 10 \
  --device cuda
```

**Thời gian ước tính**:
- GPU: ~1-2 giờ
- CPU: ~4-6 giờ

---

## 📊 SO SÁNH CÁC OPTION

| Option | Queries | Epochs | Thời gian (GPU) | Thời gian (CPU) | Khi nào dùng |
|--------|---------|--------|-----------------|-----------------|--------------|
| **1. Cực nhanh** | 500 | 5 | 10-15 min | 30-45 min | Test code, debug |
| **2. Vừa phải** | 1,000 | 10 | 20-30 min | 60-90 min | Xem kết quả sơ bộ |
| **3. Đầy đủ hơn** | 5,000 | 10 | 1-2 giờ | 4-6 giờ | Kết quả đáng tin hơn |
| **Full** | 271,345 | 50 | 25-37 giờ | Nhiều ngày | Production |

---

## 🔧 CÁC THAY ĐỔI ĐỂ TRAIN NHANH

Config `msa_quick_test.yaml` đã tối ưu để train nhanh:

### 1. Giảm kích thước model
```yaml
rl_agent:
  hidden_dim: 128              # Từ 256 → 128
  num_attention_heads: 2       # Từ 4 → 2
  num_encoder_layers: 1        # Từ 2 → 1
```

### 2. Giảm số query variants
```yaml
rl_agent:
  max_steps_per_episode: 3     # Từ 5 → 3
  num_query_variants: 2        # Từ 4 → 2
```

### 3. Giảm số candidates
```yaml
candidate_mining:
  max_candidates: 30           # Từ 50 → 30
  top_k_docs: 5                # Từ 10 → 5
```

### 4. Tắt BERT re-ranking
```yaml
bert_reranker:
  enabled: false               # Tắt hoàn toàn
```

### 5. Giảm tham số training
```yaml
training:
  batch_size: 16               # Từ 32 → 16
  episodes_per_update: 64      # Từ 128 → 64
  ppo_epochs: 2                # Từ 4 → 2
```

---

## 📈 KẾT QUẢ MONG ĐỢI

### Với 500 queries, 5 epochs:
- **Mục đích**: Verify code hoạt động
- **Kết quả**: Có thể chưa tốt, reward có thể âm hoặc gần 0
- **Chấp nhận được**: Chỉ cần không lỗi

### Với 1000 queries, 10 epochs:
- **Recall@10**: 0.25-0.30 (baseline: ~0.28)
- **Recall@40**: 0.40-0.45 (baseline: ~0.42)
- **MRR@10**: 0.18-0.22 (baseline: ~0.19)
- **Kết quả**: Có thể ngang hoặc hơi tốt hơn baseline

### Với 5000 queries, 10 epochs:
- **Recall@10**: 0.30-0.35 (baseline: ~0.28)
- **Recall@40**: 0.45-0.50 (baseline: ~0.42)
- **MRR@10**: 0.22-0.26 (baseline: ~0.19)
- **Kết quả**: Đáng tin cậy hơn, thấy được cải thiện

---

## 📝 LOG OUTPUT MẪU

```
================================================================================
Adaptive IR System - QUICK TEST Training
================================================================================
⚠️  SUBSET MODE: Training with 1,000 queries only
Random seed: 42
Device: cuda
Loading datasets...
Train queries: 1000
Val queries: 20000
Initializing search engine...
Search engine: SimpleBM25 (legacy dataset)
Building pipeline...
Loaded legacy Word2Vec embeddings from ../Query Reformulator/D_cbow_pdw_8B.pkl
  Embeddings: 374,557 words
Initialized Candidate Term Miner
Initialized RL Agent with 523,264 parameters (reduced model)
Initialized RRF Fusion (k=60)
BERT Re-ranker: DISABLED (for speed)
Pipeline initialized successfully
Starting training...
Epochs: 10
Batch size: 16
Episodes per update: 64
⚠️  Training queries: 1,000 (subset)
--------------------------------------------------------------------------------
Epoch 1/10: 100%|████████| 1000/1000 [1:32<00:00, 10.8it/s, reward=-0.0023, episodes=64]
Epoch 1/10 | Reward: -0.0023 | Loss: 0.3456
Epoch 2/10: 100%|████████| 1000/1000 [1:28<00:00, 11.3it/s, reward=0.0012, episodes=64]
Epoch 2/10 | Reward: 0.0012 | Loss: 0.2234
Validation | Recall@10: 0.2845 | Recall@40: 0.4312 | MRR@10: 0.1923
Saved checkpoint to checkpoints_msa_quick/checkpoint_epoch_2.pt
...
Epoch 10/10: 100%|████████| 1000/1000 [1:25<00:00, 11.7it/s, reward=0.0234, episodes=64]
Epoch 10/10 | Reward: 0.0234 | Loss: 0.1123
Validation | Recall@10: 0.3123 | Recall@40: 0.4678 | MRR@10: 0.2245
Saved best model with MRR@10: 0.2245
================================================================================
Training completed!
================================================================================
Total time: 15 minutes 23 seconds
```

---

## 🔍 KIỂM TRA KẾT QUẢ

### 1. Xem checkpoints

```bash
ls -lh checkpoints_msa_quick/

# Output:
# checkpoint_epoch_2.pt
# checkpoint_epoch_4.pt
# checkpoint_epoch_6.pt
# checkpoint_epoch_8.pt
# checkpoint_epoch_10.pt
# best_model.pt
```

### 2. Test model

```bash
# Test inference
python << 'EOF'
import sys
sys.path.insert(0, 'src')

from src.pipeline import AdaptiveIRPipeline
from src.utils.legacy_embeddings import LegacyEmbeddingAdapter
from src.utils.simple_searcher import SimpleBM25Searcher
from src.utils.data_loader import DatasetFactory
import yaml

# Load config
with open('configs/msa_quick_test.yaml') as f:
    config = yaml.safe_load(f)

# Load data
dataset_factory = DatasetFactory(config['data'])
dataset = dataset_factory.create_dataset('train')

# Setup searcher
searcher = SimpleBM25Searcher(dataset)

# Load embeddings
embeddings = LegacyEmbeddingAdapter('../Query Reformulator/D_cbow_pdw_8B.pkl')

# Initialize pipeline
pipeline = AdaptiveIRPipeline(
    config=config,
    search_engine=searcher,
    embedding_model=embeddings
)

# Load best model
pipeline.load_rl_checkpoint('checkpoints_msa_quick/best_model.pt')

# Test query
query = "deep learning neural networks"
result = pipeline.search(query, top_k=10, measure_latency=True)

print(f"Query: {result['query']}")
print(f"\nQuery variants:")
for i, variant in enumerate(result['query_variants'], 1):
    print(f"  {i}. {variant}")

print(f"\nTop 5 results:")
for i, (doc_id, score) in enumerate(result['results'][:5], 1):
    print(f"  {i}. [Score: {score:.4f}] Doc ID: {doc_id}")

print(f"\nLatency:")
for stage, time_ms in result['latency'].items():
    print(f"  {stage}: {time_ms:.1f}ms")
EOF
```

### 3. So sánh với baseline

```bash
# Đánh giá trên validation set
python << 'EOF'
from src.evaluation import IRMetricsAggregator

# Kết quả quick test
quick_metrics = {
    'recall@10': 0.3123,
    'recall@40': 0.4678,
    'mrr@10': 0.2245
}

# Baseline (BM25 only)
baseline_metrics = {
    'recall@10': 0.28,
    'recall@40': 0.42,
    'mrr@10': 0.19
}

print("So sánh Quick Test vs Baseline:")
print("=" * 60)
for metric in baseline_metrics:
    baseline = baseline_metrics[metric]
    quick = quick_metrics[metric]
    improvement = (quick - baseline) / baseline * 100
    status = "✓" if improvement > 0 else "✗"
    print(f"{status} {metric:15s}: {baseline:.4f} → {quick:.4f} ({improvement:+.1f}%)")
EOF
```

---

## 💡 TIPS

### 1. Nếu muốn nhanh hơn nữa

```bash
# Chỉ 200 queries, 3 epochs (~5 phút)
python train_quick_test.py \
  --num_samples 200 \
  --epochs 3 \
  --device cuda
```

### 2. Nếu bị Out of Memory

```yaml
# Sửa config:
training:
  batch_size: 8       # Giảm từ 16 → 8

rl_agent:
  hidden_dim: 64      # Giảm từ 128 → 64
```

### 3. Monitor trong khi train

```bash
# Terminal 1: Training
python train_quick_test.py --num_samples 1000 --epochs 10

# Terminal 2: Watch logs
tail -f logs_msa_quick/train.log

# Terminal 3: GPU usage (nếu dùng GPU)
watch -n 1 nvidia-smi
```

### 4. Tắt output để nhanh hơn

```bash
# Redirect output
python train_quick_test.py \
  --num_samples 1000 \
  --epochs 10 \
  > /dev/null 2>&1
```

---

## ⚙️ TÙY CHỈNH SỐ SAMPLES

### Qua command line (Khuyến nghị)

```bash
# 500 queries
python train_quick_test.py --num_samples 500 --epochs 5

# 1000 queries
python train_quick_test.py --num_samples 1000 --epochs 10

# 5000 queries
python train_quick_test.py --num_samples 5000 --epochs 10
```

### Hoặc sửa code `train.py`

Thêm vào file `train.py` sau dòng 343:

```python
# Chỉ lấy 1000 queries đầu tiên
query_ids = list(train_queries.keys())
np.random.shuffle(query_ids)
query_ids = query_ids[:1000]  # <-- THÊM DÒNG NÀY
```

---

## 🎯 LỆNH HOÀN CHỈNH

### Setup (one-time)

```bash
cd /Users/vanhkhongpeo/Documents/Github/Adaptive_information_retrival/adaptive-information-retrieval/adaptive-ir-system

# Cài packages
pip install torch numpy scikit-learn h5py rank-bm25 nltk tqdm pyyaml

# NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Quick Test (10-15 phút)

```bash
# Test nhanh nhất
python train_quick_test.py \
  --config configs/msa_quick_test.yaml \
  --num_samples 500 \
  --epochs 5 \
  --device cuda \
  2>&1 | tee quick_test.log

# Xem kết quả
grep "Validation" quick_test.log
grep "best model" quick_test.log
```

### Test Inference

```bash
# Load model và test
python inference.py \
  --config configs/msa_quick_test.yaml \
  --checkpoint checkpoints_msa_quick/best_model.pt \
  --query "machine learning algorithms"
```

---

## ❓ FAQ

### Q: Kết quả có đáng tin không?
**A**: Với 500-1000 queries: Chỉ để verify code. Với 5000+ queries: Đáng tin hơn nhưng vẫn không bằng full training.

### Q: Có nên train full sau khi test?
**A**: Nên! Kết quả quick test chỉ để verify. Train full sẽ tốt hơn nhiều.

### Q: Tại sao reward âm?
**A**: Bình thường ở epoch đầu. RL agent đang học, chưa tốt hơn baseline.

### Q: Bao lâu thì thấy improvement?
**A**: Với 1000 queries: Epoch 5-10. Với 500 queries: Có thể không thấy.

---

## 📊 BẢNG TÓM TẮT

| Mục đích | Queries | Epochs | Thời gian | Lệnh |
|----------|---------|--------|-----------|------|
| **Verify code** | 200-500 | 3-5 | 5-15 min | `--num_samples 500 --epochs 5` |
| **Xem kết quả sơ bộ** | 1,000 | 10 | 20-30 min | `--num_samples 1000 --epochs 10` |
| **Kết quả tin cậy** | 5,000 | 10 | 1-2 giờ | `--num_samples 5000 --epochs 10` |
| **Production** | 271,345 | 50 | 25-37 giờ | Dùng `train.py` thường |

---

**Chúc bạn test thành công!** 🚀

Sau khi verify code hoạt động, bạn có thể:
1. Train với nhiều queries hơn (5000-10000)
2. Tăng epochs (20-30)
3. Hoặc train full với 271K queries
