# QUICK START: TRAINING VỚI MSA DATASET

Data đã sẵn sàng! Hướng dẫn training nhanh.

---

## ✅ Checklist Data

- ✅ `msa_dataset.hdf5`: 452M - 271,345 training queries
- ✅ `msa_corpus.hdf5`: 459M - 480,722 documents
- ✅ `D_cbow_pdw_8B.pkl`: 732M - 374,557 words embeddings (500-dim)

---

## 🚀 BƯỚC 1: CÀI ĐẶT DEPENDENCIES

```bash
cd /Users/vanhkhongpeo/Documents/Github/Adaptive_information_retrival/adaptive-information-retrieval/adaptive-ir-system

# Tạo virtual environment (nếu chưa có)
python3 -m venv venv
source venv/bin/activate

# Cài packages
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## 🚀 BƯỚC 2: KIỂM TRA CONFIG

File config đã có sẵn tại `configs/msa_config.yaml`:

```yaml
# configs/msa_config.yaml
system:
  device: 'cuda'      # Đổi thành 'cpu' nếu không có GPU
  seed: 42

data:
  dataset_type: 'msa'
  data_dir: '../Query Reformulator'

embeddings:
  type: 'legacy'
  path: '../Query Reformulator/D_cbow_pdw_8B.pkl'
  embedding_dim: 500

rl_agent:
  embedding_dim: 500
  hidden_dim: 256
  max_steps_per_episode: 5
  num_query_variants: 4

training:
  num_epochs: 50
  batch_size: 32
  learning_rate: 0.0003
  episodes_per_update: 128
```

**Nếu máy yếu**, sửa config:
```yaml
system:
  device: 'cpu'       # Dùng CPU

rl_agent:
  hidden_dim: 128     # Giảm từ 256 → 128

training:
  batch_size: 16      # Giảm từ 32 → 16
  num_epochs: 20      # Test với 20 epochs trước
```

---

## 🚀 BƯỚC 3: TEST DATA LOADER

Trước khi training, test xem data load đúng không:

```bash
cd adaptive-ir-system

# Test load data
python scripts/test_legacy_data.py --data_dir "../Query Reformulator"
```

**Expected output**:
```
Testing legacy dataset: msa
✓ Dataset loaded successfully
  Queries: 271,345
  Documents: 480,722
  Embeddings: 374,557 words

Sample query: "the metabolic code"
Sample doc: "hybrid compactifications and brane gravity..."
✓ All data valid!
```

---

## 🚀 BƯỚC 4: TRAINING THẬT

### Option A: Training đầy đủ (GPU khuyến nghị)

```bash
cd adaptive-ir-system

# Training với GPU
python train.py \
  --config configs/msa_config.yaml \
  --device cuda \
  --epochs 50

# Log sẽ ở: logs_msa/train.log
# Checkpoints sẽ ở: checkpoints_msa/
```

**Thời gian ước tính**:
- **GPU (RTX 3090)**: ~30-45 phút/epoch → 50 epochs = **25-37 giờ**
- **GPU (V100)**: ~20-30 phút/epoch → 50 epochs = **17-25 giờ**
- **CPU**: ~2-4 giờ/epoch → Không khuyến nghị

### Option B: Test training với subset nhỏ (CPU OK)

```bash
# Test với 1000 queries, 5 epochs
python train.py \
  --config configs/msa_config.yaml \
  --device cpu \
  --epochs 5
```

Trong file `train.py`, thêm dòng này sau dòng 343:
```python
query_ids = query_ids[:1000]  # Chỉ lấy 1000 queries
```

---

## 🚀 BƯỚC 5: GIÁM SÁT TRAINING

### Terminal 1: Chạy training
```bash
python train.py --config configs/msa_config.yaml
```

### Terminal 2: Theo dõi logs
```bash
# Xem log realtime
tail -f logs_msa/train.log

# Hoặc dùng grep để lọc
tail -f logs_msa/train.log | grep "Epoch"
tail -f logs_msa/train.log | grep "Validation"
```

### Terminal 3: Monitor GPU (nếu dùng GPU)
```bash
watch -n 1 nvidia-smi
```

---

## 📊 BƯỚC 6: ĐÁNH GIÁ KẾT QUẢ

Sau khi training xong, kiểm tra checkpoints:

```bash
# List checkpoints
ls -lh checkpoints_msa/

# Output:
# checkpoint_epoch_5.pt
# checkpoint_epoch_10.pt
# ...
# best_model.pt  ← Model tốt nhất
```

Test model:

```bash
# Test với checkpoint tốt nhất
python inference.py \
  --config configs/msa_config.yaml \
  --checkpoint checkpoints_msa/best_model.pt \
  --query "machine learning deep neural networks"
```

---

## 📈 KẾT QUẢ MONG ĐỢI

### Baseline (BM25 only):
- Recall@10: ~0.30
- Recall@40: ~0.45
- MRR@10: ~0.20

### Sau training RL:
- Recall@10: ~0.35-0.40 (+17-33%)
- Recall@40: ~0.50-0.55 (+11-22%)
- MRR@10: ~0.25-0.30 (+25-50%)

---

## 🔧 TROUBLESHOOTING

### Lỗi: "No module named 'rank_bm25'"
```bash
pip install rank-bm25
```

### Lỗi: Out of Memory
```yaml
# Sửa config:
training:
  batch_size: 8       # Giảm từ 32
rl_agent:
  hidden_dim: 128     # Giảm từ 256
```

### Lỗi: Training quá chậm
```yaml
# Tắt BERT re-ranker trong training
bert_reranker:
  enabled: false

# Giảm candidates
candidate_mining:
  max_candidates: 50  # Từ 100
```

### Lỗi: Can't load embeddings
```python
# Test embeddings
import pickle
with open('../Query Reformulator/D_cbow_pdw_8B.pkl', 'rb') as f:
    emb = pickle.load(f, encoding='latin1')
    print(f"Loaded {len(emb)} words")
```

---

## 📝 LOG OUTPUT MẪU

```
================================================================================
Adaptive IR System - Training
================================================================================
Random seed: 42
Device: cuda
Loading datasets...
Train queries: 271345
Val queries: 20000
Initializing search engine...
Search engine: SimpleBM25 (legacy dataset)
Building pipeline...
Loaded legacy Word2Vec embeddings from ../Query Reformulator/D_cbow_pdw_8B.pkl
Initialized Candidate Term Miner
Initialized RL Agent with 1,245,632 parameters
Initialized RRF Fusion (k=60)
Initialized BERT Re-ranker
Pipeline initialized successfully
Starting training...
Epochs: 50
Batch size: 32
Episodes per update: 128
--------------------------------------------------------------------------------
Epoch 1/50: 100%|████| 271345/271345 [38:24<00:00, 117.8it/s, reward=0.0123, episodes=2048]
Epoch 1/50 | Reward: 0.0123 | Loss: 0.2341
Epoch 5/50: 100%|████| 271345/271345 [36:12<00:00, 124.9it/s, reward=0.0567, episodes=2048]
Epoch 5/50 | Reward: 0.0567 | Loss: 0.1234
Validation | Recall@10: 0.3234 | Recall@40: 0.4567 | MRR@10: 0.2123
Saved best model with MRR@10: 0.2123
...
```

---

## 🎯 LỆNH HOÀN CHỈNH

```bash
# Setup
cd /Users/vanhkhongpeo/Documents/Github/Adaptive_information_retrival/adaptive-information-retrieval/adaptive-ir-system
source venv/bin/activate  # Nếu dùng venv

# Install (one-time)
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Test data
python scripts/test_legacy_data.py --data_dir "../Query Reformulator"

# Training
python train.py \
  --config configs/msa_config.yaml \
  --device cuda \
  --epochs 50 \
  2>&1 | tee training.log

# Monitor (terminal khác)
tail -f logs_msa/train.log
watch -n 1 nvidia-smi

# Evaluate
python inference.py \
  --config configs/msa_config.yaml \
  --checkpoint checkpoints_msa/best_model.pt \
  --query "deep learning neural networks"
```

---

## 💡 TIPS

1. **Test trước với subset nhỏ**: Sửa `query_ids = query_ids[:1000]` trong `train.py`
2. **Dùng CPU cho test nhanh**: `--device cpu --epochs 5`
3. **Save checkpoints thường xuyên**: Config đã set `save_freq: 5`
4. **Monitor GPU**: Đảm bảo GPU utilization > 80%
5. **Early stopping**: Sẽ tự động dừng nếu không cải thiện sau 10 epochs

---

**Chúc bạn training thành công!** 🚀

Nếu gặp vấn đề, check:
1. `logs_msa/train.log`
2. Phần Troubleshooting ở trên
3. Hoặc hỏi tôi!
