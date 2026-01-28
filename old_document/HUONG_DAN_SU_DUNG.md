# 📚 HƯỚNG DẪN SỬ DỤNG - ADAPTIVE IR TRAINING PIPELINE

## 🎯 Tổng quan

Chương trình huấn luyện Adaptive IR System với các tính năng:
- **3 chế độ chạy**: quick, medium, full
- **Checkpoint management**: Tự động lưu và resume
- **Early stopping**: Dừng sớm nếu không cải thiện
- **Logging**: Log chi tiết ra file và console
- **Config linh hoạt**: Qua command line, file JSON, hoặc code

---

## 🚀 Cách chạy nhanh

### 1. Quick Test (~5-10 phút)
```bash
cd /kaggle/adaptive-information-retrieval/adaptive-ir-system
python train_full_epoch.py --mode quick
```

### 2. Medium Training (~30-60 phút)
```bash
python train_full_epoch.py --mode medium
```

### 3. Full Epoch Training (vài giờ)
```bash
python train_full_epoch.py --mode full
```

---

## ⚙️ Cấu hình chi tiết

### Qua Command Line

```bash
# Tùy chỉnh epochs và batch size
python train_full_epoch.py --mode medium --epochs 3 --batch-size 128

# Tùy chỉnh learning rate
python train_full_epoch.py --lr 1e-4

# Tùy chỉnh evaluation
python train_full_epoch.py --eval-every 500 --num-eval-queries 200

# Tắt early stopping
python train_full_epoch.py --no-early-stopping

# Resume từ checkpoint
python train_full_epoch.py --resume checkpoints/latest.pt
```

### Qua File JSON

```bash
# Tạo config
python -c "
from train_full_epoch import TrainingConfig
config = TrainingConfig(mode='medium')
config.epochs = 3
config.batch_size = 128
config.learning_rate = 1e-4
config.to_json('my_config.json')
"

# Chạy với config
python train_full_epoch.py --config my_config.json
```

### Chỉnh sửa trực tiếp trong code

Mở file `train_full_epoch.py`, tìm class `TrainingConfig` và sửa:

```python
@dataclass
class TrainingConfig:
    # =========================================================================
    # CHẾ ĐỘ CHẠY
    # =========================================================================
    mode: str = 'medium'  # Đổi từ 'quick' sang 'medium' hoặc 'full'
    
    # =========================================================================
    # HUẤN LUYỆN
    # =========================================================================
    epochs: int = 3               # Tăng số epochs
    batch_size: int = 128         # Tăng batch size
    learning_rate: float = 1e-4   # Giảm learning rate
```

---

## 📊 Các chế độ chi tiết

| Chế độ | Train Queries | Eval Queries | Eval Every | Thời gian ước tính |
|--------|---------------|--------------|------------|-------------------|
| quick  | 500           | 100          | 200        | 5-10 phút         |
| medium | 5,000         | 300          | 1,000      | 30-60 phút        |
| full   | ALL (~270k)   | 1,000        | 5,000      | 5-10 giờ          |

---

## 📁 Cấu trúc thư mục output

```
adaptive-ir-system/
├── checkpoints/
│   ├── config.json          # Config đã sử dụng
│   ├── latest.pt            # Checkpoint mới nhất
│   ├── best_model.pt        # Model tốt nhất
│   └── step_XXXXX.pt        # Checkpoint theo step
└── logs/
    ├── train_YYYYMMDD_HHMMSS.log   # Log file
    └── training_history.json        # Training history
```

---

## 🔧 Các tham số quan trọng

### 1. Model Architecture

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `embedding_dim` | 500 | Kích thước Word2Vec embedding |
| `hidden_dim` | 256 | Kích thước hidden layer |
| `num_heads` | 4 | Số attention heads |
| `num_layers` | 2 | Số Transformer layers |
| `dropout` | 0.1 | Dropout rate |

### 2. Training Hyperparameters

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `batch_size` | 64 | Số samples mỗi batch |
| `learning_rate` | 3e-4 | Learning rate |
| `weight_decay` | 0.01 | L2 regularization |
| `max_grad_norm` | 0.5 | Gradient clipping |

### 3. PPO Hyperparameters

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `ppo_epochs` | 3 | PPO update epochs |
| `clip_epsilon` | 0.2 | PPO clipping |
| `gamma` | 0.99 | Discount factor |
| `entropy_coef` | 0.01 | Entropy bonus |
| `update_every` | 512 | Update sau N samples |

### 4. Retrieval

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `max_candidates` | 20 | Candidates cho agent |
| `top_k_retrieve` | 100 | Top-k BM25 results |
| `rrf_k` | 60 | RRF constant |
| `bm25_k1` | 0.9 | BM25 k1 parameter |
| `bm25_b` | 0.4 | BM25 b parameter |

### 5. Early Stopping

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `early_stopping` | True | Có dùng early stopping |
| `patience` | 5 | Số lần không cải thiện |
| `min_delta` | 0.001 | Ngưỡng cải thiện tối thiểu |

---

## 📈 Theo dõi training

### 1. Xem log realtime
```bash
tail -f logs/train_*.log
```

### 2. Xem GPU usage
```bash
watch -n 1 nvidia-smi
```

### 3. Load training history
```python
import json
with open('logs/training_history.json') as f:
    history = json.load(f)

# Plot rewards
import matplotlib.pyplot as plt
steps = [h['step'] for h in history['train_history']]
rewards = [h['reward'] for h in history['train_history']]
plt.plot(steps, rewards)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.show()
```

---

## 🔄 Resume Training

```bash
# Resume từ checkpoint mới nhất
python train_full_epoch.py --resume checkpoints/latest.pt

# Resume từ best model
python train_full_epoch.py --resume checkpoints/best_model.pt

# Resume với config khác
python train_full_epoch.py --resume checkpoints/latest.pt --epochs 5
```

---

## 🧪 Evaluation riêng

```python
from train_full_epoch import *

# Load model
config = TrainingConfig.from_json('checkpoints/config.json')
data = DataManager(config)
data.load_all()

search = BM25SearchEngine(data, config)
search.build_index()

agent = QueryReformulationAgent(config).to('cuda')
checkpoint = torch.load('checkpoints/best_model.pt')
agent.load_state_dict(checkpoint['model_state_dict'])

# Evaluate
trainer = Trainer(agent, data, search, config)
metrics = trainer.evaluate('test', max_queries=1000)
print(metrics)
```

---

## ❗ Xử lý lỗi thường gặp

### 1. Out of Memory (OOM)
```bash
# Giảm batch size
python train_full_epoch.py --batch-size 32
```

### 2. Training quá chậm
```bash
# Tăng update_every để giảm PPO updates
# Sửa trong code: update_every = 1024
```

### 3. Reward không tăng
- Thử giảm learning rate: `--lr 1e-4`
- Tăng entropy coefficient trong config

### 4. Early stopping quá sớm
```bash
# Tắt early stopping
python train_full_epoch.py --no-early-stopping

# Hoặc tăng patience trong config
```

---

## 📝 Ví dụ configs

### Config cho GPU yếu (4GB)
```python
config = TrainingConfig(
    mode='quick',
    batch_size=16,
    max_candidates=10,
    update_every=256
)
```

### Config cho training dài
```python
config = TrainingConfig(
    mode='full',
    epochs=5,
    batch_size=128,
    learning_rate=1e-4,
    early_stopping=False
)
```

### Config cho debugging
```python
config = TrainingConfig(
    mode='quick',
    eval_every=50,
    log_every=10,
    save_every=100
)
```

---

## 🎯 Tips & Best Practices

1. **Bắt đầu với mode='quick'** để verify setup hoạt động
2. **Monitor GPU memory** với `nvidia-smi`
3. **Check baseline metrics** trước khi train dài
4. **Save config** để reproduce experiments
5. **Use early stopping** để tránh overfitting
6. **Log everything** để debug

---

## 📞 Troubleshooting

Nếu gặp vấn đề:
1. Check logs trong `logs/train_*.log`
2. Verify data paths trong config
3. Check GPU memory với `nvidia-smi`
4. Try với mode='quick' trước

---

*Chúc bạn training thành công! 🚀*
