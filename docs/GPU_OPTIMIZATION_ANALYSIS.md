# 📊 Phân Tích Chi Tiết Hệ Thống RL Training & Tối Ưu GPU

## 1. PHÂN TÍCH VẤN ĐỀ: TẠI SAO GPU CHỈ SỬ DỤNG 2%?

### 1.1 Luồng Training Hiện Tại (train_rl.py)

```
┌─────────────────────────────────────────────────────────────────┐
│                    LUỒNG TRAINING CŨ                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  for query in queries:           ← XỬ LÝ TUẦN TỰ TỪNG QUERY    │
│      │                                                           │
│      ▼                                                           │
│  ┌────────────────┐                                             │
│  │ Mine Candidates │ ← BM25 search + TF-IDF (CPU-bound)         │
│  └────────────────┘                                             │
│      │                                                           │
│      ▼                                                           │
│  ┌────────────────┐                                             │
│  │ Embed Query    │ ← Tính toán embedding (có thể cache)        │
│  └────────────────┘                                             │
│      │                                                           │
│      ▼                                                           │
│  ┌────────────────┐                                             │
│  │ Embed Candidates│ ← LẶP LẠI cho mỗi query (redundant!)       │
│  └────────────────┘                                             │
│      │                                                           │
│      ▼                                                           │
│  for step in range(5):           ← XỬ LÝ TUẦN TỰ TỪNG STEP     │
│      │                                                           │
│      ▼                                                           │
│  ┌────────────────┐                                             │
│  │ RL Forward     │ ← BATCH SIZE = 1 (không batching!)          │
│  │ (select_action)│   GPU chỉ xử lý 1 sample tại 1 thời điểm    │
│  └────────────────┘                                             │
│      │                                                           │
│      ▼                                                           │
│  ┌────────────────┐                                             │
│  │ Search & Eval  │ ← BM25 search lại cho mỗi step (slow!)      │
│  │ (reward)       │                                              │
│  └────────────────┘                                             │
│      │                                                           │
│      ▼                                                           │
│  ┌────────────────┐                                             │
│  │ Store to Buffer│ ← CPU memory, transfer overhead             │
│  └────────────────┘                                             │
│                                                                  │
│  if episode_count % 128 == 0:                                    │
│      ┌────────────────┐                                         │
│      │ PPO Update     │ ← Chỉ update sau 128 episodes           │
│      │ (batch=32)     │   GPU idle phần lớn thời gian!          │
│      └────────────────┘                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 CÁC BOTTLENECK CHÍNH

| Vấn đề | Mô tả | Impact |
|--------|-------|--------|
| **Sequential Processing** | Xử lý từng query một | GPU idle 90%+ |
| **No Batching in Collection** | `select_action` với batch_size=1 | GPU utilization < 5% |
| **Repeated Embeddings** | Tính embedding lại cho mỗi query/term | Redundant computation |
| **CPU-GPU Transfer** | Liên tục chuyển data giữa CPU và GPU | High latency |
| **Search Bottleneck** | BM25 search blocking trong mỗi step | CPU-bound operation |
| **Small PPO Batches** | batch_size=32 cho PPO update | Underutilize GPU memory |

### 1.3 VÍ DỤ CỤ THỂ

Giả sử có 1000 queries, mỗi episode có 5 steps:

**Code cũ:**
```python
# Mỗi query xử lý riêng lẻ
for query_id in query_ids:  # 1000 lần
    trajectory, reward = self.collect_episode(query, qrels)  # Sequential
    
    # Trong collect_episode():
    for step in range(5):  # 5 steps
        # GPU forward với batch_size=1!
        action = self.rl_agent.select_action(
            query_emb.unsqueeze(0),  # [1, 512]
            current_emb.unsqueeze(0),  # [1, 512]
            candidate_embs.unsqueeze(0),  # [1, 50, 512]
            ...
        )
        # → GPU nhận input rất nhỏ, phần lớn cores idle
```

**Thời gian ước tính:**
- 1000 queries × 5 steps = 5000 forward passes
- Mỗi forward pass: ~10ms (chủ yếu là overhead, không phải computation)
- Tổng: ~50 giây chỉ cho forward passes
- GPU utilization: < 5%

---

## 2. GIẢI PHÁP TỐI ƯU

### 2.1 KIẾN TRÚC MỚI

```
┌─────────────────────────────────────────────────────────────────┐
│                    LUỒNG TRAINING MỚI (Optimized)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │           PHASE 1: PRE-COMPUTATION (1 lần)                  ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │  • Pre-compute ALL query embeddings                         ││
│  │  • Cache trong GPU memory                                    ││
│  │  • Hash-based lookup                                         ││
│  └─────────────────────────────────────────────────────────────┘│
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │           PHASE 2: PARALLEL PREPARATION                      ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │  ThreadPoolExecutor(workers=4):                              ││
│  │    • Mine candidates cho N queries đồng thời                 ││
│  │    • Batch embed candidates                                  ││
│  │    • Chuẩn bị EpisodeData objects                           ││
│  └─────────────────────────────────────────────────────────────┘│
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │           PHASE 3: BATCHED COLLECTION                        ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │  for batch in batches(episodes, size=32):                    ││
│  │      │                                                       ││
│  │      ▼                                                       ││
│  │  ┌────────────────────────────────────────────┐             ││
│  │  │ BATCHED RL FORWARD                         │             ││
│  │  │ • query_embs: [32, 512]                    │             ││
│  │  │ • candidate_embs: [32, 50, 512]            │             ││
│  │  │ → GPU processes 32 samples simultaneously! │             ││
│  │  └────────────────────────────────────────────┘             ││
│  │      │                                                       ││
│  │      ▼                                                       ││
│  │  ┌────────────────────────────────────────────┐             ││
│  │  │ CACHED REWARD COMPUTATION                   │             ││
│  │  │ • Cache search results                      │             ││
│  │  │ • Avoid repeated searches                   │             ││
│  │  └────────────────────────────────────────────┘             ││
│  └─────────────────────────────────────────────────────────────┘│
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │           PHASE 4: OPTIMIZED PPO UPDATE                      ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │  • GPU-resident replay buffer                                ││
│  │  • Mixed precision (FP16)                                    ││
│  │  • Large batch size (64-128)                                 ││
│  │  • Mini-batch updates                                        ││
│  │  • Multi-GPU DataParallel                                    ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 CÁC TỐI ƯU CỤ THỂ

#### A. EMBEDDING CACHE

```python
class EmbeddingCache:
    """
    Cache embeddings để tránh tính toán lại.
    
    Trước: Mỗi query "machine learning" được embed 5 lần (mỗi step)
    Sau: Embed 1 lần, lookup từ cache
    
    Tiết kiệm: ~80% computation cho embeddings
    """
    
    def __init__(self, max_size=200000):
        self.cache = {}  # hash -> embedding tensor
        
    def get(self, text: str) -> torch.Tensor:
        key = hash(text)
        if key in self.cache:
            return self.cache[key]  # O(1) lookup
        
        # Compute và cache
        embedding = self.embed_model.encode(text)
        self.cache[key] = embedding
        return embedding
    
    def get_batch(self, texts: List[str]) -> torch.Tensor:
        """Batch compute cho efficiency."""
        # Check cache first
        # Batch encode missing texts
        # Much faster than individual encodes!
```

#### B. GPU-RESIDENT REPLAY BUFFER

```python
class OptimizedReplayBuffer:
    """
    Lưu trực tiếp trên GPU để tránh transfer.
    
    Trước: 
      - Store on CPU
      - Sample → transfer to GPU
      - Overhead: ~5ms per batch
    
    Sau:
      - Pre-allocate trên GPU
      - Sample trực tiếp
      - Overhead: ~0.1ms per batch
    """
    
    def __init__(self, capacity, device='cuda'):
        # Pre-allocate trên GPU
        self.query_embs = torch.zeros(capacity, 512, device=device)
        self.candidate_embs = torch.zeros(capacity, 50, 512, device=device)
        # ...
        
    def sample(self, batch_size) -> Dict[str, torch.Tensor]:
        # Không cần .to(device)!
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            'query_emb': self.query_embs[indices],  # Already on GPU
            # ...
        }
```

#### C. BATCHED EPISODE COLLECTION

```python
def collect_batch_episodes(self, episode_data_list, batch_size=32):
    """
    Xử lý nhiều episodes đồng thời.
    
    Trước: 1 query → 1 forward pass → batch_size=1
    Sau: 32 queries → 1 forward pass → batch_size=32
    
    GPU Utilization: 5% → 60%+
    """
    
    # Stack all queries into batch
    batch_query_embs = torch.stack([d.query_emb for d in episode_data_list])
    # Shape: [32, 512]
    
    # Pad and stack candidates
    batch_candidate_embs = pad_and_stack(...)
    # Shape: [32, 50, 512]
    
    # Single forward pass for entire batch!
    actions, log_probs, values = self.rl_agent.select_action(
        batch_query_embs,  # [32, 512]
        batch_candidate_embs,  # [32, 50, 512]
        ...
    )
    # GPU processes all 32 simultaneously!
```

#### D. MIXED PRECISION TRAINING (FP16)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Forward pass với FP16
with autocast():
    log_probs, values, entropy = agent.evaluate_actions(...)
    loss = policy_loss + value_loss + entropy_loss

# Backward với scaling
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(agent.parameters(), max_grad_norm)
scaler.step(optimizer)
scaler.update()

# Benefits:
# - 2x faster matrix multiplication
# - 2x less memory usage
# - Có thể tăng batch size gấp đôi
```

#### E. MULTI-GPU (DataParallel)

```python
if torch.cuda.device_count() > 1:
    agent = nn.DataParallel(agent)
    # Tự động split batch across GPUs
    # GPU 0: processes samples 0-15
    # GPU 1: processes samples 16-31
```

---

## 3. SO SÁNH HIỆU NĂNG

### 3.1 Theoretical Speedup

| Aspect | Before | After | Speedup |
|--------|--------|-------|---------|
| Episode Collection | Sequential | Batched (32x) | 10-20x |
| GPU Utilization | 2-5% | 60-80% | 15-30x |
| Embedding Computation | Repeated | Cached | 5x |
| CPU-GPU Transfer | Every sample | Pre-allocated | 50x |
| PPO Update | FP32, small batch | FP16, large batch | 2-3x |
| Memory Efficiency | Inefficient | Optimized | 2x |

### 3.2 Expected Results với 2x T4 GPUs

**Before (train_rl.py):**
- GPU Memory: 2% (300MB / 15GB)
- GPU Utilization: 2-5%
- Epoch time: ~30-60 minutes
- Total training (50 epochs): ~25-50 hours

**After (train_rl_optimized.py):**
- GPU Memory: 40-60% (6-9GB / 15GB)
- GPU Utilization: 60-80%
- Epoch time: ~3-5 minutes
- Total training (50 epochs): ~2.5-4 hours

**Speedup: 10-15x**

---

## 4. CÁCH SỬ DỤNG

### 4.1 Chạy Training Tối Ưu

```bash
cd /kaggle/adaptive-information-retrieval/adaptive-ir-system

# Training với config tối ưu
python train_optimized.py \
    --config configs/msa_optimized_gpu.yaml \
    --device cuda \
    --epochs 50 \
    --test

# Tùy chỉnh batch size (nếu OOM)
python train_optimized.py \
    --config configs/msa_optimized_gpu.yaml \
    --batch-size 32 \
    --epochs 50

# Disable mixed precision (debug)
python train_optimized.py \
    --config configs/msa_optimized_gpu.yaml \
    --no-amp
```

### 4.2 Monitor GPU Usage

```bash
# Terminal riêng
watch -n 1 nvidia-smi

# Hoặc dùng nvitop (đẹp hơn)
pip install nvitop
nvitop
```

### 4.3 Config Quan Trọng

```yaml
training:
  # Tăng batch_size để sử dụng GPU tốt hơn
  batch_size: 64           # PPO update batch
  collect_batch_size: 32    # Episode collection batch
  
  # Tăng buffer cho better sampling
  buffer_size: 50000
  
  # Mixed precision
  use_amp: true
  
  # Episodes before update
  episodes_per_update: 256
```

---

## 5. TROUBLESHOOTING

### 5.1 Out of Memory (OOM)

```bash
# Giảm batch size
python train_optimized.py --batch-size 32

# Hoặc trong config:
training:
  batch_size: 32
  collect_batch_size: 16
```

### 5.2 GPU Vẫn Thấp

Kiểm tra:
1. Data loading có blocking không?
2. BM25 search có quá chậm không?
3. Embedding model có trên GPU không?

```python
# Debug
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU Utilization: {torch.cuda.utilization()}%")
```

### 5.3 Convergence Issues

```yaml
# Tăng entropy coefficient
rl_agent:
  entropy_coef: 0.02  # Default: 0.01

# Giảm learning rate
training:
  learning_rate: 0.0001  # Default: 0.0003
```

---

## 6. KẾT LUẬN

Các tối ưu chính:

1. **Batched Processing**: Thay vì xử lý từng query, xử lý 32 queries đồng thời
2. **Embedding Cache**: Cache embeddings để tránh tính toán lại
3. **GPU-Resident Buffer**: Lưu data trên GPU, tránh transfer overhead
4. **Mixed Precision**: FP16 cho faster computation và less memory
5. **Multi-GPU**: DataParallel để sử dụng cả 2 T4 GPUs

Kết quả mong đợi:
- GPU utilization: 2% → 60-80%
- Training speed: 10-15x faster
- Có thể training 50 epochs trong 2-4 giờ thay vì 25-50 giờ
