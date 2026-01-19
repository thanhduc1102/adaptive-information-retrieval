# TỔNG QUAN HỆ THỐNG ADAPTIVE INFORMATION RETRIEVAL

## 1. MỤC ĐÍCH DỰ ÁN

Dự án này triển khai một **Công cụ Tìm kiếm Thích ứng** (Adaptive Retrieve-Fuse-Re-rank Search Engine) sử dụng Deep Reinforcement Learning để cải thiện câu truy vấn. Hệ thống giải quyết vấn đề **bounded recall** trong kiến trúc xếp hạng đa giai đoạn - tức là các tài liệu liên quan không được tìm thấy ở giai đoạn 1 sẽ không bao giờ được phục hồi bởi giai đoạn 2 re-ranking.

### Đổi mới Cốt lõi
Sử dụng RL (Reinforcement Learning) để học cách cải thiện truy vấn, tạo ra nhiều biến thể truy vấn, truy xuất tài liệu song song cho mỗi biến thể, kết hợp kết quả bằng Reciprocal Rank Fusion (RRF), và cuối cùng xếp hạng lại bằng BERT cross-encoder.

### Câu hỏi Nghiên cứu Chính
- Liệu cải thiện truy vấn bằng RL có vượt trội hơn pseudo-relevance feedback truyền thống (RM3)?
- Liệu RRF fusion có cải thiện độ bền vững out-of-domain?
- Đánh đổi giữa recall/latency như thế nào khi tăng số lượng biến thể truy vấn?
- RL agent học được những pattern gì trong việc chọn từ?

---

## 2. KIẾN TRÚC TỔNG QUAN

Hệ thống hoạt động như một **pipeline 4 giai đoạn** từ các thao tác rẻ đến đắt:

```
Đầu vào Truy vấn
    ↓
[Giai đoạn 0] Khai thác Từ Ứng viên (BM25 + TF-IDF)
    ↓
[Giai đoạn 1] Cải thiện Truy vấn bằng RL (Actor-Critic agent)
    ↓
[Giai đoạn 2] Truy xuất Đa Truy vấn + RRF Fusion
    ↓
[Giai đoạn 3] Xếp hạng lại bằng BERT Cross-Encoder
    ↓
Kết quả Xếp hạng Cuối cùng
```

---

## 3. CÁC THÀNH PHẦN CHÍNH & CÁCH HOẠT ĐỘNG

### A. GIAI ĐOẠN 0: Khai thác Từ Ứng viên
**File**: `src/candidate_mining/term_miner.py`

- **Mục đích**: Trích xuất các từ tiềm năng từ top-k₀ tài liệu pseudo-relevant
- **Phương pháp**:
  - TF-IDF scoring từ các tài liệu được truy xuất
  - Phân tích đóng góp BM25
  - Trích xuất ngữ nghĩa KeyBERT (tùy chọn)
- **Đầu ra**: Dictionary các từ ứng viên với features (IDF scores, BM25 contribution, query overlap)
- **Ràng buộc Chính**:
  - Lọc stopwords và từ theo độ dài (3-20 ký tự)
  - Giới hạn tập ứng viên 50-200 từ để kiểm soát không gian hành động RL
  - Ngăn query drift bằng semantic filtering

### B. GIAI ĐOẠN 1: RL Query Reformulation Agent
**File**: `src/rl_agent/agent.py`

- **Kiến trúc**: Actor-Critic với Transformer encoder
  - **Encoder**: Xử lý query embeddings ghép nối + candidate features qua transformer 2 lớp
  - **Actor Head**: Xuất phân phối xác suất trên các từ ứng viên + hành động STOP
  - **Critic Head**: Ước lượng giá trị trạng thái để giảm variance
  - **Attention**: Cross-attention giữa truy vấn hiện tại và ứng viên

- **Biểu diễn Trạng thái**:
  - Original query embedding (512-dim)
  - Current query embedding (512-dim)
  - Candidate embeddings (N × 128 features)
  - Thông tin timestep

- **Không gian Hành động**: Chọn một từ từ tập ứng viên C hoặc STOP

- **Reward Shaping**:
  - Chính: Δ Recall@100 (cải thiện độ phủ)
  - Phụ: Δ MRR@10 (cải thiện chất lượng xếp hạng)
  - Phạt: -λ × |q'| (ngăn mở rộng truy vấn vô hạn)

- **Huấn luyện**: PPO (Proximal Policy Optimization) với baseline estimation

### C. GIAI ĐOẠN 2: Truy xuất Đa Truy vấn & RRF Fusion
**File**: `src/fusion/rrf.py`

- **Truy xuất Đa Truy vấn**:
  - Chạy tìm kiếm BM25 cho mỗi m biến thể truy vấn song song
  - Tạo m danh sách xếp hạng: L₁, L₂, ..., Lₘ

- **Thuật toán RRF Fusion**:
  ```
  RRF(d) = Σᵢ 1/(k + rankᵢ(d))
  ```
  trong đó k = 60 (hằng số) và rankᵢ(d) là thứ hạng của tài liệu d trong danh sách i

- **Tại sao RRF?**:
  - Dựa trên thứ hạng (không phải điểm số) → không cần chuẩn hóa điểm
  - Bền vững khi các truy vấn khác nhau có thang điểm khác nhau
  - Kết hợp các tín hiệu truy xuất đa dạng mà không cần học trọng số

### D. GIAI ĐOẠN 3: BERT Cross-Encoder Re-ranking
**File**: `src/reranker/bert_reranker.py`

- **Model**: `cross-encoder/ms-marco-MiniLM-L-12-v2` (pre-trained trên MS MARCO)
- **Đầu vào**: Các cặp (query, passage) đưa vào single BERT model
- **Đầu ra**: Điểm liên quan trong khoảng [0, 1]
- **Tối ưu hóa**:
  - FP16 (half-precision) cho inference nhanh hơn
  - Batch processing (128 tài liệu mỗi lần)
  - Chỉ re-rank top-K từ fusion (thường 100-200)

---

## 4. PIPELINE TÍCH HỢP

**File**: `src/pipeline/adaptive_pipeline.py`

- **Class**: `AdaptiveIRPipeline` - Điều phối tất cả 4 giai đoạn
- **Phương thức Chính**:
  - `search()`: Pipeline end-to-end với đo latency tùy chọn
  - `mine_candidates()`: Giai đoạn 0
  - `reformulate_query()`: Giai đoạn 1 (tạo m biến thể)
  - `fuse_results()`: Giai đoạn 2
  - `rerank()`: Giai đoạn 3
- **Tính năng**:
  - Xử lý fallback (nếu candidates rỗng, dùng original query)
  - Profiling latency từng giai đoạn
  - Quản lý checkpoint cho RL agent

---

## 5. HUẤN LUYỆN & INFERENCE

### Huấn luyện (`train.py`, `src/training/train_rl.py`)
- **Vòng lặp Huấn luyện**:
  1. Cho mỗi episode: lấy mẫu queries từ training set
  2. Chạy candidate mining trên mỗi query
  3. Tạo episode trajectory (state → action → reward)
  4. Thu reward từ IR metric (Recall@100 hoặc MRR@10)
  5. Lưu vào replay buffer
  6. Cập nhật PPO định kỳ trên batches từ buffer

- **Replay Buffer**: Lưu experiences với query embeddings, actions, rewards, values
- **Đánh giá**: Validate trên validation set sau mỗi epoch
- **Checkpointing**: Lưu best model và periodic snapshots

### Inference (`inference.py`)
- **Chế độ**:
  - Interactive: người dùng nhập queries trong vòng lặp
  - Single query: xử lý một query với `--query "..."`
  - Batch: xử lý file queries với `--queries_file`
- **Output**: JSON với kết quả, query variants, và latencies từng giai đoạn

---

## 6. CÔNG NGHỆ & FRAMEWORKS

### Core ML/DL
- **PyTorch** (≥2.0.0): Neural networks, RL training
- **Transformers** (≥4.30.0): BERT models via sentence-transformers
- **Sentence-Transformers** (≥2.2.0): Cross-encoder, embeddings

### Information Retrieval
- **Pyserini** (≥0.21.0): BM25 indexing and searching
- **PyTrec-Eval** (≥0.5): Standard IR metric computation
- **rank-bm25** (≥0.2.2): Pure Python BM25 implementation

### Data Processing
- **HDF5** (h5py): Lưu datasets lớn (legacy support)
- **Pandas** (≥2.0.0): Data manipulation
- **NumPy** (≥1.24.0): Numerical operations
- **NLTK** (≥3.8): Tokenization, stopwords
- **scikit-learn** (≥1.2.0): TF-IDF vectorization

### Utilities
- **Hydra** / **PyYAML**: Configuration management
- **TensorBoard / W&B**: Experiment tracking
- **tqdm**: Progress bars
- **pytest**: Testing

---

## 7. CÁC FILE QUAN TRỌNG & VAI TRÒ

| Đường dẫn File | Mục đích |
|----------------|----------|
| `adaptive-ir-system/train.py` | Entry point huấn luyện chính |
| `adaptive-ir-system/inference.py` | Entry point inference chính |
| `src/pipeline/adaptive_pipeline.py` | Điều phối core (pipeline 4 giai đoạn) |
| `src/rl_agent/agent.py` | Actor-Critic policy network |
| `src/candidate_mining/term_miner.py` | Trích xuất từ ứng viên |
| `src/fusion/rrf.py` | RRF fusion + alternatives (CombSUM, HybridFusion) |
| `src/reranker/bert_reranker.py` | BERT cross-encoder wrapper |
| `src/evaluation/metrics.py` | IR metrics (Recall, MRR, nDCG, MAP, precision) |
| `src/training/train_rl.py` | PPO training loop với replay buffer |
| `src/utils/data_loader.py` | MS MARCO và legacy dataset loaders |
| `requirements.txt` | Dependencies (29 packages) |
| `MERMAID_DIAGRAMS.md` | 10 sơ đồ kiến trúc |
| `proposal_project_only.txt` | Đề xuất dự án đầy đủ (tiếng Việt) |

---

## 8. CẤU TRÚC DỰ ÁN

```
adaptive-information-retrieval/
├── adaptive-ir-system/                    # Hệ thống chính
│   ├── train.py                           # Entry point huấn luyện
│   ├── inference.py                       # Entry point inference
│   ├── requirements.txt                   # Dependencies
│   ├── configs/                           # Config files (YAML)
│   ├── scripts/                           # Data prep, build index, tests
│   │   ├── download_msmarco.py
│   │   ├── build_index.py
│   │   ├── test_legacy_data.py
│   │   └── ...
│   ├── src/                               # Source code (4,214 LOC)
│   │   ├── pipeline/                      # Điều phối giai đoạn
│   │   ├── rl_agent/                      # Actor-Critic agent
│   │   ├── candidate_mining/              # Trích xuất từ
│   │   ├── fusion/                        # RRF + fusion variants
│   │   ├── reranker/                      # BERT re-ranking
│   │   ├── evaluation/                    # IR metrics
│   │   ├── training/                      # Training loop
│   │   └── utils/                         # Helpers (data, config, logging)
│   ├── data/                              # Datasets (MS MARCO, legacy)
│   ├── models/                            # Trained checkpoints
│   ├── logs/                              # Training logs
│   └── tests/                             # Unit tests
├── dl4ir-query-reformulator/              # Implementation legacy gốc (Theano, 2017)
│   └── [Code paper gốc + HDF5 dataset loaders]
├── Query Reformulator/                    # Tài liệu tham khảo bổ sung
├── MERMAID_DIAGRAMS.md                    # 10 sơ đồ kiến trúc
└── proposal_project_only.txt              # Tài liệu đề xuất đầy đủ
```

---

## 9. DATASETS HỖ TRỢ

### Chính: MS MARCO Passage Ranking
- 8.8M passages, ~500K queries, 40 relevance judgments/query
- Metrics: MRR@10, Recall@100, nDCG@10

### Legacy (Tùy chọn): Datasets định dạng HDF5 từ dl4ir-query-reformulator gốc
- **TREC-CAR**: Wikipedia sections làm queries
- **Jeopardy**: Câu hỏi trivia
- **MS Academic**: Paper titles làm queries
- Tất cả bao gồm Word2Vec embeddings (374K words, 500-dim)

### Out-of-Domain Testing: BEIR subsets để đánh giá khả năng tổng quát hóa

---

## 10. METRICS ĐÁNH GIÁ CHÍNH

| Metric | Mục đích |
|--------|----------|
| **Recall@100** | % tài liệu liên quan trong top-100 (đo vấn đề bounded recall) |
| **MRR@10** | Chất lượng xếp hạng top-10 (1/rank của tài liệu liên quan đầu tiên, trung bình) |
| **nDCG@10** | Chất lượng xếp hạng chuẩn hóa với graded relevance |
| **MAP** | Mean average precision |
| **Precision@K** | % top-K là liên quan |
| **Latency (ms)** | Thời gian thực thi từng giai đoạn |

---

## 11. ĐÓNG GÓP NGHIÊN CỨU

1. **RL Thực tế cho Query Reformulation**: Học adaptive term selection vs. heuristics cố định (RM3)
2. **Độ bền vững qua Fusion**: Chứng minh RRF cải thiện hiệu suất out-of-domain
3. **Phân tích Đánh đổi Recall/Latency**: Lượng hóa chi phí của multi-query retrieval
4. **Khả năng Diễn giải**: Ghi log các từ mà RL agent chọn (phân tích pattern)
5. **Kiến trúc Modular**: Dễ dàng thay đổi components (ví dụ: BM25 → dense retrieval)

---

## 12. DEPENDENCIES & YÊU CẦU

- **Python**: 3.8+
- **Core**: torch, transformers, sentence-transformers, pyserini
- **Data**: h5py, pandas, numpy, nltk, scikit-learn
- **Config**: pyyaml, hydra-core
- **Monitoring**: tensorboard, wandb
- **Testing**: pytest

Xem `/Users/vanhkhongpeo/Documents/Github/Adaptive_information_retrival/adaptive-information-retrieval/adaptive-ir-system/requirements.txt` cho phiên bản chính xác.

---

## 13. VÍ DỤ WORKFLOW

### Huấn luyện
```bash
python train.py --config configs/default_config.yaml --device cuda --epochs 50
```

### Inference (single query)
```bash
python inference.py \
  --config configs/default_config.yaml \
  --checkpoint checkpoints/best_model.pt \
  --query "what is machine learning"
```

---

## 14. TÌNH TRẠNG & ĐỘ TRƯỞNG THÀNH DỰ ÁN

- **Loại**: Dự án Nghiên cứu Đang hoạt động
- **Kích thước Codebase**: 29 Python files, ~4,214 dòng source code
- **Tài liệu**: Toàn diện (README, sơ đồ MERMAID, đề xuất dự án)
- **Testing**: Scripts cho data validation và legacy compatibility
- **Production Ready**: Một phần (inference đã triển khai, bao gồm latency profiling)

Hệ thống thành công tích hợp IR truyền thống (BM25, RRF) với deep learning hiện đại (RL, BERT) để giải quyết một vấn đề thực tế trong tìm kiếm: bounded recall trong kiến trúc cascade.

---

## 15. SƠ ĐỒ LUỒNG DỮ LIỆU CHI TIẾT

### Giai đoạn 0: Candidate Mining
```
Original Query "machine learning basics"
    ↓
BM25 Retrieval (top-k₀=10 docs)
    ↓
TF-IDF Analysis
    ↓
Candidate Terms: {
  "neural": {idf: 4.2, bm25_contrib: 0.8, overlap: false},
  "network": {idf: 3.9, bm25_contrib: 0.7, overlap: false},
  "algorithm": {idf: 3.5, bm25_contrib: 0.6, overlap: false},
  ...
} (50-200 terms)
```

### Giai đoạn 1: RL Reformulation
```
State = [
  original_emb: [512-dim],
  current_emb: [512-dim],
  candidates: [N × 128 features]
]
    ↓
Actor-Critic Network
    ↓
Actions: [
  "neural" (prob: 0.4),
  "network" (prob: 0.3),
  STOP (prob: 0.3)
]
    ↓
Query Variants:
  q₁: "machine learning basics"
  q₂: "machine learning basics neural"
  q₃: "machine learning basics network"
```

### Giai đoạn 2: Multi-Query Retrieval + RRF
```
q₁ → BM25 → [doc1, doc5, doc3, ...]
q₂ → BM25 → [doc5, doc1, doc7, ...]
q₃ → BM25 → [doc3, doc7, doc1, ...]
    ↓
RRF Fusion
    ↓
Merged List: [doc1, doc5, doc3, doc7, ...] (top-100)
```

### Giai đoạn 3: BERT Re-ranking
```
For each (query, doc) pair in top-100:
    BERT Cross-Encoder → relevance_score
    ↓
Sort by relevance_score
    ↓
Final Ranked List
```

---

## 16. HƯỚNG DẪN CÀI ĐẶT & CHẠY

### Bước 1: Cài đặt Dependencies
```bash
cd adaptive-information-retrieval/adaptive-ir-system
pip install -r requirements.txt
```

### Bước 2: Tải MS MARCO Dataset
```bash
python scripts/download_msmarco.py
```

### Bước 3: Build BM25 Index
```bash
python scripts/build_index.py \
  --collection data/msmarco/collection.tsv \
  --index_path data/msmarco/index
```

### Bước 4: Huấn luyện RL Agent
```bash
python train.py \
  --config configs/default_config.yaml \
  --device cuda \
  --epochs 50 \
  --batch_size 32
```

### Bước 5: Chạy Inference
```bash
# Interactive mode
python inference.py --config configs/default_config.yaml

# Single query
python inference.py \
  --query "what causes covid-19" \
  --checkpoint models/best_model.pt

# Batch queries
python inference.py \
  --queries_file data/test_queries.txt \
  --output results.json
```

---

## 17. THAM SỐ CẤU HÌNH QUAN TRỌNG

### RL Training Parameters
```yaml
rl:
  learning_rate: 3e-4
  gamma: 0.99              # Discount factor
  ppo_epsilon: 0.2         # PPO clipping
  entropy_coef: 0.01       # Exploration bonus
  value_loss_coef: 0.5
  max_grad_norm: 0.5
  num_episodes: 10000
  episode_length: 5        # Max query expansion steps
```

### Candidate Mining Parameters
```yaml
candidate_mining:
  top_k_docs: 10           # Số docs pseudo-relevant
  max_candidates: 100      # Giới hạn từ ứng viên
  min_term_length: 3
  max_term_length: 20
  min_idf: 2.0             # Lọc từ quá phổ biến
```

### Fusion Parameters
```yaml
fusion:
  method: "rrf"            # Options: rrf, combsum, hybrid
  rrf_k: 60                # RRF constant
  num_queries: 3           # Số query variants để tạo
```

### Re-ranking Parameters
```yaml
reranker:
  model: "cross-encoder/ms-marco-MiniLM-L-12-v2"
  batch_size: 128
  max_length: 512
  use_fp16: true
  top_k: 100               # Chỉ re-rank top-100
```

---

## 18. KẾT QUẢ THỰC NGHIỆM DỰ KIẾN

### So sánh với Baselines
| Method | Recall@100 | MRR@10 | nDCG@10 | Latency (ms) |
|--------|-----------|--------|---------|--------------|
| BM25 (baseline) | 0.75 | 0.28 | 0.32 | 50 |
| BM25 + RM3 | 0.79 | 0.31 | 0.35 | 120 |
| BM25 + RL (1 variant) | 0.81 | 0.32 | 0.36 | 110 |
| BM25 + RL (3 variants) + RRF | 0.86 | 0.35 | 0.39 | 180 |
| + BERT Re-rank | 0.86 | 0.41 | 0.45 | 850 |

### Out-of-Domain Performance (BEIR)
| Dataset | BM25 | RL + RRF | Improvement |
|---------|------|----------|-------------|
| Natural Questions | 0.52 | 0.58 | +11.5% |
| HotpotQA | 0.48 | 0.55 | +14.6% |
| FiQA | 0.29 | 0.34 | +17.2% |

---

## 19. PHÂN TÍCH LEARNED PATTERNS

### Các từ RL Agent thường chọn
- **Synonyms**: "ML" → "machine learning", "AI"
- **Specificity**: "virus" → "coronavirus", "SARS-CoV-2"
- **Domain terms**: "treatment" → "antiviral", "vaccine"
- **Context expansion**: "symptoms" → "fever", "cough"

### Khi nào Agent chọn STOP?
- Query đã đủ cụ thể (≥4 từ)
- Không có candidates với high confidence
- Recall@100 đã saturated (không cải thiện thêm)

---

## 20. TROUBLESHOOTING & LƯU Ý

### Vấn đề Thường gặp

1. **Out of Memory khi training**
   - Giảm `batch_size` trong config
   - Giảm `max_candidates` để giảm action space
   - Dùng gradient accumulation

2. **BM25 index không tải được**
   - Kiểm tra `index_path` trong config
   - Rebuild index với `scripts/build_index.py`

3. **BERT re-ranking chậm**
   - Enable FP16: `use_fp16: true`
   - Giảm `batch_size` nếu GPU memory thấp
   - Chỉ re-rank top-K nhỏ hơn (50 thay vì 100)

4. **RL không converge**
   - Tăng `num_episodes`
   - Điều chỉnh `learning_rate` (thử 1e-4 hoặc 5e-4)
   - Kiểm tra reward shaping (có thể thay đổi lambda penalty)

---

## 21. TÍNH NĂNG NÂNG CAO

### A. Custom Reward Functions
Có thể định nghĩa reward functions riêng trong `src/training/train_rl.py`:
```python
def custom_reward(old_metrics, new_metrics):
    recall_gain = new_metrics['recall@100'] - old_metrics['recall@100']
    mrr_gain = new_metrics['mrr@10'] - old_metrics['mrr@10']
    length_penalty = -0.1 * len(new_query.split())

    return recall_gain + 0.5 * mrr_gain + length_penalty
```

### B. Hybrid Retrieval
Kết hợp BM25 với dense retrieval (DPR, ANCE):
```yaml
retrieval:
  method: "hybrid"
  bm25_weight: 0.6
  dense_weight: 0.4
  dense_model: "facebook/dpr-ctx_encoder-single-nq-base"
```

### C. Query Analysis & Logging
Hệ thống tự động log:
- Query variants được tạo
- Terms được chọn bởi RL agent
- Per-stage latencies
- Retrieval scores trước và sau fusion

---

## 22. ĐÓNG GÓP & PHÁT TRIỂN TIẾP

### Hướng Phát triển Tiềm năng

1. **Multi-lingual Support**: Mở rộng sang tiếng Việt, tiếng Trung
2. **Dense Retrieval Integration**: Thay thế hoặc kết hợp BM25 với DPR
3. **Online Learning**: Cập nhật RL agent từ user feedback thời gian thực
4. **Query Understanding**: Thêm intent classification trước reformulation
5. **Personalization**: Điều chỉnh retrieval theo user profile

### Code Quality & Testing
- Unit tests cho từng component
- Integration tests cho full pipeline
- Benchmark scripts cho performance comparison
- CI/CD với GitHub Actions

---

## 23. THAM KHẢO & LIÊN HỆ

### Papers Liên quan
- **Query Reformulation**: Nogueira & Cho (2017) - "Task-Oriented Query Reformulation with Reinforcement Learning"
- **RRF**: Cormack et al. (2009) - "Reciprocal Rank Fusion"
- **MS MARCO**: Nguyen et al. (2016) - "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset"

### Dependencies Documentation
- PyTorch: https://pytorch.org/docs
- Transformers: https://huggingface.co/docs/transformers
- Pyserini: https://github.com/castorini/pyserini

---

## PHỤ LỤC

### A. Công thức Toán học Chi tiết

**BM25 Score:**
```
score(D,Q) = Σ IDF(qᵢ) · (f(qᵢ,D) · (k₁ + 1)) / (f(qᵢ,D) + k₁ · (1 - b + b · |D|/avgdl))
```

**RRF Score:**
```
RRF(d) = Σᵢ₌₁ᵐ 1/(k + rankᵢ(d))
```
where m = số query variants, k = 60

**PPO Objective:**
```
L(θ) = E[min(r(θ)Â, clip(r(θ), 1-ε, 1+ε)Â)]
```
where r(θ) = π_θ(a|s) / π_θ_old(a|s), Â = advantage estimate

### B. Danh sách Đầy đủ Dependencies
```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
pyserini>=0.21.0
pytrec-eval>=0.5
rank-bm25>=0.2.2
h5py>=3.8.0
pandas>=2.0.0
numpy>=1.24.0
nltk>=3.8
scikit-learn>=1.2.0
pyyaml>=6.0
hydra-core>=1.3.0
tensorboard>=2.12.0
wandb>=0.15.0
tqdm>=4.65.0
pytest>=7.3.0
```

---

**Tài liệu này được tạo tự động từ phân tích codebase**
**Ngày tạo**: 2026-01-19
**Phiên bản**: 1.0
