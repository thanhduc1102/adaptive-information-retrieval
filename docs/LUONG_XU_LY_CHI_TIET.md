# LUỒNG XỬ LÝ CHI TIẾT - HỆ THỐNG TÌM KIẾM THÍCH ỨNG

## BÀI TOÁN CẦN GIẢI QUYẾT

### Vấn đề thực tế
Khi bạn tìm kiếm trên Google hoặc các công cụ tìm kiếm:
- Bạn gõ: **"triệu chứng covid"**
- Nhưng tài liệu có thể viết: "dấu hiệu nhiễm SARS-CoV-2", "biểu hiện COVID-19"
- Vì từ khóa không khớp → **tài liệu quan trọng bị bỏ sót**

### Giải pháp truyền thống
Tìm kiếm 2 giai đoạn:
1. **Giai đoạn 1**: BM25 tìm nhanh 1000 tài liệu (rẻ, nhanh)
2. **Giai đoạn 2**: BERT xếp hạng lại top 100 (đắt, chậm)

**VẤN ĐỀ**: Nếu tài liệu liên quan không có trong 1000 tài liệu ở giai đoạn 1 → Giai đoạn 2 không bao giờ thấy được → **Bounded Recall Problem**

### Giải pháp của hệ thống này
Thay vì tìm kiếm với 1 câu truy vấn, hệ thống:
1. **Tự động mở rộng** câu truy vấn thành nhiều biến thể
2. **Tìm kiếm song song** với tất cả biến thể
3. **Kết hợp kết quả** thông minh
4. **Xếp hạng lại** bằng BERT

---

## VÍ DỤ THỰC TẾ: TÌM KIẾM "COVID SYMPTOMS"

Tôi sẽ mô tả từng bước cụ thể khi người dùng tìm kiếm **"covid symptoms"**

---

## 🔹 BƯỚC 0: NGƯỜI DÙNG NHẬP TRUY VẤN

```
Input: "covid symptoms"
```

**Hệ thống nhận được**: Chuỗi text đơn giản này

---

## 🔹 GIAI ĐOẠN 0: KHAI THÁC TỪ ỨNG VIÊN

### Mục tiêu
Tìm các từ có khả năng mở rộng câu truy vấn tốt

### Quy trình chi tiết

#### Bước 0.1: Tìm kiếm sơ bộ bằng BM25
```python
# File: src/pipeline/adaptive_pipeline.py, dòng 132
doc_ids, scores = self.retrieve("covid symptoms", top_k=50)
```

**Kết quả**: Lấy 50 tài liệu có điểm BM25 cao nhất

```
Tài liệu 1: "COVID-19 causes fever, cough, and shortness of breath..."
Tài liệu 2: "SARS-CoV-2 infection symptoms include headache and fatigue..."
Tài liệu 3: "Coronavirus patients report loss of smell and taste..."
...
Tài liệu 50: "..."
```

#### Bước 0.2: Phân tích TF-IDF
```python
# File: src/candidate_mining/term_miner.py
candidates = self.candidate_miner.extract_candidates(query, documents, scores)
```

**Công việc**:
- Tính TF-IDF cho mỗi từ trong 50 tài liệu
- Lọc bỏ stopwords ("the", "a", "is"...)
- Lọc bỏ từ quá ngắn (<3 ký tự) hoặc quá dài (>20 ký tự)
- Tính đóng góp BM25 của mỗi từ

#### Bước 0.3: Tạo danh sách từ ứng viên
```python
{
  "fever": {
    "idf": 4.5,              # Điểm IDF cao = từ quan trọng
    "bm25_contrib": 0.85,    # Đóng góp cao vào điểm BM25
    "query_overlap": False,  # Không trùng với query gốc
    "tf": 25,                # Xuất hiện 25 lần
    "doc_freq": 15           # Xuất hiện trong 15/50 tài liệu
  },
  "cough": {
    "idf": 4.2,
    "bm25_contrib": 0.78,
    "query_overlap": False,
    "tf": 20,
    "doc_freq": 12
  },
  "headache": {
    "idf": 4.0,
    "bm25_contrib": 0.72,
    "query_overlap": False,
    "tf": 18,
    "doc_freq": 10
  },
  "SARS-CoV-2": {
    "idf": 5.2,
    "bm25_contrib": 0.68,
    "query_overlap": False,
    "tf": 8,
    "doc_freq": 6
  },
  "fatigue": {...},
  "shortness": {...},
  "breath": {...},
  ...
  # Tổng cộng khoảng 80-100 từ ứng viên
}
```

**Kết quả Giai đoạn 0**: Danh sách 80-100 từ tiềm năng để mở rộng query

---

## 🔹 GIAI ĐOẠN 1: RL AGENT CẢI THIỆN TRUY VẤN

### Mục tiêu
Chọn từ nào trong 80-100 từ ứng viên để thêm vào query

### Cơ chế hoạt động

#### Bước 1.1: Chuẩn bị dữ liệu đầu vào cho RL Agent

```python
# File: src/pipeline/adaptive_pipeline.py, dòng 171-183
query_emb = self._embed_text("covid symptoms")
# → Vector 512 chiều: [0.23, -0.45, 0.67, ..., 0.12]

candidate_terms = ["fever", "cough", "headache", "SARS-CoV-2", ...]
candidate_embs = [
  embed("fever"),        # [0.15, -0.22, 0.55, ...]
  embed("cough"),        # [0.18, -0.19, 0.51, ...]
  embed("headache"),     # [0.12, -0.25, 0.48, ...]
  ...
]

candidate_features = [
  [4.5, 0.85, 0, 25, 15, 0.6],  # fever: [idf, bm25, overlap, tf, df, semantic_sim]
  [4.2, 0.78, 0, 20, 12, 0.5],  # cough
  [4.0, 0.72, 0, 18, 10, 0.4],  # headache
  ...
]
```

#### Bước 1.2: RL Agent xử lý (Actor-Critic)

**Kiến trúc Neural Network**:
```
Input Layer:
├─ query_emb: [512]               ← "covid symptoms" embedding
├─ current_emb: [512]             ← Query hiện tại (ban đầu = query_emb)
└─ candidate_features: [100 × 518]  ← 100 từ, mỗi từ có 512 (emb) + 6 (features)

    ↓

Query Encoder (Linear): 512 → 256
Candidate Encoder (Linear + ReLU): 518 → 256

    ↓

Transformer Encoder (2 layers):
├─ Multi-Head Attention (4 heads)
├─ Feed-Forward Network
└─ Residual Connections

    ↓

Cross-Attention:
├─ Query: current_query_emb
├─ Keys/Values: candidate_embeddings
└─ Output: Attention-weighted candidate representations

    ↓ ─────────┬──────────────┐

Actor Head              Critic Head
├─ Linear: 256 → 256    ├─ Linear: 256 → 256
├─ ReLU                 ├─ ReLU
├─ Linear: 256 → 1      └─ Linear: 256 → 1
└─ Softmax              └─ Value estimate
    ↓                       ↓
Action probabilities    State value
[P(fever)=0.35,         V = 0.72
 P(cough)=0.28,
 P(headache)=0.15,
 ...,
 P(STOP)=0.10]
```

#### Bước 1.3: RL Agent tạo query variants

**Variant 1**:
```
Step 0: current_query = "covid symptoms"
        Agent chọn: "fever" (prob=0.35)

Step 1: current_query = "covid symptoms fever"
        Agent chọn: "cough" (prob=0.42)

Step 2: current_query = "covid symptoms fever cough"
        Agent chọn: STOP (prob=0.55)

→ Final: "covid symptoms fever cough"
```

**Variant 2**:
```
Step 0: current_query = "covid symptoms"
        Agent chọn: "SARS-CoV-2" (prob=0.25)

Step 1: current_query = "covid symptoms SARS-CoV-2"
        Agent chọn: "infection" (prob=0.38)

Step 2: current_query = "covid symptoms SARS-CoV-2 infection"
        Agent chọn: STOP (prob=0.60)

→ Final: "covid symptoms SARS-CoV-2 infection"
```

**Variant 3**:
```
Step 0: current_query = "covid symptoms"
        Agent chọn: "headache" (prob=0.18)

Step 1: current_query = "covid symptoms headache"
        Agent chọn: "fatigue" (prob=0.32)

Step 2: current_query = "covid symptoms headache fatigue"
        Agent chọn: STOP (prob=0.51)

→ Final: "covid symptoms headache fatigue"
```

#### Kết quả Giai đoạn 1: Danh sách query variants

```python
query_variants = [
  "covid symptoms",                          # Original (luôn có)
  "covid symptoms fever cough",              # Variant 1
  "covid symptoms SARS-CoV-2 infection",     # Variant 2
  "covid symptoms headache fatigue"          # Variant 3
]
```

**Tại sao làm thế này?**
- Mỗi variant nhắm đến các khía cạnh khác nhau:
  - Variant 1: Triệu chứng phổ biến (sốt, ho)
  - Variant 2: Thuật ngữ y khoa chính thức
  - Variant 3: Triệu chứng ít phổ biến hơn

---

## 🔹 GIAI ĐOẠN 2: TÌM KIẾM ĐA TRUY VẤN & RRF FUSION

### Mục tiêu
Tìm kiếm với mỗi variant và kết hợp kết quả

#### Bước 2.1: Tìm kiếm BM25 cho từng variant

```python
# File: src/pipeline/adaptive_pipeline.py, dòng 239-241
for query in query_variants:
    doc_ids, scores = self.retrieve(query, top_k=100)
    ranked_lists.append(doc_ids)
```

**Kết quả tìm kiếm**:

**Query 1**: "covid symptoms"
```
Rank 1: doc_1234 (score: 28.5)
Rank 2: doc_5678 (score: 26.3)
Rank 3: doc_9012 (score: 24.1)
...
Rank 100: doc_7777 (score: 8.2)
```

**Query 2**: "covid symptoms fever cough"
```
Rank 1: doc_5678 (score: 31.2)  ← doc này lên hạng 1
Rank 2: doc_3333 (score: 29.8)  ← doc mới xuất hiện
Rank 3: doc_1234 (score: 28.7)
...
Rank 100: doc_8888 (score: 9.1)
```

**Query 3**: "covid symptoms SARS-CoV-2 infection"
```
Rank 1: doc_4444 (score: 30.5)  ← doc mới xuất hiện
Rank 2: doc_5678 (score: 28.9)
Rank 3: doc_2222 (score: 27.3)
...
Rank 100: doc_9999 (score: 8.8)
```

**Query 4**: "covid symptoms headache fatigue"
```
Rank 1: doc_6666 (score: 29.3)  ← doc mới xuất hiện
Rank 2: doc_1234 (score: 27.8)
Rank 3: doc_5678 (score: 26.5)
...
Rank 100: doc_1111 (score: 8.5)
```

#### Bước 2.2: RRF Fusion - Kết hợp kết quả

**Công thức RRF**:
```
RRF_score(doc) = Σ 1/(k + rank_i(doc))
                 i=1..4

Trong đó:
- k = 60 (hằng số)
- rank_i(doc) = thứ hạng của doc trong query variant thứ i
- Nếu doc không xuất hiện trong query i → không cộng
```

**Ví dụ tính toán**:

**doc_5678**: Xuất hiện ở cả 4 queries
```
Query 1: rank = 2  → 1/(60+2)  = 0.0161
Query 2: rank = 1  → 1/(60+1)  = 0.0164
Query 3: rank = 2  → 1/(60+2)  = 0.0161
Query 4: rank = 3  → 1/(60+3)  = 0.0159

RRF_score = 0.0161 + 0.0164 + 0.0161 + 0.0159 = 0.0645
```

**doc_1234**: Xuất hiện ở 3 queries
```
Query 1: rank = 1  → 1/(60+1)  = 0.0164
Query 2: rank = 3  → 1/(60+3)  = 0.0159
Query 4: rank = 2  → 1/(60+2)  = 0.0161

RRF_score = 0.0164 + 0.0159 + 0.0161 = 0.0484
```

**doc_4444**: Chỉ xuất hiện ở 1 query
```
Query 3: rank = 1  → 1/(60+1)  = 0.0164

RRF_score = 0.0164
```

#### Bước 2.3: Sắp xếp theo RRF score

```python
fused_results = [
  ("doc_5678", 0.0645),  # Rank 1: Xuất hiện nhiều nhất, rank tốt
  ("doc_1234", 0.0484),  # Rank 2: Xuất hiện 3/4 queries
  ("doc_3333", 0.0325),  # Rank 3
  ("doc_6666", 0.0312),  # Rank 4
  ("doc_4444", 0.0298),  # Rank 5
  ...
  ("doc_9999", 0.0021),  # Rank 100
]
```

**Kết quả Giai đoạn 2**: Danh sách 100 tài liệu được kết hợp từ 4 queries

**Tại sao RRF tốt?**
- doc_5678 xuất hiện ở cả 4 queries → Có khả năng liên quan cao với nhiều khía cạnh
- doc_4444 chỉ xuất hiện 1 query nhưng rank 1 → Vẫn được xem xét nhưng điểm thấp hơn
- Không cần normalize scores giữa các queries (chỉ dùng thứ hạng)

---

## 🔹 GIAI ĐOẠN 3: BERT CROSS-ENCODER RE-RANKING

### Mục tiêu
Xếp hạng lại chính xác bằng BERT (đọc hiểu ngữ cảnh)

#### Bước 3.1: Lấy nội dung tài liệu

```python
# File: src/pipeline/adaptive_pipeline.py, dòng 279-287
doc_ids = ["doc_5678", "doc_1234", "doc_3333", ...]
documents = []
for doc_id in doc_ids[:100]:  # Chỉ re-rank top 100
    doc_text = self.search_engine.get_document(doc_id)
    documents.append(doc_text)
```

**Ví dụ documents**:
```python
documents = [
  "COVID-19 symptoms include fever, cough, shortness of breath...",  # doc_5678
  "Common signs of coronavirus infection are headache...",            # doc_1234
  "SARS-CoV-2 causes respiratory symptoms such as...",                # doc_3333
  ...
]
```

#### Bước 3.2: BERT Cross-Encoder đánh giá

**Mô hình**: `cross-encoder/ms-marco-MiniLM-L-12-v2`

**Input cho BERT**: Ghép query và document
```
[CLS] covid symptoms [SEP] COVID-19 symptoms include fever, cough, shortness of breath... [SEP]
```

**BERT xử lý**:
```
BERT Tokenizer
    ↓
Input IDs: [101, 2522, 4003, 102, 2522, 19, 4003, 2421, ...]
    ↓
BERT Encoder (12 layers)
├─ Self-Attention
├─ Feed-Forward
└─ Layer Normalization
    ↓
[CLS] embedding (768-dim)
    ↓
Classification Head
    ↓
Relevance Score: 0.87  (0-1 scale)
```

#### Bước 3.3: Score tất cả documents

```python
# File: src/reranker/bert_reranker.py
bert_scores = []
for doc in documents:
    query_doc_pair = f"covid symptoms [SEP] {doc}"
    score = bert_model.predict(query_doc_pair)
    bert_scores.append(score)
```

**Kết quả**:
```python
[
  ("doc_5678", 0.87),  # BERT score cao nhất
  ("doc_3333", 0.85),  # doc_3333 vượt doc_1234!
  ("doc_1234", 0.82),
  ("doc_6666", 0.79),
  ("doc_4444", 0.76),
  ...
  ("doc_9999", 0.15),
]
```

**Thay đổi quan trọng**:
- **Trước RRF**: doc_1234 rank 2, doc_3333 rank 3
- **Sau BERT**: doc_3333 vượt lên rank 2
- **Lý do**: BERT đọc hiểu nội dung sâu hơn, phát hiện doc_3333 liên quan hơn về ngữ nghĩa

#### Kết quả Giai đoạn 3: Danh sách cuối cùng

```python
final_results = [
  {
    "doc_id": "doc_5678",
    "score": 0.87,
    "title": "COVID-19 Symptoms Overview",
    "snippet": "COVID-19 symptoms include fever, cough, shortness of breath..."
  },
  {
    "doc_id": "doc_3333",
    "score": 0.85,
    "title": "Understanding SARS-CoV-2 Infection",
    "snippet": "SARS-CoV-2 causes respiratory symptoms such as..."
  },
  ...
]
```

---

## 🔹 KẾT QUẢ CUỐI CÙNG TRẢ VỀ NGƯỜI DÙNG

```json
{
  "query": "covid symptoms",
  "query_variants": [
    "covid symptoms",
    "covid symptoms fever cough",
    "covid symptoms SARS-CoV-2 infection",
    "covid symptoms headache fatigue"
  ],
  "results": [
    {
      "rank": 1,
      "doc_id": "doc_5678",
      "score": 0.87,
      "title": "COVID-19 Symptoms Overview",
      "snippet": "COVID-19 symptoms include fever, cough, shortness of breath..."
    },
    {
      "rank": 2,
      "doc_id": "doc_3333",
      "score": 0.85,
      "title": "Understanding SARS-CoV-2 Infection",
      "snippet": "SARS-CoV-2 causes respiratory symptoms such as..."
    },
    ...
  ],
  "latency": {
    "mining": 45.2,           # ms
    "reformulation": 12.8,    # ms
    "retrieval_fusion": 85.3, # ms
    "reranking": 320.5,       # ms
    "total": 463.8            # ms
  }
}
```

---

## 📊 SO SÁNH: TRƯỚC VÀ SAU

### Tìm kiếm truyền thống (chỉ BM25)
```
Query: "covid symptoms"
    ↓
BM25 Search
    ↓
Results: 100 docs
    ↓
User receives: Chỉ tài liệu chứa chính xác "covid" và "symptoms"
```

**Vấn đề**:
- Bỏ sót tài liệu viết "coronavirus", "SARS-CoV-2"
- Bỏ sót tài liệu viết "signs" thay vì "symptoms"

### Hệ thống Adaptive IR (4 giai đoạn)
```
Query: "covid symptoms"
    ↓
Candidate Mining
    ↓ Tìm được: fever, cough, SARS-CoV-2, headache, fatigue...

RL Reformulation
    ↓ Tạo ra 4 query variants

Multi-Query + RRF
    ↓ Tìm kiếm với 4 queries, kết hợp kết quả
    ↓ Tìm được nhiều tài liệu liên quan hơn

BERT Re-ranking
    ↓ Xếp hạng chính xác dựa trên ngữ nghĩa

User receives: Tài liệu đầy đủ, chính xác, đa dạng
```

**Lợi ích**:
✓ Tìm được tài liệu dùng từ đồng nghĩa
✓ Tìm được tài liệu dùng thuật ngữ chuyên môn
✓ Xếp hạng chính xác hơn nhờ BERT
✓ Giải quyết bounded recall problem

---

## 🎯 TẠI SAO RL AGENT QUAN TRỌNG?

### So sánh với cách mở rộng truy vấn truyền thống

**Cách truyền thống (RM3)**:
```
1. Lấy top-k docs
2. Tính TF-IDF
3. Chọn k từ có TF-IDF cao nhất (CỨNG NHẮC)
4. Thêm tất cả vào query
```
→ Không thông minh, không học được

**RL Agent**:
```
1. Lấy top-k docs
2. Tính features cho candidates
3. Agent QUYẾT ĐỊNH chọn từ nào (THÔNG MINH)
   - Dựa trên embedding
   - Dựa trên features
   - Dựa trên context
4. Agent HỌC từ feedback (reward)
   - Nếu chọn từ tốt → Recall tăng → Reward +
   - Nếu chọn từ xấu → Recall giảm → Reward -
5. Agent cải thiện qua thời gian
```
→ Thông minh, học được, thích nghi

### Ví dụ Agent học được gì

**Trước training**:
- Agent chọn random: "covid symptoms" → "covid symptoms the and of"
- Recall@100: 0.75

**Sau training**:
- Agent học được chọn từ có ý nghĩa: "covid symptoms" → "covid symptoms fever cough"
- Recall@100: 0.86 (+14.7%)

**Patterns agent học được**:
1. **Chọn từ đồng nghĩa**: "machine learning" → thêm "ML", "AI"
2. **Chọn từ cụ thể hơn**: "virus" → thêm "coronavirus", "SARS-CoV-2"
3. **Chọn từ mở rộng ngữ cảnh**: "treatment" → thêm "vaccine", "antiviral"
4. **Biết khi nào STOP**: Không thêm quá nhiều từ (tránh query drift)

---

## ⚙️ CÁCH CHẠY HỆ THỐNG

### 1. Training RL Agent

```bash
python train.py \
  --config configs/default_config.yaml \
  --device cuda \
  --epochs 50
```

**Quy trình training**:
```
For epoch 1 to 50:
  For each query in training set (500,000 queries):
    1. Mine candidates
    2. Agent chọn actions (thêm từ)
    3. Tìm kiếm với query mới
    4. Tính reward (Δ Recall@100)
    5. Cập nhật agent parameters (PPO)

  Validate on validation set
  Save best checkpoint
```

### 2. Inference (Tìm kiếm)

```bash
python inference.py \
  --query "covid symptoms" \
  --checkpoint models/best_model.pt
```

**Quy trình inference**:
```
1. Load trained RL agent
2. Load BM25 index
3. Load BERT re-ranker
4. Run 4-stage pipeline
5. Return results
```

---

## 📈 HIỆU SUẤT DỰ KIẾN

### Metrics

| Method | Recall@100 | MRR@10 | Latency (ms) |
|--------|-----------|---------|--------------|
| BM25 baseline | 0.75 | 0.28 | 50 |
| BM25 + RM3 | 0.79 | 0.31 | 120 |
| **BM25 + RL + RRF + BERT** | **0.86** | **0.41** | **464** |

**Trade-off**:
- Recall tăng 14.7% (0.75 → 0.86)
- MRR tăng 46.4% (0.28 → 0.41)
- Latency tăng 9.3x (50ms → 464ms)

**Khi nào dùng hệ thống này?**
- ✓ Khi độ chính xác quan trọng hơn tốc độ
- ✓ Khi cần tìm toàn diện (high recall)
- ✓ Khi query ngắn, mơ hồ
- ✗ Khi cần real-time (<100ms)
- ✗ Khi query đã rất cụ thể

---

## 🔍 TÓM TẮT LUỒNG XỬ LÝ

```
USER INPUT: "covid symptoms"
    ↓
┌─────────────────────────────────────────┐
│ STAGE 0: Candidate Mining              │
│ - BM25 search → top 50 docs             │
│ - TF-IDF analysis                       │
│ - Extract 80-100 candidate terms        │
│ Output: {fever, cough, headache, ...}   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ STAGE 1: RL Query Reformulation        │
│ - Actor-Critic Neural Network          │
│ - Select terms iteratively              │
│ - Generate 4 query variants             │
│ Output: [                               │
│   "covid symptoms",                     │
│   "covid symptoms fever cough",         │
│   "covid symptoms SARS-CoV-2 infection",│
│   "covid symptoms headache fatigue"     │
│ ]                                       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ STAGE 2: Multi-Query Retrieval + RRF   │
│ - BM25 search for each query variant    │
│ - RRF fusion: Σ 1/(k + rank_i(doc))    │
│ - Combine diverse results               │
│ Output: Top 100 fused documents         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ STAGE 3: BERT Cross-Encoder Re-ranking │
│ - BERT reads (query, doc) pairs        │
│ - Semantic relevance scoring            │
│ - Final ranking by BERT scores          │
│ Output: Top 100 re-ranked documents     │
└─────────────────────────────────────────┘
    ↓
FINAL RESULTS → USER
```

---

## ❓ CÂU HỎI THƯỜNG GẶP

### Q1: Tại sao không dùng BERT từ đầu?
**A**: BERT rất chậm (20-30ms/doc). Với 8.8M docs, cần 73 giờ để score tất cả. Do đó:
- Stage 1-2: Lọc nhanh xuống 100-1000 docs (BM25 + RRF)
- Stage 3: BERT chỉ re-rank 100 docs → Chỉ 2-3 giây

### Q2: Tại sao cần 4 query variants?
**A**: Mỗi variant nhắm khía cạnh khác nhau:
- Variant 1: Thuật ngữ chung
- Variant 2: Thuật ngữ chuyên môn
- Variant 3: Từ đồng nghĩa
- Variant 4: Ngữ cảnh mở rộng

4 variants là sweet spot giữa recall và latency.

### Q3: RL Agent học như thế nào?
**A**: PPO (Proximal Policy Optimization):
```
1. Agent chọn actions (thêm từ vào query)
2. Tính reward = Δ Recall@100
3. Cập nhật policy để maximize reward
4. Lặp lại hàng nghìn lần
```

### Q4: RRF tốt hơn CombSUM như thế nào?
**A**:
- **CombSUM**: score(doc) = Σ BM25_score_i(doc) → Phải normalize scores
- **RRF**: score(doc) = Σ 1/(k + rank_i(doc)) → Không cần normalize, robust hơn

---

**File này giải thích chi tiết luồng xử lý của hệ thống Adaptive Information Retrieval**
**Được tạo**: 2026-01-19
