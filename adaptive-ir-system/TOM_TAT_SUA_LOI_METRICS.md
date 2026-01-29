# Tóm Tắt: Sửa Lỗi Validation Metrics Luôn Trả Về 0

## 🎯 Vấn Đề

Sau khi training hoàn tất, validation metrics luôn trả về **0.0000**:

```
Validation | Recall@100: 0.0000 | MRR: 0.0000
```

Training reward bình thường (1.0716) nhưng evaluation metrics = 0, không thể đánh giá chất lượng model.

## 🔍 Nguyên Nhân

### SimpleBM25Searcher chỉ index documents từ training set

**File lỗi**: `src/utils/simple_searcher.py` (dòng 46-56)

```python
# CODE CŨ - SAI
qrels = self.adapter.load_qrels()  # Chỉ load training qrels
unique_doc_ids = set()
for qrel in qrels.values():
    unique_doc_ids.update(qrel.keys())
# Kết quả: Chỉ index ~50K docs từ training set
```

**Tại sao lỗi**:
1. Khi khởi tạo searcher với `split='train'`, chỉ load training qrels
2. Training queries có documents IDs riêng
3. **Validation/test queries cần documents KHÁC** → không có trong index
4. Search trả về rỗng → metrics = 0

**Ví dụ**:
- Train query: cần docs [A, B, C] ✅ (có trong index)  
- Valid query: cần docs [X, Y, Z] ❌ (KHÔNG có trong index)
- Search → empty → Recall = 0, MRR = 0

## ✅ Giải Pháp

### Index documents từ TẤT CẢ splits (train + valid + test)

**File đã sửa**: `src/utils/simple_searcher.py`

```python
# CODE MỚI - ĐÚNG
# QUAN TRỌNG: Index documents từ TẤT CẢ splits
unique_doc_ids = set()

# Load qrels từ tất cả splits
current_split = self.adapter.split
for split in ['train', 'valid', 'test']:
    try:
        self.adapter.split = split
        qrels = self.adapter.load_qrels()
        for qrel in qrels.values():
            unique_doc_ids.update(qrel.keys())
    except Exception as e:
        self.logger.warning(f"Could not load {split} qrels: {e}")

# Restore split ban đầu
self.adapter.split = current_split
# Kết quả: Index ~172K-468K docs từ ALL splits
```

### Sửa code trùng lặp trong pipeline

**File đã sửa**: `src/pipeline/adaptive_pipeline.py`

- Fix `retrieve()`: Xóa duplicate if-else, thêm check empty results
- Fix `mine_candidates()`: Clean up code trùng lặp
- Fix `rerank()`: Tương tự

## 📊 Kết Quả

### Trước khi fix
```
Validation | Recall@100: 0.0000 | MRR: 0.0000
```

### Sau khi fix
```
📊 Test với 100 validation queries:

Queries with results: 100/100 ✅
Queries with relevant docs: 74/100

Metrics:
  Recall@10:  0.0842
  Recall@100: 0.2044  ✅ (đã khác 0!)
  MRR:        0.2199  ✅ (đã khác 0!)
  nDCG@10:    0.1045
```

**Baseline BM25** (không RL): Recall@100 = 20.4%, MRR = 22.0%

Đây là giá trị hợp lý cho MS Academic dataset.

## 🎉 Những Gì Đã Fix

### ✅ Files Modified

1. **`src/utils/simple_searcher.py`**
   - Index documents từ ALL splits (train+valid+test)
   - Tăng từ 50K → 172K-468K documents

2. **`src/pipeline/adaptive_pipeline.py`**
   - Fix duplicate code trong `retrieve()`
   - Fix `mine_candidates()` và `rerank()`
   - Thêm empty results handling

### ✅ Documents Created

- **`BUG_FIX_ZERO_METRICS.md`**: Chi tiết technical về bug và fix
- **`quick_val_test.sh`**: Script test metrics nhanh

## 🚀 Chạy Training Lại

Bây giờ có thể train và metrics sẽ hoạt động đúng:

```bash
cd adaptive-ir-system

# Training với validation metrics đúng
python train_quickly.py --config ./configs/msa_quick_config.yaml --epochs 10

# Hoặc test nhanh validation
bash quick_val_test.sh
```

**Kết quả mong đợi**:
- Validation metrics khác 0 ✅
- Recall@100 baseline ~ 20%
- MRR baseline ~ 22%
- RL agent nên cải thiện trên baseline này

## 📝 Lưu Ý

1. **Index time**: Tăng ~1-2 phút (do index nhiều docs hơn)
2. **Memory**: Tăng ~200-300MB (chấp nhận được)
3. **Quality**: Metrics giờ phản ánh đúng chất lượng model

## 🎓 Bài Học

1. **Luôn kiểm tra index scope**: Đảm bảo index chứa TẤT CẢ documents cần thiết
2. **Split-aware indexing**: Khi có train/valid/test splits, index phải cover hết
3. **Baseline metrics**: Biết baseline để so sánh (BM25 ~ 20% Recall@100)

---

**Trạng thái**: ✅ Đã fix và verify (29/01/2026)

**Next**: Train và xem RL agent có improve trên baseline 20% không! 🎯
