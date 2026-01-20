# HƯỚNG DẪN PULL DATA TỪ GIT LFS

## Vấn đề

Các file trong folder `Query Reformulator/` hiện tại chỉ là **Git LFS pointers** (134 bytes), không phải data thật:
- `msa_dataset.hdf5`: 134B (thật ra nên là ~474MB)
- `msa_corpus.hdf5`: 134B (thật ra nên là ~500MB)
- `D_cbow_pdw_8B.pkl`: 134B (thật ra nên là ~2-4GB)

## Giải pháp

### Bước 1: Cài Git LFS

**macOS**:
```bash
brew install git-lfs
```

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install git-lfs
```

**Windows**:
- Download từ: https://git-lfs.github.com/
- Hoặc dùng: `choco install git-lfs`

### Bước 2: Initialize Git LFS

```bash
cd /Users/vanhkhongpeo/Documents/Github/Adaptive_information_retrival/adaptive-information-retrieval

# Initialize Git LFS
git lfs install
```

### Bước 3: Pull Data

```bash
# Pull tất cả LFS files
git lfs pull

# Hoặc pull từng file cụ thể
git lfs pull --include="Query Reformulator/msa_dataset.hdf5"
git lfs pull --include="Query Reformulator/msa_corpus.hdf5"
git lfs pull --include="Query Reformulator/D_cbow_pdw_8B.pkl"
```

**Lưu ý**: Quá trình này có thể mất 10-30 phút tùy tốc độ mạng (tải ~5-6GB).

### Bước 4: Verify Data

```bash
# Kiểm tra kích thước
ls -lh "Query Reformulator/"

# Nên thấy:
# msa_dataset.hdf5    ~474MB
# msa_corpus.hdf5     ~500MB+
# D_cbow_pdw_8B.pkl   ~2-4GB
```

## Nếu Git LFS Pull Không Hoạt Động

### Option A: Download từ Google Drive

Link gốc: https://drive.google.com/drive/folders/0BwmD_VLjROrfLWk3QmctMXpWRkE?usp=sharing

```bash
# 1. Download thủ công từ Google Drive qua browser
# 2. Di chuyển vào folder

cd /Users/vanhkhongpeo/Documents/Github/Adaptive_information_retrival/adaptive-information-retrieval

# Xóa pointers cũ
rm "Query Reformulator/msa_dataset.hdf5"
rm "Query Reformulator/msa_corpus.hdf5"
rm "Query Reformulator/D_cbow_pdw_8B.pkl"

# Di chuyển file mới tải
mv ~/Downloads/msa_dataset.hdf5 "Query Reformulator/"
mv ~/Downloads/msa_corpus.hdf5 "Query Reformulator/"
mv ~/Downloads/D_cbow_pdw_8B_norm.pkl "Query Reformulator/D_cbow_pdw_8B.pkl"
```

### Option B: Dùng MS MARCO thay thế

Nếu không download được legacy data, dùng MS MARCO (khuyến nghị hơn):

```bash
cd adaptive-ir-system

# Download MS MARCO (tự động, không cần Git LFS)
python scripts/download_msmarco.py --data_dir ./data/msmarco

# Build index
python scripts/build_index.py \
  --collection ./data/msmarco/collection.tsv \
  --index ./data/msmarco/index
```

## Test Data Sau Khi Pull

```bash
cd adaptive-ir-system

# Test script
python scripts/test_legacy_data.py --data_dir "../Query Reformulator"
```

Hoặc test trong Python:

```python
import h5py
import pickle

# Test dataset
print("Testing msa_dataset.hdf5...")
with h5py.File('../Query Reformulator/msa_dataset.hdf5', 'r') as f:
    print(f"  Keys: {list(f.keys())}")

# Test corpus
print("Testing msa_corpus.hdf5...")
with h5py.File('../Query Reformulator/msa_corpus.hdf5', 'r') as f:
    print(f"  Keys: {list(f.keys())}")

# Test embeddings
print("Testing D_cbow_pdw_8B.pkl...")
with open('../Query Reformulator/D_cbow_pdw_8B.pkl', 'rb') as f:
    embeddings = pickle.load(f, encoding='latin1')
    print(f"  Embeddings: {len(embeddings)} words, {list(embeddings.keys())[:5]}")
```

## Khi nào dùng Legacy Data vs MS MARCO?

| Dataset | Ưu điểm | Nhược điểm | Khi nào dùng |
|---------|---------|------------|--------------|
| **Legacy (MSA, Jeopardy)** | - Nhỏ hơn, train nhanh<br>- Có sẵn Word2Vec embeddings | - Cũ (2017)<br>- Cần Git LFS hoặc download thủ công | Testing nhanh, máy yếu |
| **MS MARCO** | - Mới nhất<br>- Dataset chuẩn<br>- Cộng đồng lớn | - Lớn hơn (10GB)<br>- Train lâu hơn | Production, research chính thức |

## Tóm tắt Lệnh

```bash
# Pull Git LFS (khuyến nghị)
cd /Users/vanhkhongpeo/Documents/Github/Adaptive_information_retrival/adaptive-information-retrieval
git lfs install
git lfs pull

# Verify
ls -lh "Query Reformulator/"

# Test
cd adaptive-ir-system
python scripts/test_legacy_data.py --data_dir "../Query Reformulator"
```
