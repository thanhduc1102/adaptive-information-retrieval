# Adaptive Information Retrieval System - Setup and Training Guide

This guide provides step-by-step instructions to set up the environment, download the necessary datasets, and start training the Adaptive IR system.

## 1. Clone the Repository

First, clone the specific branch of the project and navigate into the directory.

```bash
git clone -b nnminh322_optim https://github.com/thanhduc1102/adaptive-information-retrieval

cd adaptive-information-retrieval
```

### 2. Prepare Environment & Kaggle Credentials
Clean up old directories and configure your Kaggle API credentials.
```bash
# Remove legacy directory if it exists
rm -rf ./Query\ Reformulator/

# Setup Kaggle directory and API token
mkdir -p ~/.kaggle
echo '{"username":"YOUR_KAGGLE_USER_NAME","key":"YOUR_KAGGLE_KEY"}' > ~/.kaggle/kaggle.json
```

## 3. Download Dataset
(You must be install kaggle_cli for download this dataset. If you running in kaggle env, ignore this. Otherwise, please install kaggle_cli by Google Searching :)))) or handcraft downloading )
```bash
kaggle datasets download thanhduc1108/ir-msmarco-webmining
unzip ir-msmarco-webmining.zip
```

## 4. Install Dependencies
Navigate to the system directory and install Python dependencies using uv
```bash
cd ./adaptive-ir-system/

# Install uv package manager
pip install uv

# Install requirements
uv pip install -r requirements.txt 

# Optional: If running in a root environment (like a Kaggle server or Docker container)
# uv pip install -r requirements.txt --system

uv pip install --upgrade huggingface_hub transformers peft
uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 5. System Configuration (Java)
The system requires Java (OpenJDK 21) for Pyserini/Lucene operations.
```bash
# Update repositories and install JDK 21
apt-get update --allow-releaseinfo-change && apt-get install -y openjdk-21-jdk

# Set JAVA_HOME environment variable
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
```


## 6. Training

### Quick Training (Recommended)
Run the training script with the optimized configuration.

```bash
# Using legacy Word2Vec model (D_cbow_pdw_8B) - ~4.5h per epoch with 271k queries (2x T4 GPU)
python train_quickly.py --config ./configs/msa_quick_config.yaml --epochs 10
```

### Training with Modern Embeddings
```bash
# Using sentence-transformers (all-mpnet-base-v2) - ~25h per epoch with 271k queries (T4 GPU)
python train.py --config ./configs/msa_config.yaml --epochs 50 
```

### Key Improvements (v1.1)
- **Relevance-based Reward Signal**: Agent now learns to select terms that appear in relevant documents
- **HuggingFace Integration**: Auto-upload best model to HuggingFace Hub
- **Checkpoint Every Epoch**: All checkpoints saved for analysis
- **POC Test Script**: Verify setup before full training

## 7. POC Test (Optional but Recommended)
Before full training, verify all components work correctly:

```bash
cd adaptive-ir-system
python poc_test.py
```

This will test:
- Dataset loading
- Embedding model
- Candidate mining
- Relevance signal extraction
- Reward function
- HuggingFace config

## 8. Checkpoint Evaluation
Conveniently evaluate any checkpoint:

```bash
# Evaluate on validation set
python eval_checkpoint.py --checkpoint checkpoints_msa_optimized/best_model.pt --split valid

# Evaluate on test set  
python eval_checkpoint.py --checkpoint checkpoints_msa_optimized/best_model.pt --split test

# Download and evaluate from HuggingFace
python eval_checkpoint.py --hf-model username/model-name --split test
```

## 9. HuggingFace Upload (Optional)
To automatically upload checkpoints to HuggingFace:

1. Edit `configs/msa_quick_config.yaml`:
```yaml
huggingface:
  enabled: true
  repo_id: 'your-username/adaptive-ir-model'
  token: 'hf_your_token_here'  # Or set HF_TOKEN env var
  private: false
```

2. Or use environment variable:
```bash
export HF_TOKEN="your_huggingface_token"
```

## 10. Configuration Details

Key settings in `configs/msa_quick_config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.num_epochs` | 10 | Number of training epochs |
| `training.batch_size` | 128 | PPO mini-batch size |
| `training.collect_batch_size` | 128 | Episode collection batch size |
| `training.save_freq` | 1 | Save checkpoint every N epochs |
| `training.reward_mode` | 'improved' | Reward function mode |
| `training.use_amp` | true | Mixed precision (FP16) |
| `huggingface.enabled` | false | Auto-upload to HuggingFace |

## Checkpoints
Saved to `checkpoints_msa_optimized/`:
- `checkpoint_epoch_N.pt` - Checkpoint after epoch N
- `best_model.pt` - Best model by MRR
- `final_model.pt` - Final model after training
- `test_results.json` - Test metrics (if `--test` flag used)
