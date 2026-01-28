# Adaptive Information Retrieval System - Setup and Training Guide

This guide provides step-by-step instructions to set up the environment, download the necessary datasets, and start training the Adaptive IR system.

## 1. Clone the Repository

First, clone the specific branch of the project and navigate into the directory.

```bash
git clone -b nnminh322_optim [https://github.com/thanhduc1102/adaptive-information-retrieval](https://github.com/thanhduc1102/adaptive-information-retrieval)

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
Run the training script with the specified configuration.

```bash
# If you want to using modern modal (as sentence-transformers/all-mpnet-base-v2 2019). ~ 25h for 1 full epoch with 271k sample (at GPU T4)
python train.py --config ./configs/msa_config.yaml --epochs 50 

# If you want to using old model (D_cbow_pdw_8B base on word2vec 2013) ~ 8h for 1 full epoch with 271k sample (at GPU T4)
python train_quickly.py --config ./configs/msa_quick_config.yaml --batch-size 64 --epochs 50
```
