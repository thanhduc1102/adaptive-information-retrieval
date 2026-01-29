"""
HuggingFace Model Uploader

Automatically upload model checkpoints to HuggingFace Hub.
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class HuggingFaceUploader:
    """
    Upload model checkpoints to HuggingFace Hub.
    
    Usage:
        uploader = HuggingFaceUploader(
            repo_id="username/adaptive-ir-model",
            token="hf_xxxxx"
        )
        uploader.upload_checkpoint(
            checkpoint_path="checkpoints/best_model.pt",
            metrics={'mrr': 0.45, 'recall@100': 0.78}
        )
    """
    
    def __init__(
        self,
        repo_id: str,
        token: str = None,
        private: bool = False,
        create_repo: bool = True
    ):
        """
        Initialize uploader.
        
        Args:
            repo_id: HuggingFace repo ID (e.g., 'username/model-name')
            token: HuggingFace access token (or set HF_TOKEN env var)
            private: Whether repo should be private
            create_repo: Create repo if it doesn't exist
        """
        self.repo_id = repo_id
        self.token = token or os.environ.get('HF_TOKEN')
        self.private = private
        
        if not self.token:
            raise ValueError(
                "HuggingFace token required. "
                "Pass token= or set HF_TOKEN environment variable."
            )
        
        # Import here to avoid dependency issues
        try:
            from huggingface_hub import HfApi, login, create_repo as hf_create_repo
            self.HfApi = HfApi
            self.hf_login = login
            self.hf_create_repo = hf_create_repo
        except ImportError:
            raise ImportError(
                "huggingface_hub not installed. "
                "Install with: pip install huggingface_hub"
            )
        
        # Login
        self.hf_login(token=self.token)
        self.api = self.HfApi()
        
        # Create repo if needed
        if create_repo:
            self._ensure_repo_exists()
        
        logger.info(f"✅ HuggingFace uploader initialized for repo: {repo_id}")
    
    def _ensure_repo_exists(self):
        """Create repository if it doesn't exist."""
        try:
            self.api.repo_info(repo_id=self.repo_id)
            logger.info(f"Repository {self.repo_id} already exists")
        except Exception:
            logger.info(f"Creating new repository: {self.repo_id}")
            self.hf_create_repo(
                repo_id=self.repo_id,
                token=self.token,
                private=self.private,
                repo_type="model",
                exist_ok=True
            )
    
    def upload_checkpoint(
        self,
        checkpoint_path: str,
        metrics: Dict[str, float] = None,
        config: Dict[str, Any] = None,
        commit_message: str = None,
        subfolder: str = None
    ) -> str:
        """
        Upload a checkpoint file to HuggingFace Hub.
        
        Args:
            checkpoint_path: Path to .pt checkpoint file
            metrics: Training metrics to include in metadata
            config: Model configuration to include
            commit_message: Git commit message
            subfolder: Optional subfolder in repo
            
        Returns:
            URL of uploaded file
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Generate commit message
        if commit_message is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            metrics_str = ""
            if metrics:
                metrics_str = " | " + " | ".join(
                    f"{k}: {v:.4f}" for k, v in metrics.items()
                )
            commit_message = f"Upload {checkpoint_path.name} - {timestamp}{metrics_str}"
        
        # Determine path in repo
        if subfolder:
            path_in_repo = f"{subfolder}/{checkpoint_path.name}"
        else:
            path_in_repo = checkpoint_path.name
        
        logger.info(f"📤 Uploading {checkpoint_path} to {self.repo_id}/{path_in_repo}")
        
        # Upload checkpoint
        try:
            url = self.api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                token=self.token,
                commit_message=commit_message
            )
            logger.info(f"✅ Uploaded: {url}")
        except Exception as e:
            logger.error(f"❌ Upload failed: {e}")
            raise
        
        # Upload metadata
        if metrics or config:
            self._upload_metadata(
                checkpoint_name=checkpoint_path.stem,
                metrics=metrics,
                config=config,
                subfolder=subfolder
            )
        
        return url
    
    def _upload_metadata(
        self,
        checkpoint_name: str,
        metrics: Dict[str, float] = None,
        config: Dict[str, Any] = None,
        subfolder: str = None
    ):
        """Upload metadata JSON file."""
        metadata = {
            'checkpoint_name': checkpoint_name,
            'upload_time': datetime.now().isoformat(),
            'metrics': metrics or {},
            'config': config or {}
        }
        
        # Write to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(metadata, f, indent=2, default=str)
            temp_path = f.name
        
        try:
            path_in_repo = f"{checkpoint_name}_metadata.json"
            if subfolder:
                path_in_repo = f"{subfolder}/{path_in_repo}"
            
            self.api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                token=self.token,
                commit_message=f"Upload metadata for {checkpoint_name}"
            )
            logger.info(f"✅ Uploaded metadata: {path_in_repo}")
        finally:
            os.unlink(temp_path)
    
    def upload_directory(
        self,
        directory: str,
        pattern: str = "*.pt",
        commit_message: str = None
    ) -> list:
        """
        Upload all checkpoints in a directory.
        
        Args:
            directory: Path to checkpoints directory
            pattern: Glob pattern for files to upload
            commit_message: Git commit message
            
        Returns:
            List of uploaded file URLs
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        files = list(directory.glob(pattern))
        if not files:
            logger.warning(f"No files matching {pattern} in {directory}")
            return []
        
        logger.info(f"📁 Uploading {len(files)} files from {directory}")
        
        urls = []
        for file_path in files:
            try:
                url = self.upload_checkpoint(
                    checkpoint_path=str(file_path),
                    commit_message=commit_message
                )
                urls.append(url)
            except Exception as e:
                logger.error(f"Failed to upload {file_path}: {e}")
        
        return urls
    
    def upload_training_results(
        self,
        checkpoint_dir: str,
        final_metrics: Dict[str, float] = None,
        config: Dict[str, Any] = None,
        upload_all: bool = False
    ) -> Dict[str, str]:
        """
        Upload training results including best/final checkpoints.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            final_metrics: Final validation/test metrics
            config: Training configuration
            upload_all: Upload all checkpoints (not just best/final)
            
        Returns:
            Dictionary mapping filename to upload URL
        """
        checkpoint_dir = Path(checkpoint_dir)
        results = {}
        
        # Priority files to upload
        priority_files = ['best_model.pt', 'final_model.pt']
        
        for filename in priority_files:
            file_path = checkpoint_dir / filename
            if file_path.exists():
                try:
                    url = self.upload_checkpoint(
                        checkpoint_path=str(file_path),
                        metrics=final_metrics,
                        config=config
                    )
                    results[filename] = url
                except Exception as e:
                    logger.error(f"Failed to upload {filename}: {e}")
        
        # Upload test results if exists
        test_results_path = checkpoint_dir / "test_results.json"
        if test_results_path.exists():
            try:
                self.api.upload_file(
                    path_or_fileobj=str(test_results_path),
                    path_in_repo="test_results.json",
                    repo_id=self.repo_id,
                    token=self.token
                )
                logger.info("✅ Uploaded test_results.json")
            except Exception as e:
                logger.error(f"Failed to upload test_results.json: {e}")
        
        # Optionally upload all checkpoints
        if upload_all:
            for ckpt in checkpoint_dir.glob("checkpoint_*.pt"):
                try:
                    url = self.upload_checkpoint(str(ckpt))
                    results[ckpt.name] = url
                except Exception as e:
                    logger.error(f"Failed to upload {ckpt.name}: {e}")
        
        # Create and upload README
        self._create_model_card(checkpoint_dir, final_metrics, config)
        
        return results
    
    def _create_model_card(
        self,
        checkpoint_dir: Path,
        metrics: Dict[str, float] = None,
        config: Dict[str, Any] = None
    ):
        """Create and upload model card README."""
        readme_content = f"""---
tags:
- information-retrieval
- query-reformulation
- reinforcement-learning
- ppo
---

# Adaptive IR Query Reformulator

RL-based query reformulation model trained with PPO.

## Model Description

This model uses a Transformer-based Actor-Critic architecture to reformulate 
search queries by selecting expansion terms. It was trained using Proximal 
Policy Optimization (PPO) on the MS Academic dataset.

## Training Results

"""
        if metrics:
            readme_content += "| Metric | Value |\n|--------|-------|\n"
            for key, value in metrics.items():
                if isinstance(value, float):
                    readme_content += f"| {key} | {value:.4f} |\n"
                else:
                    readme_content += f"| {key} | {value} |\n"
        
        readme_content += """
## Usage

```python
import torch
from src.rl_agent import QueryReformulatorAgent

# Load model
checkpoint = torch.load('best_model.pt')
agent = QueryReformulatorAgent(config)
agent.load_state_dict(checkpoint['model_state_dict'])
```

## Configuration

"""
        if config:
            readme_content += f"```yaml\n{json.dumps(config, indent=2, default=str)}\n```\n"
        
        # Write to temp file and upload
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False
        ) as f:
            f.write(readme_content)
            temp_path = f.name
        
        try:
            self.api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo="README.md",
                repo_id=self.repo_id,
                token=self.token,
                commit_message="Update model card"
            )
            logger.info("✅ Uploaded README.md")
        finally:
            os.unlink(temp_path)


def setup_huggingface_uploader(config: Dict) -> Optional[HuggingFaceUploader]:
    """
    Setup HuggingFace uploader from config.
    
    Config format:
        huggingface:
            enabled: true
            repo_id: "username/model-name"
            token: "hf_xxxxx"  # or set HF_TOKEN env var
            private: false
    """
    hf_config = config.get('huggingface', {})
    
    if not hf_config.get('enabled', False):
        return None
    
    repo_id = hf_config.get('repo_id')
    token = hf_config.get('token') or os.environ.get('HF_TOKEN')
    
    if not repo_id:
        logger.warning("HuggingFace repo_id not configured")
        return None
    
    if not token:
        logger.warning("HuggingFace token not configured")
        return None
    
    try:
        return HuggingFaceUploader(
            repo_id=repo_id,
            token=token,
            private=hf_config.get('private', False)
        )
    except Exception as e:
        logger.error(f"Failed to setup HuggingFace uploader: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload checkpoints to HuggingFace")
    parser.add_argument("--repo", required=True, help="HuggingFace repo ID")
    parser.add_argument("--token", help="HuggingFace token (or set HF_TOKEN)")
    parser.add_argument("--checkpoint-dir", default="./checkpoints_msa_optimized")
    parser.add_argument("--upload-all", action="store_true")
    
    args = parser.parse_args()
    
    uploader = HuggingFaceUploader(
        repo_id=args.repo,
        token=args.token
    )
    
    results = uploader.upload_training_results(
        checkpoint_dir=args.checkpoint_dir,
        upload_all=args.upload_all
    )
    
    print("\n📦 Upload Results:")
    for filename, url in results.items():
        print(f"  {filename}: {url}")
