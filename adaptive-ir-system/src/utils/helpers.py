"""
Utility functions and helpers
"""

import os
import yaml
import json
import logging
import colorlog
import random
import numpy as np
import torch
from typing import Dict, Any, Optional
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup colored logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)-8s%(reset)s %(blue)s%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)-8s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    # print(f"Loading config from {config_path}")
    # # with open(str(config_path), 'r') as f:
    # config = yaml.safe_load()
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Dict, filepath: str, indent: int = 2):
    """Save to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(directory: str):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get torch device.
    
    Args:
        device: 'cuda', 'cpu', 'mps', or None (auto-detect)
        
    Returns:
        torch.device
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    return torch.device(device)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    
    def __init__(self, patience: int = 10, mode: str = 'max', delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait before stopping
            mode: 'max' (higher is better) or 'min' (lower is better)
            delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_number(num: int) -> str:
    """Format large numbers with commas."""
    return f"{num:,}"


class ConfigManager:
    """
    Configuration manager with environment variable support and validation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Path to config file (YAML)
        """
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = {}
        
        # Override with environment variables
        self._apply_env_overrides()
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Example: SYSTEM_DEVICE=cuda overrides system.device
        for key, value in os.environ.items():
            if key.startswith('SYSTEM_') or key.startswith('DATASET_') or key.startswith('MODEL_'):
                parts = key.lower().split('_', 1)
                if len(parts) == 2:
                    section, param = parts
                    if section in self.config:
                        self.config[section][param] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value using dot notation.
        
        Args:
            key: Config key (e.g., 'system.device')
            default: Default value if key not found
            
        Returns:
            Config value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set config value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def validate(self) -> bool:
        """Validate configuration."""
        required_keys = [
            'system.device',
            'dataset.name',
            'retrieval.engine'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                logging.error(f"Missing required config key: {key}")
                return False
        
        return True
    
    def to_dict(self) -> Dict:
        """Get full config as dictionary."""
        return self.config


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Evaluation metrics
        path: Save path
    """
    import torch
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, path)


def load_checkpoint(model, path, optimizer=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load into
        path: Checkpoint path
        optimizer: Optional optimizer to load state
        device: Device to load to
        
    Returns:
        (epoch, metrics)
    """
    import torch
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    return epoch, metrics


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities:")
    
    # Setup logging
    logger = setup_logging("INFO")
    logger.info("Logger initialized")
    logger.debug("This won't show (level=INFO)")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    # Test config manager
    print("\nTesting ConfigManager:")
    config = ConfigManager()
    config.set('system.device', 'cuda')
    config.set('model.hidden_dim', 256)
    print(f"system.device = {config.get('system.device')}")
    print(f"model.hidden_dim = {config.get('model.hidden_dim')}")
    
    # Test utilities
    print("\nTesting utility functions:")
    print(f"Format time: {format_time(3725)}")
    print(f"Format number: {format_number(1234567)}")
    
    # Test early stopping
    print("\nTesting EarlyStopping:")
    early_stop = EarlyStopping(patience=3, mode='max')
    scores = [0.5, 0.6, 0.65, 0.64, 0.63, 0.62, 0.61]
    for i, score in enumerate(scores):
        should_stop = early_stop(score)
        print(f"  Epoch {i+1}: score={score}, stop={should_stop}")
        if should_stop:
            break
