#!/usr/bin/env python3
"""
Save Current Model Checkpoint

Script để lưu checkpoint từ model hiện tại sau khi training hoàn thành.
Sử dụng khi training xong nhưng checkpoint chưa được lưu.

Usage:
    python scripts/save_current_model.py --config configs/msa_quick_config.yaml
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import argparse
import yaml
import torch

# Setup Java for Pyserini BEFORE imports
def check_and_setup_java():
    """Find and configure Java for Pyserini."""
    java_dirs = [
        '/usr/lib/jvm/java-21-openjdk-amd64',
        '/usr/lib/jvm/java-17-openjdk-amd64',
        '/usr/lib/jvm/java-11-openjdk-amd64',
        '/usr/lib/jvm/default-java',
    ]
    
    for java_dir in java_dirs:
        if os.path.exists(java_dir):
            os.environ['JAVA_HOME'] = java_dir
            print(f"✅ Set JAVA_HOME={java_dir}")
            return True
    
    print("⚠️ Java not found - Pyserini may not work")
    return False

check_and_setup_java()

from src.rl_agent import QueryReformulatorAgent
from src.utils import save_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Save current trained model')
    parser.add_argument('--config', type=str, default='configs/msa_quick_config.yaml',
                        help='Path to config file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for checkpoint (default: from config)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get checkpoint directory
    checkpoint_dir = Path(args.output_dir or config.get('training', {}).get('checkpoint_dir', './checkpoints_msa_optimized'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Checkpoint directory: {checkpoint_dir}")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device}")
    
    # Create agent with same config
    agent_config = config.get('rl_agent', {})
    embedding_dim = agent_config.get('embedding_dim', 500)
    hidden_dim = agent_config.get('hidden_dim', 256)
    num_heads = agent_config.get('num_heads', 4)
    num_layers = agent_config.get('num_layers', 2)
    max_candidates = config.get('candidate_mining', {}).get('max_candidates', 50)
    max_steps = agent_config.get('max_steps_per_episode', 5)
    num_features = 3  # tfidf, bm25_contrib, keybert
    
    print(f"\n📊 Agent Configuration:")
    print(f"  embedding_dim: {embedding_dim}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_layers: {num_layers}")
    print(f"  max_candidates: {max_candidates}")
    print(f"  max_steps: {max_steps}")
    
    # Initialize agent
    agent = QueryReformulatorAgent(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_candidates=max_candidates,
        max_steps=max_steps,
        num_features=num_features
    ).to(device)
    
    # Create optimizer (needed for checkpoint format)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.0003)
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / "final_model.pt"
    best_path = checkpoint_dir / "best_model.pt"
    
    print(f"\n💾 Saving checkpoints...")
    
    # Note: This saves the model architecture with random weights
    # If you want to save the trained weights, you need to have them in memory
    
    save_checkpoint(
        agent,
        optimizer,
        epoch=0,
        metrics={'note': 'Model architecture checkpoint - weights may not be trained'},
        path=checkpoint_path
    )
    print(f"  ✅ Saved: {checkpoint_path}")
    
    save_checkpoint(
        agent,
        optimizer,
        epoch=0,
        metrics={'note': 'Model architecture checkpoint - weights may not be trained'},
        path=best_path
    )
    print(f"  ✅ Saved: {best_path}")
    
    print(f"\n⚠️  QUAN TRỌNG:")
    print(f"   Checkpoint này chỉ lưu kiến trúc model với random weights.")
    print(f"   Để lưu weights đã train, bạn cần:")
    print(f"   1. Không tắt kernel sau khi training")
    print(f"   2. Hoặc training lại với code đã fix")
    print(f"\n🔧 Đề xuất: Chạy lại training với config đã cập nhật:")
    print(f"   python train_quickly.py --config {args.config} --epochs 1")


if __name__ == "__main__":
    main()
