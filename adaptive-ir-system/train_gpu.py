#!/usr/bin/env python3
"""
GPU-Focused Training Script for Adaptive IR System.

Strategy to maximize GPU utilization:
1. Pre-compute BM25 search results and cache them
2. Use cached results during training to avoid CPU bottleneck
3. Keep training loop focused on GPU operations

Author: Based on existing src/ modules
"""

import os
import sys
import argparse
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import pickle
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from existing src modules
from src.rl_agent import QueryReformulatorAgent, RLTrainer
from src.utils.legacy_loader import LegacyDatasetAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GPUTrainingConfig:
    """Training configuration."""
    # Data paths
    dataset_path: str = "/kaggle/input/ir-msmarco/Query Reformulator/msa_dataset.hdf5"
    corpus_path: str = "/kaggle/input/ir-msmarco/Query Reformulator/msa_corpus.hdf5"
    embeddings_path: str = "/kaggle/input/ir-msmarco/Query Reformulator/D_cbow_pdw_8B.pkl"
    
    # Model
    embed_dim: int = 500
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    
    # Training
    num_epochs: int = 10
    batch_size: int = 64  # Larger batch for GPU efficiency
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_steps: int = 5  # Max expansion terms per query
    
    # Candidates
    num_candidates: int = 30
    
    # Output
    checkpoint_dir: str = "checkpoints_gpu"
    cache_dir: str = "cache_gpu"


class PrecomputedDataset:
    """
    Dataset with pre-computed BM25 results and candidates.
    This allows training loop to focus on GPU operations.
    """
    
    def __init__(
        self,
        config: GPUTrainingConfig,
        split: str = 'train',
        max_queries: int = 10000
    ):
        self.config = config
        self.split = split
        self.max_queries = max_queries
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load embeddings
        self._load_embeddings()
        
        # Load or create cache
        cache_path = os.path.join(config.cache_dir, f'{split}_precomputed.pkl')
        os.makedirs(config.cache_dir, exist_ok=True)
        
        if os.path.exists(cache_path):
            self._load_cache(cache_path)
        else:
            self._create_cache(cache_path)
    
    def _load_embeddings(self):
        """Load word embeddings."""
        logger.info("Loading word embeddings...")
        if os.path.exists(self.config.embeddings_path):
            try:
                with open(self.config.embeddings_path, 'rb') as f:
                    self.word2vec = pickle.load(f, encoding='latin1')
                logger.info(f"Loaded {len(self.word2vec)} word embeddings")
            except Exception as e:
                logger.warning(f"Could not load embeddings: {e}")
                self.word2vec = {}
        else:
            self.word2vec = {}
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Embed text using word2vec."""
        tokens = text.lower().split()
        embeddings = []
        
        for token in tokens:
            if token in self.word2vec:
                embeddings.append(self.word2vec[token])
        
        if embeddings:
            return np.mean(embeddings, axis=0).astype(np.float32)
        else:
            return np.zeros(self.config.embed_dim, dtype=np.float32)
    
    def _load_cache(self, path: str):
        """Load precomputed cache."""
        logger.info(f"Loading precomputed cache from {path}...")
        with open(path, 'rb') as f:
            cache = pickle.load(f)
        
        self.query_ids = cache['query_ids']
        self.query_texts = cache['query_texts']
        self.query_embs = cache['query_embs']
        self.candidate_terms = cache['candidate_terms']
        self.candidate_embs = cache['candidate_embs']
        self.candidate_scores = cache['candidate_scores']
        self.baseline_metrics = cache['baseline_metrics']
        
        logger.info(f"Loaded {len(self.query_ids)} queries from cache")
    
    def _create_cache(self, cache_path: str):
        """Create precomputed cache."""
        logger.info("Creating precomputed cache...")
        
        # Load dataset
        adapter = LegacyDatasetAdapter(
            self.config.dataset_path,
            self.config.corpus_path,
            split=self.split
        )
        
        queries = adapter.load_queries()
        qrels = adapter.load_qrels()
        corpus = adapter.corpus
        
        # Build BM25 index
        from rank_bm25 import BM25Okapi
        
        logger.info("Building BM25 index...")
        corpus_docs = []
        doc_ids = []
        
        for doc_id in tqdm(range(len(corpus)), desc="Indexing"):
            doc = corpus.get_document(doc_id)
            if doc:
                corpus_docs.append(doc.lower().split())
                doc_ids.append(str(doc_id))
        
        bm25 = BM25Okapi(corpus_docs, k1=1.2, b=0.75)
        
        # Precompute for each query
        query_ids_list = list(queries.keys())[:self.max_queries]
        
        self.query_ids = []
        self.query_texts = []
        self.query_embs = []
        self.candidate_terms = []
        self.candidate_embs = []
        self.candidate_scores = []
        self.baseline_metrics = []
        
        logger.info(f"Processing {len(query_ids_list)} queries...")
        
        for qid in tqdm(query_ids_list, desc="Precomputing"):
            query = queries[qid]
            rel_docs = qrels.get(qid, {})
            relevant_set = set(rel_docs.keys())
            
            if not relevant_set:
                continue
            
            # Search and compute baseline
            query_tokens = query.lower().split()
            scores = bm25.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][:100]
            retrieved = [doc_ids[i] for i in top_indices if scores[i] > 0]
            
            # Baseline metrics
            recall_10 = len(set(retrieved[:10]) & relevant_set) / len(relevant_set)
            recall_100 = len(set(retrieved[:100]) & relevant_set) / len(relevant_set)
            mrr = 0.0
            for i, doc_id in enumerate(retrieved):
                if doc_id in relevant_set:
                    mrr = 1.0 / (i + 1)
                    break
            
            # Mine candidates from top docs
            top_20_indices = top_indices[:20]
            query_term_set = set(query_tokens)
            term_scores = {}
            
            for idx in top_20_indices:
                doc_tokens = corpus_docs[idx]
                doc_score = scores[idx]
                for token in doc_tokens:
                    if token not in query_term_set and len(token) > 2:
                        if token not in term_scores:
                            term_scores[token] = 0.0
                        term_scores[token] += doc_score
            
            # Top candidates
            sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
            top_terms = sorted_terms[:self.config.num_candidates]
            
            if not top_terms:
                continue
            
            terms = [t[0] for t in top_terms]
            term_scores_list = [t[1] for t in top_terms]
            max_score = max(term_scores_list) if term_scores_list else 1.0
            
            # Embed
            query_emb = self._embed_text(query)
            term_embs = np.stack([self._embed_text(t) for t in terms])
            
            # Pad to fixed size
            pad_size = self.config.num_candidates - len(terms)
            if pad_size > 0:
                term_embs = np.pad(term_embs, ((0, pad_size), (0, 0)))
                term_scores_list = term_scores_list + [0.0] * pad_size
                terms = terms + [''] * pad_size
            
            # Compute features (3 features)
            features = np.array([
                [s, np.log1p(s), s/max_score if max_score > 0 else 0]
                for s in term_scores_list
            ], dtype=np.float32)
            
            self.query_ids.append(qid)
            self.query_texts.append(query)
            self.query_embs.append(query_emb)
            self.candidate_terms.append(terms)
            self.candidate_embs.append(term_embs)
            self.candidate_scores.append(features)
            self.baseline_metrics.append({
                'recall@10': recall_10,
                'recall@100': recall_100,
                'mrr': mrr,
                'relevant': list(relevant_set)
            })
        
        # Convert to numpy
        self.query_embs = np.stack(self.query_embs)
        self.candidate_embs = np.stack(self.candidate_embs)
        self.candidate_scores = np.stack(self.candidate_scores)
        
        # Save cache
        logger.info(f"Saving cache to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'query_ids': self.query_ids,
                'query_texts': self.query_texts,
                'query_embs': self.query_embs,
                'candidate_terms': self.candidate_terms,
                'candidate_embs': self.candidate_embs,
                'candidate_scores': self.candidate_scores,
                'baseline_metrics': self.baseline_metrics
            }, f)
        
        logger.info(f"Cached {len(self.query_ids)} queries")
    
    def __len__(self):
        return len(self.query_ids)
    
    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Get a batch of precomputed data."""
        query_embs = torch.tensor(self.query_embs[indices], dtype=torch.float32)
        candidate_embs = torch.tensor(self.candidate_embs[indices], dtype=torch.float32)
        candidate_scores = torch.tensor(self.candidate_scores[indices], dtype=torch.float32)
        
        # Combine candidate embeddings and features
        # Expected: [batch, num_cands, embed_dim + 3]
        candidate_input = torch.cat([candidate_embs, candidate_scores], dim=-1)
        
        return {
            'query_emb': query_embs.to(self.device),
            'candidate_embs': candidate_embs.to(self.device),
            'candidate_features': candidate_scores.to(self.device),
            'baseline_metrics': [self.baseline_metrics[i] for i in indices],
            'candidate_terms': [self.candidate_terms[i] for i in indices]
        }


class GPUTrainer:
    """
    GPU-focused trainer that uses precomputed data.
    """
    
    def __init__(self, config: GPUTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load dataset
        self.train_data = PrecomputedDataset(config, split='train', max_queries=5000)
        self.valid_data = PrecomputedDataset(config, split='valid', max_queries=500)
        
        # Initialize model
        self._init_model()
        
        # Directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Metrics
        self.history = []
    
    def _init_model(self):
        """Initialize RL agent and trainer."""
        logger.info("Initializing model...")
        
        agent_config = {
            'embedding_dim': self.config.embed_dim,
            'hidden_dim': self.config.hidden_dim,
            'num_attention_heads': self.config.num_heads,
            'num_encoder_layers': self.config.num_layers,
            'dropout': 0.1
        }
        
        self.agent = QueryReformulatorAgent(agent_config).to(self.device)
        
        rl_config = {
            'learning_rate': self.config.learning_rate,
            'gamma': self.config.gamma,
            'gae_lambda': self.config.gae_lambda,
            'clip_epsilon': 0.2,
            'entropy_coef': 0.01,
            'value_loss_coef': 0.5
        }
        
        self.rl_trainer = RLTrainer(agent=self.agent, config=rl_config)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.agent.parameters())
        logger.info(f"Agent: {total_params:,} parameters")
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch with GPU-focused batch processing."""
        self.agent.train()
        
        # Shuffle indices
        indices = np.random.permutation(len(self.train_data))
        num_batches = len(indices) // self.config.batch_size
        
        total_reward = 0.0
        total_loss = 0.0
        num_updates = 0
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}")
        
        for batch_idx in pbar:
            start = batch_idx * self.config.batch_size
            batch_indices = indices[start:start + self.config.batch_size]
            
            # Get batch (on GPU)
            batch = self.train_data.get_batch(batch_indices)
            
            # Collect experiences
            experiences, rewards = self._collect_experiences(batch)
            
            total_reward += np.mean(rewards)
            
            # Update policy if we have enough experiences
            if len(experiences) >= self.config.batch_size:
                loss = self._update_policy(experiences)
                total_loss += loss
                num_updates += 1
            
            # Update progress
            pbar.set_postfix({
                'reward': total_reward / (batch_idx + 1),
                'loss': total_loss / max(1, num_updates)
            })
        
        return {
            'avg_reward': total_reward / num_batches,
            'avg_loss': total_loss / max(1, num_updates)
        }
    
    def _collect_experiences(self, batch: Dict) -> Tuple[List[Dict], List[float]]:
        """Collect experiences from batch (GPU operations)."""
        query_embs = batch['query_emb']
        candidate_embs = batch['candidate_embs']
        candidate_features = batch['candidate_features']
        candidate_terms = batch['candidate_terms']
        baseline_metrics = batch['baseline_metrics']
        
        batch_size = query_embs.shape[0]
        
        experiences = []
        rewards = []
        
        for i in range(batch_size):
            query_emb = query_embs[i:i+1]
            cand_embs = candidate_embs[i:i+1]
            cand_feats = candidate_features[i:i+1]
            terms = candidate_terms[i]
            baseline = baseline_metrics[i]
            
            # Start with query embedding as current
            current_emb = query_emb.clone()
            
            episode_reward = 0.0
            
            for step in range(self.config.max_steps):
                # Select action (GPU)
                with torch.no_grad():
                    action, log_prob, value = self.agent.select_action(
                        query_emb,
                        current_emb,
                        cand_embs,
                        cand_feats,
                        deterministic=False
                    )
                
                action_idx = action.item()
                done = action_idx >= self.config.num_candidates or not terms[action_idx]
                
                # Reward: based on candidate score (heuristic)
                if not done:
                    term_score = cand_feats[0, action_idx, 2].item()  # Normalized score
                    reward = term_score * 2.0  # Scale reward
                else:
                    reward = 0.0
                
                episode_reward += reward
                
                # Store experience
                exp = {
                    'query_emb': query_emb.squeeze(0).cpu(),
                    'current_query_emb': current_emb.squeeze(0).cpu(),
                    'candidate_embs': cand_embs.squeeze(0).cpu(),
                    'candidate_features': cand_feats.squeeze(0).cpu(),
                    'action': action_idx,
                    'log_prob': log_prob.item(),
                    'value': value.item(),
                    'reward': reward,
                    'done': done
                }
                experiences.append(exp)
                
                if done:
                    break
                
                # Update current embedding (simple: add term embedding scaled)
                term_emb = cand_embs[0, action_idx:action_idx+1, :]
                current_emb = (current_emb + term_emb * 0.3).clone()
            
            rewards.append(episode_reward)
        
        return experiences, rewards
    
    def _update_policy(self, experiences: List[Dict]) -> float:
        """Update policy using PPO (GPU)."""
        # Stack tensors
        query_embs = torch.stack([e['query_emb'] for e in experiences]).to(self.device)
        current_embs = torch.stack([e['current_query_emb'] for e in experiences]).to(self.device)
        candidate_embs = torch.stack([e['candidate_embs'] for e in experiences]).to(self.device)
        candidate_features = torch.stack([e['candidate_features'] for e in experiences]).to(self.device)
        
        actions = torch.tensor([e['action'] for e in experiences], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor([e['log_prob'] for e in experiences]).to(self.device)
        rewards = torch.tensor([e['reward'] for e in experiences]).to(self.device)
        dones = torch.tensor([e['done'] for e in experiences]).to(self.device)
        values = torch.tensor([e['value'] for e in experiences]).to(self.device)
        
        # PPO update
        rollout_buffer = {
            'query_emb': query_embs,
            'current_query_emb': current_embs,
            'candidate_embs': candidate_embs,
            'candidate_features': candidate_features,
            'actions': actions,
            'old_log_probs': old_log_probs,
            'rewards': rewards.unsqueeze(1),
            'dones': dones.unsqueeze(1),
            'values': values.unsqueeze(1)
        }
        
        loss_dict = self.rl_trainer.update(rollout_buffer)
        
        return loss_dict.get('total_loss', 0.0)
    
    def evaluate(self) -> Dict:
        """Evaluate on validation set."""
        self.agent.eval()
        
        indices = np.arange(len(self.valid_data))
        batch = self.valid_data.get_batch(indices)
        
        baseline_recall = np.mean([m['recall@10'] for m in batch['baseline_metrics']])
        baseline_mrr = np.mean([m['mrr'] for m in batch['baseline_metrics']])
        
        # Simple evaluation: just return baseline for now
        # Full evaluation would require BM25 search with reformulated queries
        
        return {
            'baseline_recall@10': baseline_recall,
            'baseline_mrr': baseline_mrr
        }
    
    def train(self):
        """Main training loop."""
        logger.info("Starting GPU-focused training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train samples: {len(self.train_data)}")
        logger.info(f"Valid samples: {len(self.valid_data)}")
        
        best_reward = -float('inf')
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            eval_metrics = self.evaluate()
            
            epoch_time = time.time() - start_time
            
            # Log
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            logger.info(f"  Avg Reward: {train_metrics['avg_reward']:.4f}")
            logger.info(f"  Avg Loss: {train_metrics['avg_loss']:.4f}")
            logger.info(f"  Baseline R@10: {eval_metrics['baseline_recall@10']:.4f}")
            logger.info(f"  Time: {epoch_time:.1f}s")
            
            # Save checkpoint
            if train_metrics['avg_reward'] > best_reward:
                best_reward = train_metrics['avg_reward']
                self._save_checkpoint('best_model.pt')
                logger.info("  New best model saved!")
            
            self._save_checkpoint(f'epoch_{epoch+1}.pt')
            
            # History
            self.history.append({
                'epoch': epoch + 1,
                'train': train_metrics,
                'eval': eval_metrics,
                'time': epoch_time
            })
            
            with open(os.path.join(self.config.checkpoint_dir, 'history.json'), 'w') as f:
                json.dump(self.history, f, indent=2)
        
        logger.info(f"Training complete! Best reward: {best_reward:.4f}")
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.rl_trainer.optimizer.state_dict()
        }, path)


def main():
    parser = argparse.ArgumentParser(description="GPU-Focused Training")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()
    
    config = GPUTrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    trainer = GPUTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
