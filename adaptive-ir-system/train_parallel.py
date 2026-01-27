#!/usr/bin/env python3
"""
Parallel Training Script for Adaptive IR System.

This script leverages existing src/ modules and adds:
1. Parallel reward computation using multiprocessing
2. Batch processing for GPU efficiency
3. Pre-computed BM25 index caching
4. Mixed precision training

Author: Based on existing src/training/train_rl.py
"""

import os
import sys
import argparse
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
from dataclasses import dataclass
import multiprocessing as mp

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from existing src modules
from src.rl_agent import QueryReformulatorAgent, RLTrainer
from src.evaluation import IRMetrics, IRMetricsAggregator
from src.utils.legacy_loader import LegacyDatasetAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# BM25 Search Worker (for multiprocessing)
# ============================================================================

class BM25SearchWorker:
    """
    Pre-initialized BM25 search worker for parallel reward computation.
    Uses shared memory for efficiency.
    """
    
    def __init__(self, corpus_docs: List[List[str]], doc_ids: List[str], k1: float = 1.2, b: float = 0.75):
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(corpus_docs, k1=k1, b=b)
        self.doc_ids = doc_ids
        self.corpus_size = len(doc_ids)
    
    def search(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        """Perform BM25 search."""
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.doc_ids[idx], float(scores[idx])))
        return results


# Global worker for multiprocessing
_global_worker: Optional[BM25SearchWorker] = None


def init_worker(corpus_docs: List[List[str]], doc_ids: List[str]):
    """Initialize global worker for each process."""
    global _global_worker
    _global_worker = BM25SearchWorker(corpus_docs, doc_ids)


def search_task(args: Tuple[str, str, List[str], int]) -> Dict:
    """
    Parallel search task for reward computation.
    
    Args:
        args: (query_id, query_text, relevant_docs, k)
    
    Returns:
        Dict with query_id, results, and metrics
    """
    global _global_worker
    query_id, query_text, relevant_docs, k = args
    
    # Search
    results = _global_worker.search(query_text, k=k)
    doc_ids = [doc_id for doc_id, _ in results]
    
    # Compute metrics manually (IRMetrics is class with static methods)
    relevant_set = set(relevant_docs)
    
    # Recall
    retrieved_at_10 = set(doc_ids[:10])
    retrieved_at_100 = set(doc_ids[:100])
    recall_10 = len(retrieved_at_10 & relevant_set) / len(relevant_set) if relevant_set else 0.0
    recall_100 = len(retrieved_at_100 & relevant_set) / len(relevant_set) if relevant_set else 0.0
    
    # MRR
    mrr = 0.0
    for i, doc_id in enumerate(doc_ids):
        if doc_id in relevant_set:
            mrr = 1.0 / (i + 1)
            break
    
    metrics = {
        'recall@10': recall_10,
        'recall@100': recall_100,
        'mrr': mrr
    }
    
    return {
        'query_id': query_id,
        'results': results,
        'metrics': metrics
    }


# ============================================================================
# Parallel Reward Computer
# ============================================================================

class ParallelRewardComputer:
    """
    Computes rewards using threading (lower overhead than multiprocessing).
    Uses a single pre-built BM25 index.
    """
    
    def __init__(
        self,
        corpus_docs: List[List[str]],
        doc_ids: List[str],
        num_workers: int = 4
    ):
        self.corpus_docs = corpus_docs
        self.doc_ids = doc_ids
        self.num_workers = num_workers
        self.pool = None
        self.bm25 = None
        
    def start(self):
        """Initialize the thread pool and BM25 index."""
        logger.info(f"Starting parallel reward computer with {self.num_workers} workers...")
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_docs, k1=1.2, b=0.75)
        self.pool = ThreadPoolExecutor(max_workers=self.num_workers)
        logger.info("Parallel reward computer ready")
    
    def stop(self):
        """Shutdown the thread pool."""
        if self.pool:
            self.pool.shutdown(wait=True)
            self.pool = None
    
    def _search_single(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        """Perform BM25 search for a single query."""
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.doc_ids[idx], float(scores[idx])))
        return results
    
    def _compute_single(self, args: Tuple[str, str, List[str], int]) -> Dict:
        """Compute metrics for a single query."""
        query_id, query_text, relevant_docs, k = args
        
        # Search
        results = self._search_single(query_text, k=k)
        doc_ids = [doc_id for doc_id, _ in results]
        
        # Compute metrics
        relevant_set = set(relevant_docs)
        
        retrieved_at_10 = set(doc_ids[:10])
        retrieved_at_100 = set(doc_ids[:100])
        recall_10 = len(retrieved_at_10 & relevant_set) / len(relevant_set) if relevant_set else 0.0
        recall_100 = len(retrieved_at_100 & relevant_set) / len(relevant_set) if relevant_set else 0.0
        
        mrr = 0.0
        for i, doc_id in enumerate(doc_ids):
            if doc_id in relevant_set:
                mrr = 1.0 / (i + 1)
                break
        
        return {
            'query_id': query_id,
            'results': results,
            'metrics': {
                'recall@10': recall_10,
                'recall@100': recall_100,
                'mrr': mrr
            }
        }
    
    def compute_batch_rewards(
        self,
        queries: List[Tuple[str, str, List[str]]],  # [(query_id, query_text, relevant_docs), ...]
        k: int = 100
    ) -> List[Dict]:
        """
        Compute rewards for a batch of queries using threads.
        
        Args:
            queries: List of (query_id, query_text, relevant_docs)
            k: Number of results to retrieve
            
        Returns:
            List of result dicts with metrics
        """
        if self.bm25 is None:
            self.start()
        
        tasks = [(qid, text, rels, k) for qid, text, rels in queries]
        
        # Use threading for parallel execution
        results = list(self.pool.map(self._compute_single, tasks))
        
        return results


# ============================================================================
# Batch Training Loop
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    dataset_path: str = "/kaggle/input/ir-msmarco/Query Reformulator/msa_dataset.hdf5"
    corpus_path: str = "/kaggle/input/ir-msmarco/Query Reformulator/msa_corpus.hdf5"
    embeddings_path: str = "/kaggle/input/ir-msmarco/Query Reformulator/D_cbow_pdw_8B.pkl"
    
    # Model
    embed_dim: int = 500
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    max_terms: int = 10
    
    # Training
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    
    # Optimization
    num_workers: int = 4
    use_amp: bool = True
    gradient_accumulation: int = 4
    
    # Output
    checkpoint_dir: str = "checkpoints_parallel"
    log_dir: str = "logs_parallel"


class ParallelTrainingLoop:
    """
    Optimized training loop with parallel reward computation.
    Uses existing QueryReformulatorAgent and RLTrainer from src/.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self._load_embeddings()
        self._load_data()
        self._build_bm25_index()
        self._initialize_agent()
        self._initialize_parallel_computer()
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp and torch.cuda.is_available() else None
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Metrics tracking
        self.train_history = []
    
    def _load_embeddings(self):
        """Load word embeddings."""
        logger.info("Loading word embeddings...")
        import pickle
        
        if os.path.exists(self.config.embeddings_path):
            try:
                with open(self.config.embeddings_path, 'rb') as f:
                    self.word2vec = pickle.load(f, encoding='latin1')
                logger.info(f"Loaded {len(self.word2vec)} word embeddings")
            except Exception as e:
                logger.warning(f"Could not load embeddings: {e}")
                self.word2vec = {}
        else:
            logger.warning("Embeddings not found, using random initialization")
            self.word2vec = {}
    
    def _load_data(self):
        """Load dataset using LegacyDatasetAdapter."""
        logger.info("Loading dataset...")
        
        self.train_adapter = LegacyDatasetAdapter(
            self.config.dataset_path,
            self.config.corpus_path,
            split='train'
        )
        
        self.valid_adapter = LegacyDatasetAdapter(
            self.config.dataset_path,
            self.config.corpus_path,
            split='valid'
        )
        
        # Load queries and qrels
        self.train_queries = self.train_adapter.load_queries()
        self.train_qrels = self.train_adapter.load_qrels()
        self.valid_queries = self.valid_adapter.load_queries()
        self.valid_qrels = self.valid_adapter.load_qrels()
        
        logger.info(f"Loaded {len(self.train_queries)} train queries, {len(self.valid_queries)} valid queries")
    
    def _build_bm25_index(self):
        """Build BM25 index from corpus with caching."""
        cache_dir = os.path.join(self.config.checkpoint_dir, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, 'bm25_index.pkl')
        
        # Try to load cached index
        if os.path.exists(cache_path):
            logger.info(f"Loading cached BM25 index from {cache_path}...")
            import pickle
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            self.corpus_docs = cached['corpus_docs']
            self.doc_ids = cached['doc_ids']
            logger.info(f"Loaded {len(self.doc_ids)} documents from cache")
        else:
            logger.info("Building BM25 index...")
            
            # Get corpus from adapter
            corpus = self.train_adapter.corpus
            
            self.corpus_docs = []
            self.doc_ids = []
            
            num_docs = len(corpus) if corpus else 0
            logger.info(f"Processing {num_docs} documents...")
            
            for doc_id in tqdm(range(num_docs), desc="Indexing corpus"):
                doc = corpus.get_document(doc_id)
                if doc:
                    tokens = doc.lower().split()
                    self.corpus_docs.append(tokens)
                    self.doc_ids.append(str(doc_id))
            
            logger.info(f"Indexed {len(self.doc_ids)} documents")
            
            # Cache index
            import pickle
            logger.info(f"Saving BM25 index to {cache_path}...")
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'corpus_docs': self.corpus_docs,
                    'doc_ids': self.doc_ids
                }, f)
            logger.info("Cache saved")
        
        logger.info(f"Indexed {len(self.doc_ids)} documents")
        
        # Also create local BM25 for candidate mining
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_docs, k1=1.2, b=0.75)
    
    def _initialize_agent(self):
        """Initialize RL agent using existing src/ module."""
        logger.info("Initializing RL agent...")
        
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
            'clip_epsilon': self.config.clip_epsilon,
            'entropy_coef': self.config.entropy_coef,
            'value_loss_coef': self.config.value_coef
        }
        self.rl_trainer = RLTrainer(agent=self.agent, config=rl_config)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.agent.parameters())
        trainable_params = sum(p.numel() for p in self.agent.parameters() if p.requires_grad)
        logger.info(f"Agent parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def _initialize_parallel_computer(self):
        """Initialize parallel reward computer."""
        self.parallel_computer = ParallelRewardComputer(
            corpus_docs=self.corpus_docs,
            doc_ids=self.doc_ids,
            num_workers=self.config.num_workers
        )
    
    def _embed_text(self, text: str) -> torch.Tensor:
        """Embed text using word2vec."""
        tokens = text.lower().split()
        embeddings = []
        
        for token in tokens:
            if token in self.word2vec:
                embeddings.append(self.word2vec[token])
        
        if embeddings:
            return torch.tensor(np.mean(embeddings, axis=0), dtype=torch.float32)
        else:
            return torch.zeros(self.config.embed_dim, dtype=torch.float32)
    
    def _mine_candidates(self, query: str, k: int = 50) -> Dict[str, float]:
        """Mine candidate expansion terms from top documents."""
        # Search with BM25
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:20]  # Top 20 docs
        
        # Extract terms from top documents
        query_terms = set(query_tokens)
        term_scores = {}
        
        for idx in top_indices:
            doc_tokens = self.corpus_docs[idx]
            doc_score = scores[idx]
            
            for token in doc_tokens:
                if token not in query_terms and len(token) > 2:
                    if token not in term_scores:
                        term_scores[token] = 0.0
                    term_scores[token] += doc_score
        
        # Return top-k candidates
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return dict(sorted_terms)
    
    def collect_batch_trajectories(
        self,
        query_ids: List[str],
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]]  # {qid: {doc_id: relevance}}
    ) -> Tuple[List[Dict], List[float]]:
        """
        Collect trajectories for a batch of queries with parallel reward computation.
        
        Returns:
            List of experiences and list of rewards
        """
        batch_experiences = []
        batch_rewards = []
        
        # Prepare batch data
        batch_data = []
        for qid in query_ids:
            query = queries[qid]
            rel_docs = qrels.get(qid, {})
            relevant = list(rel_docs.keys())  # Convert {doc_id: rel} to [doc_id, ...]
            if relevant:
                batch_data.append((qid, query, relevant))
        
        if not batch_data:
            return [], []
        
        # Get initial metrics in parallel
        initial_results = self.parallel_computer.compute_batch_rewards(
            batch_data, k=100
        )
        
        # Create lookup
        initial_metrics = {r['query_id']: r['metrics'] for r in initial_results}
        
        # Process each query
        for qid, query, relevant in batch_data:
            # Mine candidates
            candidates = self._mine_candidates(query)
            if not candidates:
                continue
            
            # Prepare tensors
            query_emb = self._embed_text(query).to(self.device)
            candidate_terms = list(candidates.keys())
            candidate_embs = torch.stack([self._embed_text(t) for t in candidate_terms]).to(self.device)
            
            # Simple candidate features (3 features: score, log_score, normalized_score)
            max_score = max(candidates.values()) if candidates.values() else 1.0
            candidate_features = torch.tensor(
                [[candidates[t], np.log1p(candidates[t]), candidates[t]/max_score] for t in candidate_terms],
                dtype=torch.float32
            ).to(self.device)
            
            # Pad to fixed size
            max_cands = 50
            if len(candidate_terms) < max_cands:
                pad_size = max_cands - len(candidate_terms)
                candidate_embs = F.pad(candidate_embs, (0, 0, 0, pad_size))
                candidate_features = F.pad(candidate_features, (0, 0, 0, pad_size))
            
            # Run episode
            current_query = query
            selected_terms = []
            episode_reward = 0.0
            metrics_before = initial_metrics.get(qid, {'recall@10': 0, 'recall@100': 0, 'mrr': 0})
            
            for step in range(self.config.max_terms):
                current_emb = self._embed_text(current_query).unsqueeze(0).to(self.device)
                
                # Select action with agent
                with torch.no_grad():
                    action, log_prob, value = self.agent.select_action(
                        query_emb.unsqueeze(0),
                        current_emb,
                        candidate_embs.unsqueeze(0),
                        candidate_features.unsqueeze(0)
                    )
                
                action_idx = action.item()
                done = action_idx >= len(candidate_terms)
                
                if not done:
                    selected_term = candidate_terms[action_idx]
                    selected_terms.append(selected_term)
                    current_query = f"{current_query} {selected_term}"
                
                # Compute reward (simplified - based on term quality)
                # For full reward, would need another search
                if done:
                    # STOP action - no reward
                    reward = 0.0
                else:
                    # Reward based on term score (heuristic for speed)
                    term_score = candidates.get(selected_term, 0)
                    reward = term_score / max(candidates.values()) if candidates.values() else 0.0
                
                episode_reward += reward
                
                # Store experience
                exp = {
                    'query_emb': query_emb.cpu(),
                    'current_query_emb': current_emb.squeeze(0).cpu(),
                    'candidate_embs': candidate_embs.cpu(),
                    'candidate_features': candidate_features.cpu(),
                    'action': action_idx,
                    'log_prob': log_prob.item(),
                    'value': value.item(),
                    'reward': reward,
                    'done': done
                }
                batch_experiences.append(exp)
                
                if done:
                    break
            
            batch_rewards.append(episode_reward)
        
        return batch_experiences, batch_rewards
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with batch processing."""
        self.agent.train()
        
        # Sample queries
        query_ids = list(self.train_queries.keys())
        np.random.shuffle(query_ids)
        
        # Limit for faster iteration
        max_queries = min(len(query_ids), 1000)
        query_ids = query_ids[:max_queries]
        
        all_experiences = []
        all_rewards = []
        
        # Process in batches
        num_batches = (len(query_ids) + self.config.batch_size - 1) // self.config.batch_size
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}")
        
        for batch_idx in pbar:
            start = batch_idx * self.config.batch_size
            end = min(start + self.config.batch_size, len(query_ids))
            batch_qids = query_ids[start:end]
            
            # Collect trajectories (with parallel reward computation)
            experiences, rewards = self.collect_batch_trajectories(
                batch_qids,
                self.train_queries,
                self.train_qrels
            )
            
            all_experiences.extend(experiences)
            all_rewards.extend(rewards)
            
            # Update progress bar
            if rewards:
                pbar.set_postfix({
                    'avg_reward': np.mean(rewards),
                    'exps': len(all_experiences)
                })
            
            # Update policy periodically
            if len(all_experiences) >= self.config.batch_size * self.config.gradient_accumulation:
                loss = self._update_policy(all_experiences)
                all_experiences = []
        
        # Final update
        if all_experiences:
            self._update_policy(all_experiences)
        
        metrics = {
            'epoch': epoch + 1,
            'avg_reward': np.mean(all_rewards) if all_rewards else 0.0,
            'num_episodes': len(all_rewards)
        }
        
        return metrics
    
    def _update_policy(self, experiences: List[Dict]) -> float:
        """Update policy using PPO."""
        if not experiences:
            return 0.0
        
        # Convert to tensors
        query_embs = torch.stack([e['query_emb'] for e in experiences]).to(self.device)
        current_embs = torch.stack([e['current_query_emb'] for e in experiences]).to(self.device)
        candidate_embs = torch.stack([e['candidate_embs'] for e in experiences]).to(self.device)
        candidate_features = torch.stack([e['candidate_features'] for e in experiences]).to(self.device)
        
        actions = torch.tensor([e['action'] for e in experiences], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor([e['log_prob'] for e in experiences], dtype=torch.float32).to(self.device)
        rewards = torch.tensor([e['reward'] for e in experiences], dtype=torch.float32).to(self.device)
        dones = torch.tensor([e['done'] for e in experiences], dtype=torch.bool).to(self.device)
        values = torch.tensor([e['value'] for e in experiences], dtype=torch.float32).to(self.device)
        
        # Prepare rollout buffer
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
        
        # PPO update with optional mixed precision
        if self.scaler and self.config.use_amp:
            with autocast():
                loss = self.rl_trainer.update(rollout_buffer)
        else:
            loss = self.rl_trainer.update(rollout_buffer)
        
        return loss
    
    def evaluate(self, num_queries: int = 100) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.agent.eval()
        
        query_ids = list(self.valid_queries.keys())[:num_queries]
        
        # Prepare batch
        batch_data = []
        for qid in query_ids:
            query = self.valid_queries[qid]
            rel_docs = self.valid_qrels.get(qid, {})
            relevant = list(rel_docs.keys())  # Convert {doc_id: rel} to [doc_id, ...]
            if relevant:
                batch_data.append((qid, query, relevant))
        
        # Baseline (original queries)
        baseline_results = self.parallel_computer.compute_batch_rewards(batch_data, k=100)
        
        # Reformulated queries
        reformulated_data = []
        for qid, query, relevant in batch_data:
            # Mine and select terms
            candidates = self._mine_candidates(query)
            if candidates:
                query_emb = self._embed_text(query).to(self.device)
                candidate_terms = list(candidates.keys())
                candidate_embs = torch.stack([self._embed_text(t) for t in candidate_terms]).to(self.device)
                max_score = max(candidates.values()) if candidates.values() else 1.0
                candidate_features = torch.tensor(
                    [[candidates[t], np.log1p(candidates[t]), candidates[t]/max_score] for t in candidate_terms],
                    dtype=torch.float32
                ).to(self.device)
                
                # Pad
                max_cands = 50
                if len(candidate_terms) < max_cands:
                    pad_size = max_cands - len(candidate_terms)
                    candidate_embs = F.pad(candidate_embs, (0, 0, 0, pad_size))
                    candidate_features = F.pad(candidate_features, (0, 0, 0, pad_size))
                
                # Select terms
                reformulated_query = query
                with torch.no_grad():
                    for _ in range(3):  # Max 3 terms
                        current_emb = self._embed_text(reformulated_query).unsqueeze(0).to(self.device)
                        action, _, _ = self.agent.select_action(
                            query_emb.unsqueeze(0),
                            current_emb,
                            candidate_embs.unsqueeze(0),
                            candidate_features.unsqueeze(0),
                            deterministic=True
                        )
                        action_idx = action.item()
                        if action_idx >= len(candidate_terms):
                            break
                        reformulated_query = f"{reformulated_query} {candidate_terms[action_idx]}"
                
                reformulated_data.append((qid, reformulated_query, relevant))
            else:
                reformulated_data.append((qid, query, relevant))
        
        # Evaluate reformulated
        reformulated_results = self.parallel_computer.compute_batch_rewards(reformulated_data, k=100)
        
        # Aggregate metrics
        baseline_metrics = {
            'recall@10': np.mean([r['metrics'].get('recall@10', 0) for r in baseline_results]),
            'recall@100': np.mean([r['metrics'].get('recall@100', 0) for r in baseline_results]),
            'mrr': np.mean([r['metrics'].get('mrr', 0) for r in baseline_results])
        }
        
        reformulated_metrics = {
            'recall@10': np.mean([r['metrics'].get('recall@10', 0) for r in reformulated_results]),
            'recall@100': np.mean([r['metrics'].get('recall@100', 0) for r in reformulated_results]),
            'mrr': np.mean([r['metrics'].get('mrr', 0) for r in reformulated_results])
        }
        
        return {
            'baseline': baseline_metrics,
            'reformulated': reformulated_metrics,
            'improvement': {
                k: reformulated_metrics[k] - baseline_metrics[k]
                for k in baseline_metrics
            }
        }
    
    def train(self):
        """Main training loop."""
        logger.info("Starting parallel training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Num workers: {self.config.num_workers}")
        
        # Start parallel computer
        self.parallel_computer.start()
        
        best_recall = 0.0
        
        try:
            for epoch in range(self.config.num_epochs):
                start_time = time.time()
                
                # Train
                train_metrics = self.train_epoch(epoch)
                
                # Evaluate
                eval_metrics = self.evaluate(num_queries=100)
                
                epoch_time = time.time() - start_time
                
                # Log
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
                logger.info(f"  Train reward: {train_metrics['avg_reward']:.4f}")
                logger.info(f"  Baseline Recall@10: {eval_metrics['baseline']['recall@10']:.4f}")
                logger.info(f"  Reformed Recall@10: {eval_metrics['reformulated']['recall@10']:.4f}")
                logger.info(f"  Improvement: {eval_metrics['improvement']['recall@10']:.4f}")
                logger.info(f"  Time: {epoch_time:.1f}s")
                
                # Save checkpoint
                current_recall = eval_metrics['reformulated']['recall@10']
                if current_recall > best_recall:
                    best_recall = current_recall
                    self.save_checkpoint(f"best_model.pt")
                    logger.info(f"  New best model saved!")
                
                # Save epoch checkpoint
                self.save_checkpoint(f"epoch_{epoch+1}.pt")
                
                # Save history
                self.train_history.append({
                    'epoch': epoch + 1,
                    'train': train_metrics,
                    'eval': eval_metrics,
                    'time': epoch_time
                })
                
                with open(os.path.join(self.config.log_dir, 'history.json'), 'w') as f:
                    json.dump(self.train_history, f, indent=2)
        
        finally:
            self.parallel_computer.stop()
        
        logger.info(f"Training complete! Best Recall@10: {best_recall:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.rl_trainer.optimizer.state_dict(),
            'config': vars(self.config)
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.rl_trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {path}")


def main():
    parser = argparse.ArgumentParser(description="Parallel RL Training for Adaptive IR")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.workers,
        use_amp=not args.no_amp
    )
    
    trainer = ParallelTrainingLoop(config)
    
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    trainer.train()


if __name__ == "__main__":
    main()
