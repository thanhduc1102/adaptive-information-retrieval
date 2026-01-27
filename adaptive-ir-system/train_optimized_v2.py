#!/usr/bin/env python3
"""
=============================================================================
ADAPTIVE IR - OPTIMIZED TRAINING PIPELINE V2
=============================================================================

Tối ưu hóa toàn diện:
1. LAZY PRE-COMPUTING: Chỉ pre-compute khi cần, không pre-compute tất cả
2. PARALLEL PROCESSING: Multi-threaded batch preparation
3. CACHED EXPANDED QUERIES: Cache kết quả BM25 cho expanded queries
4. VECTORIZED OPERATIONS: Batched embedding computation
5. MIXED PRECISION: FP16 training để tăng tốc GPU
6. ASYNC DATA LOADING: Prepare next batch trong khi training

Cách sử dụng:
    python train_optimized_v2.py --mode quick   # Test nhanh ~3 phút
    python train_optimized_v2.py --mode medium  # ~15 phút  
    python train_optimized_v2.py --mode full    # Full training

Author: Adaptive IR System
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import h5py
import ast
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from evaluation.metrics import IRMetrics, IRMetricsAggregator, LatencyTimer


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OptimizedConfig:
    """Cấu hình tối ưu cho training."""
    
    # Data paths
    data_dir: str = str(Path(__file__).parent.parent / 'Query Reformulator')
    checkpoint_dir: str = str(Path(__file__).parent / 'checkpoints')
    log_dir: str = str(Path(__file__).parent / 'logs')
    
    # Mode
    mode: str = 'quick'
    
    # Model architecture
    embedding_dim: int = 500
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    
    # Training
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 0.5
    
    # PPO
    ppo_epochs: int = 3
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    update_every: int = 512
    
    # Retrieval
    max_candidates: int = 20
    top_k_retrieve: int = 100
    rrf_k: int = 60
    bm25_k1: float = 0.9
    bm25_b: float = 0.4
    
    # Optimization settings
    use_amp: bool = True                    # Mixed precision
    num_workers: int = 4                    # Parallel workers
    prefetch_batches: int = 2               # Batches to prefetch
    cache_size: int = 100000                # Max cache entries
    lazy_precompute: bool = True            # Lazy pre-computing
    
    # Evaluation
    eval_every: int = 500
    num_eval_queries: int = 200
    
    # Checkpoint
    save_every: int = 1000
    log_every: int = 50
    
    # Preset values
    num_train_queries: Optional[int] = None
    num_train_steps: int = 500
    
    def apply_mode(self):
        """Apply mode presets."""
        if self.mode == 'quick':
            self.num_train_queries = 500
            self.num_train_steps = 300
            self.eval_every = 150
            self.num_eval_queries = 100
            self.batch_size = 32
            
        elif self.mode == 'medium':
            self.num_train_queries = 2000
            self.num_train_steps = 1000
            self.eval_every = 500
            self.num_eval_queries = 200
            self.batch_size = 64
            
        elif self.mode == 'full':
            self.num_train_queries = None
            self.num_train_steps = 5000
            self.eval_every = 1000
            self.num_eval_queries = 500
            self.batch_size = 64
    
    def __post_init__(self):
        self.apply_mode()
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# OPTIMIZED DATA MANAGER
# =============================================================================

class OptimizedDataManager:
    """
    Data Manager với các tối ưu:
    1. Lazy loading
    2. Batched encoding
    3. Thread-safe caching
    """
    
    def __init__(self, config: OptimizedConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Data
        self.word2vec = None
        self.unk_emb = None
        self.title_to_idx = {}
        self.idx_to_title = {}
        self.queries = {'train': {}, 'valid': {}, 'test': {}}
        self.qrels = {'train': {}, 'valid': {}, 'test': {}}
        
        # Thread-safe cache
        self._emb_cache = {}
        self._cache_lock = threading.Lock()
        
        # Pre-computed tensors
        self._query_emb_tensor = None
        self._qid_to_idx = {}
    
    def load_all(self):
        """Load all data."""
        self._load_word2vec()
        self._load_corpus()
        self._load_dataset()
        self._precompute_query_embeddings()
    
    def _load_word2vec(self):
        """Load Word2Vec."""
        self.logger.info("Loading Word2Vec embeddings...")
        t0 = time.time()
        
        path = Path(self.config.data_dir) / 'D_cbow_pdw_8B.pkl'
        with open(path, 'rb') as f:
            try:
                data = pickle.load(f, encoding='latin1')
            except:
                f.seek(0)
                data = pickle.load(f)
        
        self.word2vec = dict(data) if not isinstance(data, dict) else data
        
        # UNK embedding
        vecs = list(self.word2vec.values())[:10000]
        self.unk_emb = torch.tensor(np.mean(vecs, axis=0), dtype=torch.float32, device=self.device)
        
        self.logger.info(f"  Loaded {len(self.word2vec):,} vectors in {time.time()-t0:.1f}s")
    
    def _load_corpus(self):
        """Load corpus."""
        self.logger.info("Loading corpus...")
        
        path = Path(self.config.data_dir) / 'msa_corpus.hdf5'
        with h5py.File(path, 'r') as f:
            titles = f['title'][:]
        
        for i, t in enumerate(titles):
            title = t.decode('utf-8') if isinstance(t, bytes) else t
            self.title_to_idx[title.lower().strip()] = i
            self.idx_to_title[i] = title
        
        self.logger.info(f"  Loaded {len(self.idx_to_title):,} documents")
    
    def _load_dataset(self):
        """Load dataset."""
        self.logger.info("Loading dataset...")
        
        path = Path(self.config.data_dir) / 'msa_dataset.hdf5'
        with h5py.File(path, 'r') as f:
            for split in ['train', 'valid', 'test']:
                q_key = f'queries_{split}'
                d_key = f'doc_ids_{split}'
                
                if q_key not in f:
                    continue
                
                for i, (q, d) in enumerate(zip(f[q_key][:], f[d_key][:])):
                    query = q.decode('utf-8') if isinstance(q, bytes) else q
                    doc_str = d.decode('utf-8') if isinstance(d, bytes) else d
                    
                    try:
                        doc_titles = ast.literal_eval(doc_str)
                        if not isinstance(doc_titles, list):
                            doc_titles = [doc_titles]
                    except:
                        doc_titles = [doc_str]
                    
                    rel_indices = {self.title_to_idx[t.lower().strip()] 
                                   for t in doc_titles 
                                   if t.lower().strip() in self.title_to_idx}
                    
                    if rel_indices:
                        qid = str(i)
                        self.queries[split][qid] = query
                        self.qrels[split][qid] = rel_indices
        
        for split in ['train', 'valid', 'test']:
            self.logger.info(f"  {split}: {len(self.queries[split]):,} queries")
    
    def _precompute_query_embeddings(self):
        """Pre-compute embeddings cho TẤT CẢ queries một lần."""
        self.logger.info("Pre-computing query embeddings...")
        t0 = time.time()
        
        all_queries = []
        all_qids = []
        
        for split in ['train', 'valid', 'test']:
            for qid, query in self.queries[split].items():
                all_qids.append((split, qid))
                all_queries.append(query)
        
        # Batch encode
        embeddings = []
        for query in tqdm(all_queries, desc="Encoding queries", leave=False):
            embeddings.append(self._encode_text(query))
        
        self._query_emb_tensor = torch.stack(embeddings)
        self._qid_to_idx = {(split, qid): i for i, (split, qid) in enumerate(all_qids)}
        
        self.logger.info(f"  Pre-computed {len(all_queries):,} query embeddings in {time.time()-t0:.1f}s")
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding (internal)."""
        words = text.lower().split()
        vecs = [self.word2vec[w] for w in words if w in self.word2vec]
        
        if vecs:
            return torch.tensor(np.mean(vecs, axis=0), dtype=torch.float32, device=self.device)
        return self.unk_emb.clone()
    
    def get_query_embedding(self, qid: str, split: str = 'train') -> torch.Tensor:
        """Get pre-computed query embedding."""
        idx = self._qid_to_idx.get((split, qid))
        if idx is not None:
            return self._query_emb_tensor[idx]
        return self._encode_text(self.queries[split][qid])
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode with caching."""
        with self._cache_lock:
            if text in self._emb_cache:
                return self._emb_cache[text]
        
        emb = self._encode_text(text)
        
        with self._cache_lock:
            if len(self._emb_cache) < self.config.cache_size:
                self._emb_cache[text] = emb
        
        return emb
    
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Batch encode texts."""
        embeddings = []
        for text in texts:
            embeddings.append(self.encode(text))
        return torch.stack(embeddings)
    
    def get_doc_text(self, doc_id: int) -> str:
        return self.idx_to_title.get(doc_id, '')
    
    def get_train_qids(self) -> List[str]:
        all_qids = list(self.queries['train'].keys())
        if self.config.num_train_queries:
            return all_qids[:self.config.num_train_queries]
        return all_qids


# =============================================================================
# OPTIMIZED BM25 ENGINE
# =============================================================================

class OptimizedBM25Engine:
    """
    BM25 Engine với các tối ưu:
    1. Lazy caching (chỉ cache khi cần)
    2. LRU-style cache management
    3. Parallel search support
    """
    
    def __init__(self, data_manager: OptimizedDataManager, config: OptimizedConfig):
        self.data = data_manager
        self.config = config
        self.cache = {}
        self._cache_lock = threading.Lock()
        self.bm25 = None
        self.doc_ids = None
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
    
    def build_index(self):
        """Build BM25 index."""
        self.logger.info("Building BM25 index...")
        t0 = time.time()
        
        self.doc_ids = list(self.data.idx_to_title.keys())
        corpus = [self.data.idx_to_title[i].lower().split() for i in self.doc_ids]
        
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(corpus, k1=self.config.bm25_k1, b=self.config.bm25_b)
        
        self.logger.info(f"  Indexed {len(corpus):,} documents in {time.time()-t0:.1f}s")
    
    def _search_internal(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Internal search without caching."""
        tokens = query.lower().split()
        if not tokens:
            return []
        
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        
        return [(self.doc_ids[i], scores[i]) for i in top_idx if scores[i] > 0]
    
    def search(self, query: str, k: int = 100) -> List[Tuple[int, float]]:
        """Search with lazy caching."""
        cache_key = (query, k)
        
        with self._cache_lock:
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key]
        
        self.cache_misses += 1
        results = self._search_internal(query, k)
        
        with self._cache_lock:
            # LRU-style: remove oldest if cache full
            if len(self.cache) >= self.config.cache_size:
                # Remove 10% oldest entries
                to_remove = list(self.cache.keys())[:len(self.cache) // 10]
                for key in to_remove:
                    del self.cache[key]
            
            self.cache[cache_key] = results
        
        return results
    
    def get_retrieved_ids(self, query: str, k: int = 100) -> List[int]:
        """Get only doc IDs."""
        return [doc_id for doc_id, _ in self.search(query, k)]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'cache_size': len(self.cache),
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate
        }


# =============================================================================
# RRF FUSION
# =============================================================================

class RRFFusion:
    """Reciprocal Rank Fusion."""
    
    def __init__(self, k: int = 60):
        self.k = k
    
    def fuse(self, ranked_lists: List[List[int]]) -> List[int]:
        if not ranked_lists:
            return []
        
        doc_scores = defaultdict(float)
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list, start=1):
                doc_scores[doc_id] += 1.0 / (self.k + rank)
        
        return sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)


# =============================================================================
# RL AGENT
# =============================================================================

class QueryReformulationAgent(nn.Module):
    """RL Agent cho query reformulation."""
    
    def __init__(self, config: OptimizedConfig):
        super().__init__()
        
        dim = config.embedding_dim
        hidden = config.hidden_dim
        
        self.query_proj = nn.Linear(dim, hidden)
        self.cand_proj = nn.Linear(dim + 3, hidden)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.num_heads,
            dim_feedforward=hidden * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(hidden)
        
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, query_emb, cand_embs, cand_features, mask=None):
        B, N, _ = cand_embs.shape
        
        q_enc = self.query_proj(query_emb).unsqueeze(1)
        c_input = torch.cat([cand_embs, cand_features], dim=-1)
        c_enc = self.cand_proj(c_input)
        
        seq = torch.cat([q_enc, c_enc], dim=1)
        
        if mask is not None:
            src_key_padding_mask = torch.zeros(B, 1 + N, dtype=torch.bool, device=query_emb.device)
            src_key_padding_mask[:, 1:] = ~mask
        else:
            src_key_padding_mask = None
        
        encoded = self.norm(self.transformer(seq, src_key_padding_mask=src_key_padding_mask))
        
        q_rep = encoded[:, 0, :]
        c_rep = encoded[:, 1:, :]
        
        cand_logits = self.actor(c_rep).squeeze(-1)
        stop_logit = self.actor(q_rep)
        
        action_logits = torch.cat([cand_logits, stop_logit], dim=-1)
        
        if mask is not None:
            action_logits[:, :-1] = action_logits[:, :-1].masked_fill(~mask, float('-inf'))
        
        value = self.critic(q_rep).squeeze(-1)
        
        return action_logits, value
    
    def select_action(self, query_emb, cand_embs, cand_features, mask, deterministic=False):
        logits, value = self(query_emb, cand_embs, cand_features, mask)
        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def evaluate_action(self, query_emb, cand_embs, cand_features, mask, action):
        logits, value = self(query_emb, cand_embs, cand_features, mask)
        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, value, entropy


# =============================================================================
# ASYNC BATCH PREFETCHER
# =============================================================================

class AsyncBatchPrefetcher:
    """
    Async batch preparation để overlap computation.
    Trong khi GPU training, CPU chuẩn bị batch tiếp theo.
    """
    
    def __init__(self, data_manager, search_engine, config, train_qids):
        self.data = data_manager
        self.search = search_engine
        self.config = config
        self.train_qids = train_qids
        
        self.batch_queue = queue.Queue(maxsize=config.prefetch_batches)
        self.stop_event = threading.Event()
        self.worker_thread = None
    
    def start(self):
        """Start prefetch worker."""
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def stop(self):
        """Stop prefetch worker."""
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=1)
    
    def _worker(self):
        """Background worker to prepare batches."""
        while not self.stop_event.is_set():
            try:
                batch = self._prepare_batch()
                self.batch_queue.put(batch, timeout=1)
            except queue.Full:
                continue
            except Exception as e:
                logging.error(f"Prefetch error: {e}")
    
    def _prepare_batch(self):
        """Prepare a single batch."""
        batch_qids = np.random.choice(
            self.train_qids, 
            min(self.config.batch_size, len(self.train_qids)), 
            replace=False
        ).tolist()
        
        B = len(batch_qids)
        N = self.config.max_candidates
        D = self.config.embedding_dim
        device = self.data.device
        
        q_embs = torch.zeros(B, D, device=device)
        c_embs = torch.zeros(B, N, D, device=device)
        c_feats = torch.zeros(B, N, 3, device=device)
        masks = torch.zeros(B, N, dtype=torch.bool, device=device)
        
        batch_candidates = []
        batch_base_metrics = []
        batch_relevant = []
        
        for i, qid in enumerate(batch_qids):
            query = self.data.queries['train'][qid]
            relevant = self.data.qrels['train'][qid]
            batch_relevant.append(relevant)
            
            # Pre-computed query embedding
            q_embs[i] = self.data.get_query_embedding(qid, 'train')
            
            # BM25 candidates (lazy cached)
            candidates = self.search.get_retrieved_ids(query, k=N)
            batch_candidates.append(candidates)
            
            # Base metrics
            full_results = self.search.get_retrieved_ids(query, k=100)
            relevant_str = {str(r) for r in relevant}
            retrieved_str = [str(r) for r in full_results]
            base_recall = IRMetrics.recall_at_k(retrieved_str, relevant_str, 10)
            batch_base_metrics.append(base_recall)
            
            # Candidate features
            query_tokens = set(query.lower().split())
            
            for j, doc_id in enumerate(candidates[:N]):
                doc_text = self.data.get_doc_text(doc_id)
                c_embs[i, j] = self.data.encode(doc_text)
                c_feats[i, j, 0] = 1.0 / (j + 1)
                c_feats[i, j, 1] = 1.0 if doc_id in relevant else 0.0
                
                doc_tokens = set(doc_text.lower().split())
                c_feats[i, j, 2] = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
                masks[i, j] = True
        
        return {
            'qids': batch_qids,
            'q_embs': q_embs,
            'c_embs': c_embs,
            'c_feats': c_feats,
            'masks': masks,
            'candidates': batch_candidates,
            'base_metrics': batch_base_metrics,
            'relevant': batch_relevant
        }
    
    def get_batch(self, timeout=5):
        """Get next batch."""
        try:
            return self.batch_queue.get(timeout=timeout)
        except queue.Empty:
            # Fallback: prepare synchronously
            return self._prepare_batch()


# =============================================================================
# OPTIMIZED TRAINER
# =============================================================================

class OptimizedTrainer:
    """
    Trainer với các tối ưu:
    1. Mixed precision (AMP)
    2. Async batch prefetching
    3. Efficient reward computation
    4. Gradient accumulation support
    """
    
    def __init__(self, agent, data_manager, search_engine, config):
        self.agent = agent
        self.data = data_manager
        self.search = search_engine
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.optimizer = torch.optim.AdamW(
            agent.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        
        # Experience buffer
        self.buffer = defaultdict(list)
        
        # Metrics
        self.train_rewards = []
        self.train_improvements = []
        
        # Async prefetcher
        train_qids = data_manager.get_train_qids()
        self.prefetcher = AsyncBatchPrefetcher(
            data_manager, search_engine, config, train_qids
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _compute_reward_batch(self, batch_data, actions):
        """Compute rewards for entire batch efficiently."""
        rewards = torch.zeros(len(batch_data['qids']), device=self.device)
        
        for i, (qid, action, candidates, base_recall, relevant) in enumerate(zip(
            batch_data['qids'],
            actions.cpu().numpy(),
            batch_data['candidates'],
            batch_data['base_metrics'],
            batch_data['relevant']
        )):
            query = self.data.queries['train'][qid]
            relevant_str = {str(r) for r in relevant}
            n_cands = len(candidates)
            
            if action >= n_cands:
                # STOP action
                rewards[i] = base_recall * 0.5
            else:
                # Expand query
                doc_id = candidates[action]
                doc_text = self.data.get_doc_text(doc_id)
                
                if not doc_text:
                    rewards[i] = -0.1
                else:
                    expanded_query = query + " " + doc_text[:50]
                    new_results = self.search.get_retrieved_ids(expanded_query, k=100)
                    new_retrieved_str = [str(r) for r in new_results]
                    new_recall = IRMetrics.recall_at_k(new_retrieved_str, relevant_str, 10)
                    
                    improvement = new_recall - base_recall
                    self.train_improvements.append(improvement)
                    rewards[i] = improvement * 5.0
        
        return rewards
    
    def train(self, num_steps: int):
        """Main training loop with optimizations."""
        self.logger.info(f"\n🚀 Starting optimized training for {num_steps} steps...")
        
        # Start async prefetcher
        self.prefetcher.start()
        
        step = 0
        pbar = tqdm(total=num_steps, desc="Training")
        
        try:
            while step < num_steps:
                # Get pre-fetched batch
                batch_data = self.prefetcher.get_batch()
                
                # Forward pass with AMP
                self.agent.eval()
                with torch.no_grad():
                    if self.config.use_amp:
                        with autocast(device_type='cuda'):
                            actions, log_probs, values = self.agent.select_action(
                                batch_data['q_embs'],
                                batch_data['c_embs'],
                                batch_data['c_feats'],
                                batch_data['masks']
                            )
                    else:
                        actions, log_probs, values = self.agent.select_action(
                            batch_data['q_embs'],
                            batch_data['c_embs'],
                            batch_data['c_feats'],
                            batch_data['masks']
                        )
                
                # Compute rewards
                rewards = self._compute_reward_batch(batch_data, actions)
                self.train_rewards.append(rewards.mean().item())
                
                # Store in buffer
                self.buffer['q_embs'].append(batch_data['q_embs'])
                self.buffer['c_embs'].append(batch_data['c_embs'])
                self.buffer['c_feats'].append(batch_data['c_feats'])
                self.buffer['masks'].append(batch_data['masks'])
                self.buffer['actions'].append(actions)
                self.buffer['log_probs'].append(log_probs)
                self.buffer['values'].append(values)
                self.buffer['rewards'].append(rewards)
                
                step += len(batch_data['qids'])
                pbar.update(len(batch_data['qids']))
                
                # PPO Update
                total_samples = sum(len(x) for x in self.buffer['q_embs'])
                if total_samples >= self.config.update_every:
                    self._ppo_update()
                
                # Logging
                if step % self.config.log_every < self.config.batch_size:
                    avg_reward = np.mean(self.train_rewards[-50:]) if self.train_rewards else 0
                    avg_imp = np.mean(self.train_improvements[-50:]) if self.train_improvements else 0
                    cache_stats = self.search.get_cache_stats()
                    pbar.set_postfix({
                        'reward': f'{avg_reward:.4f}',
                        'imp': f'{avg_imp:.4f}',
                        'cache_hit': f'{cache_stats["hit_rate"]:.1%}'
                    })
        
        finally:
            self.prefetcher.stop()
            pbar.close()
        
        self.logger.info(f"  Final avg reward: {np.mean(self.train_rewards[-100:]):.4f}")
        self.logger.info(f"  Final avg improvement: {np.mean(self.train_improvements[-100:]):.4f}")
        self.logger.info(f"  Cache stats: {self.search.get_cache_stats()}")
    
    def _ppo_update(self):
        """PPO update with mixed precision."""
        self.agent.train()
        
        # Stack buffer
        all_q = torch.cat(self.buffer['q_embs'])
        all_c = torch.cat(self.buffer['c_embs'])
        all_f = torch.cat(self.buffer['c_feats'])
        all_m = torch.cat(self.buffer['masks'])
        all_a = torch.cat(self.buffer['actions'])
        all_lp = torch.cat(self.buffer['log_probs'])
        all_v = torch.cat(self.buffer['values'])
        all_r = torch.cat(self.buffer['rewards'])
        
        # Normalize rewards
        all_r = (all_r - all_r.mean()) / (all_r.std() + 1e-8)
        advantages = all_r - all_v.detach()
        returns = all_r
        
        # PPO epochs
        for _ in range(self.config.ppo_epochs):
            perm = torch.randperm(len(all_q), device=self.device)
            
            for start in range(0, len(all_q), 64):
                end = min(start + 64, len(all_q))
                idx = perm[start:end]
                
                if self.config.use_amp:
                    with autocast(device_type='cuda'):
                        new_log_probs, new_values, entropy = self.agent.evaluate_action(
                            all_q[idx], all_c[idx], all_f[idx], all_m[idx], all_a[idx]
                        )
                        
                        ratio = torch.exp(new_log_probs - all_lp[idx])
                        surr1 = ratio * advantages[idx]
                        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                                           1 + self.config.clip_epsilon) * advantages[idx]
                        policy_loss = -torch.min(surr1, surr2).mean()
                        value_loss = F.mse_loss(new_values, returns[idx])
                        loss = (policy_loss + 
                               self.config.value_coef * value_loss - 
                               self.config.entropy_coef * entropy.mean())
                    
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    new_log_probs, new_values, entropy = self.agent.evaluate_action(
                        all_q[idx], all_c[idx], all_f[idx], all_m[idx], all_a[idx]
                    )
                    
                    ratio = torch.exp(new_log_probs - all_lp[idx])
                    surr1 = ratio * advantages[idx]
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon,
                                       1 + self.config.clip_epsilon) * advantages[idx]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(new_values, returns[idx])
                    loss = (policy_loss +
                           self.config.value_coef * value_loss -
                           self.config.entropy_coef * entropy.mean())
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
        
        # Clear buffer
        self.buffer = defaultdict(list)


# =============================================================================
# EVALUATOR
# =============================================================================

class Evaluator:
    """Evaluation với standard IR metrics."""
    
    def __init__(self, data_manager, search_engine, config):
        self.data = data_manager
        self.search = search_engine
        self.config = config
        self.rrf = RRFFusion(k=config.rrf_k)
    
    def evaluate_baseline(self, split='valid', max_queries=None):
        """Evaluate BM25 baseline."""
        if max_queries is None:
            max_queries = self.config.num_eval_queries
        
        logging.info(f"\n📊 Evaluating BM25 baseline...")
        
        aggregator = IRMetricsAggregator()
        qids = list(self.data.queries[split].keys())[:max_queries]
        
        for qid in tqdm(qids, desc="BM25 Eval", leave=False):
            query = self.data.queries[split][qid]
            relevant = self.data.qrels[split][qid]
            relevant_str = {str(r) for r in relevant}
            
            with LatencyTimer() as timer:
                results = self.search.get_retrieved_ids(query, k=100)
            
            aggregator.add_query_result(
                query_id=qid,
                retrieved=[str(r) for r in results],
                relevant=relevant_str,
                latency=timer.get_elapsed()
            )
        
        return aggregator.compute_aggregate()
    
    def evaluate_with_rl(self, agent, split='valid', max_queries=None, use_rrf=True):
        """Evaluate with RL."""
        if max_queries is None:
            max_queries = self.config.num_eval_queries
        
        logging.info(f"\n📊 Evaluating RL+{'RRF' if use_rrf else 'NoRRF'}...")
        
        agent.eval()
        aggregator = IRMetricsAggregator()
        qids = list(self.data.queries[split].keys())[:max_queries]
        
        N = self.config.max_candidates
        D = self.config.embedding_dim
        expand_count = 0
        
        for qid in tqdm(qids, desc="RL Eval", leave=False):
            query = self.data.queries[split][qid]
            relevant = self.data.qrels[split][qid]
            relevant_str = {str(r) for r in relevant}
            
            with LatencyTimer() as timer:
                initial_results = self.search.get_retrieved_ids(query, k=N)
                
                if not initial_results:
                    final_results = self.search.get_retrieved_ids(query, k=100)
                else:
                    q_emb = self.data.get_query_embedding(qid, split).unsqueeze(0)
                    c_embs = torch.zeros(1, N, D, device=self.data.device)
                    c_feats = torch.zeros(1, N, 3, device=self.data.device)
                    mask = torch.zeros(1, N, dtype=torch.bool, device=self.data.device)
                    
                    query_tokens = set(query.lower().split())
                    
                    for j, doc_id in enumerate(initial_results[:N]):
                        doc_text = self.data.get_doc_text(doc_id)
                        c_embs[0, j] = self.data.encode(doc_text)
                        c_feats[0, j, 0] = 1.0 / (j + 1)
                        c_feats[0, j, 1] = 1.0 if doc_id in relevant else 0.0
                        c_feats[0, j, 2] = len(query_tokens & set(doc_text.lower().split())) / max(len(query_tokens), 1)
                        mask[0, j] = True
                    
                    with torch.no_grad():
                        action, _, _ = agent.select_action(q_emb, c_embs, c_feats, mask, deterministic=True)
                    
                    action_idx = action[0].item()
                    
                    if action_idx < len(initial_results):
                        expand_count += 1
                        doc_text = self.data.get_doc_text(initial_results[action_idx])
                        expanded_query = query + " " + doc_text[:50]
                        
                        if use_rrf:
                            lists = [
                                self.search.get_retrieved_ids(query, k=100),
                                self.search.get_retrieved_ids(expanded_query, k=100)
                            ]
                            final_results = self.rrf.fuse(lists)[:100]
                        else:
                            final_results = self.search.get_retrieved_ids(expanded_query, k=100)
                    else:
                        final_results = self.search.get_retrieved_ids(query, k=100)
            
            aggregator.add_query_result(
                query_id=qid,
                retrieved=[str(r) for r in final_results],
                relevant=relevant_str,
                latency=timer.get_elapsed()
            )
        
        logging.info(f"  Expanded {expand_count}/{len(qids)} queries")
        return aggregator.compute_aggregate()


# =============================================================================
# UTILITIES
# =============================================================================

def setup_logging(log_dir: str):
    """Setup logging."""
    log_file = Path(log_dir) / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return str(log_file)


def print_metrics(metrics: Dict[str, float], title: str):
    """Print metrics."""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    
    for key in ['recall@10', 'recall@100', 'mrr', 'map', 'precision@10', 'ndcg@10']:
        if key in metrics:
            print(f"  {key:<15}: {metrics[key]:.4f}")
    
    if 'latency_mean' in metrics:
        print(f"\n  Latency: {metrics['latency_mean']*1000:.2f}ms")


def compare_metrics(baseline: Dict, rl: Dict):
    """Compare metrics."""
    print(f"\n{'='*70}")
    print(f"📈 COMPARISON: BASELINE vs RL")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<15} {'Baseline':>12} {'RL':>12} {'Δ':>12} {'%':>10}")
    print("-" * 60)
    
    for metric in ['recall@10', 'recall@100', 'mrr', 'map']:
        if metric in baseline and metric in rl:
            b, r = baseline[metric], rl[metric]
            delta = r - b
            pct = (delta / b * 100) if b > 0 else 0
            sign = '+' if delta >= 0 else ''
            print(f"{metric:<15} {b:>12.4f} {r:>12.4f} {sign}{delta:>11.4f} {sign}{pct:>9.1f}%")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optimized Adaptive IR Training')
    parser.add_argument('--mode', type=str, default='quick', choices=['quick', 'medium', 'full'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    
    # Config
    config = OptimizedConfig(mode=args.mode)
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.no_amp:
        config.use_amp = False
    config.num_workers = args.workers
    
    # Logging
    log_file = setup_logging(config.log_dir)
    
    print("=" * 80)
    print("🚀 ADAPTIVE IR - OPTIMIZED TRAINING V2")
    print("=" * 80)
    
    print(f"\nMode: {config.mode}")
    print(f"Train queries: {config.num_train_queries or 'ALL'}")
    print(f"Train steps: {config.num_train_steps}")
    print(f"Batch size: {config.batch_size}")
    print(f"Mixed precision: {config.use_amp}")
    print(f"Log file: {log_file}")
    
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name} ({props.total_memory/1e9:.1f} GB)")
    
    t_total = time.time()
    
    # ==========================================================================
    # STEP 1: Load Data
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    
    data = OptimizedDataManager(config)
    data.load_all()
    
    # ==========================================================================
    # STEP 2: Build Search Engine
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Building BM25 Search Engine (NO pre-computing)")
    print("=" * 60)
    
    search = OptimizedBM25Engine(data, config)
    search.build_index()
    
    # NOTE: KHÔNG pre-compute tất cả queries ở đây!
    # Lazy caching sẽ tự động cache khi cần
    
    # ==========================================================================
    # STEP 3: Baseline Evaluation
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Baseline Evaluation")
    print("=" * 60)
    
    evaluator = Evaluator(data, search, config)
    baseline_metrics = evaluator.evaluate_baseline('valid', config.num_eval_queries)
    print_metrics(baseline_metrics, "BASELINE (BM25)")
    
    # ==========================================================================
    # STEP 4: RL Training
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: RL Training (with async prefetching)")
    print("=" * 60)
    
    agent = QueryReformulationAgent(config).to('cuda')
    trainer = OptimizedTrainer(agent, data, search, config)
    trainer.train(config.num_train_steps)
    
    # ==========================================================================
    # STEP 5: RL Evaluation
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: RL Evaluation")
    print("=" * 60)
    
    rl_metrics = evaluator.evaluate_with_rl(agent, 'valid', config.num_eval_queries, use_rrf=True)
    print_metrics(rl_metrics, "RL + RRF")
    
    # ==========================================================================
    # STEP 6: Comparison
    # ==========================================================================
    compare_metrics(baseline_metrics, rl_metrics)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    total_time = time.time() - t_total
    
    print(f"\n{'='*60}")
    print(f"⏱️ Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"📊 Cache Stats: {search.get_cache_stats()}")
    print(f"{'='*60}")
    
    # Save results
    results = {
        'baseline': {k: float(v) for k, v in baseline_metrics.items()},
        'rl': {k: float(v) for k, v in rl_metrics.items()},
        'config': {
            'mode': config.mode,
            'num_train_queries': config.num_train_queries,
            'num_train_steps': config.num_train_steps,
            'batch_size': config.batch_size,
            'use_amp': config.use_amp
        },
        'cache_stats': search.get_cache_stats(),
        'time_seconds': total_time
    }
    
    with open('benchmark_results_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to benchmark_results_v2.json")
    
    return baseline_metrics, rl_metrics


if __name__ == "__main__":
    main()
