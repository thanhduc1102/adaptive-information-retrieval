#!/usr/bin/env python3
"""
=============================================================================
ADAPTIVE IR - FULL EPOCH TRAINING PIPELINE
=============================================================================

Chương trình huấn luyện đầy đủ 1 epoch với các tính năng:
1. Cấu hình dễ chỉnh sửa qua command line hoặc file config
2. Checkpoint saving/loading
3. Logging chi tiết
4. Early stopping
5. Evaluation định kỳ

Cách sử dụng:
    # Chạy với config mặc định (quick test)
    python train_full_epoch.py
    
    # Chạy full epoch  
    python train_full_epoch.py --mode full
    
    # Custom config
    python train_full_epoch.py --epochs 3 --batch-size 128 --lr 1e-4
    
    # Resume từ checkpoint
    python train_full_epoch.py --resume checkpoints/latest.pt

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
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import h5py
import ast
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from evaluation.metrics import IRMetrics, IRMetricsAggregator, LatencyTimer


# =============================================================================
# CONFIGURATION - DỄ CHỈNH SỬA
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Cấu hình huấn luyện - Chỉnh sửa các giá trị ở đây.
    
    Để thay đổi:
    1. Sửa trực tiếp giá trị default trong class này
    2. Hoặc dùng command line: python train_full_epoch.py --batch-size 128
    3. Hoặc load từ file JSON
    """
    
    # =========================================================================
    # ĐƯỜNG DẪN DỮ LIỆU
    # =========================================================================
    data_dir: str = str(Path(__file__).parent.parent / 'Query Reformulator')
    checkpoint_dir: str = str(Path(__file__).parent / 'checkpoints')
    log_dir: str = str(Path(__file__).parent / 'logs')
    
    # =========================================================================
    # CHẾ ĐỘ CHẠY
    # =========================================================================
    mode: str = 'quick'  # 'quick', 'medium', 'full'
    
    # =========================================================================
    # KIẾN TRÚC MÔ HÌNH
    # =========================================================================
    embedding_dim: int = 500      # Kích thước word embedding (từ Word2Vec)
    hidden_dim: int = 256         # Kích thước hidden layer
    num_heads: int = 4            # Số attention heads
    num_layers: int = 2           # Số Transformer layers
    dropout: float = 0.1          # Dropout rate
    
    # =========================================================================
    # HUẤN LUYỆN
    # =========================================================================
    epochs: int = 1               # Số epochs
    batch_size: int = 64          # Batch size
    learning_rate: float = 3e-4   # Learning rate
    weight_decay: float = 0.01    # Weight decay cho AdamW
    max_grad_norm: float = 0.5    # Gradient clipping
    warmup_steps: int = 100       # Warmup steps cho scheduler
    
    # =========================================================================
    # PPO (REINFORCEMENT LEARNING)
    # =========================================================================
    ppo_epochs: int = 3           # Số epochs PPO mỗi lần update
    clip_epsilon: float = 0.2     # PPO clipping
    gamma: float = 0.99           # Discount factor
    gae_lambda: float = 0.95      # GAE lambda
    entropy_coef: float = 0.01    # Entropy bonus
    value_coef: float = 0.5       # Value loss coefficient
    update_every: int = 512       # Update sau mỗi N samples
    
    # =========================================================================
    # RETRIEVAL
    # =========================================================================
    max_candidates: int = 20      # Số candidates cho agent
    top_k_retrieve: int = 100     # Top-k cho BM25
    rrf_k: int = 60               # RRF constant
    bm25_k1: float = 0.9          # BM25 k1
    bm25_b: float = 0.4           # BM25 b
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    eval_every: int = 1000        # Evaluate sau mỗi N steps
    num_eval_queries: int = 500   # Số queries để evaluate
    
    # =========================================================================
    # CHECKPOINT & LOGGING
    # =========================================================================
    save_every: int = 2000        # Save checkpoint sau mỗi N steps
    log_every: int = 100          # Log sau mỗi N steps
    
    # =========================================================================
    # EARLY STOPPING
    # =========================================================================
    early_stopping: bool = True   # Có dùng early stopping không
    patience: int = 5             # Số lần eval không cải thiện trước khi dừng
    min_delta: float = 0.001      # Ngưỡng cải thiện tối thiểu
    
    # =========================================================================
    # CÁC PRESET CHO TỪNG CHẾ ĐỘ
    # =========================================================================
    def apply_mode(self):
        """Áp dụng preset dựa trên mode."""
        if self.mode == 'quick':
            # Test nhanh ~5-10 phút
            self.epochs = 1
            self.num_train_queries = 1000
            self.eval_every = 300
            self.num_eval_queries = 100
            self.save_every = 500
            
        elif self.mode == 'medium':
            # Test trung bình ~30-60 phút
            self.epochs = 1
            self.num_train_queries = 5000
            self.eval_every = 1000
            self.num_eval_queries = 300
            self.save_every = 2000
            
        elif self.mode == 'full':
            # Full training ~vài giờ
            self.epochs = 1
            self.num_train_queries = None  # Dùng tất cả
            self.eval_every = 5000
            self.num_eval_queries = 1000
            self.save_every = 10000
    
    # Các thuộc tính được tính từ mode
    num_train_queries: Optional[int] = None
    
    def __post_init__(self):
        self.apply_mode()
        
        # Tạo thư mục nếu chưa có
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_json(cls, path: str) -> 'TrainingConfig':
        """Load config từ file JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, path: str):
        """Save config ra file JSON."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def summary(self) -> str:
        """In tóm tắt config."""
        lines = [
            "=" * 60,
            "📋 TRAINING CONFIGURATION",
            "=" * 60,
            f"  Mode: {self.mode}",
            f"  Epochs: {self.epochs}",
            f"  Batch Size: {self.batch_size}",
            f"  Learning Rate: {self.learning_rate}",
            f"  Train Queries: {self.num_train_queries or 'ALL'}",
            f"  Eval Queries: {self.num_eval_queries}",
            f"  Eval Every: {self.eval_every} steps",
            f"  Save Every: {self.save_every} steps",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# DATA MANAGER
# =============================================================================

class DataManager:
    """Quản lý tất cả dữ liệu."""
    
    def __init__(self, config: TrainingConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Data containers
        self.word2vec = None
        self.unk_emb = None
        self.title_to_idx = {}
        self.idx_to_title = {}
        self.queries = {'train': {}, 'valid': {}, 'test': {}}
        self.qrels = {'train': {}, 'valid': {}, 'test': {}}
        
        # Caches
        self.emb_cache = {}
    
    def load_all(self):
        """Load tất cả dữ liệu."""
        self._load_word2vec()
        self._load_corpus()
        self._load_dataset()
    
    def _load_word2vec(self):
        """Load Word2Vec embeddings."""
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
        
        # Compute UNK embedding
        vecs = list(self.word2vec.values())[:10000]
        self.unk_emb = torch.tensor(np.mean(vecs, axis=0), dtype=torch.float32, device=self.device)
        
        self.logger.info(f"  Loaded {len(self.word2vec):,} vectors in {time.time()-t0:.1f}s")
    
    def _load_corpus(self):
        """Load document corpus."""
        self.logger.info("Loading corpus...")
        
        path = Path(self.config.data_dir) / 'msa_corpus.hdf5'
        with h5py.File(path, 'r') as f:
            titles = f['title'][:]
        
        for i, t in enumerate(titles):
            title = t.decode('utf-8') if isinstance(t, bytes) else t
            title_norm = title.lower().strip()
            self.title_to_idx[title_norm] = i
            self.idx_to_title[i] = title
        
        self.logger.info(f"  Loaded {len(self.idx_to_title):,} documents")
    
    def _load_dataset(self):
        """Load queries and qrels."""
        self.logger.info("Loading dataset...")
        
        path = Path(self.config.data_dir) / 'msa_dataset.hdf5'
        with h5py.File(path, 'r') as f:
            for split in ['train', 'valid', 'test']:
                q_key = f'queries_{split}'
                d_key = f'doc_ids_{split}'
                
                if q_key not in f:
                    continue
                
                queries_raw = f[q_key][:]
                doc_ids_raw = f[d_key][:]
                
                for i, (q, d) in enumerate(zip(queries_raw, doc_ids_raw)):
                    query = q.decode('utf-8') if isinstance(q, bytes) else q
                    doc_str = d.decode('utf-8') if isinstance(d, bytes) else d
                    
                    try:
                        doc_titles = ast.literal_eval(doc_str)
                        if not isinstance(doc_titles, list):
                            doc_titles = [doc_titles]
                    except:
                        doc_titles = [doc_str]
                    
                    rel_indices = set()
                    for title in doc_titles:
                        title_norm = title.lower().strip()
                        if title_norm in self.title_to_idx:
                            rel_indices.add(self.title_to_idx[title_norm])
                    
                    if rel_indices:
                        qid = str(i)
                        self.queries[split][qid] = query
                        self.qrels[split][qid] = rel_indices
        
        for split in ['train', 'valid', 'test']:
            self.logger.info(f"  {split}: {len(self.queries[split]):,} queries")
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text thành embedding."""
        if text in self.emb_cache:
            return self.emb_cache[text]
        
        words = text.lower().split()
        vecs = [self.word2vec[w] for w in words if w in self.word2vec]
        
        if vecs:
            emb = torch.tensor(np.mean(vecs, axis=0), dtype=torch.float32, device=self.device)
        else:
            emb = self.unk_emb.clone()
        
        if len(self.emb_cache) < 500000:
            self.emb_cache[text] = emb
        
        return emb
    
    def get_doc_text(self, doc_id: int) -> str:
        """Lấy text của document."""
        return self.idx_to_title.get(doc_id, '')
    
    def get_train_qids(self) -> List[str]:
        """Lấy danh sách query IDs cho training."""
        all_qids = list(self.queries['train'].keys())
        if self.config.num_train_queries:
            return all_qids[:self.config.num_train_queries]
        return all_qids


# =============================================================================
# BM25 SEARCH ENGINE
# =============================================================================

class BM25SearchEngine:
    """BM25 search engine với caching."""
    
    def __init__(self, data_manager: DataManager, config: TrainingConfig):
        self.data = data_manager
        self.config = config
        self.cache = {}
        self.bm25 = None
        self.doc_ids = None
        self.logger = logging.getLogger(__name__)
    
    def build_index(self):
        """Build BM25 index."""
        self.logger.info("Building BM25 index...")
        t0 = time.time()
        
        self.doc_ids = list(self.data.idx_to_title.keys())
        corpus = [self.data.idx_to_title[i].lower().split() for i in self.doc_ids]
        
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(corpus, k1=self.config.bm25_k1, b=self.config.bm25_b)
        
        self.logger.info(f"  Indexed {len(corpus):,} documents in {time.time()-t0:.1f}s")
    
    def precompute_queries(self, queries: Dict[str, str], k: int = 100, desc: str = "Pre-computing"):
        """Pre-compute BM25 results."""
        self.logger.info(f"Pre-computing BM25 for {len(queries):,} queries...")
        t0 = time.time()
        
        for qid, query in tqdm(queries.items(), desc=desc, leave=False):
            cache_key = (query, k)
            if cache_key not in self.cache:
                self.cache[cache_key] = self._search_internal(query, k)
        
        self.logger.info(f"  Pre-computed in {time.time()-t0:.1f}s")
    
    def _search_internal(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Internal search."""
        tokens = query.lower().split()
        if not tokens:
            return []
        
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        
        return [(self.doc_ids[i], scores[i]) for i in top_idx if scores[i] > 0]
    
    def search(self, query: str, k: int = 100) -> List[Tuple[int, float]]:
        """Search với caching."""
        cache_key = (query, k)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        results = self._search_internal(query, k)
        
        if len(self.cache) < 1000000:
            self.cache[cache_key] = results
        
        return results
    
    def get_retrieved_ids(self, query: str, k: int = 100) -> List[int]:
        """Chỉ lấy doc IDs."""
        return [doc_id for doc_id, _ in self.search(query, k)]


# =============================================================================
# RRF FUSION
# =============================================================================

class RRFFusion:
    """Reciprocal Rank Fusion."""
    
    def __init__(self, k: int = 60):
        self.k = k
    
    def fuse(self, ranked_lists: List[List[int]]) -> List[int]:
        """Fuse nhiều ranked lists."""
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
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        
        dim = config.embedding_dim
        hidden = config.hidden_dim
        
        # Query encoder
        self.query_proj = nn.Linear(dim, hidden)
        
        # Candidate encoder
        self.cand_proj = nn.Linear(dim + 3, hidden)
        
        # Transformer encoder
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
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(
        self,
        query_emb: torch.Tensor,
        cand_embs: torch.Tensor,
        cand_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        B, N, _ = cand_embs.shape
        
        q_enc = self.query_proj(query_emb).unsqueeze(1)
        
        c_input = torch.cat([cand_embs, cand_features], dim=-1)
        c_enc = self.cand_proj(c_input)
        
        seq = torch.cat([q_enc, c_enc], dim=1)
        
        if mask is not None:
            full_mask = torch.cat([
                torch.ones(B, 1, device=mask.device, dtype=torch.bool),
                mask
            ], dim=1)
            src_key_padding_mask = ~full_mask
        else:
            src_key_padding_mask = None
        
        encoded = self.norm(self.transformer(seq, src_key_padding_mask=src_key_padding_mask))
        
        q_rep = encoded[:, 0, :]
        c_rep = encoded[:, 1:, :]
        
        cand_logits = self.actor(c_rep).squeeze(-1)
        stop_logit = self.actor(q_rep)
        
        action_logits = torch.cat([cand_logits, stop_logit], dim=-1)
        
        if mask is not None:
            action_logits[:, :-1] = action_logits[:, :-1].masked_fill(~mask, -1e9)
        
        value = self.critic(q_rep).squeeze(-1)
        
        return action_logits, value
    
    def select_action(
        self,
        query_emb: torch.Tensor,
        cand_embs: torch.Tensor,
        cand_features: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action."""
        logits, value = self(query_emb, cand_embs, cand_features, mask)
        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def evaluate_action(
        self,
        query_emb: torch.Tensor,
        cand_embs: torch.Tensor,
        cand_features: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate action cho PPO update."""
        logits, value = self(query_emb, cand_embs, cand_features, mask)
        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, value, entropy


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """PPO Trainer với đầy đủ tính năng."""
    
    def __init__(
        self,
        agent: QueryReformulationAgent,
        data: DataManager,
        search: BM25SearchEngine,
        config: TrainingConfig
    ):
        self.agent = agent
        self.data = data
        self.search = search
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rrf = RRFFusion(k=config.rrf_k)
        self.logger = logging.getLogger(__name__)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            agent.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = None
        
        # Experience buffer
        self.buffer = {
            'q_embs': [], 'c_embs': [], 'c_feats': [], 'masks': [],
            'actions': [], 'log_probs': [], 'values': [], 'rewards': []
        }
        
        # Metrics tracking
        self.global_step = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        self.train_history = []
        self.eval_history = []
    
    def _prepare_batch(self, qids: List[str], split: str = 'train'):
        """Chuẩn bị batch tensors."""
        B = len(qids)
        N = self.config.max_candidates
        D = self.config.embedding_dim
        
        q_embs = torch.zeros(B, D, device=self.device)
        c_embs = torch.zeros(B, N, D, device=self.device)
        c_feats = torch.zeros(B, N, 3, device=self.device)
        masks = torch.zeros(B, N, dtype=torch.bool, device=self.device)
        
        batch_candidates = []
        batch_base_metrics = []
        
        for i, qid in enumerate(qids):
            query = self.data.queries[split][qid]
            relevant = self.data.qrels[split][qid]
            
            q_embs[i] = self.data.encode(query)
            
            candidates = self.search.get_retrieved_ids(query, k=N)
            batch_candidates.append(candidates)
            
            full_results = self.search.get_retrieved_ids(query, k=self.config.top_k_retrieve)
            relevant_str = {str(r) for r in relevant}
            retrieved_str = [str(r) for r in full_results]
            
            base_recall = IRMetrics.recall_at_k(retrieved_str, relevant_str, 10)
            batch_base_metrics.append(base_recall)
            
            query_tokens = set(query.lower().split())
            
            for j, doc_id in enumerate(candidates[:N]):
                doc_text = self.data.get_doc_text(doc_id)
                c_embs[i, j] = self.data.encode(doc_text)
                c_feats[i, j, 0] = 1.0 / (j + 1)
                c_feats[i, j, 1] = 1.0 if doc_id in relevant else 0.0
                
                doc_tokens = set(doc_text.lower().split())
                c_feats[i, j, 2] = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
                
                masks[i, j] = True
        
        return q_embs, c_embs, c_feats, masks, batch_candidates, batch_base_metrics
    
    def _compute_reward(self, qid: str, action: int, candidates: List[int], base_recall: float, split: str = 'train') -> float:
        """Tính reward."""
        query = self.data.queries[split][qid]
        relevant = self.data.qrels[split][qid]
        relevant_str = {str(r) for r in relevant}
        
        n_cands = len(candidates)
        
        if action >= n_cands:
            return base_recall * 0.5
        
        doc_id = candidates[action]
        doc_text = self.data.get_doc_text(doc_id)
        
        if not doc_text:
            return -0.1
        
        expanded_query = query + " " + doc_text[:50]
        new_results = self.search.get_retrieved_ids(expanded_query, k=self.config.top_k_retrieve)
        new_retrieved_str = [str(r) for r in new_results]
        
        new_recall = IRMetrics.recall_at_k(new_retrieved_str, relevant_str, 10)
        
        return (new_recall - base_recall) * 5.0
    
    def _ppo_update(self):
        """PPO update."""
        self.agent.train()
        
        all_q = torch.cat(self.buffer['q_embs'])
        all_c = torch.cat(self.buffer['c_embs'])
        all_f = torch.cat(self.buffer['c_feats'])
        all_m = torch.cat(self.buffer['masks'])
        all_a = torch.cat(self.buffer['actions'])
        all_lp = torch.cat(self.buffer['log_probs'])
        all_v = torch.cat(self.buffer['values'])
        all_r = torch.cat(self.buffer['rewards'])
        
        all_r = (all_r - all_r.mean()) / (all_r.std() + 1e-8)
        advantages = all_r - all_v.detach()
        returns = all_r
        
        for _ in range(self.config.ppo_epochs):
            perm = torch.randperm(len(all_q), device=self.device)
            
            for start in range(0, len(all_q), self.config.batch_size):
                end = min(start + self.config.batch_size, len(all_q))
                idx = perm[start:end]
                
                new_log_probs, new_values, entropy = self.agent.evaluate_action(
                    all_q[idx], all_c[idx], all_f[idx], all_m[idx], all_a[idx]
                )
                
                ratio = torch.exp(new_log_probs - all_lp[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(new_values, returns[idx])
                
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
        
        for k in self.buffer:
            self.buffer[k] = []
    
    def evaluate(self, split: str = 'valid', max_queries: int = None) -> Dict[str, float]:
        """Evaluate model."""
        if max_queries is None:
            max_queries = self.config.num_eval_queries
        
        self.agent.eval()
        aggregator = IRMetricsAggregator()
        qids = list(self.data.queries[split].keys())[:max_queries]
        
        expand_count = 0
        stop_count = 0
        
        N = self.config.max_candidates
        D = self.config.embedding_dim
        
        for qid in tqdm(qids, desc=f"Eval on {split}", leave=False):
            query = self.data.queries[split][qid]
            relevant = self.data.qrels[split][qid]
            relevant_str = {str(r) for r in relevant}
            
            with LatencyTimer() as timer:
                initial_results = self.search.get_retrieved_ids(query, k=N)
                
                if not initial_results:
                    final_results = self.search.get_retrieved_ids(query, k=self.config.top_k_retrieve)
                else:
                    q_emb = self.data.encode(query).unsqueeze(0)
                    c_embs = torch.zeros(1, N, D, device=self.device)
                    c_feats = torch.zeros(1, N, 3, device=self.device)
                    mask = torch.zeros(1, N, dtype=torch.bool, device=self.device)
                    
                    query_tokens = set(query.lower().split())
                    
                    for j, doc_id in enumerate(initial_results[:N]):
                        doc_text = self.data.get_doc_text(doc_id)
                        c_embs[0, j] = self.data.encode(doc_text)
                        c_feats[0, j, 0] = 1.0 / (j + 1)
                        c_feats[0, j, 1] = 1.0 if doc_id in relevant else 0.0
                        
                        doc_tokens = set(doc_text.lower().split())
                        c_feats[0, j, 2] = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
                        mask[0, j] = True
                    
                    with torch.no_grad():
                        action, _, _ = self.agent.select_action(q_emb, c_embs, c_feats, mask, deterministic=True)
                    
                    action_idx = action[0].item()
                    
                    if action_idx < len(initial_results):
                        expand_count += 1
                        doc_id = initial_results[action_idx]
                        doc_text = self.data.get_doc_text(doc_id)
                        expanded_query = query + " " + doc_text[:50]
                        
                        lists = [
                            self.search.get_retrieved_ids(query, k=self.config.top_k_retrieve),
                            self.search.get_retrieved_ids(expanded_query, k=self.config.top_k_retrieve)
                        ]
                        final_results = self.rrf.fuse(lists)[:self.config.top_k_retrieve]
                    else:
                        stop_count += 1
                        final_results = self.search.get_retrieved_ids(query, k=self.config.top_k_retrieve)
            
            retrieved_str = [str(r) for r in final_results]
            
            aggregator.add_query_result(
                query_id=qid,
                retrieved=retrieved_str,
                relevant=relevant_str,
                latency=timer.get_elapsed()
            )
        
        metrics = aggregator.compute_aggregate()
        metrics['expand_count'] = expand_count
        metrics['stop_count'] = stop_count
        
        return metrics
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'train_history': self.train_history,
            'eval_history': self.eval_history,
            'config': asdict(self.config)
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = Path(path).parent / 'best_model.pt'
            torch.save(checkpoint, best_path)
        
        self.logger.info(f"  Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.train_history = checkpoint.get('train_history', [])
        self.eval_history = checkpoint.get('eval_history', [])
        
        self.logger.info(f"  Loaded checkpoint from step {self.global_step}")
    
    def train_epoch(self, epoch: int):
        """Train 1 epoch."""
        train_qids = self.data.get_train_qids()
        total_steps = len(train_qids)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"EPOCH {epoch + 1}/{self.config.epochs}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"  Training on {total_steps:,} queries")
        
        # Shuffle
        np.random.shuffle(train_qids)
        
        epoch_rewards = []
        pbar = tqdm(range(0, total_steps, self.config.batch_size), desc=f"Epoch {epoch + 1}")
        
        for batch_start in pbar:
            batch_end = min(batch_start + self.config.batch_size, total_steps)
            batch_qids = train_qids[batch_start:batch_end]
            
            # Prepare batch
            q_embs, c_embs, c_feats, masks, candidates, base_metrics = self._prepare_batch(batch_qids)
            
            # Select actions
            self.agent.eval()
            with torch.no_grad():
                actions, log_probs, values = self.agent.select_action(q_embs, c_embs, c_feats, masks)
            
            # Compute rewards
            rewards = torch.zeros(len(batch_qids), device=self.device)
            for i, qid in enumerate(batch_qids):
                rewards[i] = self._compute_reward(qid, actions[i].item(), candidates[i], base_metrics[i])
            
            epoch_rewards.append(rewards.mean().item())
            
            # Store in buffer
            self.buffer['q_embs'].append(q_embs)
            self.buffer['c_embs'].append(c_embs)
            self.buffer['c_feats'].append(c_feats)
            self.buffer['masks'].append(masks)
            self.buffer['actions'].append(actions)
            self.buffer['log_probs'].append(log_probs)
            self.buffer['values'].append(values)
            self.buffer['rewards'].append(rewards)
            
            self.global_step += len(batch_qids)
            
            # PPO Update
            total_samples = sum(len(x) for x in self.buffer['q_embs'])
            if total_samples >= self.config.update_every:
                self._ppo_update()
            
            # Logging
            avg_reward = np.mean(epoch_rewards[-50:]) if epoch_rewards else 0
            pbar.set_postfix({'reward': f'{avg_reward:.4f}', 'step': self.global_step})
            
            # Log to file
            if self.global_step % self.config.log_every == 0:
                self.train_history.append({
                    'step': self.global_step,
                    'reward': avg_reward
                })
            
            # Evaluation
            if self.global_step % self.config.eval_every == 0:
                self.logger.info(f"\n📊 Evaluation at step {self.global_step}")
                metrics = self.evaluate('valid')
                
                self.eval_history.append({
                    'step': self.global_step,
                    'metrics': metrics
                })
                
                # Print metrics
                self.logger.info(f"  Recall@10: {metrics.get('recall@10', 0):.4f}")
                self.logger.info(f"  MRR: {metrics.get('mrr', 0):.4f}")
                self.logger.info(f"  Actions: {metrics.get('expand_count', 0)} expand, {metrics.get('stop_count', 0)} stop")
                
                # Early stopping check
                current_metric = metrics.get('recall@10', 0)
                if current_metric > self.best_metric + self.config.min_delta:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    self.save_checkpoint(
                        Path(self.config.checkpoint_dir) / f'step_{self.global_step}.pt',
                        is_best=True
                    )
                else:
                    self.patience_counter += 1
                    if self.config.early_stopping and self.patience_counter >= self.config.patience:
                        self.logger.info(f"⚠️ Early stopping triggered after {self.patience_counter} evaluations without improvement")
                        return False
            
            # Save checkpoint
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint(Path(self.config.checkpoint_dir) / 'latest.pt')
        
        # Final PPO update if buffer not empty
        if self.buffer['q_embs']:
            self._ppo_update()
        
        self.logger.info(f"\n  Epoch {epoch + 1} completed. Avg reward: {np.mean(epoch_rewards):.4f}")
        return True
    
    def train(self):
        """Main training loop."""
        self.logger.info(self.config.summary())
        
        for epoch in range(self.config.epochs):
            continue_training = self.train_epoch(epoch)
            
            if not continue_training:
                break
        
        # Final save
        self.save_checkpoint(Path(self.config.checkpoint_dir) / 'final.pt')
        
        # Final evaluation
        self.logger.info("\n" + "=" * 60)
        self.logger.info("FINAL EVALUATION")
        self.logger.info("=" * 60)
        
        final_metrics = self.evaluate('valid', max_queries=min(1000, self.config.num_eval_queries * 2))
        
        self.logger.info(f"\n📊 Final Results:")
        for key in ['recall@10', 'recall@100', 'mrr', 'map', 'precision@10']:
            if key in final_metrics:
                self.logger.info(f"  {key}: {final_metrics[key]:.4f}")
        
        # Save training history
        history_path = Path(self.config.log_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'train_history': self.train_history,
                'eval_history': self.eval_history,
                'final_metrics': final_metrics
            }, f, indent=2)
        
        self.logger.info(f"\n✅ Training history saved to {history_path}")
        
        return final_metrics


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Adaptive IR Full Epoch Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'medium', 'full'],
                        help='Training mode preset')
    
    # Training
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    
    # Evaluation
    parser.add_argument('--eval-every', type=int, default=None, help='Evaluate every N steps')
    parser.add_argument('--num-eval-queries', type=int, default=None, help='Number of eval queries')
    
    # Checkpoint
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file')
    
    # Other
    parser.add_argument('--no-early-stopping', action='store_true', help='Disable early stopping')
    
    return parser.parse_args()


def setup_logging(config: TrainingConfig):
    """Setup logging."""
    log_file = Path(config.log_dir) / f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load or create config
    if args.config:
        config = TrainingConfig.from_json(args.config)
    else:
        config = TrainingConfig(mode=args.mode)
    
    # Override with command line args
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.eval_every is not None:
        config.eval_every = args.eval_every
    if args.num_eval_queries is not None:
        config.num_eval_queries = args.num_eval_queries
    if args.no_early_stopping:
        config.early_stopping = False
    
    # Setup logging
    logger = setup_logging(config)
    
    print("=" * 80)
    print("🚀 ADAPTIVE IR - FULL EPOCH TRAINING")
    print("=" * 80)
    
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory/1e9:.1f} GB)")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Save config
    config.to_json(Path(config.checkpoint_dir) / 'config.json')
    
    # ==========================================================================
    # LOAD DATA
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    
    data = DataManager(config, device)
    data.load_all()
    
    # ==========================================================================
    # BUILD SEARCH ENGINE
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Building BM25 Search Engine")
    print("=" * 60)
    
    search = BM25SearchEngine(data, config)
    search.build_index()
    
    # Pre-compute queries
    train_qids = data.get_train_qids()
    train_queries = {qid: data.queries['train'][qid] for qid in train_qids}
    search.precompute_queries(train_queries, k=config.top_k_retrieve, desc="Pre-computing train queries")
    
    valid_queries = {qid: data.queries['valid'][qid] 
                     for qid in list(data.queries['valid'].keys())[:config.num_eval_queries]}
    search.precompute_queries(valid_queries, k=config.top_k_retrieve, desc="Pre-computing valid queries")
    
    # ==========================================================================
    # INITIALIZE MODEL
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Initializing Model")
    print("=" * 60)
    
    agent = QueryReformulationAgent(config).to(device)
    
    num_params = sum(p.numel() for p in agent.parameters())
    logger.info(f"  Model parameters: {num_params:,}")
    
    # ==========================================================================
    # BASELINE EVALUATION
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Baseline Evaluation (BM25)")
    print("=" * 60)
    
    # Quick baseline eval
    from evaluation.metrics import IRMetricsAggregator
    aggregator = IRMetricsAggregator()
    
    for qid in tqdm(list(valid_queries.keys())[:100], desc="Baseline eval"):
        query = data.queries['valid'][qid]
        relevant = data.qrels['valid'][qid]
        relevant_str = {str(r) for r in relevant}
        
        results = search.get_retrieved_ids(query, k=config.top_k_retrieve)
        retrieved_str = [str(r) for r in results]
        
        aggregator.add_query_result(query_id=qid, retrieved=retrieved_str, relevant=relevant_str, latency=0)
    
    baseline = aggregator.compute_aggregate()
    logger.info(f"  Baseline Recall@10: {baseline.get('recall@10', 0):.4f}")
    logger.info(f"  Baseline MRR: {baseline.get('mrr', 0):.4f}")
    
    # ==========================================================================
    # TRAINING
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Training")
    print("=" * 60)
    
    trainer = Trainer(agent, data, search, config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    t_start = time.time()
    final_metrics = trainer.train()
    t_total = time.time() - t_start
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nTotal time: {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"Best Recall@10: {trainer.best_metric:.4f}")
    print(f"Checkpoints saved in: {config.checkpoint_dir}")
    print(f"Logs saved in: {config.log_dir}")


if __name__ == "__main__":
    main()
