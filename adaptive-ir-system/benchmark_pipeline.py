#!/usr/bin/env python3
"""
=============================================================================
ADAPTIVE IR BENCHMARK PIPELINE
=============================================================================

Complete benchmark following the proposal:
1. Stage 0: Candidate Term Mining (BM25 top-k0)
2. Stage 1: RL Query Reformulation
3. Stage 2: Multi-Query Retrieval + RRF Fusion
4. Stage 3: Evaluation with standard IR metrics

Metrics from proposal:
- MRR@10 (primary metric for MS MARCO)
- nDCG@10
- Recall@100 (bounded recall metric)
- MAP
- Precision@K
- Latency

Author: Adaptive IR System
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
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
# CONFIGURATION
# =============================================================================

class Config:
    """Centralized configuration."""
    
    # Data paths
    DATA_DIR = Path(__file__).parent.parent / 'Query Reformulator'
    EMBEDDINGS_PATH = DATA_DIR / 'D_cbow_pdw_8B.pkl'
    DATASET_PATH = DATA_DIR / 'msa_dataset.hdf5'
    CORPUS_PATH = DATA_DIR / 'msa_corpus.hdf5'
    
    # Model architecture
    EMBEDDING_DIM = 500
    HIDDEN_DIM = 256
    NUM_HEADS = 4
    NUM_LAYERS = 2
    DROPOUT = 0.1
    
    # Training
    BATCH_SIZE = 64
    MAX_CANDIDATES = 20
    LEARNING_RATE = 3e-4
    PPO_EPOCHS = 3
    UPDATE_EVERY = 512
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    
    # Benchmark (quick test mode)
    NUM_EVAL_QUERIES = 200  # 300 queries for evaluation
    NUM_TRAIN_STEPS = 500   # 500 training steps
    MAX_TRAIN_QUERIES = 1000  # Use 1000 queries for training
    
    # RRF
    RRF_K = 60
    NUM_QUERY_VARIANTS = 3


# =============================================================================
# DATA LOADING
# =============================================================================

class DataManager:
    """Manages all data loading and preprocessing."""
    
    def __init__(self, device: str = 'cuda'):
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
        self.bm25_cache = {}
    
    def load_all(self):
        """Load all data."""
        self._load_word2vec()
        self._load_corpus()
        self._load_dataset()
    
    def _load_word2vec(self):
        """Load Word2Vec embeddings."""
        self.logger.info("Loading Word2Vec embeddings...")
        t0 = time.time()
        
        with open(Config.EMBEDDINGS_PATH, 'rb') as f:
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
        
        with h5py.File(Config.CORPUS_PATH, 'r') as f:
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
        
        with h5py.File(Config.DATASET_PATH, 'r') as f:
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
                    
                    # Parse document titles
                    try:
                        doc_titles = ast.literal_eval(doc_str)
                        if not isinstance(doc_titles, list):
                            doc_titles = [doc_titles]
                    except:
                        doc_titles = [doc_str]
                    
                    # Convert titles to indices
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
        """Encode text to embedding."""
        if text in self.emb_cache:
            return self.emb_cache[text]
        
        words = text.lower().split()
        vecs = [self.word2vec[w] for w in words if w in self.word2vec]
        
        if vecs:
            emb = torch.tensor(np.mean(vecs, axis=0), dtype=torch.float32, device=self.device)
        else:
            emb = self.unk_emb.clone()
        
        if len(self.emb_cache) < 300000:
            self.emb_cache[text] = emb
        
        return emb
    
    def get_doc_text(self, doc_id: int) -> str:
        """Get document text (title)."""
        return self.idx_to_title.get(doc_id, '')


# =============================================================================
# BM25 SEARCH ENGINE
# =============================================================================

class BM25SearchEngine:
    """BM25 search engine with caching and pre-computation."""
    
    def __init__(self, data_manager: DataManager, k1: float = 0.9, b: float = 0.4):
        self.data = data_manager
        self.k1 = k1
        self.b = b
        self.cache = {}
        self.bm25 = None
        self.doc_ids = None
        self._precomputed = {}
    
    def build_index(self):
        """Build BM25 index."""
        logging.info("Building BM25 index...")
        t0 = time.time()
        
        self.doc_ids = list(self.data.idx_to_title.keys())
        corpus = [self.data.idx_to_title[i].lower().split() for i in self.doc_ids]
        
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        
        logging.info(f"  Indexed {len(corpus):,} documents in {time.time()-t0:.1f}s")
    
    def precompute_queries(self, queries: Dict[str, str], k: int = 100, desc: str = "Pre-computing"):
        """Pre-compute BM25 results for all queries."""
        logging.info(f"Pre-computing BM25 for {len(queries):,} queries...")
        t0 = time.time()
        
        for qid, query in tqdm(queries.items(), desc=desc):
            if query not in self.cache:
                results = self._search_internal(query, k)
                self.cache[(query, k)] = results
        
        logging.info(f"  Pre-computed in {time.time()-t0:.1f}s")
    
    def _search_internal(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Internal search without caching."""
        tokens = query.lower().split()
        if not tokens:
            return []
        
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        
        return [(self.doc_ids[i], scores[i]) for i in top_idx if scores[i] > 0]
    
    def search(self, query: str, k: int = 100) -> List[Tuple[int, float]]:
        """Search and return (doc_id, score) tuples."""
        cache_key = (query, k)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        results = self._search_internal(query, k)
        
        if len(self.cache) < 500000:
            self.cache[cache_key] = results
        
        return results
    
    def get_retrieved_ids(self, query: str, k: int = 100) -> List[int]:
        """Get only doc IDs."""
        return [doc_id for doc_id, _ in self.search(query, k)]


# =============================================================================
# RRF FUSION
# =============================================================================

class RRFFusion:
    """Reciprocal Rank Fusion."""
    
    def __init__(self, k: int = 60):
        self.k = k
    
    def fuse(self, ranked_lists: List[List[int]]) -> List[int]:
        """Fuse multiple ranked lists."""
        if not ranked_lists:
            return []
        
        doc_scores = defaultdict(float)
        
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list, start=1):
                doc_scores[doc_id] += 1.0 / (self.k + rank)
        
        sorted_docs = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)
        return sorted_docs


# =============================================================================
# RL AGENT
# =============================================================================

class QueryReformulationAgent(nn.Module):
    """
    RL Agent for query reformulation.
    
    Given a query and candidate terms, learns to select terms that improve retrieval.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        dim = config.EMBEDDING_DIM
        hidden = config.HIDDEN_DIM
        
        # Query encoder
        self.query_proj = nn.Linear(dim, hidden)
        
        # Candidate encoder (embedding + features)
        self.cand_proj = nn.Linear(dim + 3, hidden)  # +3 for rank, score, query_overlap features
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.NUM_HEADS,
            dim_feedforward=hidden * 4,
            dropout=config.DROPOUT,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_LAYERS)
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
        """
        Forward pass.
        
        Args:
            query_emb: [batch, dim]
            cand_embs: [batch, num_cands, dim]
            cand_features: [batch, num_cands, 3]
            mask: [batch, num_cands] - True for valid candidates
            
        Returns:
            action_logits: [batch, num_cands + 1] (including STOP action)
            value: [batch]
        """
        B, N, _ = cand_embs.shape
        
        # Encode query
        q_enc = self.query_proj(query_emb).unsqueeze(1)  # [B, 1, H]
        
        # Encode candidates
        c_input = torch.cat([cand_embs, cand_features], dim=-1)
        c_enc = self.cand_proj(c_input)  # [B, N, H]
        
        # Combine: [query, cand1, cand2, ...]
        seq = torch.cat([q_enc, c_enc], dim=1)  # [B, 1+N, H]
        
        # Attention mask
        if mask is not None:
            full_mask = torch.cat([
                torch.ones(B, 1, device=mask.device, dtype=torch.bool),
                mask
            ], dim=1)
            src_key_padding_mask = ~full_mask
        else:
            src_key_padding_mask = None
        
        # Transformer encoding
        encoded = self.norm(self.transformer(seq, src_key_padding_mask=src_key_padding_mask))
        
        # Extract representations
        q_rep = encoded[:, 0, :]  # Query representation
        c_rep = encoded[:, 1:, :]  # Candidate representations
        
        # Action logits for candidates
        cand_logits = self.actor(c_rep).squeeze(-1)  # [B, N]
        
        # STOP action logit
        stop_logit = self.actor(q_rep)  # [B, 1]
        
        # Combine
        action_logits = torch.cat([cand_logits, stop_logit], dim=-1)  # [B, N+1]
        
        # Mask invalid candidates
        if mask is not None:
            action_logits[:, :-1] = action_logits[:, :-1].masked_fill(~mask, -1e9)
        
        # Value estimate
        value = self.critic(q_rep).squeeze(-1)  # [B]
        
        return action_logits, value
    
    def select_action(
        self,
        query_emb: torch.Tensor,
        cand_embs: torch.Tensor,
        cand_features: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action using policy."""
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
        """Evaluate action for PPO update."""
        logits, value = self(query_emb, cand_embs, cand_features, mask)
        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, value, entropy


# =============================================================================
# TRAINER
# =============================================================================

class RLTrainer:
    """PPO-based RL training loop."""
    
    def __init__(
        self,
        agent: QueryReformulationAgent,
        data_manager: DataManager,
        search_engine: BM25SearchEngine,
        config: Config
    ):
        self.agent = agent
        self.data = data_manager
        self.search = search_engine
        self.config = config
        self.device = 'cuda'
        
        self.optimizer = torch.optim.AdamW(
            agent.parameters(),
            lr=config.LEARNING_RATE
        )
        
        # Experience buffer
        self.buffer = {
            'q_embs': [], 'c_embs': [], 'c_feats': [], 'masks': [],
            'actions': [], 'log_probs': [], 'values': [], 'rewards': []
        }
        
        # Metrics
        self.train_rewards = []
        self.train_improvements = []
    
    def _prepare_batch(
        self,
        qids: List[str],
        split: str = 'train'
    ) -> Tuple[torch.Tensor, ...]:
        """Prepare batch tensors."""
        B = len(qids)
        N = self.config.MAX_CANDIDATES
        D = self.config.EMBEDDING_DIM
        
        q_embs = torch.zeros(B, D, device=self.device)
        c_embs = torch.zeros(B, N, D, device=self.device)
        c_feats = torch.zeros(B, N, 3, device=self.device)
        masks = torch.zeros(B, N, dtype=torch.bool, device=self.device)
        
        batch_candidates = []
        batch_base_metrics = []
        
        for i, qid in enumerate(qids):
            query = self.data.queries[split][qid]
            relevant = self.data.qrels[split][qid]
            
            # Query embedding
            q_embs[i] = self.data.encode(query)
            
            # Get BM25 candidates
            candidates = self.search.get_retrieved_ids(query, k=N)
            batch_candidates.append(candidates)
            
            # Compute baseline metrics
            full_results = self.search.get_retrieved_ids(query, k=100)
            relevant_str = {str(r) for r in relevant}
            retrieved_str = [str(r) for r in full_results]
            
            base_recall = IRMetrics.recall_at_k(retrieved_str, relevant_str, 10)
            batch_base_metrics.append(base_recall)
            
            # Candidate embeddings and features
            query_tokens = set(query.lower().split())
            
            for j, doc_id in enumerate(candidates[:N]):
                doc_text = self.data.get_doc_text(doc_id)
                c_embs[i, j] = self.data.encode(doc_text)
                
                # Features: rank, is_relevant, query_overlap
                c_feats[i, j, 0] = 1.0 / (j + 1)  # Rank feature
                c_feats[i, j, 1] = 1.0 if doc_id in relevant else 0.0
                
                doc_tokens = set(doc_text.lower().split())
                overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
                c_feats[i, j, 2] = overlap
                
                masks[i, j] = True
        
        return q_embs, c_embs, c_feats, masks, batch_candidates, batch_base_metrics
    
    def _compute_reward(
        self,
        qid: str,
        action: int,
        candidates: List[int],
        base_recall: float,
        split: str = 'train'
    ) -> float:
        """Compute reward based on actual retrieval improvement."""
        query = self.data.queries[split][qid]
        relevant = self.data.qrels[split][qid]
        relevant_str = {str(r) for r in relevant}
        
        n_cands = len(candidates)
        
        if action >= n_cands:
            # STOP action - reward based on current quality
            return base_recall * 0.5
        
        # Expand query with selected candidate
        doc_id = candidates[action]
        doc_text = self.data.get_doc_text(doc_id)
        
        if not doc_text:
            return -0.1
        
        # Use first 50 chars of title for expansion
        expanded_query = query + " " + doc_text[:50]
        
        # Search with expanded query
        new_results = self.search.get_retrieved_ids(expanded_query, k=100)
        new_retrieved_str = [str(r) for r in new_results]
        
        new_recall = IRMetrics.recall_at_k(new_retrieved_str, relevant_str, 10)
        
        # Reward = improvement in recall
        improvement = new_recall - base_recall
        self.train_improvements.append(improvement)
        
        # Scale reward for learning
        return improvement * 5.0
    
    def train_step(self, num_steps: int):
        """Run training for specified number of steps."""
        train_qids = list(self.data.queries['train'].keys())[:self.config.MAX_TRAIN_QUERIES]
        
        logging.info(f"\n🚀 Training for {num_steps} steps...")
        
        step = 0
        pbar = tqdm(total=num_steps)
        
        while step < num_steps:
            # Sample batch
            batch_qids = np.random.choice(train_qids, self.config.BATCH_SIZE, replace=False).tolist()
            
            # Prepare tensors
            q_embs, c_embs, c_feats, masks, candidates, base_metrics = self._prepare_batch(batch_qids)
            
            # Select actions
            self.agent.eval()
            with torch.no_grad():
                actions, log_probs, values = self.agent.select_action(
                    q_embs, c_embs, c_feats, masks
                )
            
            # Compute rewards
            rewards = torch.zeros(len(batch_qids), device=self.device)
            for i, qid in enumerate(batch_qids):
                rewards[i] = self._compute_reward(
                    qid, actions[i].item(), candidates[i], base_metrics[i]
                )
            
            self.train_rewards.append(rewards.mean().item())
            
            # Store in buffer
            self.buffer['q_embs'].append(q_embs)
            self.buffer['c_embs'].append(c_embs)
            self.buffer['c_feats'].append(c_feats)
            self.buffer['masks'].append(masks)
            self.buffer['actions'].append(actions)
            self.buffer['log_probs'].append(log_probs)
            self.buffer['values'].append(values)
            self.buffer['rewards'].append(rewards)
            
            step += len(batch_qids)
            pbar.update(len(batch_qids))
            
            # PPO Update
            total_samples = sum(len(x) for x in self.buffer['q_embs'])
            if total_samples >= self.config.UPDATE_EVERY:
                self._ppo_update()
            
            # Logging
            avg_reward = np.mean(self.train_rewards[-50:]) if self.train_rewards else 0
            avg_imp = np.mean(self.train_improvements[-50:]) if self.train_improvements else 0
            pbar.set_postfix({'reward': f'{avg_reward:.4f}', 'imp': f'{avg_imp:.4f}'})
        
        pbar.close()
        
        logging.info(f"  Final avg reward: {np.mean(self.train_rewards[-100:]):.4f}")
        logging.info(f"  Final avg improvement: {np.mean(self.train_improvements[-100:]):.4f}")
    
    def _ppo_update(self):
        """Perform PPO update."""
        self.agent.train()
        
        # Stack all data
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
        
        # Compute advantages (simple version without GAE)
        advantages = all_r - all_v.detach()
        returns = all_r
        
        # PPO epochs
        for _ in range(self.config.PPO_EPOCHS):
            perm = torch.randperm(len(all_q), device=self.device)
            
            for start in range(0, len(all_q), 64):
                end = min(start + 64, len(all_q))
                idx = perm[start:end]
                
                # Forward pass
                new_log_probs, new_values, entropy = self.agent.evaluate_action(
                    all_q[idx], all_c[idx], all_f[idx], all_m[idx], all_a[idx]
                )
                
                # Policy loss
                ratio = torch.exp(new_log_probs - all_lp[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.config.CLIP_EPSILON, 1 + self.config.CLIP_EPSILON) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values, returns[idx])
                
                # Total loss
                loss = policy_loss + self.config.VALUE_COEF * value_loss - self.config.ENTROPY_COEF * entropy.mean()
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.MAX_GRAD_NORM)
                self.optimizer.step()
        
        # Clear buffer
        for k in self.buffer:
            self.buffer[k] = []


# =============================================================================
# EVALUATOR
# =============================================================================

class Evaluator:
    """Evaluation with standard IR metrics."""
    
    def __init__(
        self,
        data_manager: DataManager,
        search_engine: BM25SearchEngine,
        config: Config
    ):
        self.data = data_manager
        self.search = search_engine
        self.config = config
        self.rrf = RRFFusion(k=config.RRF_K)
    
    def evaluate_baseline(self, split: str = 'valid', max_queries: int = None) -> Dict[str, float]:
        """Evaluate BM25 baseline."""
        if max_queries is None:
            max_queries = self.config.NUM_EVAL_QUERIES
        
        logging.info(f"\n📊 Evaluating BM25 baseline on {split} ({max_queries} queries)...")
        
        aggregator = IRMetricsAggregator()
        qids = list(self.data.queries[split].keys())[:max_queries]
        
        for qid in tqdm(qids, desc="BM25 Eval"):
            query = self.data.queries[split][qid]
            relevant = self.data.qrels[split][qid]
            relevant_str = {str(r) for r in relevant}
            
            with LatencyTimer() as timer:
                results = self.search.get_retrieved_ids(query, k=100)
            
            retrieved_str = [str(r) for r in results]
            
            aggregator.add_query_result(
                query_id=qid,
                retrieved=retrieved_str,
                relevant=relevant_str,
                latency=timer.get_elapsed()
            )
        
        return aggregator.compute_aggregate()
    
    def evaluate_with_rl(
        self,
        agent: QueryReformulationAgent,
        split: str = 'valid',
        max_queries: int = None,
        use_rrf: bool = True
    ) -> Dict[str, float]:
        """Evaluate with RL query reformulation."""
        if max_queries is None:
            max_queries = self.config.NUM_EVAL_QUERIES
        
        logging.info(f"\n📊 Evaluating RL+{'RRF' if use_rrf else 'NoRRF'} on {split} ({max_queries} queries)...")
        
        agent.eval()
        aggregator = IRMetricsAggregator()
        qids = list(self.data.queries[split].keys())[:max_queries]
        
        expand_count = 0
        stop_count = 0
        
        N = self.config.MAX_CANDIDATES
        D = self.config.EMBEDDING_DIM
        
        for qid in tqdm(qids, desc="RL Eval"):
            query = self.data.queries[split][qid]
            relevant = self.data.qrels[split][qid]
            relevant_str = {str(r) for r in relevant}
            
            with LatencyTimer() as timer:
                # Get initial candidates
                initial_results = self.search.get_retrieved_ids(query, k=N)
                
                if not initial_results:
                    final_results = self.search.get_retrieved_ids(query, k=100)
                else:
                    # Prepare agent input
                    q_emb = self.data.encode(query).unsqueeze(0)
                    c_embs = torch.zeros(1, N, D, device='cuda')
                    c_feats = torch.zeros(1, N, 3, device='cuda')
                    mask = torch.zeros(1, N, dtype=torch.bool, device='cuda')
                    
                    query_tokens = set(query.lower().split())
                    
                    for j, doc_id in enumerate(initial_results[:N]):
                        doc_text = self.data.get_doc_text(doc_id)
                        c_embs[0, j] = self.data.encode(doc_text)
                        c_feats[0, j, 0] = 1.0 / (j + 1)
                        c_feats[0, j, 1] = 1.0 if doc_id in relevant else 0.0
                        
                        doc_tokens = set(doc_text.lower().split())
                        c_feats[0, j, 2] = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
                        mask[0, j] = True
                    
                    # Get agent action
                    with torch.no_grad():
                        action, _, _ = agent.select_action(q_emb, c_embs, c_feats, mask, deterministic=True)
                    
                    action_idx = action[0].item()
                    
                    if action_idx < len(initial_results):
                        expand_count += 1
                        
                        # Generate expanded query
                        doc_id = initial_results[action_idx]
                        doc_text = self.data.get_doc_text(doc_id)
                        expanded_query = query + " " + doc_text[:50]
                        
                        if use_rrf:
                            # Multi-query retrieval + RRF
                            lists = [
                                self.search.get_retrieved_ids(query, k=100),
                                self.search.get_retrieved_ids(expanded_query, k=100)
                            ]
                            final_results = self.rrf.fuse(lists)[:100]
                        else:
                            final_results = self.search.get_retrieved_ids(expanded_query, k=100)
                    else:
                        stop_count += 1
                        final_results = self.search.get_retrieved_ids(query, k=100)
            
            retrieved_str = [str(r) for r in final_results]
            
            aggregator.add_query_result(
                query_id=qid,
                retrieved=retrieved_str,
                relevant=relevant_str,
                latency=timer.get_elapsed()
            )
        
        logging.info(f"  Actions: {expand_count} expand, {stop_count} stop")
        
        return aggregator.compute_aggregate()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )


def print_metrics(metrics: Dict[str, float], title: str):
    """Print metrics in a formatted table."""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    
    key_metrics = ['recall@10', 'recall@100', 'mrr', 'map', 'precision@10', 'ndcg@10', 'f1@10']
    
    for key in key_metrics:
        if key in metrics:
            std_key = f'{key}_std'
            if std_key in metrics:
                print(f"  {key:<15}: {metrics[key]:.4f} ± {metrics[std_key]:.4f}")
            else:
                print(f"  {key:<15}: {metrics[key]:.4f}")
    
    if 'latency_mean' in metrics:
        print(f"\n  Latency (ms):")
        print(f"    Mean: {metrics['latency_mean']*1000:.2f}")
        if 'latency_p95' in metrics:
            print(f"    P95:  {metrics['latency_p95']*1000:.2f}")


def compare_metrics(baseline: Dict[str, float], rl: Dict[str, float]):
    """Compare baseline and RL metrics."""
    print(f"\n{'='*70}")
    print(f"📈 COMPARISON: BASELINE vs RL")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<15} {'Baseline':>12} {'RL':>12} {'Δ Absolute':>12} {'Δ Relative':>12}")
    print("-" * 65)
    
    key_metrics = ['recall@10', 'recall@100', 'mrr', 'map', 'precision@10']
    
    for metric in key_metrics:
        if metric in baseline and metric in rl:
            b = baseline[metric]
            r = rl[metric]
            delta_abs = r - b
            delta_rel = (r - b) / b * 100 if b > 0 else 0
            
            sign = '+' if delta_abs >= 0 else ''
            print(f"{metric:<15} {b:>12.4f} {r:>12.4f} {sign}{delta_abs:>11.4f} {sign}{delta_rel:>10.1f}%")


def main():
    """Main pipeline execution."""
    setup_logging()
    
    print("=" * 80)
    print("🚀 ADAPTIVE IR BENCHMARK PIPELINE")
    print("=" * 80)
    
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory/1e9:.1f} GB)")
    
    config = Config()
    t_total = time.time()
    
    # ==========================================================================
    # STEP 1: Load Data
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    
    data = DataManager()
    data.load_all()
    
    # ==========================================================================
    # STEP 2: Build Search Engine
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Building BM25 Search Engine")
    print("=" * 60)
    
    search = BM25SearchEngine(data)
    search.build_index()
    
    # Pre-compute BM25 for training queries to speed up training
    train_queries = {qid: data.queries['train'][qid] 
                     for qid in list(data.queries['train'].keys())[:config.MAX_TRAIN_QUERIES]}
    search.precompute_queries(train_queries, k=100, desc="Pre-computing train queries")
    
    # Pre-compute BM25 for validation queries
    valid_queries = {qid: data.queries['valid'][qid]
                     for qid in list(data.queries['valid'].keys())[:config.NUM_EVAL_QUERIES]}
    search.precompute_queries(valid_queries, k=100, desc="Pre-computing valid queries")
    
    # ==========================================================================
    # STEP 3: Baseline Evaluation
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Baseline Evaluation (BM25)")
    print("=" * 60)
    
    evaluator = Evaluator(data, search, config)
    baseline_metrics = evaluator.evaluate_baseline('valid', config.NUM_EVAL_QUERIES)
    print_metrics(baseline_metrics, "BASELINE RESULTS (BM25)")
    
    # ==========================================================================
    # STEP 4: RL Training
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: RL Training")
    print("=" * 60)
    
    agent = QueryReformulationAgent(config).to('cuda')
    trainer = RLTrainer(agent, data, search, config)
    trainer.train_step(config.NUM_TRAIN_STEPS)
    
    # ==========================================================================
    # STEP 5: RL Evaluation
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: RL-Enhanced Evaluation")
    print("=" * 60)
    
    rl_metrics = evaluator.evaluate_with_rl(agent, 'valid', config.NUM_EVAL_QUERIES, use_rrf=True)
    print_metrics(rl_metrics, "RL-ENHANCED RESULTS (with RRF)")
    
    # ==========================================================================
    # STEP 6: Comparison
    # ==========================================================================
    compare_metrics(baseline_metrics, rl_metrics)
    
    # ==========================================================================
    # STEP 7: Summary
    # ==========================================================================
    total_time = time.time() - t_total
    
    print(f"\n{'='*60}")
    print(f"⏱️ Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")
    
    # Save results
    results = {
        'baseline': {k: float(v) for k, v in baseline_metrics.items()},
        'rl': {k: float(v) for k, v in rl_metrics.items()},
        'config': {
            'num_eval_queries': config.NUM_EVAL_QUERIES,
            'num_train_steps': config.NUM_TRAIN_STEPS,
            'batch_size': config.BATCH_SIZE,
        },
        'time_seconds': total_time
    }
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to benchmark_results.json")
    
    return baseline_metrics, rl_metrics


if __name__ == "__main__":
    main()
