"""
End-to-End Pipeline

Integrates all 4 stages:
1. Candidate Term Mining
2. RL Query Reformulation  
3. Multi-Query Retrieval + RRF Fusion
4. BERT Cross-Encoder Re-ranking
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import time
import logging

from ..candidate_mining import CandidateTermMiner
from ..rl_agent import QueryReformulatorAgent
from ..fusion import RecipRankFusion
from ..reranker import BERTReranker
from ..evaluation import IRMetricsAggregator, LatencyTimer


class AdaptiveIRPipeline:
    """
    Complete Adaptive Retrieve-Fuse-Re-rank pipeline.
    """
    
    def __init__(
        self,
        config: Dict,
        search_engine,
        embedding_model=None
    ):
        """
        Args:
            config: Configuration dictionary
            search_engine: Search engine instance (BM25/Pyserini)
            embedding_model: Optional embedding model for candidate features
        """
        self.config = config
        self.search_engine = search_engine
        self.embedding_model = embedding_model
        self.device = config.get('system', {}).get('device', 'cuda')
        
        # Initialize components
        self.logger = logging.getLogger(__name__)
        
        # Stage 0: Candidate Mining
        if config.get('candidate_mining', {}).get('enabled', True):
            self.candidate_miner = CandidateTermMiner(config['candidate_mining'])
            self.logger.info("Initialized Candidate Term Miner")
        else:
            self.candidate_miner = None
        
        # Stage 1: RL Agent
        if config.get('rl_agent', {}).get('enabled', True):
            self.rl_agent = QueryReformulatorAgent(config['rl_agent'])
            self.rl_agent.to(self.device)
            self.rl_agent.eval()
            self.logger.info(f"Initialized RL Agent with {sum(p.numel() for p in self.rl_agent.parameters()):,} parameters")
        else:
            self.rl_agent = None
        
        # Stage 2: RRF Fusion
        if config.get('rrf_fusion', {}).get('enabled', True):
            k_constant = config['rrf_fusion'].get('k_constant', 60)
            self.rrf_fusion = RecipRankFusion(k=k_constant)
            self.logger.info(f"Initialized RRF Fusion (k={k_constant})")
        else:
            self.rrf_fusion = None
        
        # Stage 3: BERT Re-ranker
        if config.get('bert_reranker', {}).get('enabled', True):
            self.bert_reranker = BERTReranker(config['bert_reranker'])
            self.logger.info("Initialized BERT Re-ranker")
        else:
            self.bert_reranker = None
        
        # Retrieval parameters
        self.top_k = config.get('retrieval', {}).get('top_k', 100)
        self.num_query_variants = config.get('rl_agent', {}).get('num_query_variants', 4)
        self.max_steps = config.get('rl_agent', {}).get('max_steps_per_episode', 5)
        
        self.logger.info("Pipeline initialized successfully")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Basic BM25 retrieval.
        
        Args:
            query: Query string
            top_k: Number of results
            
        Returns:
            (doc_ids, scores)
        """
        if top_k is None:
            top_k = self.top_k
        
        results = self.search_engine.search(query, top_k)
        
        # --- PATCHED FOR PYSERINI ---
        if results and hasattr(results[0], 'docid'):
            # Pyserini returns objects with attributes
            doc_ids = [r.docid for r in results]
            scores = [r.score for r in results]
        else:
            # Legacy returns dictionaries
            # --- PATCHED FOR PYSERINI ---
            if results and hasattr(results[0], 'docid'):
                # Pyserini returns objects with attributes
                doc_ids = [r.docid for r in results]
                scores = [r.score for r in results]
            else:
                # Legacy returns dictionaries
                doc_ids = [r['doc_id'] for r in results]
                scores = [r['score'] for r in results]
        
        return doc_ids, scores
    
    def mine_candidates(
        self,
        query: str,
        top_k0: int = 50
    ) -> Dict[str, Dict[str, float]]:
        """
        Stage 0: Mine candidate terms from top-k0 documents.
        
        Args:
            query: Original query
            top_k0: Number of documents for mining
            
        Returns:
            Candidate terms with features
        """
        if not self.candidate_miner:
            return {}
        
        # Retrieve top-k0 documents
        doc_ids, scores = self.retrieve(query, top_k=top_k0)
        
        # Get document texts
        documents = []
        for doc_id in doc_ids:
            # --- PATCHED FOR PYSERINI ---
            if hasattr(self.search_engine, 'doc'):
                # Pyserini Logic
                doc_obj = self.search_engine.doc(doc_id)
                if doc_obj:
                    try:
                        import json
                        # Pyserini thường trả về JSON string, ta cần lấy trường 'contents'
                        doc_text = json.loads(doc_obj.raw()).get('contents', doc_obj.raw())
                    except:
                        doc_text = doc_obj.raw()
                else:
                    doc_text = ''
            else:
                # Legacy Logic
                # --- PATCHED FOR PYSERINI ---
                if hasattr(self.search_engine, 'doc'):
                    # Pyserini Logic
                    doc_obj = self.search_engine.doc(doc_id)
                    if doc_obj:
                        try:
                            import json
                            # Pyserini thường trả về JSON string, ta cần lấy trường 'contents'
                            doc_text = json.loads(doc_obj.raw()).get('contents', doc_obj.raw())
                        except:
                            doc_text = doc_obj.raw()
                    else:
                        doc_text = ''
                else:
                    # Legacy Logic
                    doc_text = self.search_engine.get_document(doc_id)
            if doc_text:
                documents.append(doc_text)
        
        # Extract candidates
        candidates = self.candidate_miner.extract_candidates(query, documents, scores)
        
        return candidates
    
    def reformulate_query(
        self,
        query: str,
        candidates: Dict[str, Dict[str, float]],
        num_variants: Optional[int] = None
    ) -> List[str]:
        """
        Stage 1: Generate query variants using RL agent.
        
        Args:
            query: Original query
            candidates: Candidate terms with features
            num_variants: Number of query variants to generate
            
        Returns:
            List of reformulated queries
        """
        if not self.rl_agent or not candidates:
            # Fallback: return original query
            return [query]
        
        if num_variants is None:
            num_variants = self.num_query_variants
        
        # Prepare inputs for RL agent
        query_emb = self._embed_text(query)
        
        candidate_terms = list(candidates.keys())
        candidate_embs = torch.stack([self._embed_text(term) for term in candidate_terms])
        
        # Extract features
        feature_matrix = self.candidate_miner.get_candidate_features(candidates)
        candidate_features = torch.tensor(feature_matrix, dtype=torch.float32)
        
        # Move to device
        query_emb = query_emb.to(self.device)
        candidate_embs = candidate_embs.to(self.device)
        candidate_features = candidate_features.to(self.device)
        
        ### nnminh
        with torch.no_grad():
            q0_enc_cached, cand_enc_cached = self.rl_agent.encode_static(
                query_emb.unsqueeze(0), # Thêm dim batch
                candidate_embs.unsqueeze(0), 
                candidate_features.unsqueeze(0)
            )
        
        query_variants = [query]  
        
        with torch.no_grad():
            for _ in range(num_variants - 1):
                current_query = query
                selected_terms = []
                
                for step in range(self.max_steps):
                    current_emb = self._embed_text(current_query).unsqueeze(0).to(self.device)
                    
                    action, log_prob, value = self.rl_agent.select_action(
                        query_emb.unsqueeze(0),
                        current_emb,
                        candidate_embs.unsqueeze(0),
                        candidate_features.unsqueeze(0),
                        deterministic=False,
                        # add cached encodings
                        q0_enc=q0_enc_cached,
                        cand_enc=cand_enc_cached
                    )
        ### done\
            
            
            
                    action_idx = action.item()
                    
                    # Check if STOP action
                    if action_idx >= len(candidate_terms):
                        break
                    
                    # Add selected term
                    selected_term = candidate_terms[action_idx]
                    if selected_term not in selected_terms:
                        selected_terms.append(selected_term)
                        current_query = current_query + " " + selected_term
                
                if current_query != query:
                    query_variants.append(current_query)
        
        return query_variants
    
    def fuse_results(
        self,
        query_variants: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Stage 2: Multi-query retrieval + RRF fusion.
        
        Args:
            query_variants: List of query strings
            
        Returns:
            Fused ranked list: [(doc_id, rrf_score), ...]
        """
        # Retrieve for each query variant
        ranked_lists = []
        
        for query in query_variants:
            doc_ids, scores = self.retrieve(query, top_k=self.top_k)
            ranked_lists.append(doc_ids)
        
        # Fuse using RRF
        if self.rrf_fusion:
            fused = self.rrf_fusion.fuse(ranked_lists)
        else:
            # Fallback: use first query's results
            fused = [(doc_id, 1.0 / (i + 1)) for i, doc_id in enumerate(ranked_lists[0])]
        
        return fused
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Stage 3: BERT cross-encoder re-ranking.
        
        Args:
            query: Original query
            candidates: List of (doc_id, score) from fusion
            top_k: Number of candidates to re-rank
            
        Returns:
            Re-ranked list: [(doc_id, bert_score), ...]
        """
        if not self.bert_reranker:
            return candidates
        
        if top_k is None:
            top_k = self.config.get('bert_reranker', {}).get('top_k_rerank', 100)
        
        # Take top-k candidates
        candidates = candidates[:top_k]
        
        # Get document texts
        doc_ids = [doc_id for doc_id, _ in candidates]
        documents = []
        valid_doc_ids = []
        
        for doc_id in doc_ids:
            # --- PATCHED FOR PYSERINI ---
            if hasattr(self.search_engine, 'doc'):
                # Pyserini Logic
                doc_obj = self.search_engine.doc(doc_id)
                if doc_obj:
                    try:
                        import json
                        # Pyserini thường trả về JSON string, ta cần lấy trường 'contents'
                        doc_text = json.loads(doc_obj.raw()).get('contents', doc_obj.raw())
                    except:
                        doc_text = doc_obj.raw()
                else:
                    doc_text = ''
            else:
                # Legacy Logic
                # --- PATCHED FOR PYSERINI ---
                if hasattr(self.search_engine, 'doc'):
                    # Pyserini Logic
                    doc_obj = self.search_engine.doc(doc_id)
                    if doc_obj:
                        try:
                            import json
                            # Pyserini thường trả về JSON string, ta cần lấy trường 'contents'
                            doc_text = json.loads(doc_obj.raw()).get('contents', doc_obj.raw())
                        except:
                            doc_text = doc_obj.raw()
                    else:
                        doc_text = ''
                else:
                    # Legacy Logic
                    doc_text = self.search_engine.get_document(doc_id)
            if doc_text:
                documents.append(doc_text)
                valid_doc_ids.append(doc_id)
        
        if not documents:
            return candidates
        
        # Re-rank
        reranked = self.bert_reranker.rerank(query, documents, valid_doc_ids)
        
        return reranked
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        measure_latency: bool = False
    ) -> Dict:
        """
        Full end-to-end search pipeline.
        
        Args:
            query: Query string
            top_k: Number of final results
            measure_latency: Measure latency for each stage
            
        Returns:
            Dictionary with results and metrics
        """
        if top_k is None:
            top_k = self.top_k
        
        result = {
            'query': query,
            'results': [],
            'latency': {}
        }
        
        # Stage 0: Candidate Mining
        with LatencyTimer() as timer:
            candidates = self.mine_candidates(query)
        
        if measure_latency:
            result['latency']['mining'] = timer.get_elapsed()
        
        # Stage 1: Query Reformulation
        with LatencyTimer() as timer:
            query_variants = self.reformulate_query(query, candidates)
        
        if measure_latency:
            result['latency']['reformulation'] = timer.get_elapsed()
        
        result['query_variants'] = query_variants
        
        # Stage 2: Multi-Query Retrieval + RRF Fusion
        with LatencyTimer() as timer:
            fused_results = self.fuse_results(query_variants)
        
        if measure_latency:
            result['latency']['retrieval_fusion'] = timer.get_elapsed()
        
        # Stage 3: BERT Re-ranking
        with LatencyTimer() as timer:
            reranked_results = self.rerank(query, fused_results, top_k)
        
        if measure_latency:
            result['latency']['reranking'] = timer.get_elapsed()
            result['latency']['total'] = sum(result['latency'].values())
        
        # Format final results
        result['results'] = reranked_results[:top_k]
        
        return result
    def _embed_text(self, text: str) -> torch.Tensor:
        """
        Embed text using embedding model.
        
        Args:
            text: Text string
            
        Returns:
            Embedding tensor [embedding_dim]
        """
        if self.embedding_model is None:
            embedding_dim = self.config.get('rl_agent', {}).get('embedding_dim', 512)
            return torch.randn(embedding_dim)
        
        try:
            return self.embedding_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        except TypeError:
            return self.embedding_model.encode(text, convert_to_tensor=True)
       
    def load_rl_checkpoint(self, checkpoint_path: str):
        """Load RL agent from checkpoint."""
        if self.rl_agent:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.rl_agent.load_state_dict(checkpoint['model_state_dict'])
            self.rl_agent.eval()
            self.logger.info(f"Loaded RL agent from {checkpoint_path}")
    
    def enable_training_mode(self):
        """Set RL agent to training mode."""
        if self.rl_agent:
            self.rl_agent.train()
    
    def enable_eval_mode(self):
        """Set RL agent to evaluation mode."""
        if self.rl_agent:
            self.rl_agent.eval()


if __name__ == "__main__":
    # Test pipeline (requires actual components)
    print("Adaptive IR Pipeline module loaded successfully")
    print("To test the pipeline, you need:")
    print("  1. A search engine instance")
    print("  2. Trained RL agent checkpoint")
    print("  3. Configuration file")