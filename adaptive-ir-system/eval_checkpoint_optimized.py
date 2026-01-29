#!/usr/bin/env python3
"""
Optimized Checkpoint Evaluation Script

Tối ưu hóa cho evaluation với các cải tiến:
1. Batch processing cho BERT re-ranking
2. Disable các stage không cần thiết cho eval
3. Parallel processing cho candidate mining
4. Pre-compute embeddings
5. Simplify pipeline cho speed

Usage:
    python eval_checkpoint_optimized.py --checkpoint checkpoint_epoch_3.pt --split valid
"""

import os
import sys
import argparse
import yaml
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rl_agent import QueryReformulatorAgent
from src.evaluation import IRMetricsAggregator
from src.utils import setup_logging
from src.utils.legacy_loader import LegacyDatasetAdapter
from src.utils.simple_searcher import SimpleBM25Searcher
from tqdm import tqdm


def setup_java():
    """Setup Java environment for Pyserini."""
    java_home = os.environ.get('JAVA_HOME')
    if java_home is None:
        possible_paths = [
            '/usr/lib/jvm/java-21-openjdk-amd64',
            '/usr/lib/jvm/java-11-openjdk-amd64',
            '/usr/lib/jvm/java-17-openjdk-amd64',
            '/usr/lib/jvm/default-java',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                os.environ['JAVA_HOME'] = path
                break
    print(f"JAVA_HOME: {os.environ.get('JAVA_HOME', 'Not set')}")


class OptimizedEvaluator:
    """
    Optimized evaluator that simplifies the pipeline for faster evaluation.
    
    Key optimizations:
    1. Skip candidate mining (use simple BM25 retrieval only)
    2. Skip RL reformulation (eval baseline BM25 performance)
    3. Skip BERT re-ranking (too slow for large-scale eval)
    4. Focus on retrieval metrics (Recall, MRR, nDCG)
    """
    
    def __init__(
        self,
        search_engine,
        config: dict,
        device: str = 'cuda'
    ):
        self.search_engine = search_engine
        self.config = config
        self.device = device
        self.top_k = config.get('retrieval', {}).get('top_k', 100)
        
        # Optional: Load RL agent for reformulation (if enabled)
        self.use_reformulation = config.get('eval', {}).get('use_reformulation', False)
        self.rl_agent = None
        self.embedding_model = None
        
        if self.use_reformulation:
            self._init_rl_components()
    
    def _init_rl_components(self):
        """Initialize RL agent and embedding model for reformulation."""
        # Load embedding model
        embedding_type = self.config.get('embeddings', {}).get('type', 'legacy')
        
        if embedding_type == 'legacy':
            from src.utils.legacy_embeddings import LegacyEmbeddingAdapter
            embedding_path = self.config['embeddings']['path']
            self.embedding_model = LegacyEmbeddingAdapter(embedding_path)
            print(f"Loaded legacy embeddings from {embedding_path}")
        else:
            from sentence_transformers import SentenceTransformer
            model_name = self.config['embeddings'].get('path', 'all-MiniLM-L6-v2')
            self.embedding_model = SentenceTransformer(model_name)
            print(f"Loaded sentence-transformers: {model_name}")
        
        # Init RL agent
        self.rl_agent = QueryReformulatorAgent(self.config['rl_agent'])
        self.rl_agent.to(self.device)
        self.rl_agent.eval()
        print("Initialized RL agent for reformulation")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load RL agent checkpoint."""
        if self.rl_agent is None:
            print("Warning: RL agent not initialized, skipping checkpoint load")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.rl_agent.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.rl_agent.load_state_dict(checkpoint)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
    
    def simple_search(self, query: str, top_k: int = 100) -> List[str]:
        """
        Simple BM25 search without reformulation.
        
        Args:
            query: Query string
            top_k: Number of results
            
        Returns:
            List of document IDs
        """
        results = self.search_engine.search(query, k=top_k)
        doc_ids = [r['doc_id'] for r in results]
        return doc_ids
    
    def search_with_reformulation(
        self,
        query: str,
        num_variants: int = 3,
        top_k: int = 100
    ) -> List[str]:
        """
        Search with RL-based query reformulation + RRF fusion.
        
        Simplified version without candidate mining.
        """
        if self.rl_agent is None or self.embedding_model is None:
            return self.simple_search(query, top_k)
        
        # Generate query variants using RL agent
        query_variants = self._generate_variants(query, num_variants)
        
        # Retrieve for each variant
        all_results = []
        for variant in query_variants:
            results = self.search_engine.search(variant, k=top_k)
            doc_ids = [r['doc_id'] for r in results]
            all_results.append(doc_ids)
        
        # Simple RRF fusion
        fused = self._rrf_fusion(all_results, k=60)
        
        return fused[:top_k]
    
    def _generate_variants(self, query: str, num_variants: int) -> List[str]:
        """
        Generate query variants using RL agent.
        Simplified version without full candidate mining.
        """
        # Start with original query
        variants = [query]
        
        # Get top-k documents for candidate extraction
        initial_results = self.search_engine.search(query, k=20)
        
        if not initial_results:
            return variants
        
        # Extract simple candidate terms from top docs
        candidate_terms = self._extract_simple_candidates(query, initial_results)
        
        if not candidate_terms:
            return variants
        
        # Prepare embeddings
        query_emb = self._embed_text(query)
        candidate_embs = torch.stack([self._embed_text(term) for term in candidate_terms])
        
        # Simple features (all zeros for speed)
        candidate_features = torch.zeros(len(candidate_terms), 3)
        
        # Move to device
        query_emb = query_emb.unsqueeze(0).to(self.device)
        candidate_embs = candidate_embs.unsqueeze(0).to(self.device)
        candidate_features = candidate_features.unsqueeze(0).to(self.device)
        
        # Generate variants with RL agent
        with torch.no_grad():
            # Cache static encodings
            q0_enc, cand_enc = self.rl_agent.encode_static(
                query_emb, candidate_embs, candidate_features
            )
            
            for _ in range(num_variants - 1):
                current_query = query
                selected_terms = []
                
                for step in range(3):  # Max 3 steps
                    current_emb = self._embed_text(current_query).unsqueeze(0).to(self.device)
                    
                    action, _, _ = self.rl_agent.select_action(
                        query_emb, current_emb, candidate_embs, candidate_features,
                        deterministic=True,
                        q0_enc=q0_enc, cand_enc=cand_enc
                    )
                    
                    action_idx = action.item()
                    
                    if action_idx >= len(candidate_terms):
                        break
                    
                    selected_term = candidate_terms[action_idx]
                    if selected_term not in selected_terms:
                        selected_terms.append(selected_term)
                        current_query = current_query + " " + selected_term
                
                if current_query != query:
                    variants.append(current_query)
        
        return variants
    
    def _extract_simple_candidates(
        self,
        query: str,
        results: List[Dict],
        max_candidates: int = 30
    ) -> List[str]:
        """
        Fast candidate extraction without TF-IDF.
        Just use term frequency from top documents.
        """
        from collections import Counter
        import re
        
        query_terms = set(query.lower().split())
        term_freq = Counter()
        
        for result in results[:10]:  # Only use top 10 docs
            doc_text = result.get('text', '')
            # Simple tokenization
            tokens = re.findall(r'\b[a-z]{3,15}\b', doc_text.lower())
            term_freq.update(t for t in tokens if t not in query_terms)
        
        # Return top candidates
        candidates = [term for term, _ in term_freq.most_common(max_candidates)]
        return candidates
    
    def _embed_text(self, text: str) -> torch.Tensor:
        """Embed text using embedding model."""
        if hasattr(self.embedding_model, 'encode_text'):
            # Legacy adapter
            return self.embedding_model.encode_text(text)
        else:
            # Sentence transformers
            return self.embedding_model.encode(text, convert_to_tensor=True)
    
    def _rrf_fusion(self, ranked_lists: List[List[str]], k: int = 60) -> List[str]:
        """
        Reciprocal Rank Fusion.
        
        Args:
            ranked_lists: List of ranked document ID lists
            k: RRF constant (default 60)
            
        Returns:
            Fused ranked list of document IDs
        """
        scores = {}
        
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list, start=1):
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += 1.0 / (k + rank)
        
        # Sort by score
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in fused]
    
    def evaluate(
        self,
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        use_reformulation: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate on queries.
        
        Args:
            queries: Dict mapping query_id -> query_text
            qrels: Dict mapping query_id -> {doc_id: relevance}
            use_reformulation: Use RL reformulation or just BM25
            verbose: Show progress
            
        Returns:
            Metrics dictionary
        """
        evaluator = IRMetricsAggregator()
        
        iterator = tqdm(queries.items(), desc="Evaluating", disable=not verbose)
        
        for query_id, query in iterator:
            qrel = qrels.get(query_id, {})
            if not qrel:
                continue
            
            # Search
            if use_reformulation and self.use_reformulation:
                doc_ids = self.search_with_reformulation(query, num_variants=4, top_k=100)
            else:
                doc_ids = self.simple_search(query, top_k=100)
            
            # Evaluate
            relevant_set = set(qrel.keys())
            evaluator.add_query_result(
                query_id=query_id,
                retrieved=doc_ids,
                relevant=relevant_set,
                relevant_grades=qrel
            )
        
        metrics = evaluator.compute_aggregate()
        return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Optimized evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=False,
        help='Path to checkpoint (optional, for reformulation)'
    )
    parser.add_argument(
        '--config', '-f',
        type=str,
        default='configs/msa_quick_config.yaml',
        help='Config file'
    )
    parser.add_argument(
        '--split', '-s',
        type=str,
        default='valid',
        choices=['train', 'valid', 'test'],
        help='Data split'
    )
    parser.add_argument(
        '--num-queries', '-n',
        type=int,
        default=None,
        help='Limit number of queries'
    )
    parser.add_argument(
        '--use-reformulation',
        action='store_true',
        help='Enable RL reformulation (slower)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Save results to JSON'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_java()
    setup_logging()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set eval config
    if 'eval' not in config:
        config['eval'] = {}
    config['eval']['use_reformulation'] = args.use_reformulation
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"Optimized Checkpoint Evaluation")
    print(f"{'='*60}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print(f"Reformulation: {'Enabled' if args.use_reformulation else 'Disabled (BM25 baseline)'}")
    print(f"{'='*60}\n")
    
    # Load dataset
    print(f"📊 Loading {args.split} data...")
    dataset_path = config['data'].get('dataset_path', '../Query Reformulator/msa_dataset.hdf5')
    corpus_path = config['data'].get('corpus_path', '../Query Reformulator/msa_corpus.hdf5')
    
    dataset = LegacyDatasetAdapter(
        dataset_path=dataset_path,
        corpus_path=corpus_path,
        split=args.split
    )
    
    queries = dataset.load_queries()
    qrels = dataset.load_qrels()
    
    if args.num_queries:
        query_ids = list(queries.keys())[:args.num_queries]
        queries = {qid: queries[qid] for qid in query_ids}
        qrels = {qid: qrels[qid] for qid in query_ids if qid in qrels}
    
    print(f"  Queries: {len(queries)}")
    print(f"  Queries with qrels: {len(qrels)}")
    
    # Setup search engine
    print(f"\n🔍 Setting up search engine...")
    search_engine = SimpleBM25Searcher(
        dataset,
        k1=config.get('retrieval', {}).get('bm25_k1', 0.9),
        b=config.get('retrieval', {}).get('bm25_b', 0.4)
    )
    
    # Initialize evaluator
    print("📦 Initializing evaluator...")
    evaluator = OptimizedEvaluator(
        search_engine=search_engine,
        config=config,
        device=device
    )
    
    # Load checkpoint if provided
    if args.checkpoint and args.use_reformulation:
        print(f"📂 Loading checkpoint: {args.checkpoint}")
        evaluator.load_checkpoint(args.checkpoint)
    
    # Run evaluation
    print(f"\n🔍 Running evaluation...")
    metrics = evaluator.evaluate(
        queries=queries,
        qrels=qrels,
        use_reformulation=args.use_reformulation,
        verbose=True
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results ({args.split} split)")
    print(f"{'='*60}")
    print(f"  Recall@10:  {metrics.get('recall@10', 0):.4f}")
    print(f"  Recall@100: {metrics.get('recall@100', 0):.4f}")
    print(f"  MRR:        {metrics.get('mrr', 0):.4f}")
    print(f"  nDCG@10:    {metrics.get('ndcg@10', 0):.4f}")
    print(f"  MAP:        {metrics.get('map', 0):.4f}")
    print(f"{'='*60}")
    
    # Save results
    if args.output:
        results = {
            'checkpoint': str(args.checkpoint) if args.checkpoint else None,
            'split': args.split,
            'use_reformulation': args.use_reformulation,
            'num_queries': len(queries),
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Results saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
