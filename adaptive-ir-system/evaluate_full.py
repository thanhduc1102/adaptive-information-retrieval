#!/usr/bin/env python3
"""
Full Evaluation Script with All Stages

Complete evaluation của Adaptive IR pipeline với:
1. BM25 Baseline
2. BM25 + RM3 (Pseudo-Relevance Feedback) 
3. BM25 + RL Reformulation + RRF
4. Full Pipeline (+ BERT Re-ranking)

Metrics:
- Recall@K (10, 100)
- MRR@10
- nDCG@10
- MAP

Usage:
    # Full evaluation on validation set
    python evaluate_full.py --split valid --num-queries 1000
    
    # Evaluate specific stages
    python evaluate_full.py --split valid --stages baseline,rl_fusion
    
    # Fast mode (skip BERT)
    python evaluate_full.py --split valid --num-queries 500 --no-bert
"""

import os
import sys
import yaml
import argparse
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import time
import numpy as np

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

# Setup Java
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-21-openjdk-amd64'


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class RM3Expander:
    """
    Simple RM3 Pseudo-Relevance Feedback implementation.
    
    Algorithm:
    1. Get top-k documents with original query
    2. Extract top terms from these documents
    3. Expand query with weighted terms
    """
    
    def __init__(self, fb_docs: int = 10, fb_terms: int = 10, original_weight: float = 0.5):
        self.fb_docs = fb_docs
        self.fb_terms = fb_terms
        self.original_weight = original_weight
    
    def expand(self, query: str, documents: List[str]) -> str:
        """
        Expand query using RM3.
        
        Args:
            query: Original query
            documents: Feedback documents
            
        Returns:
            Expanded query
        """
        from collections import Counter
        import re
        
        # Get stopwords
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                         'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                         'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                         'can', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by',
                         'from', 'as', 'into', 'through', 'during', 'before', 'after',
                         'above', 'below', 'between', 'under', 'again', 'further',
                         'then', 'once', 'here', 'there', 'when', 'where', 'why',
                         'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                         'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                         'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or'}
        
        # Count terms in feedback documents
        term_freq = Counter()
        query_terms = set(query.lower().split())
        
        for doc in documents[:self.fb_docs]:
            words = re.findall(r'\b[a-z]{3,}\b', doc.lower())
            for word in words:
                if word not in stop_words and word not in query_terms:
                    term_freq[word] += 1
        
        # Get top expansion terms
        expansion_terms = [term for term, _ in term_freq.most_common(self.fb_terms)]
        
        # Build expanded query
        expanded = query + " " + " ".join(expansion_terms)
        
        return expanded


class FullEvaluator:
    """
    Complete evaluator for all pipeline stages.
    """
    
    def __init__(
        self,
        config: dict,
        checkpoint_path: str,
        device: str = 'cuda',
        use_bert: bool = True
    ):
        self.config = config
        self.device = device
        self.use_bert = use_bert
        
        self._init_components(checkpoint_path)
    
    def _init_components(self, checkpoint_path: str):
        """Initialize all pipeline components."""
        print("Initializing components...")
        
        # Dataset
        from src.utils.legacy_loader import LegacyDatasetAdapter
        data_dir = self.config['data']['data_dir']
        
        self.train_dataset = LegacyDatasetAdapter(
            dataset_path=os.path.join(data_dir, 'msa_dataset.hdf5'),
            corpus_path=os.path.join(data_dir, 'msa_corpus.hdf5'),
            split='train'
        )
        
        self.valid_dataset = LegacyDatasetAdapter(
            dataset_path=os.path.join(data_dir, 'msa_dataset.hdf5'),
            corpus_path=os.path.join(data_dir, 'msa_corpus.hdf5'),
            split='valid'
        )
        
        self.test_dataset = LegacyDatasetAdapter(
            dataset_path=os.path.join(data_dir, 'msa_dataset.hdf5'),
            corpus_path=os.path.join(data_dir, 'msa_corpus.hdf5'),
            split='test'
        )
        
        print(f"  Train: {len(self.train_dataset.load_queries())} queries")
        print(f"  Valid: {len(self.valid_dataset.load_queries())} queries")
        print(f"  Test: {len(self.test_dataset.load_queries())} queries")
        
        # Search engine (needs all docs)
        from src.utils.simple_searcher import SimpleBM25Searcher
        self.search_engine = SimpleBM25Searcher(
            self.train_dataset,
            k1=self.config.get('retrieval', {}).get('bm25_k1', 0.9),
            b=self.config.get('retrieval', {}).get('bm25_b', 0.4)
        )
        
        # Embedding model
        embedding_type = self.config.get('embeddings', {}).get('type', 'legacy')
        if embedding_type == 'legacy':
            from src.utils.legacy_embeddings import LegacyEmbeddingAdapter
            self.embedding_model = LegacyEmbeddingAdapter(self.config['embeddings']['path'])
        else:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(
                self.config['embeddings'].get('path', 'all-MiniLM-L6-v2')
            )
        
        # RL Agent
        from src.rl_agent import QueryReformulatorAgent
        self.rl_agent = QueryReformulatorAgent(self.config['rl_agent'])
        self.rl_agent.to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.rl_agent.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.rl_agent.load_state_dict(checkpoint)
            print(f"  Loaded checkpoint: {checkpoint_path}")
        
        self.rl_agent.eval()
        
        # Candidate Miner
        from src.candidate_mining import CandidateTermMiner
        self.candidate_miner = CandidateTermMiner(self.config.get('candidate_mining', {}))
        
        # RRF Fusion
        from src.fusion import RecipRankFusion
        self.rrf_fusion = RecipRankFusion(k=self.config.get('rrf_fusion', {}).get('k_constant', 60))
        
        # RM3 Expander
        self.rm3_expander = RM3Expander(fb_docs=10, fb_terms=10, original_weight=0.5)
        
        # BERT Re-ranker (optional)
        if self.use_bert:
            from src.reranker import BERTReranker
            self.bert_reranker = BERTReranker(self.config.get('bert_reranker', {}))
        else:
            self.bert_reranker = None
        
        print("  All components initialized!")
    
    def get_dataset(self, split: str):
        """Get dataset by split name."""
        if split == 'train':
            return self.train_dataset
        elif split == 'valid':
            return self.valid_dataset
        elif split == 'test':
            return self.test_dataset
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def _embed_text(self, text: str) -> torch.Tensor:
        """Embed text."""
        emb = self.embedding_model.encode(text, convert_to_tensor=True)
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        return emb
    
    def evaluate_bm25_baseline(
        self,
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        top_k: int = 100
    ) -> Dict:
        """
        Evaluate BM25 baseline.
        """
        from src.evaluation.metrics import IRMetrics
        
        metrics = defaultdict(list)
        latencies = []
        
        for qid, query in queries.items():
            if qid not in qrels:
                continue
            
            start = time.time()
            results = self.search_engine.search(query, k=top_k)
            latency = time.time() - start
            latencies.append(latency)
            
            retrieved = [r['doc_id'] for r in results]
            relevant_set = set(qrels[qid].keys())
            relevant_dict = qrels[qid]
            
            metrics['recall@10'].append(IRMetrics.recall_at_k(retrieved, relevant_set, 10))
            metrics['recall@100'].append(IRMetrics.recall_at_k(retrieved, relevant_set, 100))
            metrics['mrr@10'].append(IRMetrics.reciprocal_rank(retrieved[:10], relevant_set))
            metrics['map'].append(IRMetrics.average_precision(retrieved, relevant_set))
            metrics['ndcg@10'].append(IRMetrics.ndcg_at_k(retrieved, relevant_dict, 10))
        
        return {
            'name': 'BM25 Baseline',
            'metrics': {k: np.mean(v) for k, v in metrics.items()},
            'latency_mean': np.mean(latencies),
            'latency_std': np.std(latencies),
            'num_queries': len(metrics['recall@10'])
        }
    
    def evaluate_bm25_rm3(
        self,
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        top_k: int = 100
    ) -> Dict:
        """
        Evaluate BM25 + RM3 (Pseudo-Relevance Feedback).
        """
        from src.evaluation.metrics import IRMetrics
        
        metrics = defaultdict(list)
        latencies = []
        
        for qid, query in queries.items():
            if qid not in qrels:
                continue
            
            start = time.time()
            
            # Initial retrieval
            initial_results = self.search_engine.search(query, k=50)
            
            # Get feedback documents
            fb_docs = []
            for r in initial_results[:10]:
                doc_text = self.train_dataset.get_document(r['doc_id'])
                if doc_text:
                    fb_docs.append(doc_text)
            
            # Expand query
            expanded_query = self.rm3_expander.expand(query, fb_docs)
            
            # Re-retrieve with expanded query
            results = self.search_engine.search(expanded_query, k=top_k)
            
            latency = time.time() - start
            latencies.append(latency)
            
            retrieved = [r['doc_id'] for r in results]
            relevant_set = set(qrels[qid].keys())
            relevant_dict = qrels[qid]
            
            metrics['recall@10'].append(IRMetrics.recall_at_k(retrieved, relevant_set, 10))
            metrics['recall@100'].append(IRMetrics.recall_at_k(retrieved, relevant_set, 100))
            metrics['mrr@10'].append(IRMetrics.reciprocal_rank(retrieved[:10], relevant_set))
            metrics['map'].append(IRMetrics.average_precision(retrieved, relevant_set))
            metrics['ndcg@10'].append(IRMetrics.ndcg_at_k(retrieved, relevant_dict, 10))
        
        return {
            'name': 'BM25 + RM3',
            'metrics': {k: np.mean(v) for k, v in metrics.items()},
            'latency_mean': np.mean(latencies),
            'latency_std': np.std(latencies),
            'num_queries': len(metrics['recall@10'])
        }
    
    def evaluate_rl_fusion(
        self,
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        top_k: int = 100,
        num_variants: int = 4
    ) -> Dict:
        """
        Evaluate BM25 + RL Reformulation + RRF Fusion.
        """
        from src.evaluation.metrics import IRMetrics
        
        metrics = defaultdict(list)
        latencies = []
        
        for qid, query in queries.items():
            if qid not in qrels:
                continue
            
            start = time.time()
            
            # Stage 0: Candidate mining
            initial_results = self.search_engine.search(query, k=50)
            documents = []
            doc_scores = []
            for r in initial_results:
                doc_text = self.train_dataset.get_document(r['doc_id'])
                if doc_text:
                    documents.append(doc_text)
                    doc_scores.append(r['score'])
            
            candidates = self.candidate_miner.extract_candidates(query, documents, doc_scores)
            
            # Stage 1: RL Reformulation
            query_variants = self._generate_variants(query, candidates, num_variants)
            
            # Stage 2: Multi-query retrieval + RRF
            ranked_lists = []
            for variant in query_variants:
                results = self.search_engine.search(variant, k=top_k)
                doc_ids = [r['doc_id'] for r in results]
                ranked_lists.append(doc_ids)
            
            fused = self.rrf_fusion.fuse(ranked_lists)
            
            latency = time.time() - start
            latencies.append(latency)
            
            retrieved = [d[0] for d in fused[:top_k]]
            relevant_set = set(qrels[qid].keys())
            relevant_dict = qrels[qid]
            
            metrics['recall@10'].append(IRMetrics.recall_at_k(retrieved, relevant_set, 10))
            metrics['recall@100'].append(IRMetrics.recall_at_k(retrieved, relevant_set, 100))
            metrics['mrr@10'].append(IRMetrics.reciprocal_rank(retrieved[:10], relevant_set))
            metrics['map'].append(IRMetrics.average_precision(retrieved, relevant_set))
            metrics['ndcg@10'].append(IRMetrics.ndcg_at_k(retrieved, relevant_dict, 10))
        
        return {
            'name': f'RL Reformulation + RRF (m={num_variants})',
            'metrics': {k: np.mean(v) for k, v in metrics.items()},
            'latency_mean': np.mean(latencies),
            'latency_std': np.std(latencies),
            'num_queries': len(metrics['recall@10'])
        }
    
    def evaluate_full_pipeline(
        self,
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        top_k: int = 100,
        num_variants: int = 4,
        top_k_rerank: int = 50
    ) -> Dict:
        """
        Evaluate Full Pipeline (with BERT re-ranking).
        """
        from src.evaluation.metrics import IRMetrics
        
        if self.bert_reranker is None:
            print("  Warning: BERT re-ranker not available, skipping full pipeline")
            return None
        
        metrics = defaultdict(list)
        latencies = []
        
        for qid, query in queries.items():
            if qid not in qrels:
                continue
            
            start = time.time()
            
            # Stages 0-2: Same as RL Fusion
            initial_results = self.search_engine.search(query, k=50)
            documents = []
            doc_scores = []
            for r in initial_results:
                doc_text = self.train_dataset.get_document(r['doc_id'])
                if doc_text:
                    documents.append(doc_text)
                    doc_scores.append(r['score'])
            
            candidates = self.candidate_miner.extract_candidates(query, documents, doc_scores)
            query_variants = self._generate_variants(query, candidates, num_variants)
            
            ranked_lists = []
            for variant in query_variants:
                results = self.search_engine.search(variant, k=top_k)
                doc_ids = [r['doc_id'] for r in results]
                ranked_lists.append(doc_ids)
            
            fused = self.rrf_fusion.fuse(ranked_lists)
            
            # Stage 3: BERT Re-ranking
            top_candidates = fused[:top_k_rerank]
            doc_ids = [d[0] for d in top_candidates]
            doc_texts = []
            valid_doc_ids = []
            
            for doc_id in doc_ids:
                doc_text = self.train_dataset.get_document(doc_id)
                if doc_text:
                    doc_texts.append(doc_text)
                    valid_doc_ids.append(doc_id)
            
            if doc_texts:
                reranked = self.bert_reranker.rerank(query, doc_texts, valid_doc_ids)
                retrieved = [d[0] for d in reranked[:top_k]]
            else:
                retrieved = [d[0] for d in fused[:top_k]]
            
            latency = time.time() - start
            latencies.append(latency)
            
            relevant_set = set(qrels[qid].keys())
            relevant_dict = qrels[qid]
            
            metrics['recall@10'].append(IRMetrics.recall_at_k(retrieved, relevant_set, 10))
            metrics['recall@100'].append(IRMetrics.recall_at_k(retrieved, relevant_set, 100))
            metrics['mrr@10'].append(IRMetrics.reciprocal_rank(retrieved[:10], relevant_set))
            metrics['map'].append(IRMetrics.average_precision(retrieved, relevant_set))
            metrics['ndcg@10'].append(IRMetrics.ndcg_at_k(retrieved, relevant_dict, 10))
        
        return {
            'name': f'Full Pipeline (RL + RRF + BERT)',
            'metrics': {k: np.mean(v) for k, v in metrics.items()},
            'latency_mean': np.mean(latencies),
            'latency_std': np.std(latencies),
            'num_queries': len(metrics['recall@10'])
        }
    
    def _generate_variants(
        self,
        query: str,
        candidates: Dict,
        num_variants: int
    ) -> List[str]:
        """Generate query variants using RL agent."""
        if not candidates:
            return [query]
        
        # Prepare embeddings
        query_emb = self._embed_text(query)
        candidate_terms = list(candidates.keys())[:50]
        
        if not candidate_terms:
            return [query]
        
        candidate_embs = torch.stack([self._embed_text(term) for term in candidate_terms])
        candidate_features = self.candidate_miner.get_candidate_features(
            {k: candidates[k] for k in candidate_terms}
        )
        candidate_features = torch.tensor(candidate_features, dtype=torch.float32)
        
        # Move to device
        query_emb = query_emb.unsqueeze(0).to(self.device)
        candidate_embs = candidate_embs.unsqueeze(0).to(self.device)
        candidate_features = candidate_features.unsqueeze(0).to(self.device)
        
        # Generate variants
        query_variants = [query]
        
        with torch.no_grad():
            q0_enc, cand_enc = self.rl_agent.encode_static(
                query_emb, candidate_embs, candidate_features
            )
            
            for _ in range(num_variants - 1):
                current_query = query
                selected_terms = []
                
                for step in range(3):
                    current_emb = self._embed_text(current_query).unsqueeze(0).to(self.device)
                    
                    action, _, _ = self.rl_agent.select_action(
                        query_emb, current_emb, candidate_embs, candidate_features,
                        deterministic=False,
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
                    query_variants.append(current_query)
        
        return query_variants
    
    def run_full_evaluation(
        self,
        split: str = 'valid',
        num_queries: Optional[int] = None,
        stages: List[str] = None
    ) -> Dict:
        """
        Run complete evaluation.
        
        Args:
            split: Dataset split
            num_queries: Number of queries (None = all)
            stages: Which stages to evaluate
            
        Returns:
            Evaluation results
        """
        from tqdm import tqdm
        
        # Get data
        dataset = self.get_dataset(split)
        queries = dataset.load_queries()
        qrels = dataset.load_qrels()
        
        # Limit queries if specified
        if num_queries:
            query_ids = list(queries.keys())[:num_queries]
            queries = {qid: queries[qid] for qid in query_ids}
            qrels = {qid: qrels[qid] for qid in query_ids if qid in qrels}
        
        print(f"\nEvaluating on {len(queries)} queries from {split} split")
        
        # Default stages
        if stages is None:
            stages = ['baseline', 'rm3', 'rl_fusion']
            if self.use_bert:
                stages.append('full_pipeline')
        
        results = {
            'split': split,
            'num_queries': len(queries),
            'timestamp': datetime.now().isoformat(),
            'stages': {}
        }
        
        # Run evaluations
        if 'baseline' in stages:
            print("\n📊 Evaluating BM25 Baseline...")
            results['stages']['baseline'] = self.evaluate_bm25_baseline(queries, qrels)
        
        if 'rm3' in stages:
            print("\n📊 Evaluating BM25 + RM3...")
            results['stages']['rm3'] = self.evaluate_bm25_rm3(queries, qrels)
        
        if 'rl_fusion' in stages:
            print("\n📊 Evaluating RL + RRF Fusion...")
            results['stages']['rl_fusion'] = self.evaluate_rl_fusion(queries, qrels)
        
        if 'full_pipeline' in stages and self.use_bert:
            print("\n📊 Evaluating Full Pipeline (with BERT)...")
            results['stages']['full_pipeline'] = self.evaluate_full_pipeline(queries, qrels)
        
        return results


def print_results_table(results: Dict):
    """Print results in table format."""
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS")
    print("=" * 80)
    print(f"Split: {results['split']}, Queries: {results['num_queries']}")
    print("-" * 80)
    
    # Header
    print(f"{'Method':<30} {'R@10':<10} {'R@100':<10} {'MRR@10':<10} {'MAP':<10} {'Latency':<12}")
    print("-" * 80)
    
    # Rows
    for stage_name, stage_results in results['stages'].items():
        if stage_results is None:
            continue
        
        m = stage_results['metrics']
        lat = stage_results['latency_mean'] * 1000  # ms
        
        print(f"{stage_results['name']:<30} "
              f"{m['recall@10']:<10.4f} "
              f"{m['recall@100']:<10.4f} "
              f"{m['mrr@10']:<10.4f} "
              f"{m['map']:<10.4f} "
              f"{lat:<10.1f}ms")
    
    print("-" * 80)
    
    # Improvement analysis
    if 'baseline' in results['stages'] and 'rl_fusion' in results['stages']:
        base = results['stages']['baseline']['metrics']
        rl = results['stages']['rl_fusion']['metrics']
        
        print("\n📈 Improvement over Baseline:")
        for metric in ['recall@10', 'recall@100', 'mrr@10']:
            delta = rl[metric] - base[metric]
            pct = (delta / base[metric] * 100) if base[metric] > 0 else 0
            sign = "+" if delta >= 0 else ""
            print(f"   {metric}: {sign}{delta:.4f} ({sign}{pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Full Pipeline Evaluation')
    parser.add_argument('--config', type=str, default='configs/msa_quick_config.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_msa_optimized/best_model.pt')
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid', 'test'])
    parser.add_argument('--num-queries', type=int, default=None)
    parser.add_argument('--stages', type=str, default=None,
                        help='Comma-separated stages: baseline,rm3,rl_fusion,full_pipeline')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--no-bert', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("🔬 Adaptive IR - Full Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = FullEvaluator(
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
        use_bert=not args.no_bert
    )
    
    # Parse stages
    stages = args.stages.split(',') if args.stages else None
    
    # Run evaluation
    results = evaluator.run_full_evaluation(
        split=args.split,
        num_queries=args.num_queries,
        stages=stages
    )
    
    # Print results
    print_results_table(results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {output_path}")


if __name__ == "__main__":
    main()
