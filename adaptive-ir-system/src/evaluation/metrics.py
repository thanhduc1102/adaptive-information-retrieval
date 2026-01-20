"""
Evaluation Metrics for Information Retrieval

Implements standard IR metrics:
- Recall@K
- MRR@K (Mean Reciprocal Rank)
- nDCG@K (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- Precision@K
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import time


class IRMetrics:
    """
    Information Retrieval evaluation metrics.
    """
    
    @staticmethod
    def recall_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Recall@K: Fraction of relevant docs in top-K.
        
        Args:
            retrieved: List of retrieved doc IDs (ranked)
            relevant: Set of relevant doc IDs
            k: Cutoff position
            
        Returns:
            Recall@K score
        """
        if not relevant:
            return 0.0
        
        retrieved_at_k = set(retrieved[:k])
        relevant_retrieved = retrieved_at_k & relevant
        
        return len(relevant_retrieved) / len(relevant)
    
    @staticmethod
    def precision_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Precision@K: Fraction of relevant docs among top-K retrieved.
        """
        if k == 0:
            return 0.0
        
        retrieved_at_k = set(retrieved[:k])
        relevant_retrieved = retrieved_at_k & relevant
        
        return len(relevant_retrieved) / k
    
    @staticmethod
    def average_precision(
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """
        Average Precision for a single query.
        
        AP = (1/|relevant|) * Σ P(k) * rel(k)
        where P(k) is precision at k, rel(k) is 1 if doc at k is relevant
        """
        if not relevant:
            return 0.0
        
        score = 0.0
        num_relevant = 0
        
        for k, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                num_relevant += 1
                precision_at_k = num_relevant / k
                score += precision_at_k
        
        return score / len(relevant) if relevant else 0.0
    
    @staticmethod
    def reciprocal_rank(
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """
        Reciprocal Rank: 1/rank of first relevant document.
        
        RR = 1/k where k is position of first relevant doc (1-indexed)
        """
        for k, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / k
        return 0.0
    
    @staticmethod
    def dcg_at_k(
        retrieved: List[str],
        relevant: Dict[str, int],
        k: int
    ) -> float:
        """
        Discounted Cumulative Gain at K.
        
        DCG@K = Σ (2^rel(i) - 1) / log2(i + 1)
        
        Args:
            retrieved: List of retrieved doc IDs
            relevant: Dict mapping doc_id -> relevance grade
            k: Cutoff
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], 1):
            rel = relevant.get(doc_id, 0)
            dcg += (2 ** rel - 1) / np.log2(i + 1)
        return dcg
    
    @staticmethod
    def ndcg_at_k(
        retrieved: List[str],
        relevant: Dict[str, int],
        k: int
    ) -> float:
        """
        Normalized Discounted Cumulative Gain at K.
        
        nDCG@K = DCG@K / IDCG@K
        where IDCG is ideal DCG (if docs were sorted by relevance)
        """
        # Compute DCG
        dcg = IRMetrics.dcg_at_k(retrieved, relevant, k)
        
        # Compute IDCG (ideal DCG)
        ideal_retrieved = sorted(relevant.keys(), key=lambda x: relevant[x], reverse=True)
        idcg = IRMetrics.dcg_at_k(ideal_retrieved, relevant, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def f1_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """F1 score at K."""
        precision = IRMetrics.precision_at_k(retrieved, relevant, k)
        recall = IRMetrics.recall_at_k(retrieved, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)


class IRMetricsAggregator:
    """
    Aggregates metrics across multiple queries.
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.query_metrics = {}
        self.latencies = []
    
    def add_query_result(
        self,
        query_id: str,
        retrieved: List[str],
        relevant: Set[str],
        relevant_grades: Optional[Dict[str, int]] = None,
        latency: Optional[float] = None
    ):
        """
        Add results for a single query.
        
        Args:
            query_id: Query identifier
            retrieved: Retrieved document IDs (ranked)
            relevant: Set of relevant document IDs
            relevant_grades: Optional relevance grades for nDCG
            latency: Optional query latency in seconds
        """
        # Compute metrics
        query_metrics = {
            'recall@10': IRMetrics.recall_at_k(retrieved, relevant, 10),
            'recall@100': IRMetrics.recall_at_k(retrieved, relevant, 100),
            'precision@10': IRMetrics.precision_at_k(retrieved, relevant, 10),
            'mrr': IRMetrics.reciprocal_rank(retrieved, relevant),
            'map': IRMetrics.average_precision(retrieved, relevant),
            'f1@10': IRMetrics.f1_at_k(retrieved, relevant, 10),
        }
        
        # Add nDCG if grades provided
        if relevant_grades:
            query_metrics['ndcg@10'] = IRMetrics.ndcg_at_k(retrieved, relevant_grades, 10)
            query_metrics['ndcg@100'] = IRMetrics.ndcg_at_k(retrieved, relevant_grades, 100)
        
        # Store query-level metrics
        self.query_metrics[query_id] = query_metrics
        
        # Aggregate
        for metric, value in query_metrics.items():
            self.metrics[metric].append(value)
        
        # Store latency
        if latency is not None:
            self.latencies.append(latency)
    
    def compute_aggregate(self) -> Dict[str, float]:
        """
        Compute aggregated metrics across all queries.
        
        Returns:
            Dictionary of metric_name -> mean_value
        """
        aggregated = {}
        
        for metric, values in self.metrics.items():
            if values:
                aggregated[metric] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
        
        # Add latency stats
        if self.latencies:
            aggregated['latency_mean'] = np.mean(self.latencies)
            aggregated['latency_std'] = np.std(self.latencies)
            aggregated['latency_p50'] = np.percentile(self.latencies, 50)
            aggregated['latency_p95'] = np.percentile(self.latencies, 95)
            aggregated['latency_p99'] = np.percentile(self.latencies, 99)
        
        return aggregated
    
    def get_query_metrics(self, query_id: str) -> Optional[Dict[str, float]]:
        """Get metrics for a specific query."""
        return self.query_metrics.get(query_id)
    
    def get_all_query_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get all query-level metrics."""
        return self.query_metrics

    def compute_single_query(
        self,
        relevant: Dict[str, int],
        retrieved: List[str],
        query_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compute metrics for a single query without adding to aggregator.

        Args:
            relevant: Dict mapping doc_id -> relevance grade (or set of relevant doc_ids)
            retrieved: List of retrieved document IDs (ranked)
            query_id: Optional query identifier

        Returns:
            Dictionary of metrics for this query
        """
        # Convert to set if dict provided
        if isinstance(relevant, dict):
            relevant_set = set(relevant.keys())
            relevant_grades = relevant
        else:
            relevant_set = set(relevant)
            relevant_grades = {doc_id: 1 for doc_id in relevant_set}

        # Compute metrics
        metrics = {
            'recall@10': IRMetrics.recall_at_k(retrieved, relevant_set, 10),
            'recall@100': IRMetrics.recall_at_k(retrieved, relevant_set, 100),
            'precision@10': IRMetrics.precision_at_k(retrieved, relevant_set, 10),
            'mrr': IRMetrics.reciprocal_rank(retrieved, relevant_set),
            'map': IRMetrics.average_precision(retrieved, relevant_set),
            'f1@10': IRMetrics.f1_at_k(retrieved, relevant_set, 10),
        }

        # Add nDCG
        if relevant_grades:
            metrics['ndcg@10'] = IRMetrics.ndcg_at_k(retrieved, relevant_grades, 10)
            metrics['ndcg@100'] = IRMetrics.ndcg_at_k(retrieved, relevant_grades, 100)

        return metrics

    def print_summary(self):
        """Print summary of metrics."""
        aggregated = self.compute_aggregate()
        
        print("=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Number of queries: {len(self.query_metrics)}")
        print()
        
        # Recall metrics
        print("Recall Metrics:")
        if 'recall@10' in aggregated:
            print(f"  Recall@10:  {aggregated['recall@10']:.4f} ± {aggregated.get('recall@10_std', 0):.4f}")
        if 'recall@100' in aggregated:
            print(f"  Recall@100: {aggregated['recall@100']:.4f} ± {aggregated.get('recall@100_std', 0):.4f}")
        print()
        
        # Ranking metrics
        print("Ranking Metrics:")
        if 'mrr' in aggregated:
            print(f"  MRR@10:     {aggregated['mrr']:.4f} ± {aggregated.get('mrr_std', 0):.4f}")
        if 'map' in aggregated:
            print(f"  MAP:        {aggregated['map']:.4f} ± {aggregated.get('map_std', 0):.4f}")
        if 'ndcg@10' in aggregated:
            print(f"  nDCG@10:    {aggregated['ndcg@10']:.4f} ± {aggregated.get('ndcg@10_std', 0):.4f}")
        print()
        
        # Precision metrics
        print("Precision Metrics:")
        if 'precision@10' in aggregated:
            print(f"  P@10:       {aggregated['precision@10']:.4f} ± {aggregated.get('precision@10_std', 0):.4f}")
        if 'f1@10' in aggregated:
            print(f"  F1@10:      {aggregated['f1@10']:.4f} ± {aggregated.get('f1@10_std', 0):.4f}")
        print()
        
        # Latency metrics
        if self.latencies:
            print("Latency Metrics (ms):")
            print(f"  Mean:       {aggregated['latency_mean']*1000:.2f}")
            print(f"  Std:        {aggregated['latency_std']*1000:.2f}")
            print(f"  P50:        {aggregated['latency_p50']*1000:.2f}")
            print(f"  P95:        {aggregated['latency_p95']*1000:.2f}")
            print(f"  P99:        {aggregated['latency_p99']*1000:.2f}")
            print()
        
        print("=" * 60)


class LatencyTimer:
    """Context manager for measuring latency."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
    
    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed if self.elapsed is not None else 0.0


if __name__ == "__main__":
    # Test metrics
    print("Testing IR Metrics:")
    
    # Example: 2 queries
    aggregator = IRMetricsAggregator()
    
    # Query 1
    retrieved1 = ['doc1', 'doc3', 'doc5', 'doc2', 'doc7']
    relevant1 = {'doc1', 'doc2', 'doc4'}
    
    aggregator.add_query_result('q1', retrieved1, relevant1, latency=0.05)
    
    # Query 2
    retrieved2 = ['doc2', 'doc4', 'doc1', 'doc8', 'doc9']
    relevant2 = {'doc1', 'doc2', 'doc3', 'doc4'}
    
    aggregator.add_query_result('q2', retrieved2, relevant2, latency=0.06)
    
    # Print summary
    aggregator.print_summary()
    
    # Test individual metrics
    print("\nIndividual Metric Tests:")
    print(f"Recall@3 (q1): {IRMetrics.recall_at_k(retrieved1, relevant1, 3):.4f}")
    print(f"MRR (q1): {IRMetrics.reciprocal_rank(retrieved1, relevant1):.4f}")
    print(f"MAP (q1): {IRMetrics.average_precision(retrieved1, relevant1):.4f}")
