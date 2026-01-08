"""
Reciprocal Rank Fusion (RRF)

Fuses multiple ranked lists into a single ranking using reciprocal rank scores.
Based on Cormack et al. (2009).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class RecipRankFusion:
    """
    Reciprocal Rank Fusion for merging multiple ranked lists.
    
    Formula:
        RRF(d) = Σᵢ 1/(k + rankᵢ(d))
    
    where k is a constant (typically 60) and rankᵢ(d) is the rank of 
    document d in the i-th ranked list.
    """
    
    def __init__(self, k: int = 60):
        """
        Args:
            k: Constant for RRF formula (default: 60)
        """
        self.k = k
    
    def fuse(
        self,
        ranked_lists: List[List[str]],
        scores_lists: Optional[List[List[float]]] = None
    ) -> List[Tuple[str, float]]:
        """
        Fuse multiple ranked lists using RRF.
        
        Args:
            ranked_lists: List of ranked lists, where each inner list contains doc IDs
                         [[doc1, doc2, ...], [doc3, doc1, ...], ...]
            scores_lists: Optional original scores for each document (not used in RRF)
            
        Returns:
            List of (doc_id, rrf_score) tuples, sorted by RRF score descending
        """
        if not ranked_lists:
            return []
        
        # Aggregate RRF scores
        doc_scores = defaultdict(float)
        
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list, start=1):
                rrf_score = 1.0 / (self.k + rank)
                doc_scores[doc_id] += rrf_score
        
        # Sort by RRF score descending
        sorted_results = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results
    
    def fuse_with_metadata(
        self,
        ranked_results: List[List[Dict]],
    ) -> List[Dict]:
        """
        Fuse ranked lists with metadata (doc text, etc.).
        
        Args:
            ranked_results: List of ranked results with metadata
                [
                    [{'doc_id': 'd1', 'score': 0.9, 'text': '...'},
                     {'doc_id': 'd2', 'score': 0.8, ...}],
                    [{'doc_id': 'd3', 'score': 0.95, ...}, ...]
                ]
        
        Returns:
            Fused ranked list with metadata and RRF scores
        """
        if not ranked_results:
            return []
        
        # Build doc_id -> metadata mapping
        doc_metadata = {}
        doc_scores = defaultdict(float)
        
        for ranked_list in ranked_results:
            for rank, doc_info in enumerate(ranked_list, start=1):
                doc_id = doc_info['doc_id']
                
                # Store metadata (from first occurrence)
                if doc_id not in doc_metadata:
                    doc_metadata[doc_id] = doc_info.copy()
                
                # Accumulate RRF score
                rrf_score = 1.0 / (self.k + rank)
                doc_scores[doc_id] += rrf_score
        
        # Create final results
        fused_results = []
        for doc_id, rrf_score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            result = doc_metadata[doc_id].copy()
            result['rrf_score'] = rrf_score
            fused_results.append(result)
        
        return fused_results
    
    def fuse_from_queries(
        self,
        query_results: Dict[str, List[Tuple[str, float]]],
        query_weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float]]:
        """
        Fuse results from multiple query variants.
        
        Args:
            query_results: Dictionary mapping query -> ranked list
                {
                    'query1': [('doc1', 0.9), ('doc2', 0.8), ...],
                    'query2': [('doc3', 0.95), ('doc1', 0.85), ...],
                    ...
                }
            query_weights: Optional weights for each query variant
            
        Returns:
            Fused ranked list
        """
        if not query_results:
            return []
        
        doc_scores = defaultdict(float)
        
        # Default: equal weights
        if query_weights is None:
            query_weights = {q: 1.0 for q in query_results.keys()}
        
        # Normalize weights
        total_weight = sum(query_weights.values())
        query_weights = {q: w / total_weight for q, w in query_weights.items()}
        
        for query, ranked_list in query_results.items():
            weight = query_weights.get(query, 1.0)
            
            for rank, (doc_id, score) in enumerate(ranked_list, start=1):
                rrf_score = weight / (self.k + rank)
                doc_scores[doc_id] += rrf_score
        
        # Sort by RRF score
        sorted_results = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results


class CombSUM:
    """
    Alternative fusion method: CombSUM
    
    Combines scores by summing normalized scores from each ranker.
    """
    
    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: Whether to normalize scores before combining
        """
        self.normalize = normalize
    
    def fuse(
        self,
        ranked_lists: List[List[str]],
        scores_lists: List[List[float]]
    ) -> List[Tuple[str, float]]:
        """
        Fuse using score summation.
        
        Args:
            ranked_lists: List of doc ID lists
            scores_lists: Corresponding scores
            
        Returns:
            Fused ranked list
        """
        if not ranked_lists or not scores_lists:
            return []
        
        doc_scores = defaultdict(float)
        
        for ranked_list, scores in zip(ranked_lists, scores_lists):
            # Normalize scores if requested
            if self.normalize and scores:
                max_score = max(scores) if max(scores) > 0 else 1.0
                scores = [s / max_score for s in scores]
            
            # Accumulate scores
            for doc_id, score in zip(ranked_list, scores):
                doc_scores[doc_id] += score
        
        # Sort by combined score
        sorted_results = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results


class HybridFusion:
    """
    Hybrid fusion combining RRF and score-based methods.
    """
    
    def __init__(self, rrf_weight: float = 0.7, score_weight: float = 0.3, k: int = 60):
        """
        Args:
            rrf_weight: Weight for RRF component
            score_weight: Weight for score component
            k: RRF constant
        """
        self.rrf = RecipRankFusion(k=k)
        self.combsum = CombSUM(normalize=True)
        self.rrf_weight = rrf_weight
        self.score_weight = score_weight
    
    def fuse(
        self,
        ranked_lists: List[List[str]],
        scores_lists: List[List[float]]
    ) -> List[Tuple[str, float]]:
        """Hybrid fusion of RRF and CombSUM."""
        # Get RRF scores
        rrf_results = self.rrf.fuse(ranked_lists)
        rrf_dict = dict(rrf_results)
        
        # Get CombSUM scores
        combsum_results = self.combsum.fuse(ranked_lists, scores_lists)
        combsum_dict = dict(combsum_results)
        
        # Normalize both
        if rrf_dict:
            max_rrf = max(rrf_dict.values())
            rrf_dict = {k: v / max_rrf for k, v in rrf_dict.items()}
        
        if combsum_dict:
            max_combsum = max(combsum_dict.values())
            combsum_dict = {k: v / max_combsum for k, v in combsum_dict.items()}
        
        # Combine
        all_docs = set(rrf_dict.keys()) | set(combsum_dict.keys())
        hybrid_scores = {}
        
        for doc_id in all_docs:
            rrf_score = rrf_dict.get(doc_id, 0)
            combsum_score = combsum_dict.get(doc_id, 0)
            hybrid_scores[doc_id] = (
                self.rrf_weight * rrf_score + 
                self.score_weight * combsum_score
            )
        
        # Sort
        sorted_results = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results


if __name__ == "__main__":
    # Test RRF
    print("Testing Reciprocal Rank Fusion:")
    
    # Example: 3 query variants return different rankings
    ranked_lists = [
        ['doc1', 'doc2', 'doc3', 'doc4', 'doc5'],
        ['doc3', 'doc1', 'doc5', 'doc2', 'doc6'],
        ['doc2', 'doc3', 'doc1', 'doc7', 'doc4']
    ]
    
    rrf = RecipRankFusion(k=60)
    fused = rrf.fuse(ranked_lists)
    
    print("\nFused results:")
    for i, (doc_id, score) in enumerate(fused[:10], 1):
        print(f"  {i}. {doc_id}: {score:.4f}")
    
    # Test with metadata
    print("\n\nTesting with metadata:")
    ranked_results = [
        [
            {'doc_id': 'doc1', 'score': 0.9, 'text': 'Deep learning...'},
            {'doc_id': 'doc2', 'score': 0.8, 'text': 'Neural networks...'},
        ],
        [
            {'doc_id': 'doc2', 'score': 0.95, 'text': 'Neural networks...'},
            {'doc_id': 'doc3', 'score': 0.85, 'text': 'Machine learning...'},
        ]
    ]
    
    fused_meta = rrf.fuse_with_metadata(ranked_results)
    print("Fused with metadata:")
    for i, result in enumerate(fused_meta[:5], 1):
        print(f"  {i}. {result['doc_id']}: RRF={result['rrf_score']:.4f}")
