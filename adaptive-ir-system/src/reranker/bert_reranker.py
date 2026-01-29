"""
BERT Cross-Encoder Re-ranker

Uses pre-trained cross-encoder models for passage re-ranking.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from sentence_transformers import CrossEncoder
import numpy as np
from tqdm import tqdm


class BERTReranker:
    """
    BERT Cross-Encoder for re-ranking passages.
    
    Uses sentence-transformers CrossEncoder models fine-tuned on MS MARCO.
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config.get('model_name', 'cross-encoder/ms-marco-MiniLM-L-12-v2')
        self.max_length = config.get('max_length', 512)
        self.batch_size = config.get('batch_size', 128)
        self.use_fp16 = config.get('use_fp16', True)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading BERT re-ranker: {self.model_name}")
        self.model = CrossEncoder(
            self.model_name,
            max_length=self.max_length,
            device=self.device
        )
        
        # Enable FP16 if requested and available
        if self.use_fp16 and self.device == 'cuda':
            self.model.model.half()
            print("Enabled FP16 inference")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents for a query.
        
        Args:
            query: Query string
            documents: List of document texts
            doc_ids: Optional document IDs (if not provided, uses indices)
            top_k: Number of top results to return (None = all)
            
        Returns:
            List of (doc_id, score) tuples sorted by relevance score
        """
        if not documents:
            return []
        
        # Use indices as doc_ids if not provided
        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(documents))]
        
        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]
        
        # Predict relevance scores
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            convert_to_numpy=True
        )
        
        # Combine with doc_ids and sort
        results = list(zip(doc_ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k if specified
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def rerank_batch(
        self,
        queries: List[str],
        documents_list: List[List[str]],
        doc_ids_list: Optional[List[List[str]]] = None,
        top_k: Optional[int] = None,
        show_progress: bool = True
    ) -> List[List[Tuple[str, float]]]:
        """
        Re-rank documents for multiple queries.
        
        Args:
            queries: List of query strings
            documents_list: List of document lists (one per query)
            doc_ids_list: Optional list of doc_id lists
            top_k: Number of top results per query
            show_progress: Show progress bar
            
        Returns:
            List of re-ranked results (one list per query)
        """
        if not queries or not documents_list:
            return []
        
        results = []
        iterator = tqdm(zip(queries, documents_list), total=len(queries), disable=not show_progress)
        
        for i, (query, documents) in enumerate(iterator):
            doc_ids = doc_ids_list[i] if doc_ids_list else None
            reranked = self.rerank(query, documents, doc_ids, top_k)
            results.append(reranked)
        
        return results
    
    def rerank_with_metadata(
        self,
        query: str,
        candidates: List[Dict],
        text_field: str = 'text',
        id_field: str = 'doc_id',
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Re-rank candidates with metadata.
        
        Args:
            query: Query string
            candidates: List of candidate dicts with metadata
            text_field: Field name for document text
            id_field: Field name for document ID
            top_k: Number of top results
            
        Returns:
            Re-ranked candidates with added 'rerank_score' field
        """
        if not candidates:
            return []
        
        # Extract texts and IDs
        documents = [c[text_field] for c in candidates]
        doc_ids = [c[id_field] for c in candidates]
        
        # Re-rank
        reranked = self.rerank(query, documents, doc_ids, top_k=None)
        
        # Create ID to score mapping
        id_to_score = {doc_id: score for doc_id, score in reranked}
        
        # Add scores to candidates and sort
        for candidate in candidates:
            doc_id = candidate[id_field]
            candidate['rerank_score'] = id_to_score.get(doc_id, 0.0)
        
        # Sort by rerank score
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Return top-k if specified
        if top_k is not None:
            candidates = candidates[:top_k]
        
        return candidates
    
    def compute_scores_only(
        self,
        query: str,
        documents: List[str]
    ) -> np.ndarray:
        """
        Compute relevance scores without sorting.
        
        Args:
            query: Query string
            documents: List of documents
            
        Returns:
            Array of relevance scores
        """
        if not documents:
            return np.array([])
        
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            convert_to_numpy=True
        )
        
        return scores


class EnsembleReranker:
    """
    Ensemble of multiple re-ranking models.
    """
    
    def __init__(self, model_names: List[str], weights: Optional[List[float]] = None, config: Dict = None):
        """
        Args:
            model_names: List of model names to ensemble
            weights: Optional weights for each model (default: equal)
            config: Shared config
        """
        self.model_names = model_names
        
        if weights is None:
            weights = [1.0 / len(model_names)] * len(model_names)
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
        
        self.weights = weights
        
        # Load models
        default_config = config or {}
        self.models = []
        
        for model_name in model_names:
            model_config = default_config.copy()
            model_config['model_name'] = model_name
            self.models.append(BERTReranker(model_config))
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """Ensemble re-ranking."""
        if not documents:
            return []
        
        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(documents))]
        
        # Get scores from each model
        all_scores = []
        for model in self.models:
            scores = model.compute_scores_only(query, documents)
            all_scores.append(scores)
        
        # Normalize and combine
        ensemble_scores = np.zeros(len(documents))
        
        for scores, weight in zip(all_scores, self.weights):
            # Min-max normalization
            if scores.max() > scores.min():
                scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                scores_norm = scores
            
            ensemble_scores += weight * scores_norm
        
        # Create results
        results = list(zip(doc_ids, ensemble_scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            results = results[:top_k]
        
        return results


if __name__ == "__main__":
    # Test re-ranker
    config = {
        'model_name': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
        'max_length': 512,
        'batch_size': 32,
        'use_fp16': False,
        'device': 'cpu'
    }
    
    reranker = BERTReranker(config)
    
    query = "what is deep learning"
    documents = [
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
        "Python is a popular programming language for data science.",
        "Neural networks are inspired by the human brain and consist of interconnected nodes.",
        "Machine learning algorithms can learn from data without being explicitly programmed."
    ]
    
    print("Re-ranking documents for query:", query)
    results = reranker.rerank(query, documents, top_k=3)
    
    print("\nTop-3 results:")
    for i, (doc_id, score) in enumerate(results, 1):
        print(f"{i}. Doc {doc_id}: {score:.4f}")
        print(f"   {documents[int(doc_id)][:80]}...")
