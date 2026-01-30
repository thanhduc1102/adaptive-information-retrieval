"""
RM3 (Relevance Model 3) Query Expansion

Implementation of pseudo-relevance feedback based on:
- Lavrenko & Croft (2001): Relevance-Based Language Models

RM3 improves recall by expanding query with terms from top-k pseudo-relevant documents.
"""

import math
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import logging


class RM3QueryExpansion:
    """
    RM3 Query Expansion for Pseudo-Relevance Feedback.
    
    Algorithm:
    1. Retrieve top fbDocs documents using original query
    2. Build language model from these pseudo-relevant docs
    3. Score terms by P(term|R) where R is relevance class
    4. Select top fbTerms by score
    5. Interpolate expanded terms with original query
    """
    
    def __init__(
        self,
        searcher,
        fbTerms: int = 10,              # Number of feedback terms to add
        fbDocs: int = 10,               # Number of feedback documents
        originalQueryWeight: float = 0.5,  # Weight for original query (α)
        min_term_freq: int = 2,         # Minimum term frequency in feedback docs
        stopwords: set = None
    ):
        """
        Args:
            searcher: Search engine (SimpleBM25Searcher or Pyserini)
            fbTerms: Number of expansion terms
            fbDocs: Number of pseudo-relevant documents
            originalQueryWeight: Interpolation weight (0-1)
                0 = only expansion terms
                1 = only original query
                0.5 = equal weight (typical)
            min_term_freq: Filter rare terms
            stopwords: Set of stopwords to exclude
        """
        self.searcher = searcher
        self.fbTerms = fbTerms
        self.fbDocs = fbDocs
        self.originalQueryWeight = originalQueryWeight
        self.min_term_freq = min_term_freq
        self.stopwords = stopwords or self._get_default_stopwords()
        self.logger = logging.getLogger(__name__)
    
    def _get_default_stopwords(self) -> set:
        """Common English stopwords."""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
    
    def expand_query(self, query: str) -> str:
        """
        Expand query using RM3.
        
        Args:
            query: Original query string
            
        Returns:
            Expanded query string with weighted original + feedback terms
        """
        # Step 1: Retrieve pseudo-relevant documents
        results = self.searcher.search(query, k=self.fbDocs)
        
        if not results:
            self.logger.warning(f"No feedback documents found for query: {query}")
            return query
        
        # Step 2: Extract and collect terms from feedback docs
        doc_texts = []
        for result in results:
            if hasattr(result, 'doc_id'):
                doc_id = result.doc_id
            else:
                doc_id = result['doc_id']
            
            # Get document text
            if hasattr(self.searcher, 'get_document'):
                doc_text = self.searcher.get_document(doc_id)
            elif hasattr(self.searcher, 'doc'):
                doc_obj = self.searcher.doc(doc_id)
                doc_text = doc_obj.raw() if doc_obj else ""
            else:
                doc_text = ""
            
            if doc_text:
                doc_texts.append(doc_text)
        
        if not doc_texts:
            self.logger.warning("Could not retrieve document texts")
            return query
        
        # Step 3: Build relevance language model
        term_scores = self._compute_rm3_scores(doc_texts)
        
        # Step 4: Select top fbTerms
        expansion_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        expansion_terms = expansion_terms[:self.fbTerms]
        
        # Step 5: Interpolate with original query
        expanded_query = self._interpolate_query(query, expansion_terms)
        
        self.logger.debug(f"Original: {query}")
        self.logger.debug(f"Expanded: {expanded_query}")
        self.logger.debug(f"Added terms: {[t for t, _ in expansion_terms]}")
        
        return expanded_query
    
    def _compute_rm3_scores(self, doc_texts: List[str]) -> Dict[str, float]:
        """
        Compute RM3 term scores: P(term|R).
        
        Simplified version:
        P(w|R) ∝ ∑_{d ∈ R} P(w|d) P(d|q)
        
        Where:
        - P(w|d) = term frequency in doc / doc length
        - P(d|q) = 1/|R| (uniform, or use retrieval score)
        """
        term_scores = defaultdict(float)
        total_docs = len(doc_texts)
        
        # Collect term frequencies across all feedback docs
        doc_term_freqs = []
        doc_lengths = []
        
        for doc_text in doc_texts:
            tokens = self._tokenize(doc_text)
            term_freq = Counter(tokens)
            doc_term_freqs.append(term_freq)
            doc_lengths.append(len(tokens))
        
        # Compute P(w|R)
        for i, (term_freq, doc_len) in enumerate(zip(doc_term_freqs, doc_lengths)):
            if doc_len == 0:
                continue
            
            # P(d|q) = 1/|R| (uniform assumption)
            p_d_given_q = 1.0 / total_docs
            
            for term, freq in term_freq.items():
                # Filter stopwords and short terms
                if term in self.stopwords or len(term) < 3:
                    continue
                
                # P(w|d)
                p_w_given_d = freq / doc_len
                
                # P(w|R) += P(w|d) * P(d|q)
                term_scores[term] += p_w_given_d * p_d_given_q
        
        # Filter low-frequency terms
        # Count how many docs each term appears in
        doc_freq = defaultdict(int)
        for term_freq in doc_term_freqs:
            for term in term_freq:
                doc_freq[term] += 1
        
        # Remove terms that appear in < min_term_freq docs
        filtered_scores = {
            term: score 
            for term, score in term_scores.items()
            if doc_freq[term] >= self.min_term_freq
        }
        
        return filtered_scores
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (lowercase + split)."""
        # Basic tokenization - can be improved with proper tokenizer
        text = text.lower()
        # Remove punctuation
        import string
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        return tokens
    
    def _interpolate_query(
        self, 
        original_query: str, 
        expansion_terms: List[Tuple[str, float]]
    ) -> str:
        """
        Interpolate original query with expansion terms.
        
        RM3 formula:
        Q' = α·Q + (1-α)·∑ w_i·t_i
        
        In practice, we concatenate:
        Q' = "original_query expansion_term1 expansion_term2 ..."
        
        Weighting can be done at retrieval time (not implemented here).
        """
        # Extract original query terms
        original_terms = original_query.lower().split()
        
        # Extract expansion terms (excluding those already in query)
        expansion_term_list = [
            term for term, _ in expansion_terms 
            if term not in original_terms
        ]
        
        # Simple concatenation (weight handling would require BM25 modification)
        if self.originalQueryWeight == 1.0:
            # Only original query
            expanded = original_query
        elif self.originalQueryWeight == 0.0:
            # Only expansion terms
            expanded = " ".join(expansion_term_list)
        else:
            # Concatenate (implicit interpolation through retrieval)
            expanded = original_query + " " + " ".join(expansion_term_list)
        
        return expanded
    
    def batch_expand(self, queries: List[str]) -> List[str]:
        """Expand multiple queries."""
        return [self.expand_query(q) for q in queries]


if __name__ == "__main__":
    # Example usage
    print("RM3 Query Expansion Module")
    print("Usage:")
    print("""
from src.baselines.rm3 import RM3QueryExpansion
from src.utils.simple_searcher import SimpleBM25Searcher

# Setup searcher
searcher = SimpleBM25Searcher(dataset_adapter)

# Create RM3 expander
rm3 = RM3QueryExpansion(
    searcher,
    fbTerms=10,
    fbDocs=10,
    originalQueryWeight=0.5
)

# Expand query
expanded = rm3.expand_query("machine learning algorithms")
print(f"Expanded: {expanded}")
    """)
