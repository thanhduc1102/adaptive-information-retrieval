"""
Candidate Term Mining Module

Extracts candidate terms/phrases from top-k retrieved documents
for query reformulation.
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')


class CandidateTermMiner:
    """
    Extracts candidate terms from pseudo-relevant documents.
    
    Methods:
        - TF-IDF scoring
        - BM25 contribution analysis
        - (Optional) KeyBERT semantic extraction
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary with mining parameters
        """
        self.config = config
        self.top_k0 = config.get('top_k0', 50)
        self.max_candidates = config.get('max_candidates', 200)
        self.min_candidates = config.get('min_candidates', 50)
        self.methods = config.get('methods', ['tfidf', 'bm25_contrib'])
        self.use_stopwords = config.get('stopwords', True)
        self.min_term_length = config.get('min_term_length', 3)
        self.max_term_length = config.get('max_term_length', 20)
        
        # Initialize stopwords
        if self.use_stopwords:
            self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = set()
        
        # TF-IDF vectorizer
        self.tfidf = None
        
    def extract_candidates(
        self,
        query: str,
        documents: List[str],
        doc_scores: List[float] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract candidate terms from top-k documents.
        
        Args:
            query: Original query string
            documents: List of retrieved document texts
            doc_scores: Optional BM25 scores for each document
            
        Returns:
            Dictionary mapping terms to their feature scores
            {
                'term1': {'tfidf': 0.5, 'bm25_contrib': 0.3, ...},
                'term2': {...},
                ...
            }
        """
        candidates = {}
        query_tokens = set(self._tokenize(query))
        
        # Method 1: TF-IDF
        if 'tfidf' in self.methods:
            tfidf_candidates = self._extract_tfidf(documents)
            for term, score in tfidf_candidates.items():
                if term not in candidates:
                    candidates[term] = {}
                candidates[term]['tfidf'] = score
        
        # Method 2: BM25 Contribution
        if 'bm25_contrib' in self.methods and doc_scores is not None:
            bm25_candidates = self._extract_bm25_contrib(
                query, documents, doc_scores
            )
            for term, score in bm25_candidates.items():
                if term not in candidates:
                    candidates[term] = {}
                candidates[term]['bm25_contrib'] = score
        
        # Method 3: KeyBERT (optional)
        if 'keybert' in self.methods:
            try:
                keybert_candidates = self._extract_keybert(documents)
                for term, score in keybert_candidates.items():
                    if term not in candidates:
                        candidates[term] = {}
                    candidates[term]['keybert'] = score
            except ImportError:
                print("Warning: KeyBERT not installed. Skipping KeyBERT extraction.")
        
        # Filter candidates
        filtered = self._filter_candidates(candidates, query_tokens)
        
        # Aggregate scores and rank
        ranked_candidates = self._rank_candidates(filtered)
        
        # Return top-N
        return dict(ranked_candidates[:self.max_candidates])
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and clean text."""
        # Lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Filter length
        tokens = [
            t for t in tokens 
            if self.min_term_length <= len(t) <= self.max_term_length
        ]
        return tokens
    
    def _extract_tfidf(self, documents: List[str]) -> Dict[str, float]:
        """Extract terms using TF-IDF scoring. Thread-safe version."""
        if not documents:
            return {}
        
        # Create new TF-IDF vectorizer for thread safety
        tfidf = TfidfVectorizer(
            max_features=self.max_candidates * 2,
            stop_words='english' if self.use_stopwords else None,
            token_pattern=r'\b[a-z]{' + str(self.min_term_length) + r',' + str(self.max_term_length) + r'}\b'
        )
        
        try:
            tfidf_matrix = tfidf.fit_transform(documents)
        except ValueError:
            return {}
        
        # Get feature names
        feature_names = tfidf.get_feature_names_out()
        
        # Average TF-IDF across documents
        mean_tfidf = tfidf_matrix.mean(axis=0).A1
        
        # Create term -> score mapping
        term_scores = {
            term: float(score)
            for term, score in zip(feature_names, mean_tfidf)
            if score > 0
        }
        
        return term_scores
    
    def _extract_bm25_contrib(
        self,
        query: str,
        documents: List[str],
        doc_scores: List[float]
    ) -> Dict[str, float]:
        """
        Extract terms based on BM25 contribution to document scores.
        
        This approximates how much each term contributed to the BM25 score.
        """
        if not documents or not doc_scores:
            return {}
        
        query_terms = self._tokenize(query)
        term_contributions = defaultdict(float)
        
        for doc, score in zip(documents, doc_scores):
            doc_tokens = self._tokenize(doc)
            doc_term_freq = Counter(doc_tokens)
            
            # Approximate contribution of each term
            for term, freq in doc_term_freq.items():
                if term not in query_terms:
                    # Weight by doc score and term frequency
                    contribution = score * np.log1p(freq)
                    term_contributions[term] += contribution
        
        # Normalize
        if term_contributions:
            max_contrib = max(term_contributions.values())
            if max_contrib > 0:
                term_contributions = {
                    k: v / max_contrib 
                    for k, v in term_contributions.items()
                }
        
        return dict(term_contributions)
    
    def _extract_keybert(self, documents: List[str]) -> Dict[str, float]:
        """Extract semantic keyphrases using KeyBERT."""
        from keybert import KeyBERT
        
        # Initialize KeyBERT
        kw_model = KeyBERT()
        
        # Extract keywords from concatenated docs
        combined_text = ' '.join(documents[:10])  # Limit to first 10 docs
        
        try:
            keywords = kw_model.extract_keywords(
                combined_text,
                keyphrase_ngram_range=(1, 2),
                top_n=self.max_candidates
            )
            return {kw: float(score) for kw, score in keywords}
        except Exception as e:
            print(f"KeyBERT extraction failed: {e}")
            return {}
    
    def _filter_candidates(
        self,
        candidates: Dict[str, Dict[str, float]],
        query_tokens: Set[str]
    ) -> Dict[str, Dict[str, float]]:
        """Filter candidates based on various criteria."""
        filtered = {}
        
        for term, features in candidates.items():
            # Skip if already in query
            if term in query_tokens:
                continue
            
            # Skip if stopword
            if term in self.stopwords:
                continue
            
            # Skip if too short/long
            if not (self.min_term_length <= len(term) <= self.max_term_length):
                continue
            
            # Skip if not mostly alphabetic
            if not any(c.isalpha() for c in term):
                continue
            
            filtered[term] = features
        
        return filtered
    
    def _rank_candidates(
        self,
        candidates: Dict[str, Dict[str, float]]
    ) -> List[Tuple[str, Dict[str, float]]]:
        """
        Rank candidates by aggregated score.
        
        Uses weighted average of available features.
        """
        scored_candidates = []
        
        for term, features in candidates.items():
            # Aggregate score (equal weights for now)
            scores = list(features.values())
            if scores:
                agg_score = np.mean(scores)
                features['aggregate'] = agg_score
                scored_candidates.append((term, features))
        
        # Sort by aggregate score
        scored_candidates.sort(key=lambda x: x[1]['aggregate'], reverse=True)
        
        return scored_candidates
    
    def get_candidate_features(
        self,
        candidates: Dict[str, Dict[str, float]],
        query_embedding: np.ndarray = None
    ) -> np.ndarray:
        """
        Convert candidate dict to feature matrix for RL agent.
        
        Args:
            candidates: Dict mapping terms to feature dicts
            query_embedding: Optional query embedding for similarity
            
        Returns:
            Feature matrix of shape (num_candidates, num_features)
        """
        feature_names = ['tfidf', 'bm25_contrib', 'keybert']
        num_features = len(feature_names)
        
        terms = list(candidates.keys())
        features = np.zeros((len(terms), num_features))
        
        for i, term in enumerate(terms):
            for j, feat_name in enumerate(feature_names):
                features[i, j] = candidates[term].get(feat_name, 0.0)
        
        return features


if __name__ == "__main__":
    # Example usage
    config = {
        'top_k0': 10,
        'max_candidates': 50,
        'methods': ['tfidf', 'bm25_contrib'],
        'stopwords': True
    }
    
    miner = CandidateTermMiner(config)
    
    query = "deep learning neural networks"
    documents = [
        "Deep learning is a subset of machine learning based on artificial neural networks.",
        "Neural networks consist of layers of interconnected nodes processing information.",
        "Convolutional neural networks are particularly effective for image recognition tasks."
    ]
    doc_scores = [10.5, 8.2, 7.8]
    
    candidates = miner.extract_candidates(query, documents, doc_scores)
    
    print(f"Extracted {len(candidates)} candidates:")
    for term, features in list(candidates.items())[:10]:
        print(f"  {term}: {features}")
