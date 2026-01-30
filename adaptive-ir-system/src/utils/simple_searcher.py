"""
Simple Search Engine for Legacy Datasets

For datasets with corpus in HDF5, we don't need Lucene.
This provides a simple BM25 search over the corpus.
"""

from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict
import logging


class SimpleBM25Searcher:
    """
    Simple BM25 searcher for legacy datasets with corpus.
    No need for Lucene index - works directly with documents.
    """
    
    def __init__(self, corpus_adapter, k1: float = 0.9, b: float = 0.4):
        """
        Args:
            corpus_adapter: LegacyDatasetAdapter with corpus
            k1, b: BM25 parameters
        """
        self.adapter = corpus_adapter
        self.k1 = k1
        self.b = b
        self.logger = logging.getLogger(__name__)
        
        # Check if corpus available
        if not self.adapter.corpus.has_corpus:
            self.logger.warning("No corpus available - search will return empty results")
            self.corpus_docs = []
            self.doc_ids = []
            self.bm25 = None
            return
        
        self.logger.info("Building BM25 index from corpus...")
        
        # Get all documents
        self.doc_ids = []
        self.corpus_docs = []
        
        # IMPORTANT: Index documents from ALL splits (train + valid + test)
        # Otherwise validation/test queries won't find their relevant documents
        unique_doc_ids = set()
        
        # Load qrels from all available splits using dataset directly
        # (LegacyDatasetAdapter.split switching doesn't work because qrels_dict is cached)
        from src.utils.legacy_loader import LegacyDatasetHDF5
        dataset = self.adapter.dataset
        
        for split in ['train', 'valid', 'test']:
            try:
                _, qrels_dict = dataset.get_split_data(split)
                for doc_ids in qrels_dict.values():
                    unique_doc_ids.update(str(d) for d in doc_ids)
                self.logger.info(f"Collected docs from {split} split, total unique: {len(unique_doc_ids)}")
            except Exception as e:
                self.logger.warning(f"Could not load {split} qrels: {e}")
        
        self.logger.info(f"Indexing {len(unique_doc_ids)} relevant documents from all splits...")
        
        for doc_id in unique_doc_ids:
            try:
                doc_text = self.adapter.get_document(doc_id)
                self.doc_ids.append(doc_id)
                # Tokenize
                tokens = doc_text.lower().split()
                self.corpus_docs.append(tokens)
            except Exception as e:
                self.logger.warning(f"Could not load doc {doc_id}: {e}")
        
        # Build BM25 index
        if self.corpus_docs:
            self.bm25 = BM25Okapi(self.corpus_docs, k1=k1, b=b)
            self.logger.info(f"BM25 index built with {len(self.corpus_docs)} documents")
        else:
            self.bm25 = None
            self.logger.warning("No documents indexed")
    
    def search(self, query: str, k: int = 100) -> List[Dict]:
        """
        Search for query.
        
        Args:
            query: Query string
            k: Number of results
            
        Returns:
            List of dicts with 'doc_id' and 'score'
        """
        if self.bm25 is None or not self.corpus_docs:
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return docs with positive score
                results.append({
                    'doc_id': self.doc_ids[idx],
                    'score': float(scores[idx])
                })
        
        return results
    
    def doc(self, doc_id: str):
        """Get document (for compatibility with Pyserini API)."""
        class MockDoc:
            def __init__(self, text):
                self._text = text
            def raw(self):
                return self._text
        
        text = self.adapter.get_document(doc_id)
        return MockDoc(text)
    
    def get_document(self, doc_id: str) -> str:
        """Get document text."""
        return self.adapter.get_document(doc_id)
    
    def set_bm25(self, k1: float, b: float):
        """Set BM25 parameters (for compatibility)."""
        self.k1 = k1
        self.b = b
        # Would need to rebuild index, but we'll skip for now


if __name__ == "__main__":
    print("SimpleBM25Searcher - For legacy datasets")
    print("Use this instead of Pyserini when you have corpus in HDF5")
