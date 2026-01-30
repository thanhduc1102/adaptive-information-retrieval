"""
Legacy Dataset Loader

Wrapper for loading HDF5 datasets from original dl4ir-query-reformulator.
Compatible with: Jeopardy, MS Academic, TREC-CAR datasets.
"""

import h5py
import ast
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging


class LegacyDatasetHDF5:
    """
    Wrapper for legacy HDF5 dataset format.
    
    Dataset structure:
    - queries_train, queries_valid, queries_test
    - doc_ids_train, doc_ids_valid, doc_ids_test
    """
    
    def __init__(self, path: str):
        """
        Args:
            path: Path to dataset HDF5 file
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        
        self.f = h5py.File(path, 'r')
        self.logger = logging.getLogger(__name__)
        
        # Check available splits
        self.available_splits = []
        for split in ['train', 'valid', 'test']:
            if f'queries_{split}' in self.f:
                self.available_splits.append(split)
        
        self.logger.info(f"Loaded dataset: {self.path.name}")
        self.logger.info(f"Available splits: {', '.join(self.available_splits)}")
    
    def get_queries(self, dset: List[str] = ['train', 'valid', 'test']) -> List[List[str]]:
        """
        Return queries for specified splits.
        
        Args:
            dset: List of splits ('train', 'valid', 'test')
            
        Returns:
            List of query lists for each split
        """
        outs = []
        for dname in dset:
            key = f'queries_{dname}'
            if key in self.f:
                queries = [q.decode('utf-8') if isinstance(q, bytes) else q 
                          for q in self.f[key]]
                outs.append(queries)
            else:
                self.logger.warning(f"Split '{dname}' not found in dataset")
                outs.append([])
        
        return outs
    
    def get_doc_ids(self, dset: List[str] = ['train', 'valid', 'test']) -> List[List[List[str]]]:
        """
        Return <query, references> pairs for specified splits.
        
        Args:
            dset: List of splits ('train', 'valid', 'test')
            
        Returns:
            List of doc_id lists for each split
        """
        outs = []
        for dname in dset:
            key = f'doc_ids_{dname}'
            if key in self.f:
                # Handle both string and bytes
                doc_ids_raw = self.f[key]
                doc_ids = []
                
                for item in doc_ids_raw:
                    if isinstance(item, bytes):
                        item = item.decode('utf-8')
                    # Parse string representation of list
                    try:
                        parsed = ast.literal_eval(item)
                        doc_ids.append(parsed)
                    except:
                        doc_ids.append([item])
                
                outs.append(doc_ids)
            else:
                self.logger.warning(f"Split '{dname}' not found in dataset")
                outs.append([])
        
        return outs
    
    def get_split_data(self, split: str = 'train') -> Tuple[List[str], Dict[int, List[str]]]:
        """
        Get queries and qrels for a single split.
        
        Args:
            split: 'train', 'valid', or 'test'
            
        Returns:
            (queries_list, qrels_dict)
            qrels_dict: {query_idx: [doc_id1, doc_id2, ...]}
        """
        queries = self.get_queries([split])[0]
        doc_ids = self.get_doc_ids([split])[0]
        
        # Convert to dict format
        qrels = {}
        for qid, doc_list in enumerate(doc_ids):
            qrels[qid] = doc_list
        
        return queries, qrels
    
    def close(self):
        """Close HDF5 file."""
        self.f.close()


class LegacyCorpusHDF5:
    """
    Wrapper for legacy HDF5 corpus format.
    
    Corpus structure:
    - text: Document texts
    - title: Document titles
    """
    
    def __init__(self, path: str):
        """
        Args:
            path: Path to corpus HDF5 file
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Corpus not found: {path}")
        
        self.f = h5py.File(path, 'r')
        self.logger = logging.getLogger(__name__)
        
        # Check if this is a proper corpus file
        if 'text' in self.f:
            self.num_docs = len(self.f['text'])
            self.has_corpus = True
            self.logger.info(f"Loaded corpus: {self.path.name} ({self.num_docs:,} documents)")
        else:
            # This is a dataset file without corpus
            self.num_docs = 0
            self.has_corpus = False
            self.logger.warning(f"File {self.path.name} does not contain corpus data ('text' key missing)")
    
    def get_article_text(self, article_id: int) -> str:
        """
        Get document text by ID.
        
        Args:
            article_id: Document index
            
        Returns:
            Document text
        """
        if not self.has_corpus:
            return ""
        
        text = self.f['text'][article_id]
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        return text
    
    def get_article_title(self, article_id: int) -> str:
        """
        Get document title by ID.
        
        Args:
            article_id: Document index
            
        Returns:
            Document title
        """
        if not self.has_corpus:
            # Return doc_id as string if no corpus
            return str(article_id)
        
        if 'title' not in self.f:
            return str(article_id)
        
        title = self.f['title'][article_id]
        if isinstance(title, bytes):
            title = title.decode('utf-8')
        return title
    
    def get_document(self, article_id: int) -> str:
        """
        Get full document (title + text).
        
        Args:
            article_id: Document index
            
        Returns:
            Combined title and text
        """
        if not self.has_corpus:
            # Return doc_id as placeholder
            return f"Document {article_id}"
        
        title = self.get_article_title(article_id)
        text = self.get_article_text(article_id)
        
        if text:
            return f"{title} {text}"
        return title
    
    def get_titles_pos(self) -> Dict[str, int]:
        """
        Get mapping from title to document ID.
        
        Returns:
            {title: doc_id}
        """
        titles = self.f['title'][:]
        return {(t.decode('utf-8') if isinstance(t, bytes) else t): i 
                for i, t in enumerate(titles)}
    
    def get_pos_titles(self) -> Dict[int, str]:
        """
        Get mapping from document ID to title.
        
        Returns:
            {doc_id: title}
        """
        titles = self.f['title'][:]
        return {i: (t.decode('utf-8') if isinstance(t, bytes) else t) 
                for i, t in enumerate(titles)}
    
    def get_text_iter(self):
        """Get iterator over document texts."""
        return self.f['text']
    
    def get_title_iter(self):
        """Get iterator over document titles."""
        return self.f['title']
    
    def __len__(self):
        return self.num_docs
    
    def close(self):
        """Close HDF5 file."""
        self.f.close()


class LegacyDatasetAdapter:
    """
    Adapter to use legacy datasets with new pipeline.
    Converts legacy format to new pipeline's expected format.
    
    Handles datasets with or without corpus files.
    """
    
    def __init__(
        self,
        dataset_path: str,
        corpus_path: Optional[str] = None,
        split: str = 'train'
    ):
        """
        Args:
            dataset_path: Path to dataset HDF5
            corpus_path: Path to corpus HDF5 (optional, can be None or same as dataset_path)
            split: 'train', 'valid', or 'test'
        """
        self.dataset = LegacyDatasetHDF5(dataset_path)
        
        # Handle corpus
        if corpus_path is None or corpus_path == dataset_path:
            # Try to use dataset as corpus
            try:
                self.corpus = LegacyCorpusHDF5(dataset_path)
                if not self.corpus.has_corpus:
                    self.logger = logging.getLogger(__name__)
                    self.logger.warning(f"No corpus data available for {Path(dataset_path).name}")
                    self.logger.warning("Documents will return placeholders. Consider using external search engine.")
                if not self.corpus.has_corpus:
                    self.logger = logging.getLogger(__name__)
                    self.logger.warning(f"No corpus data available for {Path(dataset_path).name}")
                    self.logger.warning("Documents will return placeholders. Consider using external search engine.")
            except Exception as e:
                self.logger = logging.getLogger(__name__)
                self.logger.error(f"Could not load corpus: {e}")
                self.corpus = None
        else:
            # Separate corpus file
            self.corpus = LegacyCorpusHDF5(corpus_path)
        
        self.split = split
        
        # Load split data
        self.queries_list, self.qrels_dict = self.dataset.get_split_data(split)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loaded {split} split: {len(self.queries_list)} queries")
    
    def load_queries(self) -> Dict[str, str]:
        """
        Load queries in standard format.
        
        Returns:
            {query_id: query_text}
        """
        return {str(i): query for i, query in enumerate(self.queries_list)}
    
    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        """
        Load qrels in standard format.
        
        Returns:
            {query_id: {doc_id: relevance}}
        """
        qrels = {}
        for qid, doc_ids in self.qrels_dict.items():
            qrels[str(qid)] = {str(doc_id): 1 for doc_id in doc_ids}
        return qrels
    
    def load_collection(self) -> Dict[str, str]:
        """
        Load full corpus.
        
        Returns:
            {doc_id: document_text}
        """
        collection = {}
        for doc_id in range(len(self.corpus)):
            collection[str(doc_id)] = self.corpus.get_document(doc_id)
        return collection
    
    def get_document(self, doc_id: str) -> str:
        """
        Get single document.
        
        Args:
            doc_id: Document ID (can be string index or title)
            
        Returns:
            Document text or placeholder if corpus not available
        """
        if self.corpus is None or not self.corpus.has_corpus:
            # Return placeholder when corpus not available
            return f"Document ID: {doc_id} (corpus not loaded)"
        
        try:
            # doc_id might be numeric string or title string
            if isinstance(doc_id, str):
                # First try to convert to int if it's numeric
                try:
                    doc_idx = int(doc_id)
                    return self.corpus.get_document(doc_idx)
                except ValueError:
                    # Non-numeric doc_id - likely a title
                    # Try to find in title->index mapping
                    if not hasattr(self, '_title_to_idx'):
                        self._title_to_idx = self.corpus.get_titles_pos()
                    
                    if doc_id in self._title_to_idx:
                        doc_idx = self._title_to_idx[doc_id]
                        return self.corpus.get_document(doc_idx)
                    else:
                        # Return the title as document text (for BM25 matching)
                        return f"{doc_id}"
            else:
                doc_idx = int(doc_id)
                return self.corpus.get_document(doc_idx)
        except Exception as e:
            return f"Document ID: {doc_id} (error: {e})"
    
    def close(self):
        """Close files."""
        self.dataset.close()
        if self.corpus is not None:
            self.corpus.close()


if __name__ == "__main__":
    # Test loading
    import sys
    
    # Example paths (update to your actual paths)
    dataset_path = "../../../Query Reformulator/trec_car_dataset.hdf5"
    corpus_path = "../../../Query Reformulator/trec_car_corpus.hdf5"  # If exists
    
    print("Testing Legacy Dataset Loader...")
    print("=" * 60)
    
    # Load dataset
    dataset = LegacyDatasetHDF5(dataset_path)
    
    # Get train queries
    queries_train = dataset.get_queries(['train'])[0]
    doc_ids_train = dataset.get_doc_ids(['train'])[0]
    
    print(f"Train queries: {len(queries_train)}")
    print(f"First query: {queries_train[0]}")
    print(f"First query's relevant docs: {doc_ids_train[0][:3]}")
    
    dataset.close()
    
    print("\n✓ Legacy dataset loader working!")
