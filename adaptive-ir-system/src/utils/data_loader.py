"""
Data Loaders for MS MARCO and other datasets

Handles loading queries, documents, and relevance judgments.
"""

import os
import h5py
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import pickle
import logging


class MSMARCODataset:
    """
    MS MARCO Passage Ranking dataset loader.
    
    Dataset structure:
        - collection.tsv: pid \t passage_text
        - queries.train.tsv: qid \t query_text
        - queries.dev.small.tsv: qid \t query_text
        - qrels.train.tsv: qid 0 pid relevance
        - qrels.dev.small.tsv: qid 0 pid relevance
    """
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Directory containing MS MARCO files
        """
        self.data_dir = data_dir
        self.collection = {}
        self.queries = {}
        self.qrels = {}
        
    def load_collection(self, filename: str = "collection.tsv", max_docs: Optional[int] = None):
        """
        Load document collection.
        
        Args:
            filename: Collection filename
            max_docs: Maximum number of documents to load (None = all)
        """
        filepath = os.path.join(self.data_dir, filename)
        print(f"Loading collection from {filepath}...")
        
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if max_docs and count >= max_docs:
                    break
                
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    doc_id, text = parts
                    self.collection[doc_id] = text
                    count += 1
                
                if count % 100000 == 0:
                    print(f"  Loaded {count} documents...")
        
        print(f"Loaded {len(self.collection)} documents")
    
    def load_queries(self, split: str = "train", max_queries: Optional[int] = None):
        """
        Load queries for a split.
        
        Args:
            split: 'train', 'dev', or 'test'
            max_queries: Maximum queries to load
        """
        if split == "train":
            filename = "queries.train.tsv"
        elif split == "dev":
            filename = "queries.dev.small.tsv"
        else:
            filename = f"queries.{split}.tsv"
        
        filepath = os.path.join(self.data_dir, filename)
        print(f"Loading queries from {filepath}...")
        
        queries = {}
        count = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if max_queries and count >= max_queries:
                    break
                
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    qid, query = parts
                    queries[qid] = query
                    count += 1
        
        print(f"Loaded {len(queries)} queries for split: {split}")
        self.queries[split] = queries
        return queries
    
    def load_qrels(self, split: str = "train"):
        """
        Load relevance judgments (qrels).
        
        Args:
            split: 'train', 'dev', or 'test'
        """
        if split == "train":
            filename = "qrels.train.tsv"
        elif split == "dev":
            filename = "qrels.dev.small.tsv"
        else:
            filename = f"qrels.{split}.tsv"
        
        filepath = os.path.join(self.data_dir, filename)
        print(f"Loading qrels from {filepath}...")
        
        qrels = defaultdict(dict)
        count = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, _, doc_id, relevance = parts[:4]
                    qrels[qid][doc_id] = int(relevance)
                    count += 1
        
        print(f"Loaded {count} relevance judgments for {len(qrels)} queries")
        self.qrels[split] = dict(qrels)
        return dict(qrels)
    
    def get_query(self, qid: str, split: str = "train") -> Optional[str]:
        """Get query text by ID."""
        return self.queries.get(split, {}).get(qid)
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """Get document text by ID."""
        return self.collection.get(doc_id)
    
    def get_relevant_docs(self, qid: str, split: str = "train") -> Set[str]:
        """Get set of relevant document IDs for a query."""
        return set(self.qrels.get(split, {}).get(qid, {}).keys())
    
    def get_relevant_docs_with_grades(self, qid: str, split: str = "train") -> Dict[str, int]:
        """Get relevant documents with relevance grades."""
        return self.qrels.get(split, {}).get(qid, {})
    
    def save_cache(self, cache_file: str):
        """Save loaded data to cache file."""
        cache = {
            'collection': self.collection,
            'queries': self.queries,
            'qrels': self.qrels
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        print(f"Saved cache to {cache_file}")
    
    def load_cache(self, cache_file: str):
        """Load data from cache file."""
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        self.collection = cache['collection']
        self.queries = cache['queries']
        self.qrels = cache['qrels']
        print(f"Loaded cache from {cache_file}")


class HDF5Dataset:
    """
    Loader for legacy HDF5 datasets (Jeopardy, TREC-CAR, MS Academic).
    """
    
    def __init__(self, dataset_path: str, corpus_path: str):
        """
        Args:
            dataset_path: Path to dataset HDF5 file
            corpus_path: Path to corpus HDF5 file
        """
        self.dataset_path = dataset_path
        self.corpus_path = corpus_path
        
        self.queries = {}
        self.doc_ids = {}
        self.corpus = {}
    
    def load_queries(self, split: str = "train"):
        """Load queries from HDF5."""
        with h5py.File(self.dataset_path, 'r') as f:
            key = f'queries_{split}'
            if key in f:
                queries = [q.decode('utf-8') if isinstance(q, bytes) else q for q in f[key]]
                self.queries[split] = queries
                print(f"Loaded {len(queries)} queries for {split}")
                return queries
        return []
    
    def load_doc_ids(self, split: str = "train"):
        """Load ground-truth document IDs."""
        with h5py.File(self.dataset_path, 'r') as f:
            key = f'doc_ids_{split}'
            if key in f:
                doc_ids = []
                for item in f[key]:
                    if isinstance(item, bytes):
                        item = item.decode('utf-8')
                    # Parse string representation of list
                    import ast
                    try:
                        doc_ids.append(ast.literal_eval(item))
                    except:
                        doc_ids.append([item])
                
                self.doc_ids[split] = doc_ids
                print(f"Loaded {len(doc_ids)} doc_id lists for {split}")
                return doc_ids
        return []
    
    def load_corpus(self):
        """Load document corpus."""
        with h5py.File(self.corpus_path, 'r') as f:
            print("Loading corpus...")
            
            if 'text' in f:
                texts = f['text']
                titles = f['title'] if 'title' in f else None
                
                for i in range(len(texts)):
                    text = texts[i]
                    if isinstance(text, bytes):
                        text = text.decode('utf-8')
                    
                    title = ""
                    if titles:
                        title = titles[i]
                        if isinstance(title, bytes):
                            title = title.decode('utf-8')
                    
                    self.corpus[title] = text
                
                print(f"Loaded {len(self.corpus)} documents")
    
    def get_query(self, idx: int, split: str = "train") -> Optional[str]:
        """Get query by index."""
        queries = self.queries.get(split, [])
        if 0 <= idx < len(queries):
            return queries[idx]
        return None
    
    def get_relevant_docs(self, idx: int, split: str = "train") -> List[str]:
        """Get relevant document IDs for query at index."""
        doc_ids = self.doc_ids.get(split, [])
        if 0 <= idx < len(doc_ids):
            return doc_ids[idx]
        return []
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """Get document text by ID (title)."""
        return self.corpus.get(doc_id)


class DatasetFactory:
    """
    Factory for creating dataset instances.
    Supports: MS MARCO, legacy HDF5 datasets (Jeopardy, TREC-CAR, MS Academic)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.dataset_type = config.get('dataset_type', 'msmarco')
        self.data_dir = Path(config.get('data_dir', './data'))
        self.logger = logging.getLogger(__name__)
    
    def create_dataset(self, split: str):
        """
        Create dataset loader for specified split.
        
        Args:
            split: 'train', 'dev', or 'test'
            
        Returns:
            Dataset instance
        """
        if self.dataset_type == 'msmarco':
            return MSMARCODataset(str(self.data_dir))
        
        elif self.dataset_type in ['hdf5', 'legacy', 'jeopardy', 'trec-car', 'msa']:
            # Legacy HDF5 format from original dl4ir-query-reformulator
            from .legacy_loader import LegacyDatasetAdapter
            
            # Determine dataset and corpus files
            if self.dataset_type == 'jeopardy':
                dataset_file = 'jeopardy_dataset.hdf5'
                corpus_file = 'jeopardy_corpus.hdf5'
            elif self.dataset_type == 'trec-car':
                dataset_file = 'trec_car_dataset.hdf5'
                corpus_file = 'trec_car_corpus.hdf5'
            elif self.dataset_type == 'msa':
                dataset_file = 'msa_dataset.hdf5'
                corpus_file = 'msa_corpus.hdf5'
            else:
                # Use config-specified files
                dataset_file = self.config.get('dataset_file', 'trec_car_dataset.hdf5')
                corpus_file = self.config.get('corpus_file', None)
            
            dataset_path = self.data_dir / dataset_file
            
            # Corpus path might not exist for all datasets
            if corpus_file:
                corpus_path = self.data_dir / corpus_file
            else:
                corpus_path = dataset_path  # Use same file
            
            # Map split names (dev -> valid for legacy datasets)
            legacy_split = 'valid' if split == 'dev' else split
            
            self.logger.info(f"Loading legacy dataset: {dataset_file}")
            
            return LegacyDatasetAdapter(str(dataset_path), str(corpus_path), legacy_split)
        
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")


if __name__ == "__main__":
    # Test MS MARCO loader
    print("Testing MS MARCO Dataset Loader:")
    
    # Note: This requires actual MS MARCO data files
    # For testing purposes, you would need to download the dataset first
    
    data_dir = "./data/msmarco"
    
    if os.path.exists(data_dir):
        dataset = MSMARCODataset(data_dir)
        
        # Load small sample
        dataset.load_queries('dev', max_queries=100)
        dataset.load_qrels('dev')
        dataset.load_collection(max_docs=10000)
        
        # Get example query
        qids = list(dataset.queries['dev'].keys())
        if qids:
            qid = qids[0]
            query = dataset.get_query(qid, 'dev')
            relevant = dataset.get_relevant_docs(qid, 'dev')
            
            print(f"\nExample query (QID={qid}):")
            print(f"  Query: {query}")
            print(f"  Relevant docs: {relevant}")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Please download MS MARCO dataset first.")
