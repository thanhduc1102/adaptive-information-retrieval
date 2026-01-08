"""
Legacy Embeddings Loader

Load pretrained Word2Vec embeddings from D_cbow_pdw_8B.pkl
"""

import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional
import logging


class LegacyEmbeddingsLoader:
    """
    Load and manage legacy Word2Vec embeddings.
    
    Format: Python dictionary {word: embedding_vector}
    - 374,000 words
    - 500-dimensional embeddings
    - Pre-normalized vectors
    """
    
    def __init__(self, embeddings_path: str):
        """
        Args:
            embeddings_path: Path to D_cbow_pdw_8B.pkl
        """
        self.path = Path(embeddings_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading embeddings from {self.path.name}...")
        
        # Load pickle
        with open(self.path, 'rb') as f:
            self.word2vec = pickle.load(f, encoding='latin1')  # Python 2 compatibility
        
        # Get embedding dimension
        sample_word = list(self.word2vec.keys())[0]
        self.embedding_dim = len(self.word2vec[sample_word])
        
        self.logger.info(f"Loaded {len(self.word2vec):,} words, dim={self.embedding_dim}")
        
        # Create special tokens
        self.unk_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        self.pad_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
    
    def get_embedding(self, word: str) -> np.ndarray:
        """
        Get embedding for a word.
        
        Args:
            word: Word string
            
        Returns:
            Embedding vector (500-dim)
        """
        # Try exact match
        if word in self.word2vec:
            return self.word2vec[word]
        
        # Try lowercase
        word_lower = word.lower()
        if word_lower in self.word2vec:
            return self.word2vec[word_lower]
        
        # Return UNK
        return self.unk_embedding
    
    def get_embeddings_batch(self, words: list) -> np.ndarray:
        """
        Get embeddings for multiple words.
        
        Args:
            words: List of words
            
        Returns:
            Array of shape (len(words), embedding_dim)
        """
        embeddings = []
        for word in words:
            embeddings.append(self.get_embedding(word))
        return np.array(embeddings, dtype=np.float32)
    
    def embed_text(self, text: str, method: str = 'mean') -> np.ndarray:
        """
        Embed text using word embeddings.
        
        Args:
            text: Text string
            method: 'mean', 'sum', or 'max'
            
        Returns:
            Text embedding vector
        """
        words = text.lower().split()
        
        if not words:
            return self.unk_embedding
        
        embeddings = self.get_embeddings_batch(words)
        
        if method == 'mean':
            return embeddings.mean(axis=0)
        elif method == 'sum':
            return embeddings.sum(axis=0)
        elif method == 'max':
            return embeddings.max(axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def create_embedding_matrix(
        self,
        vocab: list,
        add_special_tokens: bool = True
    ) -> np.ndarray:
        """
        Create embedding matrix for a vocabulary.
        
        Args:
            vocab: List of words
            add_special_tokens: Add PAD and UNK tokens
            
        Returns:
            Embedding matrix of shape (len(vocab), embedding_dim)
        """
        embeddings = []
        
        if add_special_tokens:
            embeddings.append(self.pad_embedding)  # PAD = 0
            embeddings.append(self.unk_embedding)  # UNK = 1
        
        for word in vocab:
            embeddings.append(self.get_embedding(word))
        
        return np.array(embeddings, dtype=np.float32)
    
    def get_vocabulary(self) -> list:
        """Get list of all words in vocabulary."""
        return list(self.word2vec.keys())
    
    def __len__(self):
        return len(self.word2vec)
    
    def __contains__(self, word):
        return word in self.word2vec or word.lower() in self.word2vec


class LegacyEmbeddingAdapter:
    """
    Adapter to use legacy embeddings with PyTorch models.
    """
    
    def __init__(self, embeddings_path: str):
        self.loader = LegacyEmbeddingsLoader(embeddings_path)
        self.embedding_dim = self.loader.embedding_dim
    
    def encode(
        self,
        text: str,
        convert_to_tensor: bool = False
    ) -> np.ndarray or torch.Tensor:
        """
        Encode text to embedding.
        
        Args:
            text: Input text
            convert_to_tensor: Return PyTorch tensor
            
        Returns:
            Embedding vector
        """
        embedding = self.loader.embed_text(text, method='mean')
        
        if convert_to_tensor:
            return torch.from_numpy(embedding)
        
        return embedding
    
    def to_pytorch_embedding_layer(
        self,
        vocab: list,
        freeze: bool = True
    ) -> torch.nn.Embedding:
        """
        Create PyTorch embedding layer.
        
        Args:
            vocab: Vocabulary list
            freeze: Freeze embeddings (not trainable)
            
        Returns:
            nn.Embedding layer
        """
        embedding_matrix = self.loader.create_embedding_matrix(vocab, add_special_tokens=True)
        embedding_layer = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(embedding_matrix),
            freeze=freeze,
            padding_idx=0
        )
        return embedding_layer


if __name__ == "__main__":
    # Test loading
    embeddings_path = "../../../Query Reformulator/D_cbow_pdw_8B.pkl"
    
    print("Testing Legacy Embeddings Loader...")
    print("=" * 60)
    
    loader = LegacyEmbeddingsLoader(embeddings_path)
    
    print(f"Vocabulary size: {len(loader):,}")
    print(f"Embedding dimension: {loader.embedding_dim}")
    
    # Test word embeddings
    test_words = ['machine', 'learning', 'computer', 'unknownword123']
    for word in test_words:
        emb = loader.get_embedding(word)
        in_vocab = word in loader
        print(f"{word}: {'✓' if in_vocab else '✗'} (norm: {np.linalg.norm(emb):.4f})")
    
    # Test text embedding
    text = "machine learning algorithms"
    text_emb = loader.embed_text(text)
    print(f"\nText embedding: {text_emb.shape} (norm: {np.linalg.norm(text_emb):.4f})")
    
    print("\n✓ Legacy embeddings loader working!")
