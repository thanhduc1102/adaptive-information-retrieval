"""
Build Pyserini Index for MS MARCO

Creates BM25 index for fast retrieval.
"""

import os
import sys
from pathlib import Path
import argparse
import subprocess


def build_index(
    collection_path: str,
    index_path: str,
    threads: int = 8
):
    """
    Build Pyserini index.
    
    Args:
        collection_path: Path to collection.tsv
        index_path: Output index directory
        threads: Number of threads
    """
    collection_path = Path(collection_path)
    index_path = Path(index_path)
    
    if not collection_path.exists():
        print(f"Error: Collection not found at {collection_path}")
        return
    
    if index_path.exists():
        print(f"Index already exists at {index_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            return
    
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Building index from {collection_path}...")
    print(f"Output: {index_path}")
    print(f"Threads: {threads}")
    
    # Pyserini command
    cmd = [
        'python', '-m', 'pyserini.index.lucene',
        '--collection', 'TsvCollection',
        '--input', str(collection_path.parent),
        '--index', str(index_path),
        '--generator', 'DefaultLuceneDocumentGenerator',
        '--threads', str(threads),
        '--storePositions',
        '--storeDocvectors',
        '--storeRaw'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Index built successfully at {index_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error building index: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build Pyserini index for MS MARCO')
    parser.add_argument(
        '--collection',
        type=str,
        default='./data/msmarco/collection.tsv',
        help='Path to collection.tsv'
    )
    parser.add_argument(
        '--index',
        type=str,
        default='./data/msmarco/index',
        help='Output index directory'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=8,
        help='Number of indexing threads'
    )
    
    args = parser.parse_args()
    
    build_index(args.collection, args.index, args.threads)
