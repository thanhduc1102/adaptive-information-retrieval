"""
Download MS MARCO Passage Ranking Dataset

Downloads:
- Collection (8.8M passages)
- Train queries + qrels
- Dev queries + qrels  
- Eval queries

Official: https://microsoft.github.io/msmarco/
"""

import os
import sys
import wget
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse


# MS MARCO URLs
URLS = {
    'collection': 'https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz',
    'queries_train': 'https://msmarco.blob.core.windows.net/msmarcoranking/queries.train.tsv',
    'queries_dev': 'https://msmarco.blob.core.windows.net/msmarcoranking/queries.dev.tsv',
    'queries_eval': 'https://msmarco.blob.core.windows.net/msmarcoranking/queries.eval.tsv',
    'qrels_train': 'https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv',
    'qrels_dev': 'https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv',
    'top1000_train': 'https://msmarco.blob.core.windows.net/msmarcoranking/top1000.train.txt.gz',
    'top1000_dev': 'https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.txt.gz',
}


def download_file(url: str, dest_path: Path, extract: bool = False):
    """
    Download file with progress bar.
    
    Args:
        url: URL to download
        dest_path: Destination path
        extract: Whether to extract gzip files
    """
    if dest_path.exists() and not extract:
        print(f"File already exists: {dest_path}")
        return
    
    print(f"Downloading {url}...")
    
    # Download
    temp_path = dest_path.parent / (dest_path.name + '.tmp')
    
    try:
        wget.download(url, str(temp_path))
        print()  # New line after wget progress
        
        # Extract if gzip
        if extract and temp_path.suffix == '.gz':
            print(f"Extracting {temp_path}...")
            with gzip.open(temp_path, 'rb') as f_in:
                with open(dest_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            temp_path.unlink()
        else:
            temp_path.rename(dest_path)
        
        print(f"Saved to {dest_path}")
    
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        raise


def download_msmarco(data_dir: str, subsets: list = None):
    """
    Download MS MARCO dataset.
    
    Args:
        data_dir: Directory to save data
        subsets: List of subsets to download (default: all)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if subsets is None:
        subsets = list(URLS.keys())
    
    print(f"Downloading MS MARCO to {data_dir}")
    print(f"Subsets: {', '.join(subsets)}")
    print()
    
    for subset in subsets:
        if subset not in URLS:
            print(f"Warning: Unknown subset '{subset}', skipping")
            continue
        
        url = URLS[subset]
        
        # Determine destination path
        if subset == 'collection':
            dest_path = data_dir / 'collection.tar.gz'
            extract = False  # Extract manually after download
        else:
            filename = url.split('/')[-1]
            
            # Extract .gz files
            if filename.endswith('.gz'):
                dest_path = data_dir / filename[:-3]
                extract = True
            else:
                dest_path = data_dir / filename
                extract = False
        
        download_file(url, dest_path, extract)
    
    # Extract collection.tar.gz if downloaded
    collection_tar = data_dir / 'collection.tar.gz'
    if collection_tar.exists():
        print(f"Extracting {collection_tar}...")
        import tarfile
        with tarfile.open(collection_tar, 'r:gz') as tar:
            tar.extractall(data_dir)
        print(f"Collection extracted to {data_dir}")
        
        # Optionally remove tar file to save space
        # collection_tar.unlink()
    
    print("\nDownload complete!")
    print(f"Data directory: {data_dir}")


def verify_dataset(data_dir: str):
    """
    Verify downloaded dataset.
    
    Args:
        data_dir: Data directory
    """
    data_dir = Path(data_dir)
    
    expected_files = [
        'collection.tsv',
        'queries.train.tsv',
        'queries.dev.tsv',
        'qrels.train.tsv',
        'qrels.dev.tsv',
    ]
    
    print("Verifying dataset...")
    all_present = True
    
    for filename in expected_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024 / 1024
            print(f"✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {filename} (missing)")
            all_present = False
    
    if all_present:
        print("\n✓ All files present!")
    else:
        print("\n✗ Some files are missing. Re-run download.")
    
    return all_present


def print_statistics(data_dir: str):
    """
    Print dataset statistics.
    
    Args:
        data_dir: Data directory
    """
    data_dir = Path(data_dir)
    
    print("\nDataset Statistics:")
    
    # Collection
    collection_path = data_dir / 'collection.tsv'
    if collection_path.exists():
        num_docs = sum(1 for _ in open(collection_path, encoding='utf-8'))
        print(f"  Collection: {num_docs:,} passages")
    
    # Queries
    for split in ['train', 'dev', 'eval']:
        queries_path = data_dir / f'queries.{split}.tsv'
        if queries_path.exists():
            num_queries = sum(1 for _ in open(queries_path, encoding='utf-8'))
            print(f"  Queries ({split}): {num_queries:,}")
    
    # Qrels
    for split in ['train', 'dev']:
        qrels_path = data_dir / f'qrels.{split}.tsv'
        if qrels_path.exists():
            num_qrels = sum(1 for _ in open(qrels_path, encoding='utf-8'))
            print(f"  Qrels ({split}): {num_qrels:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download MS MARCO dataset')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/msmarco',
        help='Directory to save data'
    )
    parser.add_argument(
        '--subsets',
        type=str,
        nargs='+',
        default=None,
        help='Subsets to download (default: all)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify downloaded files'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print dataset statistics'
    )
    
    args = parser.parse_args()
    
    if not args.verify and not args.stats:
        # Download
        download_msmarco(args.data_dir, args.subsets)
    
    if args.verify:
        verify_dataset(args.data_dir)
    
    if args.stats:
        print_statistics(args.data_dir)
