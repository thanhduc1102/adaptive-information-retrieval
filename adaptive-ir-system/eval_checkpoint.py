#!/usr/bin/env python3
"""
Checkpoint Evaluation Script

Convenient script to evaluate a trained checkpoint on any split.

Usage:
    # Evaluate on validation set
    python eval_checkpoint.py --checkpoint checkpoints_msa_optimized/best_model.pt --split valid
    
    # Evaluate on test set
    python eval_checkpoint.py --checkpoint checkpoints_msa_optimized/best_model.pt --split test
    
    # Evaluate with custom config
    python eval_checkpoint.py --checkpoint path/to/model.pt --config configs/custom.yaml --split test
    
    # Download and evaluate from HuggingFace
    python eval_checkpoint.py --hf-model username/model-name --split test
"""

import os
import sys
import argparse
import yaml
import json
import torch
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rl_agent import QueryReformulatorAgent
from src.pipeline import AdaptiveIRPipeline
from src.evaluation import IRMetricsAggregator
from src.utils import setup_logging, load_checkpoint
from src.utils.legacy_loader import LegacyDatasetAdapter


def setup_java():
    """Setup Java environment for Pyserini."""
    java_home = os.environ.get('JAVA_HOME')
    if java_home is None:
        possible_paths = [
            '/usr/lib/jvm/java-11-openjdk-amd64',
            '/usr/lib/jvm/java-17-openjdk-amd64',
            '/usr/lib/jvm/default-java',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                os.environ['JAVA_HOME'] = path
                break
    print(f"JAVA_HOME: {os.environ.get('JAVA_HOME', 'Not set')}")


def load_hf_checkpoint(repo_id: str, filename: str = "best_model.pt") -> str:
    """
    Download checkpoint from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repo ID
        filename: Checkpoint filename
        
    Returns:
        Local path to downloaded checkpoint
    """
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"📥 Downloading {filename} from {repo_id}...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir="./hf_cache"
        )
        print(f"✅ Downloaded to: {local_path}")
        return local_path
    except ImportError:
        raise ImportError("huggingface_hub not installed. Run: pip install huggingface_hub")
    except Exception as e:
        raise RuntimeError(f"Failed to download from HuggingFace: {e}")


def evaluate_checkpoint(
    checkpoint_path: str,
    config: dict,
    split: str = 'valid',
    num_queries: int = None,
    verbose: bool = True
) -> dict:
    """
    Evaluate a checkpoint on specified split.
    
    Args:
        checkpoint_path: Path to .pt checkpoint
        config: Configuration dictionary
        split: 'train', 'valid', or 'test'
        num_queries: Limit number of queries (None = all)
        verbose: Print progress
        
    Returns:
        Metrics dictionary
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Checkpoint Evaluation")
        print(f"{'='*60}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Split: {split}")
        print(f"Device: {device}")
        print(f"{'='*60}\n")
    
    # Load dataset first
    if verbose:
        print(f"📊 Loading {split} data...")
    
    data_config = config.get('data', {})
    dataset_path = data_config.get('dataset_path', '../Query Reformulator/msa_dataset.hdf5')
    corpus_path = data_config.get('corpus_path', '../Query Reformulator/msa_corpus.hdf5')
    
    dataset = LegacyDatasetAdapter(
        dataset_path=dataset_path,
        corpus_path=corpus_path,
        split=split
    )
    
    queries = dataset.load_queries()
    qrels = dataset.load_qrels()
    
    if num_queries:
        query_ids = list(queries.keys())[:num_queries]
        queries = {qid: queries[qid] for qid in query_ids}
        qrels = {qid: qrels[qid] for qid in query_ids if qid in qrels}
    
    if verbose:
        print(f"  Queries: {len(queries)}")
        print(f"  Queries with qrels: {len(qrels)}")
    
    # Setup search engine
    if verbose:
        print(f"\n🔍 Setting up search engine...")
    
    dataset_type = config['data'].get('dataset_type', 'msa')
    
    if dataset_type in ['msa', 'trec-car', 'jeopardy', 'legacy', 'hdf5']:
        from src.utils.simple_searcher import SimpleBM25Searcher
        
        search_engine = SimpleBM25Searcher(
            dataset,
            k1=config.get('retrieval', {}).get('bm25_k1', 0.9),
            b=config.get('retrieval', {}).get('bm25_b', 0.4)
        )
    else:
        from pyserini.search.lucene import LuceneSearcher
        
        index_path = config['data'].get('index_path', './data/msa_index')
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index not found at {index_path}")
        
        search_engine = LuceneSearcher(index_path)
        search_engine.set_bm25(
            config.get('retrieval', {}).get('bm25_k1', 0.9),
            config.get('retrieval', {}).get('bm25_b', 0.4)
        )
    
    # Initialize pipeline
    if verbose:
        print("📦 Initializing pipeline...")
    
    pipeline = AdaptiveIRPipeline(
        config=config,
        search_engine=search_engine,
        embedding_model=None
    )
    
    # Load checkpoint
    if verbose:
        print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        pipeline.rl_agent.load_state_dict(checkpoint['model_state_dict'])
    else:
        pipeline.rl_agent.load_state_dict(checkpoint)
    
    pipeline.rl_agent.to(device)
    pipeline.rl_agent.eval()
    
    # Print checkpoint info
    if verbose and 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if verbose and 'metrics' in checkpoint:
        print(f"  Saved metrics: {checkpoint['metrics']}")
    
    # Evaluate
    if verbose:
        print(f"\n🔍 Running evaluation...")
    
    evaluator = IRMetricsAggregator()
    
    from tqdm import tqdm
    
    for query_id, query in tqdm(queries.items(), desc="Evaluating", disable=not verbose):
        qrel = qrels.get(query_id, {})
        if not qrel:
            continue
        
        # Run search
        result = pipeline.search(query, top_k=100)
        doc_ids = [doc_id for doc_id, _ in result['results']]
        
        relevant_set = set(qrel.keys())
        evaluator.add_query_result(
            query_id=query_id,
            retrieved=doc_ids,
            relevant=relevant_set,
            relevant_grades=qrel
        )
    
    metrics = evaluator.compute_aggregate()
    
    # Print results
    if verbose:
        print(f"\n{'='*60}")
        print(f"Results ({split} split)")
        print(f"{'='*60}")
        print(f"  Recall@10:  {metrics.get('recall@10', 0):.4f}")
        print(f"  Recall@100: {metrics.get('recall@100', 0):.4f}")
        print(f"  MRR:        {metrics.get('mrr', 0):.4f}")
        print(f"  nDCG@10:    {metrics.get('ndcg@10', 0):.4f}")
        print(f"  MAP:        {metrics.get('map', 0):.4f}")
        print(f"{'='*60}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Checkpoint source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--checkpoint', '-c',
        type=str,
        help='Path to local checkpoint file (.pt)'
    )
    source_group.add_argument(
        '--hf-model',
        type=str,
        help='HuggingFace model ID (e.g., username/model-name)'
    )
    
    # Config
    parser.add_argument(
        '--config', '-f',
        type=str,
        default='configs/msa_quick_config.yaml',
        help='Path to config YAML file (default: configs/msa_quick_config.yaml)'
    )
    
    # Evaluation options
    parser.add_argument(
        '--split', '-s',
        type=str,
        default='valid',
        choices=['train', 'valid', 'test'],
        help='Data split to evaluate (default: valid)'
    )
    parser.add_argument(
        '--num-queries', '-n',
        type=int,
        default=None,
        help='Limit number of queries (default: all)'
    )
    parser.add_argument(
        '--hf-filename',
        type=str,
        default='best_model.pt',
        help='Filename to download from HuggingFace (default: best_model.pt)'
    )
    
    # Output
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_java()
    setup_logging()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get checkpoint path
    if args.hf_model:
        checkpoint_path = load_hf_checkpoint(args.hf_model, args.hf_filename)
    else:
        checkpoint_path = args.checkpoint
        if not Path(checkpoint_path).exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    
    # Run evaluation
    metrics = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        split=args.split,
        num_queries=args.num_queries,
        verbose=not args.quiet
    )
    
    # Save results
    if args.output:
        results = {
            'checkpoint': str(checkpoint_path),
            'split': args.split,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📁 Results saved to: {args.output}")
    
    # Return success
    return 0


if __name__ == "__main__":
    sys.exit(main())
