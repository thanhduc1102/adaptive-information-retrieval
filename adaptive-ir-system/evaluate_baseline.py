#!/usr/bin/env python3
"""
=============================================================================
BASELINE EVALUATION SCRIPT
=============================================================================

Đánh giá BM25 baseline trên tất cả metrics, lưu kết quả riêng.
Sử dụng các module đã xây dựng trong src/

Cách sử dụng:
    python evaluate_baseline.py                    # Đánh giá trên valid
    python evaluate_baseline.py --split test       # Đánh giá trên test
    python evaluate_baseline.py --max-queries 500  # Giới hạn số queries

Author: Adaptive IR System
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import (
    setup_logging, set_seed,
    LegacyDatasetAdapter, LegacyEmbeddingAdapter
)
from src.utils.simple_searcher import SimpleBM25Searcher
from src.evaluation import IRMetricsAggregator, LatencyTimer


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate BM25 Baseline')
    parser.add_argument('--data-dir', type=str, 
                        default=str(Path(__file__).parent.parent / 'Query Reformulator'),
                        help='Data directory')
    parser.add_argument('--dataset', type=str, default='msa',
                        choices=['msa', 'trec-car', 'jeopardy'],
                        help='Dataset type')
    parser.add_argument('--split', type=str, default='valid',
                        choices=['train', 'valid', 'test'],
                        help='Data split to evaluate')
    parser.add_argument('--max-queries', type=int, default=None,
                        help='Maximum queries to evaluate (None = all)')
    parser.add_argument('--output-dir', type=str, default='./baseline_results',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(args.output_dir) / f'baseline_{args.split}_{datetime.now():%Y%m%d_%H%M%S}.log'
    setup_logging(log_file=str(log_file))
    logger = logging.getLogger(__name__)
    set_seed(args.seed)
    
    print("=" * 70)
    print("📊 BASELINE EVALUATION (BM25)")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Max queries: {args.max_queries or 'ALL'}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    t_start = time.time()
    
    # ==========================================================================
    # Load Data
    # ==========================================================================
    logger.info("Loading dataset...")
    
    dataset_path = Path(args.data_dir) / f'{args.dataset}_dataset.hdf5'
    corpus_path = Path(args.data_dir) / f'{args.dataset}_corpus.hdf5'
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Use LegacyDatasetAdapter from src/utils
    # Note: split is passed in constructor
    adapter = LegacyDatasetAdapter(
        dataset_path=str(dataset_path),
        corpus_path=str(corpus_path) if corpus_path.exists() else None,
        split=args.split
    )
    
    # Load queries and qrels (adapter already loaded for the split)
    queries = adapter.load_queries()
    qrels = adapter.load_qrels()
    
    logger.info(f"Loaded {len(queries)} queries for {args.split}")
    
    # Limit queries if specified
    if args.max_queries:
        query_ids = list(queries.keys())[:args.max_queries]
        queries = {qid: queries[qid] for qid in query_ids}
        qrels = {qid: qrels[qid] for qid in query_ids if qid in qrels}
    
    # ==========================================================================
    # Build Search Engine
    # ==========================================================================
    logger.info("Building BM25 index...")
    t_index = time.time()
    
    searcher = SimpleBM25Searcher(adapter, k1=0.9, b=0.4)
    
    logger.info(f"Index built in {time.time() - t_index:.1f}s")
    
    # ==========================================================================
    # Evaluate
    # ==========================================================================
    logger.info("Evaluating...")
    
    aggregator = IRMetricsAggregator()
    
    for qid in tqdm(queries.keys(), desc=f"Evaluating {args.split}"):
        query = queries[qid]
        qrel = qrels.get(qid, {})
        
        if not qrel:
            continue
        
        # Search
        with LatencyTimer() as timer:
            results = searcher.search(query, k=100)
        
        # Get doc IDs
        retrieved = [str(r['doc_id']) for r in results]
        relevant = set(str(d) for d in qrel.keys())
        
        # Convert qrel to grades for nDCG
        relevant_grades = {str(k): v for k, v in qrel.items()}
        
        aggregator.add_query_result(
            query_id=qid,
            retrieved=retrieved,
            relevant=relevant,
            relevant_grades=relevant_grades,
            latency=timer.get_elapsed()
        )
    
    # Compute metrics
    metrics = aggregator.compute_aggregate()
    
    # ==========================================================================
    # Print Results
    # ==========================================================================
    print("\n" + "=" * 70)
    print(f"📊 BASELINE RESULTS - {args.dataset.upper()} ({args.split})")
    print("=" * 70)
    
    # Recall metrics
    print("\n📈 Recall Metrics:")
    for k in [10, 100]:
        key = f'recall@{k}'
        if key in metrics:
            print(f"  Recall@{k}: {metrics[key]:.4f} ± {metrics.get(f'{key}_std', 0):.4f}")
    
    # Ranking metrics  
    print("\n📈 Ranking Metrics:")
    if 'mrr' in metrics:
        print(f"  MRR:      {metrics['mrr']:.4f} ± {metrics.get('mrr_std', 0):.4f}")
    if 'map' in metrics:
        print(f"  MAP:      {metrics['map']:.4f} ± {metrics.get('map_std', 0):.4f}")
    if 'ndcg@10' in metrics:
        print(f"  nDCG@10:  {metrics['ndcg@10']:.4f} ± {metrics.get('ndcg@10_std', 0):.4f}")
    
    # Precision metrics
    print("\n📈 Precision Metrics:")
    if 'precision@10' in metrics:
        print(f"  P@10:     {metrics['precision@10']:.4f} ± {metrics.get('precision@10_std', 0):.4f}")
    if 'f1@10' in metrics:
        print(f"  F1@10:    {metrics['f1@10']:.4f} ± {metrics.get('f1@10_std', 0):.4f}")
    
    # Latency metrics
    if 'latency_mean' in metrics:
        print("\n⏱️ Latency Metrics (ms):")
        print(f"  Mean:     {metrics['latency_mean']*1000:.2f}")
        print(f"  P50:      {metrics.get('latency_p50', 0)*1000:.2f}")
        print(f"  P95:      {metrics.get('latency_p95', 0)*1000:.2f}")
    
    total_time = time.time() - t_start
    print(f"\n⏱️ Total evaluation time: {total_time:.1f}s")
    print("=" * 70)
    
    # ==========================================================================
    # Save Results
    # ==========================================================================
    result_file = Path(args.output_dir) / f'baseline_{args.dataset}_{args.split}.json'
    
    results = {
        'dataset': args.dataset,
        'split': args.split,
        'num_queries': len(aggregator.query_metrics),
        'metrics': {k: float(v) for k, v in metrics.items()},
        'config': {
            'bm25_k1': 0.9,
            'bm25_b': 0.4,
            'top_k': 100
        },
        'timestamp': datetime.now().isoformat(),
        'time_seconds': total_time
    }
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {result_file}")
    
    # Also save per-query metrics
    per_query_file = Path(args.output_dir) / f'baseline_{args.dataset}_{args.split}_perquery.json'
    with open(per_query_file, 'w') as f:
        json.dump(aggregator.get_all_query_metrics(), f, indent=2)
    
    print(f"✅ Per-query metrics saved to {per_query_file}")
    
    return metrics


if __name__ == "__main__":
    main()
