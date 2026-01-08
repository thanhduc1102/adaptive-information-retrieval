"""
Inference Script

Run the trained Adaptive IR system on queries.
"""

import os
import sys
import yaml
import argparse
import torch
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.utils import setup_logging, ConfigManager
from src.pipeline import AdaptiveIRPipeline
from pyserini.search.lucene import LuceneSearcher


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_search_engine(config: dict):
    """Initialize search engine."""
    index_path = config['data']['index_path']
    
    if not Path(index_path).exists():
        raise FileNotFoundError(f"Index not found at {index_path}")
    
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25(
        config['retrieval'].get('bm25_k1', 0.9),
        config['retrieval'].get('bm25_b', 0.4)
    )
    
    return searcher


def main(args):
    """Main inference function."""
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("Adaptive IR System - Inference")
    logger.info("=" * 60)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    config['system']['device'] = device
    logger.info(f"Device: {device}")
    
    # Setup search engine
    logger.info("Initializing search engine...")
    search_engine = setup_search_engine(config)
    
    # Load embedding model
    embedding_model = None
    if config['rl_agent'].get('use_pretrained_embeddings', True):
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(
            config['rl_agent'].get('embedding_model', 'all-MiniLM-L6-v2')
        )
        logger.info(f"Loaded embedding model")
    
    # Initialize pipeline
    logger.info("Building pipeline...")
    pipeline = AdaptiveIRPipeline(
        config=config,
        search_engine=search_engine,
        embedding_model=embedding_model
    )
    
    # Load checkpoint
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        pipeline.load_rl_checkpoint(args.checkpoint)
    else:
        logger.warning("No checkpoint specified, using untrained RL agent")
    
    pipeline.enable_eval_mode()
    
    logger.info("=" * 60)
    
    # Interactive mode
    if args.interactive:
        logger.info("Interactive mode (Ctrl+C to exit)")
        logger.info("-" * 60)
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if not query:
                    continue
                
                # Run search
                result = pipeline.search(query, top_k=args.top_k, measure_latency=True)
                
                # Display results
                print(f"\nOriginal query: {result['query']}")
                print(f"Query variants: {result['query_variants']}")
                print(f"\nTop {min(args.top_k, len(result['results']))} results:")
                print("-" * 60)
                
                for i, (doc_id, score) in enumerate(result['results'][:args.top_k], 1):
                    doc_text = search_engine.doc(doc_id).raw()
                    
                    # Truncate long documents
                    if len(doc_text) > 200:
                        doc_text = doc_text[:200] + "..."
                    
                    print(f"{i}. [Score: {score:.4f}] Doc {doc_id}")
                    print(f"   {doc_text}")
                    print()
                
                # Latency
                if 'latency' in result:
                    print(f"Latency: {result['latency']['total']*1000:.1f} ms")
                    print(f"  - Mining: {result['latency'].get('mining', 0)*1000:.1f} ms")
                    print(f"  - Reformulation: {result['latency'].get('reformulation', 0)*1000:.1f} ms")
                    print(f"  - Retrieval+Fusion: {result['latency'].get('retrieval_fusion', 0)*1000:.1f} ms")
                    print(f"  - Re-ranking: {result['latency'].get('reranking', 0)*1000:.1f} ms")
            
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
    
    # Batch mode
    elif args.queries_file:
        logger.info(f"Processing queries from {args.queries_file}")
        
        # Load queries
        queries = {}
        with open(args.queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    qid, query = parts
                    queries[qid] = query
        
        logger.info(f"Loaded {len(queries)} queries")
        
        # Run search
        results = {}
        
        for qid, query in queries.items():
            result = pipeline.search(query, top_k=args.top_k)
            
            results[qid] = {
                'query': query,
                'query_variants': result['query_variants'],
                'results': result['results']
            }
        
        # Save results
        output_path = Path(args.output) if args.output else Path('results.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
    
    # Single query
    else:
        query = args.query
        
        logger.info(f"Query: {query}")
        
        # Run search
        result = pipeline.search(query, top_k=args.top_k, measure_latency=True)
        
        # Display
        print(f"\nOriginal query: {result['query']}")
        print(f"Query variants: {result['query_variants']}")
        print(f"\nTop {min(args.top_k, len(result['results']))} results:")
        print("-" * 60)
        
        for i, (doc_id, score) in enumerate(result['results'][:args.top_k], 1):
            print(f"{i}. Doc {doc_id} (Score: {score:.4f})")
        
        if 'latency' in result:
            print(f"\nLatency: {result['latency']['total']*1000:.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with Adaptive IR system')
    
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/default_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to trained checkpoint'
    )
    parser.add_argument(
        '--query',
        type=str,
        default=None,
        help='Single query to search'
    )
    parser.add_argument(
        '--queries_file',
        type=str,
        default=None,
        help='File with queries (TSV format: qid\\tquery)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for batch results'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Number of results to return'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU mode'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.queries_file and not args.query:
        parser.error("Must specify --interactive, --query, or --queries_file")
    
    main(args)
