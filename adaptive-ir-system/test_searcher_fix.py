"""
Test SimpleBM25Searcher fix - ensure it indexes docs from all splits
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.legacy_loader import LegacyDatasetAdapter
from src.utils.simple_searcher import SimpleBM25Searcher

def test_searcher():
    print("=" * 60)
    print("Testing SimpleBM25Searcher with multi-split indexing")
    print("=" * 60)
    
    # Load training dataset adapter
    print("\n1. Loading dataset adapter (train split)...")
    adapter_train = LegacyDatasetAdapter(
        dataset_path='../Query Reformulator/msa_dataset.hdf5',
        corpus_path='../Query Reformulator/msa_corpus.hdf5',
        split='train'
    )
    
    # Build searcher (should index docs from all splits)
    print("\n2. Building searcher (should index train+valid+test docs)...")
    searcher = SimpleBM25Searcher(adapter_train, k1=0.9, b=0.4)
    
    print(f"\n✅ Searcher built with {len(searcher.doc_ids)} documents indexed")
    
    # Now test with validation query
    print("\n3. Testing with validation queries...")
    adapter_val = LegacyDatasetAdapter(
        dataset_path='../Query Reformulator/msa_dataset.hdf5',
        corpus_path='../Query Reformulator/msa_corpus.hdf5',
        split='valid'
    )
    
    val_queries = adapter_val.load_queries()
    val_qrels = adapter_val.load_qrels()
    
    # Test first 10 validation queries
    print(f"\nTesting first 10 validation queries (out of {len(val_queries)})...")
    
    total_results = 0
    queries_with_results = 0
    
    for i, (qid, query) in enumerate(list(val_queries.items())[:10]):
        results = searcher.search(query, k=100)
        
        if results:
            queries_with_results += 1
            total_results += len(results)
            
            # Check if any retrieved docs are relevant
            qrel = val_qrels.get(qid, {})
            relevant_retrieved = sum(1 for r in results if r['doc_id'] in qrel)
            
            print(f"  Query {i+1}: {len(results)} results, {relevant_retrieved} relevant")
        else:
            print(f"  Query {i+1}: ❌ NO RESULTS (BUG!)")
    
    print(f"\n📊 Summary:")
    print(f"  Queries with results: {queries_with_results}/10")
    print(f"  Avg results per query: {total_results/10:.1f}")
    
    if queries_with_results == 10:
        print("\n✅ SUCCESS! All validation queries returned results")
        return True
    else:
        print(f"\n❌ FAILED! {10 - queries_with_results} queries returned no results")
        return False

if __name__ == "__main__":
    success = test_searcher()
    sys.exit(0 if success else 1)
