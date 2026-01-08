#!/usr/bin/env python
"""
Simple test for legacy data loading - run this in your conda environment
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing legacy data loading...")
print("="*60)

try:
    from src.utils.legacy_loader import LegacyDatasetHDF5, LegacyCorpusHDF5, LegacyDatasetAdapter
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 1: Load TREC-CAR dataset
print("\n1. Testing TREC-CAR dataset...")
try:
    dataset = LegacyDatasetHDF5("../Query Reformulator/trec_car_dataset.hdf5")
    queries = dataset.get_queries(['train'])[0]
    print(f"   ✓ Loaded {len(queries):,} queries")
    dataset.close()
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Try to load TREC-CAR as corpus (should handle gracefully)
print("\n2. Testing TREC-CAR as corpus (no 'text' key)...")
try:
    corpus = LegacyCorpusHDF5("../Query Reformulator/trec_car_dataset.hdf5")
    print(f"   ✓ Loaded, has_corpus={corpus.has_corpus}, num_docs={corpus.num_docs}")
    
    # Try to get a document
    if corpus.has_corpus:
        doc = corpus.get_document(0)
        print(f"   Doc 0: {doc[:50]}")
    else:
        # Should return placeholder
        doc = corpus.get_document(0)
        print(f"   Placeholder: {doc}")
    
    corpus.close()
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Load MS Academic corpus (has 'text' key)
print("\n3. Testing MS Academic corpus (has 'text' key)...")
try:
    corpus = LegacyCorpusHDF5("../Query Reformulator/msa_corpus.hdf5")
    print(f"   ✓ Loaded, has_corpus={corpus.has_corpus}, num_docs={corpus.num_docs:,}")
    
    doc = corpus.get_document(0)
    print(f"   Doc 0: {doc[:80]}...")
    
    corpus.close()
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Adapter with TREC-CAR (dataset as corpus)
print("\n4. Testing Adapter with TREC-CAR...")
try:
    adapter = LegacyDatasetAdapter(
        "../Query Reformulator/trec_car_dataset.hdf5",
        "../Query Reformulator/trec_car_dataset.hdf5",
        'train'
    )
    
    queries = adapter.load_queries()
    qrels = adapter.load_qrels()
    
    print(f"   ✓ Queries: {len(queries):,}")
    print(f"   ✓ Qrels: {len(qrels):,}")
    
    # Test getting a document
    if queries:
        qid = list(queries.keys())[0]
        query = queries[qid]
        relevant = qrels.get(qid, {})
        print(f"   Example: {query} -> {len(relevant)} relevant docs")
        
        # Try to get first relevant doc
        if relevant:
            doc_id = list(relevant.keys())[0]
            doc = adapter.get_document(doc_id)
            print(f"   Doc: {doc[:60]}")
    
    adapter.close()
    print("   ✓ Adapter working!")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Adapter with MS Academic (separate corpus)
print("\n5. Testing Adapter with MS Academic...")
try:
    adapter = LegacyDatasetAdapter(
        "../Query Reformulator/msa_dataset.hdf5",
        "../Query Reformulator/msa_corpus.hdf5",
        'train'
    )
    
    queries = adapter.load_queries()
    qrels = adapter.load_qrels()
    
    print(f"   ✓ Queries: {len(queries):,}")
    print(f"   ✓ Qrels: {len(qrels):,}")
    
    adapter.close()
    print("   ✓ Adapter working!")
    
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*60)
print("Testing complete!")
