"""
Test Legacy Data Loading

Verify that legacy HDF5 datasets and embeddings load correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.legacy_loader import LegacyDatasetHDF5, LegacyCorpusHDF5, LegacyDatasetAdapter
from src.utils.legacy_embeddings import LegacyEmbeddingsLoader, LegacyEmbeddingAdapter


def test_dataset_loading(dataset_path: str):
    """Test loading HDF5 dataset."""
    print(f"\n{'='*60}")
    print(f"Testing Dataset: {Path(dataset_path).name}")
    print('='*60)
    
    try:
        dataset = LegacyDatasetHDF5(dataset_path)
        
        # Get train queries
        queries_train = dataset.get_queries(['train'])[0]
        doc_ids_train = dataset.get_doc_ids(['train'])[0]
        
        print(f"✓ Loaded dataset")
        print(f"  Available splits: {', '.join(dataset.available_splits)}")
        print(f"  Train queries: {len(queries_train):,}")
        
        if queries_train:
            print(f"\n  Example query: {queries_train[0]}")
            print(f"  Relevant docs: {doc_ids_train[0][:5]}")
        
        # Test all splits
        for split in ['train', 'valid', 'test']:
            queries = dataset.get_queries([split])[0]
            if queries:
                print(f"  {split.capitalize()}: {len(queries):,} queries")
        
        dataset.close()
        return True
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False


def test_corpus_loading(corpus_path: str):
    """Test loading HDF5 corpus."""
    print(f"\n{'='*60}")
    print(f"Testing Corpus: {Path(corpus_path).name}")
    print('='*60)
    
    try:
        corpus = LegacyCorpusHDF5(corpus_path)
        
        print(f"✓ Loaded corpus")
        print(f"  Documents: {len(corpus):,}")
        
        # Get sample documents
        if len(corpus) > 0:
            doc_id = 0
            title = corpus.get_article_title(doc_id)
            text = corpus.get_article_text(doc_id)
            
            print(f"\n  Example document (ID={doc_id}):")
            print(f"  Title: {title[:100]}")
            print(f"  Text: {text[:150]}...")
        
        corpus.close()
        return True
        
    except Exception as e:
        print(f"✗ Error loading corpus: {e}")
        return False


def test_adapter(dataset_path: str, corpus_path: str, split: str = 'train'):
    """Test LegacyDatasetAdapter."""
    print(f"\n{'='*60}")
    print(f"Testing Adapter (split={split})")
    print('='*60)
    
    try:
        adapter = LegacyDatasetAdapter(dataset_path, corpus_path, split)
        
        # Load queries
        queries = adapter.load_queries()
        print(f"✓ Loaded {len(queries):,} queries")
        
        # Load qrels
        qrels = adapter.load_qrels()
        print(f"✓ Loaded qrels for {len(qrels):,} queries")
        
        # Example query
        if queries:
            qid = list(queries.keys())[0]
            query = queries[qid]
            relevant_docs = qrels.get(qid, {})
            
            print(f"\n  Example (QID={qid}):")
            print(f"  Query: {query}")
            print(f"  Relevant docs: {len(relevant_docs)}")
        
        adapter.close()
        return True
        
    except Exception as e:
        print(f"✗ Error with adapter: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embeddings(embeddings_path: str):
    """Test loading Word2Vec embeddings."""
    print(f"\n{'='*60}")
    print(f"Testing Embeddings: {Path(embeddings_path).name}")
    print('='*60)
    
    try:
        loader = LegacyEmbeddingsLoader(embeddings_path)
        
        print(f"✓ Loaded embeddings")
        print(f"  Vocabulary: {len(loader):,} words")
        print(f"  Dimension: {loader.embedding_dim}")
        
        # Test word embeddings
        test_words = ['machine', 'learning', 'computer', 'algorithm', 'unknownword123']
        print(f"\n  Word embedding tests:")
        for word in test_words:
            emb = loader.get_embedding(word)
            in_vocab = word in loader
            norm = (emb ** 2).sum() ** 0.5
            print(f"    {word:15s}: {'✓' if in_vocab else '✗'} (norm: {norm:.4f})")
        
        # Test text embedding
        text = "machine learning algorithms for information retrieval"
        text_emb = loader.embed_text(text, method='mean')
        norm = (text_emb ** 2).sum() ** 0.5
        print(f"\n  Text embedding: shape={text_emb.shape}, norm={norm:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LEGACY DATA LOADING TESTS")
    print("="*60)
    
    # Base directory
    data_dir = Path("../Query Reformulator")
    
    results = {}
    
    # Test datasets
    datasets = {
        'TREC-CAR': data_dir / 'trec_car_dataset.hdf5',
        'Jeopardy': data_dir / 'jeopardy_dataset.hdf5',
        'MS Academic': data_dir / 'msa_dataset.hdf5'
    }
    
    for name, path in datasets.items():
        if path.exists():
            results[name + ' Dataset'] = test_dataset_loading(str(path))
        else:
            print(f"\n⚠ {name} dataset not found: {path}")
            results[name + ' Dataset'] = False
    
    # Test corpora
    corpora = {
        'MS Academic': data_dir / 'msa_corpus.hdf5'
    }
    
    for name, path in corpora.items():
        if path.exists():
            results[name + ' Corpus'] = test_corpus_loading(str(path))
        else:
            print(f"\n⚠ {name} corpus not found: {path}")
            results[name + ' Corpus'] = False
    
    # Test adapter with TREC-CAR (if available)
    trec_dataset = data_dir / 'trec_car_dataset.hdf5'
    if trec_dataset.exists():
        # TREC-CAR doesn't have separate corpus file, use dataset file for both
        results['Adapter (TREC-CAR)'] = test_adapter(str(trec_dataset), str(trec_dataset), 'train')
    
    # Test adapter with MS Academic (has separate corpus)
    msa_dataset = data_dir / 'msa_dataset.hdf5'
    msa_corpus = data_dir / 'msa_corpus.hdf5'
    if msa_dataset.exists() and msa_corpus.exists():
        results['Adapter (MSA)'] = test_adapter(str(msa_dataset), str(msa_corpus), 'train')
    
    # Test embeddings
    embeddings_path = data_dir / 'D_cbow_pdw_8B.pkl'
    if embeddings_path.exists():
        results['Embeddings'] = test_embeddings(str(embeddings_path))
    else:
        print(f"\n⚠ Embeddings not found: {embeddings_path}")
        results['Embeddings'] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status:8s} {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    elif passed > 0:
        print(f"\n⚠ {total - passed} test(s) failed")
    else:
        print("\n✗ All tests failed")


if __name__ == "__main__":
    main()
