#!/bin/bash

# Quick validation test - just evaluate without training

cd /kaggle/adaptive-information-retrieval/adaptive-ir-system

echo "============================================"
echo "Quick Validation Test (No Training)"
echo "============================================"

python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import torch
import yaml
from src.utils.legacy_loader import LegacyDatasetAdapter
from src.utils.simple_searcher import SimpleBM25Searcher
from src.evaluation.metrics import IRMetricsAggregator
from tqdm import tqdm

print('\n📚 Loading config...')
with open('./configs/msa_quick_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print('📚 Loading validation dataset...')
adapter = LegacyDatasetAdapter(
    dataset_path='../Query Reformulator/msa_dataset.hdf5',
    corpus_path='../Query Reformulator/msa_corpus.hdf5',
    split='valid'
)

queries = adapter.load_queries()
qrels = adapter.load_qrels()
print(f'  Loaded {len(queries)} queries')

print('\n🔍 Building search engine...')
searcher = SimpleBM25Searcher(adapter, k1=0.9, b=0.4)
print(f'  Indexed {len(searcher.doc_ids)} documents')

print('\n📊 Evaluating 100 sample queries...')
evaluator = IRMetricsAggregator()

sample_queries = dict(list(queries.items())[:100])
num_with_results = 0
num_with_relevant = 0

for qid, query in tqdm(sample_queries.items(), desc='Evaluating'):
    qrel = qrels.get(qid, {})
    if not qrel:
        continue
    
    results = searcher.search(query, k=100)
    
    if results:
        num_with_results += 1
        doc_ids = [r['doc_id'] for r in results]
        
        # Check if any relevant
        relevant_set = set(qrel.keys())
        relevant_retrieved = set(doc_ids) & relevant_set
        if relevant_retrieved:
            num_with_relevant += 1
        
        evaluator.add_query_result(
            query_id=qid,
            retrieved=doc_ids,
            relevant=relevant_set,
            relevant_grades=qrel
        )

metrics = evaluator.compute_aggregate()

print('\n' + '=' * 60)
print('📊 Results:')
print('=' * 60)
print(f'Queries with results: {num_with_results}/100')
print(f'Queries with relevant docs: {num_with_relevant}/100')
print(f'Recall@10: {metrics.get(\"recall@10\", 0):.4f}')
print(f'Recall@100: {metrics.get(\"recall@100\", 0):.4f}')
print(f'MRR: {metrics.get(\"mrr\", 0):.4f}')
print(f'nDCG@10: {metrics.get(\"ndcg@10\", 0):.4f}')
print('=' * 60)

if metrics.get('recall@100', 0) > 0:
    print('\n✅ SUCCESS! Metrics are non-zero')
    sys.exit(0)
else:
    print('\n❌ FAILED! Metrics are still zero')
    sys.exit(1)
"
