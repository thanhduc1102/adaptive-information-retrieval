#!/usr/bin/env python
"""Check HDF5 file structure"""
import h5py
import sys
from pathlib import Path

def check_hdf5_structure(filepath):
    """Print structure of HDF5 file"""
    print(f"\n{'='*60}")
    print(f"File: {Path(filepath).name}")
    print('='*60)
    
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"Keys: {list(f.keys())}")
            
            for key in f.keys():
                dataset = f[key]
                print(f"\n{key}:")
                print(f"  Type: {type(dataset)}")
                print(f"  Shape: {dataset.shape if hasattr(dataset, 'shape') else 'N/A'}")
                print(f"  Dtype: {dataset.dtype if hasattr(dataset, 'dtype') else 'N/A'}")
                
                # Show first item
                if hasattr(dataset, '__getitem__') and len(dataset) > 0:
                    first_item = dataset[0]
                    if isinstance(first_item, bytes):
                        first_item = first_item.decode('utf-8')
                    print(f"  First item: {str(first_item)[:100]}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    data_dir = Path("../Query Reformulator")
    
    files = [
        "trec_car_dataset.hdf5",
        "jeopardy_dataset.hdf5",
        "msa_dataset.hdf5",
        "msa_corpus.hdf5"
    ]
    
    for filename in files:
        filepath = data_dir / filename
        if filepath.exists():
            check_hdf5_structure(filepath)
        else:
            print(f"\n⚠ Not found: {filepath}")
