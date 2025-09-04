#!/usr/bin/env python3
"""
Pinecone Latency Test Script
Compares performance of metadata-only vs full vector queries
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import argparse
from typing import Dict, Any, List
from pinecone import Pinecone
from src.settings import settings


def run_comprehensive_latency_test(index_name: str, namespace: str = "", test_sizes: List[int] = None):
    """Run comprehensive latency tests comparing metadata-only vs full queries."""
    
    print("=== Pinecone Latency Test: Metadata-only vs Full Query ===")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(index_name)
    
    # Get index stats
    stats = index.describe_index_stats()
    total_vectors = stats.get('total_vector_count', 0)
    dimension = stats.get('dimension', 768)
    
    print(f"Index: {index_name}")
    print(f"Namespace: {namespace or '__default__'}")
    print(f"Total vectors: {total_vectors}")
    print(f"Vector dimension: {dimension}")
    
    if total_vectors == 0:
        print("❌ No vectors found in index. Cannot run test.")
        return
    
    print()
    
    # Default test sizes if not provided
    if test_sizes is None:
        test_sizes = [100, 500, 1000, min(5000, total_vectors)]
    
    # Create a zero vector for querying (will match all vectors)
    zero_vector = [0.0] * dimension
    
    results = []
    
    for size in test_sizes:
        if size > total_vectors:
            print(f"--- Skipping top_k={size} (only {total_vectors} vectors available) ---")
            continue
            
        print(f"--- Testing with top_k={size} ---")
        
        # Test 1: Metadata only query
        start_time = time.time()
        result_meta = index.query(
            vector=zero_vector,
            top_k=size,
            include_metadata=True,
            include_values=False,  # No vector values
            namespace=namespace
        )
        meta_time = time.time() - start_time
        
        # Test 2: Full query (metadata + values)
        start_time = time.time()
        result_full = index.query(
            vector=zero_vector,
            top_k=size,
            include_metadata=True,
            include_values=True,   # Include vector values
            namespace=namespace
        )
        full_time = time.time() - start_time
        
        # Results analysis
        meta_matches = len(result_meta.get('matches', []))
        full_matches = len(result_full.get('matches', []))
        
        # Calculate response sizes
        meta_response_size = len(json.dumps(result_meta, default=str))
        full_response_size = len(json.dumps(result_full, default=str))
        
        # Calculate metrics
        speedup = full_time / meta_time if meta_time > 0 else 0
        size_reduction_pct = ((full_response_size - meta_response_size) / full_response_size * 100) if full_response_size > 0 else 0
        
        # Estimate vector data size (rough approximation)
        if meta_matches > 0:
            vector_size_per_item = (full_response_size - meta_response_size) / meta_matches
            estimated_vector_bytes = vector_size_per_item * dimension * 4  # 4 bytes per float32
        else:
            estimated_vector_bytes = 0
        
        print(".3f")
        print(".3f")
        print(".2f")
        print(".1f")
        print(f"  Est. vector size per item: {estimated_vector_bytes:.0f} bytes")
        
        # Store results
        results.append({
            'top_k': size,
            'meta_time': meta_time,
            'full_time': full_time,
            'speedup': speedup,
            'meta_size': meta_response_size,
            'full_size': full_response_size,
            'size_reduction': size_reduction_pct,
            'matches': meta_matches
        })
        
        print()
    
    # Summary
    print("=== SUMMARY ===")
    if results:
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        avg_size_reduction = sum(r['size_reduction'] for r in results) / len(results)
        
        print(".2f")
        print(".1f")
        print(f"Average matches returned: {sum(r['matches'] for r in results) / len(results):.0f}")
    
    return results


def run_fetch_latency_test(index_name: str, namespace: str = "", batch_size: int = 100):
    """Test fetch latency with metadata-only vs full fetch."""
    
    print("=== Pinecone Fetch Latency Test ===")
    
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(index_name)
    
    # Get some vector IDs to fetch
    ids = []
    try:
        for vec in index.list(namespace=namespace, limit=batch_size):
            vid = vec.get("id") if isinstance(vec, dict) else vec
            if isinstance(vid, str):
                ids.append(vid)
            if len(ids) >= batch_size:
                break
    except Exception as e:
        print(f"❌ Could not list vectors: {e}")
        return
    
    if not ids:
        print("❌ No vectors found to fetch.")
        return
    
    print(f"Testing fetch with {len(ids)} vectors...")
    
    # Test metadata-only fetch
    start_time = time.time()
    result_meta = index.fetch(ids=ids, namespace=namespace, include_values=False)
    meta_fetch_time = time.time() - start_time
    
    # Test full fetch
    start_time = time.time()
    result_full = index.fetch(ids=ids, namespace=namespace, include_values=True)
    full_fetch_time = time.time() - start_time
    
    # Results
    meta_size = len(json.dumps(result_meta, default=str))
    full_size = len(json.dumps(result_full, default=str))
    
    speedup = full_fetch_time / meta_fetch_time if meta_fetch_time > 0 else 0
    size_reduction = ((full_size - meta_size) / full_size * 100) if full_size > 0 else 0
    
    print(".3f")
    print(".3f")
    print(".2f")
    print(".1f")
    print()


def main():
    parser = argparse.ArgumentParser(description="Test Pinecone query/fetch latency")
    parser.add_argument("--index", default=settings.PINECONE_INDEX, help="Pinecone index name")
    parser.add_argument("--namespace", default="", help="Namespace to test")
    parser.add_argument("--test-sizes", nargs="+", type=int, default=[100, 500, 1000, 5000], 
                       help="Test sizes for queries")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for fetch tests")
    parser.add_argument("--query-only", action="store_true", help="Only run query tests")
    parser.add_argument("--fetch-only", action="store_true", help="Only run fetch tests")
    
    args = parser.parse_args()
    
    if not args.fetch_only:
        run_comprehensive_latency_test(args.index, args.namespace, args.test_sizes)
    
    if not args.query_only:
        run_fetch_latency_test(args.index, args.namespace, args.batch_size)


if __name__ == "__main__":
    main()
