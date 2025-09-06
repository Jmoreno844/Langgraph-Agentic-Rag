#!/usr/bin/env python3
"""
Optimized Pinecone Inspector with Metadata-Only Operations
Demonstrates significant latency improvements with metadata-only queries
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import argparse
from typing import Optional, Dict, Any, List
from pinecone import Pinecone
from src.settings import settings


def extract_count(stats: Dict[str, Any]) -> int:
    total = stats.get("total_vector_count")
    if isinstance(total, int):
        return int(total)
    namespaces = stats.get("namespaces") or {}
    try:
        return sum(int(ns.get("vector_count", 0)) for ns in namespaces.values())
    except Exception:
        return 0


def build_filter(
    doc_id: Optional[str], etag: Optional[str]
) -> Optional[Dict[str, Any]]:
    if not doc_id and not etag:
        return None
    f: Dict[str, Any] = {}
    if doc_id:
        f["doc_id"] = {"$eq": doc_id}
    if etag:
        f["etag"] = {"$eq": etag}
    return f


def run_performance_comparison(index, namespace: str = "", filter_dict: Optional[Dict] = None):
    """Compare performance of metadata-only vs full queries."""
    
    print("\n=== Performance Comparison ===")
    
    # Get index stats
    stats_all = index.describe_index_stats()
    total_vectors = extract_count(stats_all)
    dimension = stats_all.get("dimension", 768)
    
    if total_vectors == 0:
        print("No vectors found in index.")
        return
    
    # Create query vector (zero vector to match all)
    zero_vector = [0.0] * dimension
    
    # Test with available data
    test_k = min(1000, total_vectors)  # Don't exceed available vectors
    
    print(f"Testing with top_k={test_k} (dimension={dimension})")
    
    # Metadata-only query
    start_time = time.time()
    result_meta = index.query(
        vector=zero_vector,
        top_k=test_k,
        include_metadata=True,
        include_values=False,  # KEY: No vector embeddings
        filter=filter_dict,
        namespace=namespace,
    )
    meta_time = time.time() - start_time
    
    # Full query (metadata + embeddings)
    start_time = time.time()
    result_full = index.query(
        vector=zero_vector,
        top_k=test_k,
        include_metadata=True,
        include_values=True,   # KEY: Include vector embeddings
        filter=filter_dict,
        namespace=namespace,
    )
    full_time = time.time() - start_time
    
    # Analyze results
    meta_matches = len(result_meta.get("matches", []))
    full_matches = len(result_full.get("matches", []))
    
    # Response sizes
    meta_size = len(json.dumps(result_meta, default=str))
    full_size = len(json.dumps(result_full, default=str))
    
    # Calculate improvements
    speedup = full_time / meta_time if meta_time > 0 else 0
    size_reduction = ((full_size - meta_size) / full_size * 100) if full_size > 0 else 0
    
    print(".3f")
    print(".3f")
    print(".2f")
    print(".1f")
    
    # Extrapolate to 5K results
    if meta_matches > 0:
        scale_factor = 5000 / meta_matches
        estimated_meta_time = meta_time * scale_factor
        estimated_full_time = full_time * scale_factor
        
        print(f"\n=== Extrapolated to 5K results ===")
        print(".1f")
        print(".1f")
        print(".1f")
    
    return {
        'meta_time': meta_time,
        'full_time': full_time,
        'speedup': speedup,
        'size_reduction': size_reduction
    }


def main():
    parser = argparse.ArgumentParser(
        description="Optimized Pinecone inspector with metadata-only operations"
    )
    parser.add_argument(
        "--index", default=settings.PINECONE_INDEX, help="Pinecone index name"
    )
    parser.add_argument(
        "--doc-id", dest="doc_id", default=None, help="Filter by doc_id (metadata)"
    )
    parser.add_argument(
        "--etag", dest="etag", default=None, help="Filter by etag (metadata)"
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Namespace to inspect (default: __default__) (optional)",
    )
    parser.add_argument(
        "--performance-test", action="store_true", 
        help="Run performance comparison test"
    )
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Use metadata-only queries for all operations"
    )
    
    args = parser.parse_args()

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(args.index)

    namespace = args.namespace
    if namespace is None:
        namespace = ""

    print(f"Index: {args.index}")
    print(f"Namespace: {namespace or '__default__'}")
    print(f"Metadata-only mode: {args.metadata_only}")

    # Overall stats
    stats_all = index.describe_index_stats()
    print("\n=== Overall Stats ===")
    print(json.dumps(stats_all, default=str, indent=2))
    print(f"Total vectors (derived): {extract_count(stats_all)}")

    # Performance test
    if args.performance_test:
        filt = build_filter(args.doc_id, args.etag)
        run_performance_comparison(index, namespace, filt)

    # Filtered stats (may not be supported on serverless)
    filt = build_filter(args.doc_id, args.etag)
    if filt:
        try:
            stats_filtered = index.describe_index_stats(filter=filt)
            print("\n=== Filtered Stats ===")
            print(
                json.dumps(
                    {"filter": filt, "stats": stats_filtered}, default=str, indent=2
                )
            )
            print(f"Filtered vectors (derived): {extract_count(stats_filtered)}")
        except Exception as exc:
            print("\nFiltered stats not supported:", repr(exc))

    # Query-based listing (recommended approach)
    print("\n=== Query-based Listing ===")
    try:
        dimension = int(stats_all.get("dimension") or 768)
        zero_vector = [0.0] * dimension
        
        # Use metadata-only for faster listing
        include_values = not args.metadata_only
        
        result = index.query(
            vector=zero_vector,
            top_k=1000,  # Adjust as needed
            include_metadata=True,
            include_values=include_values,
            filter=filt,
            namespace=namespace,
        )
        
        matches = result.get("matches", [])
        print(f"Found {len(matches)} matches")
        
        # Show sample results
        for i, match in enumerate(matches[:5]):  # Show first 5
            mid = match.get("id")
            score = match.get("score")
            meta = match.get("metadata", {})
            
            print(f"Match {i+1}: id={mid} score={score:.3f}")
            if meta:
                # Truncate text for display
                text = meta.get("text", "")
                if len(text) > 100:
                    text = text[:100] + "..."
                print(f"  doc_id={meta.get('doc_id')} chunk={meta.get('chunk_number')} text='{text}'")
        
        if len(matches) > 5:
            print(f"... and {len(matches)-5} more matches")
            
    except Exception as exc:
        print("Query-based listing failed:", repr(exc))


if __name__ == "__main__":
    main()
