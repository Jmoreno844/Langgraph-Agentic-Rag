import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Pinecone vectorstore contents"
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
        "--limit", type=int, default=1000, help="Max vectors to consider when listing"
    )
    args = parser.parse_args()

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(args.index)

    namespace = args.namespace
    if namespace is None:
        # For serverless default namespace, Pinecone uses an empty string
        namespace = ""

    print(f"Index: {args.index}")
    print(f"Namespace: {namespace or '__default__'}")

    # Overall stats
    stats_all = index.describe_index_stats()
    print("\n=== Overall Stats ===")
    print(json.dumps(stats_all, default=str, indent=2))
    print(f"Total vectors (derived): {extract_count(stats_all)}")

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

    # Listing via list() + fetch()
    print("\n=== Listing via fetch() ===")
    try:
        ids: List[str] = []
        for vec in index.list(namespace=namespace):
            vid = vec.get("id") if isinstance(vec, dict) else vec
            if isinstance(vid, str):
                ids.append(vid)
            if len(ids) >= args.limit:
                break
        print(f"Collected {len(ids)} ids (capped by --limit)")
        shown = 0
        # Fetch in batches of 100
        for i in range(0, len(ids), 100):
            batch = ids[i : i + 100]
            resp = index.fetch(ids=batch, namespace=namespace)
            vectors = resp.get("vectors", {}) if isinstance(resp, dict) else {}
            for vid, v in vectors.items():
                meta = v.get("metadata") if isinstance(v, dict) else None
                if not isinstance(meta, dict):
                    continue
                if args.doc_id and meta.get("doc_id") != args.doc_id:
                    continue
                if args.etag and meta.get("etag") != args.etag:
                    continue
                print(
                    f"id={vid} doc_id={meta.get('doc_id')} etag={meta.get('etag')} chunk_number={meta.get('chunk_number')}"
                )
                shown += 1
        print(f"Listed {shown} vectors matching the filters.")
    except Exception as exc:
        print("Listing via fetch failed:", repr(exc))

    # Query fallback if nothing listed
    if filt and shown == 0:
        print("\n=== Query fallback ===")
        try:
            dim = int(stats_all.get("dimension") or 0)
            zero = [0.0] * dim if dim > 0 else None
            if zero is None:
                print("Cannot build zero-vector; unknown dimension")
                return
            res = index.query(
                vector=zero,
                top_k=1000,
                include_metadata=True,
                filter=filt,
                namespace=namespace,
            )
            matches = res.get("matches") if isinstance(res, dict) else []
            print(f"Query returned {len(matches)} matches")
            for m in matches:
                mid = m.get("id") if isinstance(m, dict) else None
                meta = m.get("metadata") if isinstance(m, dict) else None
                if isinstance(meta, dict):
                    print(
                        f"id={mid} doc_id={meta.get('doc_id')} etag={meta.get('etag')} chunk_number={meta.get('chunk_number')}"
                    )
        except Exception as exc:
            print("Query fallback failed:", repr(exc))


if __name__ == "__main__":
    main()
