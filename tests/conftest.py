import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path so `import src.*` works
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide safe default env vars for tests (overridden by real .env if present)
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("AWS_DB_URL", "postgresql://user:pass@localhost:5432/postgres")
os.environ.setdefault("VOYAGE_API_KEY", "test")
os.environ.setdefault("PINECONE_API_KEY", "test")
os.environ.setdefault("PINECONE_INDEX", "test-index")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_S3_RAG_DOCUMENTS_BUCKET", "test-bucket")
