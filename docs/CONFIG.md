# Configuration

Configuration is loaded via Pydantic from a `.env` file at the project root (`src/settings.py`).

## Environment variables

- `OPENAI_API_KEY` (required): used for chat models and embeddings.
- `VOYAGE_API_KEY` (required): used for reranking retrieved chunks.
- `AWS_ACCESS_KEY_ID` (required): S3 access key.
- `AWS_SECRET_ACCESS_KEY` (required): S3 secret key.
- `AWS_S3_RAG_DOCUMENTS_BUCKET` (required): S3 bucket that stores source documents.
- `AWS_DB_URL` (required): Postgres connection string for the LangGraph checkpointer.
  - Example: `postgresql://user:password@host:5432/dbname`
- `PINECONE_API_KEY` (optional, not used by default): only needed if switching to Pinecone vector store.
- `PINECONE_INDEX` (optional, not used by default): Pinecone index name.

Notes:

- S3 region defaults to `us-east-1` in `src/services/s3_service.py`.
- Additional envs are allowed by `Settings.Config.extra = "allow"`.

## .env example

```
OPENAI_API_KEY=sk-...
VOYAGE_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_RAG_DOCUMENTS_BUCKET=your-bucket
AWS_DB_URL=postgresql://user:pass@host:5432/dbname
# Optional (not used by default)
PINECONE_API_KEY=
PINECONE_INDEX=
```

## Secrets

- Do not commit secrets. Keep `.env` local.
- Use your CI/CD secret store for deployments.
