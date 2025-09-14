# Configuration

Configuration is loaded via Pydantic from a `.env` file at the project root (`src/settings.py`).

## Core Service Variables

- `OPENAI_API_KEY` (required): used for chat models and embeddings.
- `VOYAGE_API_KEY` (required): used for reranking retrieved chunks.
- `PINECONE_API_KEY` (required): API key for the Pinecone vector store.
- `PINECONE_INDEX` (required): Name of the Pinecone index to use.
- `AWS_ACCESS_KEY_ID` (required): S3 access key.
- `AWS_SECRET_ACCESS_KEY` (required): S3 secret key.
- `AWS_S3_RAG_DOCUMENTS_BUCKET` (required): S3 bucket that stores source documents.
- `AWS_DB_URL` (required): Async Postgres connection string for the LangGraph checkpointer.
  - Example: `postgresql+psycopg://user:password@host:5432/dbname`

## Observability & Evaluation Variables

- `LANGCHAIN_TRACING_V2` (optional, recommended): Set to `"true"` to enable Langsmith tracing.
- `LANGCHAIN_API_KEY` (optional, recommended): Your Langsmith API key.
- `LANGCHAIN_PROJECT` (optional, recommended): The project name to log traces under in Langsmith.
- `DEEPEVAL_SYNTH_MODEL` (optional): The OpenAI model to use for generating synthetic evaluation data with DeepEval (e.g., `gpt-4o-mini`).

## .env example

```
# Core Services
OPENAI_API_KEY=sk-...
VOYAGE_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX=your-index-name
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_RAG_DOCUMENTS_BUCKET=your-bucket
AWS_DB_URL=postgresql+psycopg://user:pass@host:5432/dbname

# Observability & Evaluation (Recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=your-project-name
DEEPEVAL_SYNTH_MODEL=gpt-4o-mini
```

## Notes

- S3 region defaults to `us-east-1` in `src/services/s3_service.py`.
- Additional envs are allowed by `Settings.Config.extra = "allow"`.
- Do not commit secrets. Keep `.env` local and use a secret store for deployments.
