# LangGraph Agentic RAG

An agentic RAG chatbot backend built with LangGraph and FastAPI. It ingests documents from S3, synchronizes them with a Pinecone vector store, and serves a streaming chat endpoint that uses a sophisticated hybrid search and reranking retrieval strategy.

## How it Works

The core of the application is a graph built with LangGraph that orchestrates the flow of a user's request. It handles topic validation, document retrieval, context extraction, and final response generation.

![LangGraph in Action](docs/assets/langgraph-in-action.gif)

## Features

- RAG graph with conditional tool routing for retrieval and product queries.
- S3-backed document management API that automatically syncs with Pinecone.
- Hybrid search (Pinecone vector search + BM25) with VoyageAI reranking for retrieval.
- FastAPI app with streaming `/chat` endpoint.
- Async-first design with `AsyncPostgresSaver` for persistent, stateful conversations.
- Observability with Langsmith tracing.
- RAG evaluation suite using DeepEval.
- Pytest unit tests for key nodes.

## Quickstart

1.  **Prereqs**

    - Python 3.12
    - An S3 bucket with documents to ingest
    - API keys: OpenAI, AWS, VoyageAI, Pinecone
    - A running Postgres database

2.  **Install**

    ```bash
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Configure environment**

    - Create a `.env` file at the project root (see `docs/CONFIG.md` for details).
    - Minimum required for local run:

    ```bash
    # Core services
    OPENAI_API_KEY=...
    VOYAGE_API_KEY=...
    PINECONE_API_KEY=...
    PINECONE_INDEX=...
    AWS_ACCESS_KEY_ID=...
    AWS_SECRET_ACCESS_KEY=...
    AWS_S3_RAG_DOCUMENTS_BUCKET=your-bucket
    AWS_DB_URL=postgresql+psycopg://user:pass@host:5432/dbname

    # Observability (recommended)
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=...
    LANGCHAIN_PROJECT=your-langsmith-project-name
    ```

4.  **Download AWS RDS Certificate**

    For secure database connections, download the AWS RDS root certificate bundle.

    ```bash
    wget https://s3.amazonaws.com/rds-downloads/rds-ca-2019-root.pem
    ```

    Place the `rds-ca-2019-root.pem` file in the root of the project directory. It is already included in `.gitignore`.

5.  **Run the API**

    ```bash
    # Kill process on port 8000 if it's running
    lsof -ti :8000 | xargs -r kill -9
    # Start the server
    uvicorn src.app.main:app --reload
    ```

    - Open API docs at http://127.0.0.1:8000/docs

## API Overview

### Chat API

- **`POST /chat`**: Streams model responses over Server-Sent Events (SSE).

  - Manages conversation state using the provided `session_id`.

  - Request body example:
    ```json
    {
      "session_id": "demo-session-1",
      "messages": [
        {
          "role": "user",
          "content": "What is the warranty on the GeoDrone Pro?"
        }
      ]
    }
    ```

### Documents API

Manages documents in an S3 bucket and ensures they are synchronized with the Pinecone vector store.

- **`GET /documents`**: List all documents.
- **`POST /documents`**: Upload a new document.
- **`PUT /documents/{key}`**: Update or replace a document.
- **`DELETE /documents/{key}`**: Delete a document.
- **`POST /documents/sync`**: Manually trigger a full sync between S3 and Pinecone.

### Products API

Provides standard CRUD operations for managing product data in the Postgres database.

- **`GET /products`**: List and filter products.
- **`POST /products`**: Create a new product.
- **`GET /products/{product_id}`**: Get a single product by its ID.
- **`PUT /products/{product_id}`**: Replace a product's data.
- **`PATCH /products/{product_id}`**: Partially update a product.
- **`DELETE /products/{product_id}`**: Delete a product.

## Testing & Evaluation

### Langsmith Tracing & Evaluation

The application is instrumented for Langsmith, providing deep visibility into the execution of the RAG graph. This is invaluable for debugging and performance tuning.

![Langsmith Traces](docs/assets/trace.png)

Additionally, the repository includes a RAG evaluation suite using DeepEval to measure the quality and performance of the retrieval and generation steps.

![Langsmith Evals](docs/assets/langsmith-evals.png)

### Local Testing

- **Unit Tests**: Fast, mocked tests for individual components.
  ```bash
  pytest -q
  ```
- **RAG Evaluation**: Quality and performance metrics using DeepEval.
  - See `evals/EVALUATION.md` for setup and execution details.

## Project Docs

- Architecture: `docs/ARCHITECTURE.md`
- Configuration: `docs/CONFIG.md`
- Testing guide: `docs/TESTING.md`
- Runbook (common issues): `docs/RUNBOOK.md`

## Notes

- The RAG retriever uses Pinecone for hybrid search (dense vectors + sparse BM25 keywords) and VoyageAI for reranking to provide highly relevant context.
