# Architecture

## High-level flow

- User sends a message to `/chat`.
- The LangGraph app runs the stateful graph with an `AsyncPostgresSaver` checkpointer.
- `generate_answer_or_rag` decides whether to answer directly or route to retrieval via a tool call.
- If tools are requested, the `tools` node runs one or more tools (e.g., `retriever_tool`, `query_products_tool`).
- After tools run, `extract_context` processes their output.
- Finally, `generate_answer` composes a prompt with retrieved context and returns the final answer.

## Components

- **Graph**: `src/graph/graph.py`
  - Nodes: `generate_answer_or_rag`, `tools` (ToolNode), `extract_context`, `generate_answer`.
  - Conditional routing via `tools_condition`.
  - State type: `CustomMessagesState` with `has_been_rewritten` flag.
- **Nodes**
  - `src/graph/nodes/generate_answer_or_rag.py`
    - Model: `openai:gpt-4o-mini`.
    - Prepends a `SystemMessage` with detailed instructions when the input has not been rewritten.
    - Binds multiple tools (`retriever_tool`, `query_products_tool`, `list_product_categories_tool`) for potential use.
  - `src/graph/nodes/generate_answer.py`
    - Model: `openai:gpt-4o-mini`.
    - Builds a concise QA prompt from the question and retrieved context.
- **Tools**
  - `src/graph/tools/hybrid_retriever.py`: The `retriever_tool` implements a sophisticated retrieval strategy:
    1. **Hybrid Search**: It combines results from both dense (vector) search and sparse (BM25 keyword) search to ensure both semantic relevance and keyword matching.
    2. **Reranking**: The combined search results are then passed to a `VoyageAIRerank` model, which re-orders the documents to prioritize the most relevant chunks for the final answer.
  - `src/graph/tools/query_products.py`: `query_products_tool` queries the product database based on filters.
  - `src/graph/tools/list_product_categories.py`: `list_product_categories_tool` returns available product categories.
- **Vector Store & Ingestion**
  - `src/services/vectorstores/pinecone_service.py`: The default vector store used for document retrieval.
  - `src/app/features/documents/service.py`: Handles S3 document uploads, chunking, embedding, and upserting into Pinecone.
- **Runtime / API**
  - `src/graph/runtime.py`: Compiles the graph with `AsyncPostgresSaver`, which supports persistent, stateful conversations for async streams. It includes a fallback to an in-memory saver if the database is unavailable.
  - `src/app/main.py`: FastAPI app; mounts `/chat` and S3-backed `/documents` endpoints.

## Observability & Evaluation

- **Tracing**: Langsmith is used for detailed, real-time tracing of graph execution, prompts, and tool calls.
- **Evaluation**: DeepEval is used for RAG quality evaluation, measuring faithfulness, answer relevancy, and contextual recall/relevancy against a synthetic dataset. See `evals/README.md`.

## External services

- OpenAI (chat + embeddings)
- VoyageAI (reranking)
- AWS S3 (document storage)
- Postgres (LangGraph checkpointer)
- Langsmith (tracing and observability)
- DeepEval (RAG evaluation)
- Pinecone (vector store)

## Data flow (RAG)

1.  Documents are uploaded via the `/documents` API and stored in S3 (`AWS_S3_RAG_DOCUMENTS_BUCKET`).
2.  The service splits documents into chunks, embeds them (OpenAI), and upserts them into a Pinecone index.
3.  At runtime, `retriever_tool` surfaces relevant chunks from Pinecone.
4.  `generate_answer` uses these chunks to ground its answer.
