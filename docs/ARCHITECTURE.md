# Architecture

## High-level flow

- User sends a message to `/chat`.
- The LangGraph app runs the stateful graph with a Postgres checkpointer.
- `generate_answer_or_rag` decides whether to answer directly or route to retrieval via a tool.
- If tools are requested, the `retriever` node runs; then `generate_answer` composes a prompt with retrieved context and returns the final answer.

## Components

- Graph: `src/graph/graph.py`
  - Nodes: `generate_answer_or_rag`, `retriever` (ToolNode), `generate_answer`
  - Conditional routing via `tools_condition`
  - State type: `CustomMessagesState` with `has_been_rewritten` flag
- Nodes
  - `src/graph/nodes/generate_answer_or_rag.py`
    - Model: `openai:gpt-4o-mini`
    - Prepends a `SystemMessage` when the input has not been rewritten
    - Binds the retriever tool for potential tool-use
  - `src/graph/nodes/generate_answer.py`
    - Model: `openai:gpt-4o-mini`
    - Builds a concise QA prompt from the question and retrieved context
  - `src/graph/nodes/rewrite_question.py`
    - Model: `openai:gpt-4.1` (optional, currently not wired into the default graph)
- Retriever Tool
  - `src/graph/tools/retriever.py` → uses `in_memory` retriever
  - `src/graph/vectorstores/in_memory.py`
    - `InMemoryVectorStore` + `OpenAIEmbeddings`
    - Reranking with VoyageAI (`rerank-lite-1`)
- Ingestion
  - `src/graph/ingestion/loader_splitter.py`: loads raw docs from S3 and splits into chunks (RecursiveCharacterTextSplitter, chunk_size=100, overlap=50)
- Runtime / API
  - `src/graph/runtime.py`: compiles the graph with `PostgresSaver` (requires `AWS_DB_URL`)
  - `src/app/main.py`: FastAPI app; mounts `/chat` and S3-backed `/documents` endpoints

## External services

- OpenAI (chat + embeddings)
- VoyageAI (reranking)
- AWS S3 (document storage)
- Postgres (LangGraph checkpointer)

## Data flow (RAG)

1. Documents reside in S3 (`AWS_S3_RAG_DOCUMENTS_BUCKET`).
2. Loader pulls and splits documents into chunks.
3. Chunks are embedded (OpenAI) and stored in an in-memory vector store.
4. At runtime, the retriever surfaces relevant chunks; `generate_answer` uses them to answer the question.

> Note: A Pinecone-based vector store exists (`src/graph/vectorstores/pinecone_vectorstore.py`) but is not used by default.
