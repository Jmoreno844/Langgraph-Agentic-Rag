# src/api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.graph.runtime import build_app, cleanup


class ChatRequest(BaseModel):
    session_id: str
    messages: list  # [{role, content}]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph_app = build_app()
    yield
    # Clean up database connection on shutdown
    cleanup()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/v1/chat")
async def chat(body: ChatRequest):
    payload = {
        "messages": body.messages[-1:],
        "has_been_rewritten": False,
    }
    config = {"configurable": {"thread_id": body.session_id}}

    def gen():
        # initial comment to establish SSE
        yield "event: start\n\n"
        for chunk in app.state.graph_app.stream(payload, config=config):
            yield f"data: {str(chunk)}\n\n"
        # explicit end event
        yield "event: end\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
