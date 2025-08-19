# src/api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from src.graph.runtime import build_app, cleanup
from src.app.features.documents.api import router as documents_router


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
app.include_router(documents_router)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/chat")
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


class ErrorResponse(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail


@app.exception_handler(ErrorResponse)
async def error_handler(_: Request, exc: ErrorResponse):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
