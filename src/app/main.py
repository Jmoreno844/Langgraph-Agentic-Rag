# src/api/main.py
from dotenv import load_dotenv

load_dotenv(override=True)
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.graph.runtime import build_app, cleanup, build_app_async, acleanup

# Temporarily disable problematic routers
# from src.app.features.documents.api import router as documents_router
from src.app.features.products.api import router as products_router
from src.app.features.documents.api import router as documents_router
from src.app.core.guardrails_setup import initialize_guardrails
from src.app.features.chat.api import router as chat_router
from src.db.automigrate import apply_migrations_safely, ensure_products_table_exists


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_guardrails()

    # Apply DB migrations automatically before building the app
    apply_migrations_safely()
    ensure_products_table_exists()

    # Prefer async builder (AsyncPostgresSaver) with fallback to sync
    app.state.graph_app = await build_app_async()
    yield
    # Clean up database connection on shutdown
    await acleanup()
    cleanup()


app = FastAPI(lifespan=lifespan)
app.include_router(documents_router)
app.include_router(products_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    return {"message": "Hello World - Basic server running!"}


class ErrorResponse(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail


@app.exception_handler(ErrorResponse)
async def error_handler(_: Request, exc: ErrorResponse):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
