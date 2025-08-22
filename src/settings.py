# src/settings.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    AWS_DB_URL: str  # e.g. postgresql://postgres:...@db.<proj>.supabase.co:5432/postgres?sslmode=require
    VOYAGE_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_INDEX: str
    PINECONE_EMBEDDINGS_MODEL: str = "llama-text-embed-v2"
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_S3_RAG_DOCUMENTS_BUCKET: str

    class Config:
        extra = "allow"
        env_file = ".env"


settings = Settings()
