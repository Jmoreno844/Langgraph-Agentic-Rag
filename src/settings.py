# src/settings.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    AWS_DB_URL: str  # e.g. postgresql://postgres:...@db.<proj>.supabase.co:5432/postgres?sslmode=require

    class Config:
        extra = "allow"
        env_file = ".env"


settings = Settings()
