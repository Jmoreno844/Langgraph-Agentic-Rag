from __future__ import annotations

from pathlib import Path
from alembic import command
from alembic.config import Config

from src.settings import settings


def apply_migrations_safely() -> None:
    """Apply Alembic migrations to the latest head.

    This is safe to run on every startup; Alembic will be a no-op when up-to-date.
    Any errors are logged but do not prevent the app from starting.
    """
    try:
        project_root = Path(__file__).resolve().parents[2]
        alembic_dir = project_root / "alembic"

        cfg = Config()
        cfg.set_main_option("script_location", str(alembic_dir))
        # Ensure the DB URL used by the app is also used for migrations
        cfg.set_main_option("sqlalchemy.url", settings.AWS_DB_URL)

        command.upgrade(cfg, "head")
        print("✅ Alembic migrations applied (or already up-to-date)")
    except Exception as exc:
        print(f"⚠️ Skipping Alembic auto-migration: {exc}")


def ensure_products_table_exists() -> None:
    """Best-effort safety net to ensure `products` table exists.

    If migrations were applied to a different schema/DB or a race occurred, create the
    table using the SQLAlchemy model metadata. This is idempotent.
    """
    try:
        from sqlalchemy import inspect
        from src.db.session import get_engine
        from src.app.features.products.models import Base, Product

        engine = get_engine()
        inspector = inspect(engine)
        schema = Product.__table__.schema or "public"
        tables = inspector.get_table_names(schema=schema)
        if Product.__tablename__ not in tables:
            # Create only the products table within the configured schema
            print(f"⚠️  '{schema}.{Product.__tablename__}' missing. Creating it now...")
            Base.metadata.create_all(bind=engine, tables=[Product.__table__])
            print("✅  Products table created")
        else:
            print("✅  Products table exists")
    except Exception as exc:
        print(f"⚠️  Could not verify/create products table: {exc}")
