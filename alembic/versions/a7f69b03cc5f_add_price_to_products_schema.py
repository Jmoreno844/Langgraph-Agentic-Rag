"""add price to  products_schema

Revision ID: a7f69b03cc5f
Revises: 0001_create_products_table
Create Date: 2025-08-22 11:35:19.313965
"""

from alembic import op
import sqlalchemy as sa
import os

# revision identifiers, used by Alembic.
revision = "a7f69b03cc5f"
down_revision = "0001_create_products_table"
branch_labels = None
depends_on = None

DB_SCHEMA = os.getenv("DB_SCHEMA", "public")


def upgrade() -> None:
    op.add_column(
        "products", sa.Column("price", sa.Float(), nullable=False), schema=DB_SCHEMA
    )


def downgrade() -> None:
    op.drop_column("products", "price", schema=DB_SCHEMA)
