"""Create products table

Revision ID: 0001_create_products_table
Revises:
Create Date: 2024-08-23 11:20:00.000000

"""

from alembic import op
import sqlalchemy as sa
import os

# revision identifiers, used by Alembic.
revision = "0001_create_products_table"
down_revision = None
branch_labels = None
depends_on = None

DB_SCHEMA = os.getenv("DB_SCHEMA", "public")


def upgrade() -> None:
    op.create_table(
        "products",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("aliases", sa.JSON(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        schema=DB_SCHEMA,
    )
    op.create_index(
        op.f("ix_public_products_category"),
        "products",
        ["category"],
        unique=False,
        schema=DB_SCHEMA,
    )
    op.create_index(
        op.f("ix_public_products_model"),
        "products",
        ["model"],
        unique=False,
        schema=DB_SCHEMA,
    )
    op.create_index(
        op.f("ix_public_products_name"),
        "products",
        ["name"],
        unique=False,
        schema=DB_SCHEMA,
    )
    op.create_index(
        op.f("ix_public_products_status"),
        "products",
        ["status"],
        unique=False,
        schema=DB_SCHEMA,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_public_products_status"), table_name="products", schema=DB_SCHEMA
    )
    op.drop_index(
        op.f("ix_public_products_name"), table_name="products", schema=DB_SCHEMA
    )
    op.drop_index(
        op.f("ix_public_products_model"), table_name="products", schema=DB_SCHEMA
    )
    op.drop_index(
        op.f("ix_public_products_category"), table_name="products", schema=DB_SCHEMA
    )
    op.drop_table("products", schema=DB_SCHEMA)
