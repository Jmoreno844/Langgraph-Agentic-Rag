# src/graph/tools/query_products.py
from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from src.app.features.products.service import list_products


class ProductQuery(BaseModel):
    """Query the products catalog. All filters are optional.

    Parameters:
    - category: filter by product category (exact match)
    - status: filter by product status (e.g., active)
    - model: filter by model identifier/name
    - min_price: minimum price (inclusive)
    - max_price: maximum price (inclusive)
    - search: substring search in product name (case-insensitive)
    - limit: number of items to return (default 20, max 50)
    - offset: pagination offset (default 0)
    """

    category: Optional[str] = Field(default=None)
    status: Optional[str] = Field(default=None)
    model: Optional[str] = Field(default=None)
    min_price: Optional[float] = Field(default=None, ge=0)
    max_price: Optional[float] = Field(default=None, ge=0)
    search: Optional[str] = Field(default=None, min_length=1)
    limit: int = Field(default=20, ge=1, le=50)
    offset: int = Field(default=0, ge=0)


@tool("query_products", args_schema=ProductQuery)
def query_products_tool(
    *,
    category: Optional[str] = None,
    status: Optional[str] = None,
    model: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    search: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> List[dict]:
    """Fetch products from the database according to the provided filters.

    Returns a list of products with fields: id, name, category, price, status, model, aliases, tags, created_at, updated_at.
    """
    products = list_products(
        category=category,
        status=status,
        model=model,
        min_price=min_price,
        max_price=max_price,
        search=search,
        limit=min(limit, 50),
        offset=offset,
    )
    return [p.model_dump() for p in products]


__all__ = ["query_products_tool", "ProductQuery"]
