# src/graph/tools/list_product_categories.py
from __future__ import annotations

from typing import List
from langchain_core.tools import tool

from src.app.features.products.service import list_product_categories
from src.db.session import get_async_db


@tool("list_product_categories")
async def list_product_categories_tool() -> List[dict]:
    """List available product categories with the number of items in each.

    Returns a list of {category: string, count: number}.
    """
    async with get_async_db() as db:
        return await list_product_categories(db=db)


__all__ = ["list_product_categories_tool"]
