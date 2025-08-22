# src/graph/tools/list_product_categories.py
from __future__ import annotations

from typing import List
from langchain_core.tools import tool

from src.app.features.products.service import list_product_categories


@tool("list_product_categories")
def list_product_categories_tool() -> List[dict]:
    """List available product categories with the number of items in each.

    Returns a list of {category: string, count: number}.
    """
    return list_product_categories()


__all__ = ["list_product_categories_tool"]
