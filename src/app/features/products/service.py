from __future__ import annotations

from typing import List, Optional

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Product
from .schemas import ProductCreate, ProductUpdate, ProductOut, DeleteResult


async def list_product_categories(db: AsyncSession) -> List[dict]:
    """Return a list of available product categories with item counts.

    Each item is a dict: {"category": str, "count": int}
    """
    query = (
        select(Product.category, func.count(Product.id))
        .group_by(Product.category)
        .order_by(Product.category.asc())
    )
    rows = (await db.execute(query)).all()
    return [{"category": category, "count": int(count)} for category, count in rows]


def _product_to_out(product: Product) -> ProductOut:
    return ProductOut(
        id=product.id,
        name=product.name,
        category=product.category,
        price=product.price,
        status=product.status,
        model=product.model,
        aliases=product.aliases,
        tags=product.tags,
        created_at=product.created_at,
        updated_at=product.updated_at,
    )


async def list_products(
    *,
    db: AsyncSession,
    category: Optional[str] = None,
    status: Optional[str] = None,
    model: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[ProductOut]:
    query = select(Product)

    if category:
        query = query.where(Product.category == category)
    if status:
        query = query.where(Product.status == status)
    if model:
        query = query.where(Product.model == model)
    if min_price is not None:
        query = query.where(Product.price >= min_price)
    if max_price is not None:
        query = query.where(Product.price <= max_price)
    if search:
        like = f"%{search}%"
        query = query.where(Product.name.ilike(like))

    query = query.order_by(Product.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    products = result.scalars().all()
    return [_product_to_out(p) for p in products]


async def get_product(db: AsyncSession, product_id: int) -> ProductOut:
    product = await db.get(Product, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return _product_to_out(product)


async def create_product(db: AsyncSession, data: ProductCreate) -> ProductOut:
    product = Product(
        name=data.name,
        category=data.category,
        price=data.price,
        status=data.status,
        model=data.model,
        aliases=data.aliases,
        tags=data.tags,
    )
    db.add(product)
    await db.commit()
    await db.refresh(product)
    return _product_to_out(product)


async def update_product(
    db: AsyncSession, product_id: int, data: ProductUpdate
) -> ProductOut:
    product = await db.get(Product, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    if data.name is not None:
        product.name = data.name
    if data.category is not None:
        product.category = data.category
    if data.price is not None:
        product.price = data.price
    if data.status is not None:
        product.status = data.status
    if data.model is not None:
        product.model = data.model
    if data.aliases is not None:
        product.aliases = data.aliases
    if data.tags is not None:
        product.tags = data.tags

    db.add(product)
    await db.commit()
    await db.refresh(product)
    return _product_to_out(product)


async def replace_product(
    db: AsyncSession, product_id: int, data: ProductCreate
) -> ProductOut:
    product = await db.get(Product, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    product.name = data.name
    product.category = data.category
    product.price = data.price
    product.status = data.status
    product.model = data.model
    product.aliases = data.aliases
    product.tags = data.tags

    db.add(product)
    await db.commit()
    await db.refresh(product)
    return _product_to_out(product)


async def delete_product(db: AsyncSession, product_id: int) -> DeleteResult:
    product = await db.get(Product, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    await db.delete(product)
    await db.commit()
    return DeleteResult(id=product_id, deleted=True)
