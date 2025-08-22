from __future__ import annotations

from typing import List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.db.session import get_session
from .models import Product
from .schemas import ProductCreate, ProductUpdate, ProductOut, DeleteResult


def _get_session() -> Session:
    return get_session()


def list_product_categories() -> List[dict]:
    """Return a list of available product categories with item counts.

    Each item is a dict: {"category": str, "count": int}
    """
    session: Session = _get_session()
    try:
        rows = (
            session.query(Product.category, func.count(Product.id))
            .group_by(Product.category)
            .order_by(Product.category.asc())
            .all()
        )
        return [{"category": category, "count": int(count)} for category, count in rows]
    finally:
        session.close()


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


def list_products(
    *,
    category: Optional[str] = None,
    status: Optional[str] = None,
    model: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[ProductOut]:
    session: Session = _get_session()
    try:
        query = session.query(Product)

        if category:
            query = query.filter(Product.category == category)
        if status:
            query = query.filter(Product.status == status)
        if model:
            query = query.filter(Product.model == model)
        if min_price is not None:
            query = query.filter(Product.price >= min_price)
        if max_price is not None:
            query = query.filter(Product.price <= max_price)
        if search:
            like = f"%{search}%"
            query = query.filter(Product.name.ilike(like))

        products = (
            query.order_by(Product.created_at.desc()).offset(offset).limit(limit).all()
        )
        return [_product_to_out(p) for p in products]
    finally:
        session.close()


def get_product(product_id: int) -> ProductOut:
    session: Session = _get_session()
    try:
        product = session.get(Product, product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        return _product_to_out(product)
    finally:
        session.close()


def create_product(data: ProductCreate) -> ProductOut:
    session: Session = _get_session()
    try:
        product = Product(
            name=data.name,
            category=data.category,
            price=data.price,
            status=data.status,
            model=data.model,
            aliases=data.aliases,
            tags=data.tags,
        )
        session.add(product)
        session.commit()
        session.refresh(product)
        return _product_to_out(product)
    finally:
        session.close()


def update_product(product_id: int, data: ProductUpdate) -> ProductOut:
    session: Session = _get_session()
    try:
        product = session.get(Product, product_id)
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

        session.add(product)
        session.commit()
        session.refresh(product)
        return _product_to_out(product)
    finally:
        session.close()


def replace_product(product_id: int, data: ProductCreate) -> ProductOut:
    session: Session = _get_session()
    try:
        product = session.get(Product, product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        product.name = data.name
        product.category = data.category
        product.price = data.price
        product.status = data.status
        product.model = data.model
        product.aliases = data.aliases
        product.tags = data.tags

        session.add(product)
        session.commit()
        session.refresh(product)
        return _product_to_out(product)
    finally:
        session.close()


def delete_product(product_id: int) -> DeleteResult:
    session: Session = _get_session()
    try:
        product = session.get(Product, product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        session.delete(product)
        session.commit()
        return DeleteResult(id=product_id, deleted=True)
    finally:
        session.close()
