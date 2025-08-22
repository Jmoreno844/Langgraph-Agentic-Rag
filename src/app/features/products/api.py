from fastapi import APIRouter, Query, Path, status
from typing import Optional, List

from .schemas import ProductOut, ProductCreate, ProductUpdate, DeleteResult
from .service import (
    list_products as svc_list_products,
    get_product as svc_get_product,
    create_product as svc_create_product,
    update_product as svc_update_product,
    replace_product as svc_replace_product,
    delete_product as svc_delete_product,
)

router = APIRouter()


@router.get(
    "/products",
    response_model=List[ProductOut],
    response_model_exclude_none=True,
    summary="List products",
    tags=["products"],
)
async def get_products(
    category: Optional[str] = Query(None, description="Filter by category"),
    status_: Optional[str] = Query(
        None, alias="status", description="Filter by status"
    ),
    model: Optional[str] = Query(None, description="Filter by model"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price"),
    search: Optional[str] = Query(
        None, description="Case-insensitive name contains search"
    ),
    limit: int = Query(100, ge=1, le=500, description="Max items to return"),
    offset: int = Query(0, ge=0, description="Items to skip before collecting results"),
):
    return svc_list_products(
        category=category,
        status=status_,
        model=model,
        min_price=min_price,
        max_price=max_price,
        search=search,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/products/{product_id}",
    response_model=ProductOut,
    response_model_exclude_none=True,
    summary="Get a product by id",
    tags=["products"],
)
async def get_product(
    product_id: int = Path(..., ge=1, description="Product id"),
):
    return svc_get_product(product_id)


@router.post(
    "/products",
    response_model=ProductOut,
    status_code=status.HTTP_201_CREATED,
    response_model_exclude_none=True,
    summary="Create a new product",
    tags=["products"],
)
async def create_product(payload: ProductCreate):
    return svc_create_product(payload)


@router.put(
    "/products/{product_id}",
    response_model=ProductOut,
    response_model_exclude_none=True,
    summary="Replace a product (full update)",
    tags=["products"],
)
async def replace_product(
    product_id: int = Path(..., ge=1, description="Product id"),
    payload: ProductCreate = ...,
):
    return svc_replace_product(product_id, payload)


@router.patch(
    "/products/{product_id}",
    response_model=ProductOut,
    response_model_exclude_none=True,
    summary="Update a product (partial)",
    tags=["products"],
)
async def update_product(
    product_id: int = Path(..., ge=1, description="Product id"),
    payload: ProductUpdate = ...,
):
    return svc_update_product(product_id, payload)


@router.delete(
    "/products/{product_id}",
    response_model=DeleteResult,
    summary="Delete a product",
    tags=["products"],
)
async def delete_product(
    product_id: int = Path(..., ge=1, description="Product id"),
):
    return svc_delete_product(product_id)
