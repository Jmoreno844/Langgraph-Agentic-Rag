from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ProductBase(BaseModel):
    name: str = Field(..., description="Human-friendly product name")
    category: Optional[str] = Field(
        None,
        description="High-level category, e.g., Aircraft, Controller, Payload, Software, Service",
    )
    price: float = Field(..., ge=0, description="Unit price in the store currency")
    status: Optional[str] = Field(
        None, description="Lifecycle state, e.g., current, deprecated, discontinued"
    )
    model: Optional[str] = Field(None, description="Model identifier or SKU")
    aliases: Optional[List[str]] = Field(
        None, description="Alternative names or abbreviations"
    )
    tags: Optional[List[str]] = Field(None, description="Freeform tags for filtering")


class ProductCreate(ProductBase):
    """Payload to create a product."""


class ProductUpdate(BaseModel):
    """Payload to update a product (all fields optional)."""

    name: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = Field(None, ge=0)
    status: Optional[str] = None
    model: Optional[str] = None
    aliases: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class ProductOut(ProductBase):
    id: int = Field(..., description="Database identifier")
    created_at: datetime = Field(..., description="Creation timestamp (UTC)")
    updated_at: Optional[datetime] = Field(
        None, description="Last update timestamp (UTC) if updated"
    )


class DeleteResult(BaseModel):
    id: int = Field(..., description="Deleted product id")
    deleted: bool = Field(..., description="Whether the product was deleted")
