from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, JSON, DateTime, func, Float
import os

DB_SCHEMA = os.getenv("DB_SCHEMA", "public")
Base = declarative_base()


class Product(Base):
    __tablename__ = "products"
    __table_args__ = {"schema": DB_SCHEMA}

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, index=True)
    category = Column(
        String, index=True
    )  # Aircraft, Controller, Payload, Software, Service
    price = Column(Float, nullable=False)
    status = Column(String, index=True)  # current, deprecated, discontinued
    model = Column(String, index=True)
    aliases = Column(JSON)  # ["GDP", "Geo Drone Pro"]
    tags = Column(JSON)  # ["survey", "inspection"]
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, onupdate=func.now())
