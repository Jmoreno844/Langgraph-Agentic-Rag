import os
import json
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.features.products.models import Product, Base
from settings import settings


def create_session():
    """Create and return a new SQLAlchemy session."""
    engine = create_engine(settings.AWS_DB_URL, echo=False)
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    return DBSession()


def parse_product_data(file_path):
    """Parse the product data from the given file."""
    with open(file_path, "r") as f:
        content = f.read()

    products = []
    product_entries = content.strip().split("\n\n")

    for entry in product_entries:
        product_data = {}
        lines = entry.strip().split("\n")
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                if key == "product":
                    product_data["name"] = value
                elif key == "category":
                    product_data["category"] = value
                elif key == "status":
                    product_data["status"] = value
                elif key == "model":
                    product_data["model"] = value
                elif key == "price":
                    product_data["price"] = float(value)
                elif key == "aliases":
                    product_data["aliases"] = json.loads(value)
                elif key == "tags":
                    product_data["tags"] = json.loads(value)

        if "name" in product_data and "price" in product_data:
            products.append(product_data)

    return products


def seed_products(session, products_data):
    """Seed the database with product data."""
    # Clear existing data
    session.query(Product).delete()

    for data in products_data:
        product = Product(**data)
        session.add(product)
    session.commit()


def main():
    """Main function to run the seeder."""
    session = create_session()
    file_path = "tests/data/product_list.txt"
    products_data = parse_product_data(file_path)
    seed_products(session, products_data)
    print(f"Seeded {len(products_data)} products.")


if __name__ == "__main__":
    main()
