import os
import csv
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv(override=True)
client = genai.Client()
model = os.environ["GEMINI_MODEL"]


# order product
class Product(BaseModel):
    product: str | None = Field(
        default=None,
        description="Name of the products not the name of the person who is doing the order",
    )
    amount: int | None


class Orders(BaseModel):
    address: str | None = Field(default=None, description="null if not mentioned")
    payment_method: Literal["Cash", "Card", "Credit Card", "Debit Card", "Bank Transfer"] | None = (
        Field(default=None, description="Method of payment")
    )
    order: bool
    products: list[Product]


def parse_orders(order: str) -> Orders:
    """Extract structured data from a free text order"""
    response = client.models.generate_content(
        model=model,
        contents=order,
        config=types.GenerateContentConfig(
            system_instruction="""Extract the product orders. Use only information present in the text.""",
            temperature=0,
            response_mime_type="application/json",
            response_schema=Orders,
        ),
    )

    return response.parsed


if __name__ == "__main__":
    messages = [
        "hey!! can you send 2 beef empanadas and a large coke to 43rd street #10-20, paying cash",
        "Hi, order for Laura: 1 bandeja paisa and 2 blackberry juices. 50th street #30-15 apt 301. Paying by card",
        "can I get 3 arepas with cheese pls, I'll pay by bank transfer",
        "What time do you open tomorrow?",
    ]
    orders = []
    for message in messages:
        # Phase 1: extract structured data from each message.
        orders.append(parse_orders(message))

    with open("orders.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["product_name", "amount", "address", "payment_method"]
        )
        writer.writeheader()
        for order in orders:
            for item in order.products:
                print(item)
                writer.writerow(
                    {
                        "product_name": item.product,
                        "amount": item.amount,
                        "address": order.address,
                        "payment_method": order.payment_method,
                    }
                )

    # valid orders
    valid_orders = 0
    for order in orders:
        if order.order:
            valid_orders += 1

    print(f"Number of valid orders: {valid_orders}")
