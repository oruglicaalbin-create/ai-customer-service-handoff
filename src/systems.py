"""Simulated bank back-end systems (fake data only).

In a real bank these would be separate systems (CRM, card platform, core
banking), each behind its own API. Here each is a small file in data/, and each
function below plays the role of one API call. In step 5 these functions become
the "tools" the LLM can call.

Every function returns plain dicts/lists (easy to turn into JSON for the LLM)
and returns {"error": ...} instead of crashing, so the assistant can react to
a missing customer the way a human agent would.
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_json(filename: str) -> list[dict]:
    return json.loads((DATA_DIR / filename).read_text(encoding="utf-8"))


def get_customer(customer_id: str) -> dict:
    """CRM system: basic customer profile and account status."""
    for customer in _load_json("customers.json"):
        if customer["customer_id"] == customer_id:
            return customer
    return {"error": f"No customer with id {customer_id}"}


def get_cards(customer_id: str) -> dict:
    """Card system: all cards of a customer with their status."""
    cards = [c for c in _load_json("cards.json") if c["customer_id"] == customer_id]
    if not cards:
        return {"error": f"No cards found for customer {customer_id}"}
    return {"customer_id": customer_id, "cards": cards}


def get_transactions(customer_id: str, status: str | None = None, limit: int = 10) -> dict:
    """Core banking: the customer's most recent transactions, newest first.

    status: optional filter, one of completed, pending, declined, reverted, failed.
    """
    df = pd.read_csv(DATA_DIR / "transactions.csv")
    df = df[df["customer_id"] == customer_id]
    if status is not None:
        df = df[df["status"] == status]
    df = df.sort_values("date", ascending=False).head(limit)
    return {"customer_id": customer_id, "transactions": df.to_dict(orient="records")}


if __name__ == "__main__":
    # Demo: what the assistant would see for a few scenarios.
    print("C002 (was charged twice?):")
    for t in get_transactions("C002")["transactions"]:
        print(f"  {t['date']}  {t['description']:<20} {t['amount_eur']:>8.2f}  {t['status']}")

    print("\nC008 (where is my card?):")
    for c in get_cards("C008")["cards"]:
        print(f"  {c['type']:<9} *{c['last4']}  {c['status']:<12} delivery: {c['delivery_estimate']}")

    print("\nC010 declined transactions only:", get_transactions("C010", status="declined"))
    print("\nUnknown customer:", get_customer("C999"))
