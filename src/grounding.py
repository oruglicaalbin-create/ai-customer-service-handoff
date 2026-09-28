"""Automatic screens for invented facts in replies. No API calls.

Two checks:
  - numbers: every number in a reply must appear in the bank fact sheet, the
    customer's own data, or the customer's message.
  - app paths: every menu path like "Cards > Unblock PIN" must appear in the fact sheet.

These are screens, not proofs: they cannot see invented procedures or promises
written in plain words, and small numbers can match by coincidence.
"""
import json
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FACTS = (DATA_DIR / "bank_facts.md").read_text(encoding="utf-8")
_CUSTOMERS = json.loads((DATA_DIR / "customers.json").read_text(encoding="utf-8"))
_CARDS = json.loads((DATA_DIR / "cards.json").read_text(encoding="utf-8"))
_TRANSACTIONS = pd.read_csv(DATA_DIR / "transactions.csv")
_CAP = r"[A-Z][a-zA-Z]*(?: [A-Z][a-zA-Z]*)*"  # Capitalised words, e.g. "Unblock PIN"


def numbers(text: str) -> set[str]:
    """All numbers in a text, normalised: '10,000' and '10 000' -> '10000', '129,90' -> '129.9'."""
    found = re.findall(r"\d{1,3}(?:[, ]\d{3})+(?![\d,.])|\d+(?:[.,]\d+)?", str(text))
    normalised = set()
    for x in found:
        if re.fullmatch(r"\d{1,3}(?:[, ]\d{3})+", x):
            x = x.replace(",", "").replace(" ", "")  # thousands separators
        normalised.add(str(float(x.replace(",", "."))).rstrip("0").rstrip("."))
    return normalised


_FACT_NUMBERS = numbers(FACTS)


def _customer_numbers(customer_id: str) -> set[str]:
    data = [c for c in _CUSTOMERS if c["customer_id"] == customer_id]
    data += [c for c in _CARDS if c["customer_id"] == customer_id]
    tx = _TRANSACTIONS[_TRANSACTIONS["customer_id"] == customer_id]
    found = numbers(json.dumps(data) + tx.to_csv(index=False))
    return found | {n.lstrip("-") for n in found}


def ungrounded_numbers(reply: str, message: str, customer_id: str) -> list[str]:
    allowed = _FACT_NUMBERS | _customer_numbers(customer_id) | numbers(message)
    return sorted(numbers(reply) - allowed)


def ungrounded_paths(reply: str) -> list[str]:
    paths = [re.sub(r"\s*>\s*", " > ", m.group(0))
             for m in re.finditer(rf"{_CAP}(?:\s*>\s*{_CAP})+", str(reply))]
    return sorted({p for p in paths if p not in FACTS})
