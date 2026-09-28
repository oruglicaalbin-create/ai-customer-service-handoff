"""Run the agent on a sample of BANKING77 messages and save the results.

Usage:
  python src/run_agent.py dev            15 messages from the TRAIN split (for prompt checks)
  python src/run_agent.py test 2         2 messages per intent from the TEST split (154 total)
  python src/run_agent.py test 4         4 per intent (308); the first 2 per intent are the same
                                         messages as in "test 2", so they come from the cache

Before any API call it prints how many messages are not cached yet and the
estimated cost, and waits for you to type y.
Output: results/agent_<name>.csv
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

from agent import handle_message, is_cached
from load_data import load_banking77

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CUSTOMER_IDS = [f"C{i:03d}" for i in range(1, 11)]
EST_COST_PER_MESSAGE_USD = 0.0075  # measured average from the demo runs
SEED = 42


def assign_customer(message: str) -> str:
    """'Random' but fixed customer per message: derived from a hash of the text,
    so the same message always gets the same customer, whatever the sample size."""
    digest = hashlib.md5(message.encode()).hexdigest()
    return CUSTOMER_IDS[int(digest, 16) % len(CUSTOMER_IDS)]


def make_sample(split: str, per_intent: int | None, n_total: int | None = None) -> pd.DataFrame:
    df = load_banking77(split).sample(frac=1, random_state=SEED)  # shuffle once, fixed seed
    if per_intent is not None:
        df = df.groupby("intent").head(per_intent)  # first k per intent -> samples are nested
    if n_total is not None:
        df = df.head(n_total)
    df = df.reset_index(drop=True)
    df["customer_id"] = df["text"].map(assign_customer)
    return df


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if mode == "dev":
        sample, name = make_sample("train", per_intent=None, n_total=15), "dev"
    elif mode == "test":
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        sample, name = make_sample("test", per_intent=k), f"test_{k}_per_intent"
    else:
        sys.exit("First argument must be 'dev' or 'test'")

    uncached = sum(not is_cached(t, c) for t, c in zip(sample["text"], sample["customer_id"]))
    estimate = uncached * EST_COST_PER_MESSAGE_USD
    print(f"Sample '{name}': {len(sample)} messages, {uncached} not cached yet.")
    if uncached > 0:
        print(f"Estimated cost: ${estimate:.2f}")
        if input("Continue and pay for the API calls? [y/N] ").strip().lower() != "y":
            sys.exit("Cancelled, nothing was spent.")

    rows = []
    paid = 0.0
    for i, row in sample.iterrows():
        result = handle_message(row["text"], row["customer_id"])
        if not result["from_cache"]:
            paid += result["cost_usd"]
        rows.append({"true_intent": row["intent"], **result})
        print(f"  {i + 1}/{len(sample)}  paid so far ${paid:.3f}", end="\r")
    print()

    results = pd.DataFrame(rows)
    results["correct"] = results["intent"] == results["true_intent"]
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"agent_{name}.csv"
    results.to_csv(out, index=False)

    print(f"Paid in this run: ${paid:.4f}")
    print(f"Intent accuracy: {results['correct'].mean():.1%}")
    print("Decisions:", results["decision"].value_counts().to_dict())
    print(f"Saved to {out.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
