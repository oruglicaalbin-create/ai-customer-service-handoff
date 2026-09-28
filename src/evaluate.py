"""Step 6: evaluate the agent against the baseline. No API calls, reads saved results.

Run: python src/evaluate.py            (uses results/agent_test_2_per_intent.csv)
     python src/evaluate.py 4          (uses the 4-per-intent run, if it exists)
Output: printed report + results/evaluation_metrics.json
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

from policy import must_handoff

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"

k = sys.argv[1] if len(sys.argv) > 1 else "2"
agent = pd.read_csv(RESULTS_DIR / f"agent_test_{k}_per_intent.csv")
baseline_all = pd.read_csv(RESULTS_DIR / "baseline_test_predictions.csv")

# Baseline predictions for exactly the same messages (join on text + true intent).
baseline = baseline_all.drop_duplicates(["text", "true_intent"]).rename(columns={"text": "message"})
both = agent.merge(
    baseline[["message", "true_intent", "predicted_intent", "confidence", "correct"]],
    on=["message", "true_intent"], suffixes=("_llm", "_baseline"),
)
assert len(both) == len(agent), "every agent message should have a baseline prediction"
n = len(both)


def calibration(conf: pd.Series, correct: pd.Series) -> pd.DataFrame:
    bands = pd.cut(conf, [0, 0.5, 0.7, 0.9, 1.0], include_lowest=True)
    return pd.DataFrame({"correct": correct, "band": bands}).groupby("band", observed=True)["correct"].agg(
        accuracy="mean", messages="size")


# --- 1. Intent accuracy and cost -------------------------------------------
llm_acc = both["correct_llm"].mean()
base_acc = both["correct_baseline"].mean()
llm_cost = both["cost_usd"].mean()
print(f"=== 1. Intent accuracy on the same {n} test messages ===")
print(f"  LLM (Claude Haiku 4.5):   {llm_acc:.1%}   cost ${llm_cost:.4f}/message")
print(f"  Baseline (TF-IDF + LR):   {base_acc:.1%}   cost ~$0")
print(f"  (baseline on all 3080 test messages: {baseline_all['correct'].mean():.1%})")
both_right = (both["correct_llm"] & both["correct_baseline"]).mean()
only_llm = (both["correct_llm"] & ~both["correct_baseline"]).mean()
only_base = (~both["correct_llm"] & both["correct_baseline"]).mean()
print(f"  both right {both_right:.1%} | only LLM right {only_llm:.1%} | only baseline right {only_base:.1%}")

# --- 2. Is the confidence trustworthy? (calibration) -----------------------
print("\n=== 2. Calibration: accuracy within each confidence band ===")
print("LLM (self-reported confidence):")
print(calibration(both["confidence_llm"], both["correct_llm"]).round(3).to_string())
print("Baseline (model probability):")
print(calibration(both["confidence_baseline"], both["correct_baseline"]).round(3).to_string())
print("Distinct LLM confidence values:", sorted(float(x) for x in both["confidence_llm"].round(2).unique()))

# --- 3. Decisions -----------------------------------------------------------
print("\n=== 3. Decisions ===")
print("  ", both["decision"].value_counts().to_dict())
print("   Intent accuracy by decision:", both.groupby("decision")["correct_llm"].mean().round(3).to_dict())
overrides = both["decision_source"].str.startswith("policy").sum()
print(f"   Decisions changed by the policy rule (LLM did not hand off by itself): {overrides}")

# Safety: messages whose TRUE intent must go to a human. Did they reach a human?
must = both[both["true_intent"].map(must_handoff)]
reached_human = (must["decision"] == "handoff").mean()
not_handed_off = must[must["decision"] != "handoff"]
print(f"   Must-handoff messages (by true intent): {len(must)}; handed off {reached_human:.1%}; "
      f"not handed off: {len(not_handed_off)}")
for _, r in not_handed_off.iterrows():
    print(f"     ! '{r['message']}'  true={r['true_intent']} pred={r['intent']} decision={r['decision']}")

# --- 4. Grounding check: numbers in replies must come from facts or data ----
def numbers(text: str) -> set[str]:
    """All numbers in a text, normalised: '10,000' and '10 000' -> '10000', '129,90' -> '129.9'."""
    found = re.findall(r"\d{1,3}(?:[, ]\d{3})+(?![\d,.])|\d+(?:[.,]\d+)?", str(text))
    normalised = set()
    for x in found:
        if re.fullmatch(r"\d{1,3}(?:[, ]\d{3})+", x):
            x = x.replace(",", "").replace(" ", "")  # thousands separators
        normalised.add(str(float(x.replace(",", "."))).rstrip("0").rstrip("."))
    return normalised


facts_numbers = numbers((DATA_DIR / "bank_facts.md").read_text(encoding="utf-8"))
customers = json.loads((DATA_DIR / "customers.json").read_text(encoding="utf-8"))
cards = json.loads((DATA_DIR / "cards.json").read_text(encoding="utf-8"))
transactions = pd.read_csv(DATA_DIR / "transactions.csv")


def customer_numbers(cid: str) -> set[str]:
    data = [c for c in customers if c["customer_id"] == cid] + [c for c in cards if c["customer_id"] == cid]
    tx = transactions[transactions["customer_id"] == cid]
    text = json.dumps(data) + tx.to_csv(index=False)
    return numbers(text) | {n.lstrip("-") for n in numbers(text)}


ungrounded = []
for _, r in both.iterrows():
    allowed = facts_numbers | customer_numbers(r["customer_id"]) | numbers(r["message"])
    extra = numbers(r["customer_reply"]) - allowed
    if extra:
        ungrounded.append((r["message"], r["customer_id"], sorted(extra), r["customer_reply"]))
print(f"\n=== 4. Grounding: replies containing numbers not found in facts or customer data: "
      f"{len(ungrounded)}/{n} ===")
for message, cid, extra, reply in ungrounded:
    print(f"  [{cid}] {message}\n     unknown numbers {extra}: {reply[:160]}")

# --- 5. Most common LLM confusions -----------------------------------------
wrong = both[~both["correct_llm"]]
print("\n=== 5. Most common LLM confusions (true -> predicted) ===")
print(wrong.groupby(["true_intent", "intent"]).size().sort_values(ascending=False).head(10).to_string())

metrics = {
    "messages": n,
    "llm_accuracy": round(llm_acc, 4),
    "baseline_accuracy_same_messages": round(base_acc, 4),
    "baseline_accuracy_full_test": round(baseline_all["correct"].mean(), 4),
    "llm_cost_per_message_usd": round(llm_cost, 5),
    "llm_total_cost_usd": round(both["cost_usd"].sum(), 4),
    "decisions": both["decision"].value_counts().to_dict(),
    "policy_overrides": int(overrides),
    "must_handoff_messages": len(must),
    "must_handoff_reached_human": round(reached_human, 4),
    "replies_with_ungrounded_numbers": len(ungrounded),
}
(RESULTS_DIR / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
