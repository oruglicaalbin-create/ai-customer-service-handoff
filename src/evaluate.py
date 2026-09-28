"""Step 6: evaluate the agent against the baseline. No API calls, reads saved results.

Run: python src/evaluate.py            (uses results/agent_test_2_per_intent.csv)
     python src/evaluate.py 4          (uses the 4-per-intent run, if it exists)
Output: printed report + results/evaluation_metrics.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

from grounding import ungrounded_numbers, ungrounded_paths
from policy import must_handoff

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

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

# --- 4. Grounding screens: numbers and app menu paths ----------------------
# (screens only: they cannot see invented procedures written in plain words)
ungrounded, invented_paths = [], []
for _, r in both.iterrows():
    extra = ungrounded_numbers(r["customer_reply"], r["message"], r["customer_id"])
    if extra:
        ungrounded.append((r["message"], r["customer_id"], extra, r["customer_reply"]))
    paths = ungrounded_paths(r["customer_reply"])
    if paths:
        invented_paths.append((r["message"], r["decision"], paths))
print(f"\n=== 4. Grounding screens ===")
print(f"  replies with numbers not found in facts, customer data or message: {len(ungrounded)}/{n}")
for message, cid, extra, reply in ungrounded:
    print(f"    [{cid}] {message}\n       unknown numbers {extra}: {reply[:160]}")
print(f"  replies with app menu paths not in the fact sheet: {len(invented_paths)}/{n}")
for message, decision, paths in invented_paths:
    print(f"    {paths} ({decision}) <- '{message}'")

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
    "replies_with_invented_app_paths": len(invented_paths),
    "invented_app_paths_in_resolve_decisions": sum(d == "resolve" for _, d, _ in invented_paths),
}
(RESULTS_DIR / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
