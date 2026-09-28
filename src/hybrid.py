"""The recommended hybrid, run end to end on the same 154 test messages.

  1. The classifier (TF-IDF + LR) picks the intent and gives a confidence.
  2. Policy intents, and messages below the threshold, go to a human (cost c2).
  3. The rest go to the LLM, which is TOLD the classifier's intent; it gathers
     the customer's data and resolves, asks one question, or hands off.
  4. An LLM judge (src/judge.py) grades every resolve/clarify reply.

Cost of each outcome (€, assumptions in the constants below):
  human (routed by the classifier)            c2
  LLM hands off by itself                     c2 + c_auto
  resolve, judged good                        c_auto
  resolve, judged bad                         c1 (c1_safety if the true intent must go to a human) + c_auto
  clarify, judged necessary                   c_clarify + c_auto
  clarify, judged unnecessary                 c_clarify + c2 + c_auto  (a pointless round, then a human)

The threshold 0.62 was fixed beforehand on out-of-fold training data (src/threshold.py),
so it is not tuned on these messages. Other thresholds are shown for context.

Usage: python src/hybrid.py      (asks before spending; cached results are free)
Output: results/hybrid_154.csv, results/hybrid_metrics.json
"""
import json
import sys

import numpy as np
import pandas as pd

from agent import handle_message_with_intent, is_cached_hybrid
from grounding import ungrounded_numbers, ungrounded_paths
from judge import is_judged, judge_reply
from policy import must_handoff
from threshold import C1, C1_SAFETY, C2, C_AUTO, RESULTS_DIR

C_CLARIFY = 2.0          # € an extra automated round with the customer (assumption; sensitivity below)
PRESET_THRESHOLD = 0.62  # chosen on out-of-fold training data with the costs above
FLOOR = 0.545            # run the LLM on everything the classifier would automate at any t >= FLOOR
EST_LLM_USD, EST_JUDGE_USD = 0.0075, 0.006


def outcome_cost(row, t, c_clarify=C_CLARIFY):
    if row["rule_handoff"] or row["confidence"] < t:
        return C2
    decision = row["decision"]
    if decision == "handoff":
        return C2 + C_AUTO
    good = row["verdict"] == "good"
    if decision == "resolve":
        return C_AUTO if good else (C1_SAFETY if row["safety"] else C1) + C_AUTO
    return c_clarify + C_AUTO + (0 if good else C2)  # clarify


def summarise(df, t, c_clarify=C_CLARIFY):
    automated = ~df["rule_handoff"] & (df["confidence"] >= t)
    llm = df[automated]
    return {
        "threshold": t,
        "cost_per_message_eur": round(float(df.apply(outcome_cost, axis=1, t=t, c_clarify=c_clarify).mean()), 3),
        "sent_to_llm": round(float(automated.mean()), 3),
        "resolved_good": int(((llm["decision"] == "resolve") & (llm["verdict"] == "good")).sum()),
        "resolved_bad": int(((llm["decision"] == "resolve") & (llm["verdict"] == "bad")).sum()),
        "clarify_good": int(((llm["decision"] == "clarify") & (llm["verdict"] == "good")).sum()),
        "clarify_bad": int(((llm["decision"] == "clarify") & (llm["verdict"] == "bad")).sum()),
        "llm_handoff": int((llm["decision"] == "handoff").sum()),
        "safety_misroutes": int((automated & df["safety"]).sum()),
        "fully_automated_share": round(float(((automated) & (df["decision"] == "resolve")
                                              & (df["verdict"] == "good")).mean()), 3),
    }


def judge_sensitivity(df):
    """Does the conclusion depend on the judge? Recompute under more lenient readings.
    Manual overrides (results/judge_overrides.csv) are verdicts a human reviewer disagreed with."""
    lenient_clarify = df.copy()
    lenient_clarify.loc[lenient_clarify["decision"] == "clarify", "verdict"] = "good"
    overrides = pd.read_csv(RESULTS_DIR / "judge_overrides.csv")
    with_overrides = lenient_clarify.copy()
    for o in overrides.itertuples():
        mask = (with_overrides["message"] == o.message) & (with_overrides["customer_id"] == o.customer_id)
        with_overrides.loc[mask, "verdict"] = o.override_verdict
    return {name: summarise(d, PRESET_THRESHOLD)["cost_per_message_eur"]
            for name, d in [("as_judged", df), ("every_clarify_counted_as_necessary", lenient_clarify),
                            ("plus_manual_overrides", with_overrides)]}


def main():
    sample = pd.read_csv(RESULTS_DIR / "agent_test_2_per_intent.csv")[["message", "customer_id", "true_intent"]]
    base = pd.read_csv(RESULTS_DIR / "baseline_test_predictions.csv").drop_duplicates(["text", "true_intent"])
    df = sample.merge(base.rename(columns={"text": "message"})[["message", "true_intent", "predicted_intent",
                                                                "confidence", "correct"]],
                      on=["message", "true_intent"])
    df["rule_handoff"] = df["predicted_intent"].map(must_handoff)
    df["safety"] = df["true_intent"].map(must_handoff)
    candidates = df[~df["rule_handoff"] & (df["confidence"] >= FLOOR)]

    todo_llm = sum(not is_cached_hybrid(r.message, r.customer_id, r.predicted_intent)
                   for r in candidates.itertuples())
    estimate = todo_llm * (EST_LLM_USD + EST_JUDGE_USD)
    print(f"{len(df)} messages; {len(candidates)} would be automated at t >= {FLOOR}; "
          f"{todo_llm} not cached yet. Estimated cost (LLM + judge): ${estimate:.2f}")
    if todo_llm and input("Continue and pay for the API calls? [y/N] ").strip().lower() != "y":
        sys.exit("Cancelled, nothing was spent.")

    paid, rows = 0.0, {}
    for i, r in enumerate(candidates.itertuples(), 1):
        out = handle_message_with_intent(r.message, r.customer_id, r.predicted_intent)
        paid += 0 if out["from_cache"] else out["cost_usd"]
        verdict = {"verdict": None, "problem": None, "explanation": None}
        if out["decision"] in ("resolve", "clarify"):
            j = judge_reply(r.message, r.customer_id, r.true_intent, out["decision"], out["customer_reply"])
            paid += 0 if j["from_cache"] else j["cost_usd"]
            verdict = {k: j[k] for k in ("verdict", "problem", "explanation")}
        rows[r.Index] = {"decision": out["decision"], "customer_reply": out["customer_reply"],
                         "handoff_summary": out["handoff_summary"], **verdict,
                         "bad_numbers": ungrounded_numbers(out["customer_reply"], r.message, r.customer_id),
                         "bad_paths": ungrounded_paths(out["customer_reply"])}
        print(f"  {i}/{len(candidates)}  paid so far ${paid:.3f}", end="\r")
    print(f"\nPaid in this run: ${paid:.4f}")

    llm_cols = pd.DataFrame.from_dict(rows, orient="index")
    df = df.join(llm_cols)
    df.to_csv(RESULTS_DIR / "hybrid_154.csv", index=False)

    from threshold import arrays, summary as routing_summary
    routing = arrays(df, "confidence", "predicted_intent", "correct")
    auto = df[~df["rule_handoff"] & (df["confidence"] >= PRESET_THRESHOLD)]
    metrics = {
        "assumptions": {"c1": C1, "c1_safety": C1_SAFETY, "c2": C2, "c_auto": C_AUTO, "c_clarify": C_CLARIFY},
        "preset_threshold": summarise(df, PRESET_THRESHOLD),
        "routing_only_upper_bound_same_154": routing_summary(*routing, PRESET_THRESHOLD),
        "other_thresholds": {f"t={t}": summarise(df, t) for t in [0.545, 0.7, 0.8, 0.9]},
        "c_clarify_sensitivity": {f"c_clarify={c}": summarise(df, PRESET_THRESHOLD, c)["cost_per_message_eur"]
                                  for c in [0.5, 1, 2, 5]},
        "judge_strictness_sensitivity": judge_sensitivity(df),
        "judge_problems_among_automated": auto["problem"].value_counts().to_dict(),
        "screens_among_automated": {"replies_with_bad_numbers": int(auto["bad_numbers"].map(len).gt(0).sum()),
                                    "replies_with_invented_paths": int(auto["bad_paths"].map(len).gt(0).sum())},
    }
    (RESULTS_DIR / "hybrid_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
