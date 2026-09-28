"""Step 7: choose the confidence threshold that minimises expected cost.

Decision rule for each message:
  - predicted intent is on the always-handoff list  -> human     (cost c2)
  - confidence >= threshold                          -> automated
  - otherwise                                        -> human     (cost c2)
Cost of an automated message: c_auto if the intent is right, c1 if it is wrong,
and c1_safety if the TRUE intent is one that must go to a human (a fraud, dispute
or AML message that slipped through). Humans are assumed always right.

Theory: automating a message that is right with probability q costs
(1-q)*c1 + c_auto on average, a human costs c2. So automate when
q > 1 - (c2 - c_auto)/c1. With perfect calibration that is the best threshold.

Important: this measures the value of the classifier's ROUTING. It assumes that
every automated message with a correct intent gets a correct reply, so it is an
upper bound on what the full system can achieve (see src/hybrid.py for the
measured version). The cost is averaged over BANKING77's test set, which has
exactly 40 messages per intent; a real bank's traffic mix is different.

The baseline's threshold is chosen on held-out ("out-of-fold") predictions for
the TRAIN set and evaluated on the TEST set. The LLM has only 154 test
messages, so its threshold is chosen on those (in-sample, optimistic), and a
bootstrap shows how fragile that result is.

No API calls. Run: python src/threshold.py
Output: results/threshold_metrics.json, results/cost_curve.png, results/sensitivity.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from baseline import make_model
from load_data import load_banking77
from policy import must_handoff

C1 = 30.0          # € a wrong automated answer (assumption, agreed with the business)
C1_SAFETY = 300.0  # € a must-handoff message (fraud/dispute/AML) that gets automated (assumption)
C2 = 5.0           # € a human handling the message (assumption)
C_AUTO = 0.01      # € API cost of an automated reply (measured ~$0.008 per message)
THRESHOLDS = np.round(np.arange(0, 1.0001, 0.005), 3)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CACHE_FILE = ROOT / "cache" / "baseline_oof_predictions.csv"

# Chart style (validated palette: blue / orange pass colour-blindness checks)
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#52514e"
INK, SURFACE, GRID = "#0b0b0b", "#fcfcfb", "#e6e5e1"


def arrays(df, conf_col, pred_col, correct_col, true_col="true_intent"):
    """(confidence, correct, handed off by rule, true intent must go to a human) as numpy arrays."""
    return (df[conf_col].to_numpy(), df[correct_col].to_numpy(bool),
            df[pred_col].map(must_handoff).to_numpy(bool), df[true_col].map(must_handoff).to_numpy(bool))


def cost_per_message(conf, correct, handoff_by_rule, safety, t,
                     c1=C1, c1_safety=C1_SAFETY, c2=C2, c_auto=C_AUTO):
    """Average cost per message for threshold t (numpy arrays in, one number out)."""
    human = handoff_by_rule | (conf < t)
    wrong_cost = np.where(safety, c1_safety, c1)
    cost = np.where(human, c2, np.where(correct, c_auto, wrong_cost + c_auto))
    return cost.mean()


def best_threshold(conf, correct, handoff_by_rule, safety, **costs):
    curve = np.array([cost_per_message(conf, correct, handoff_by_rule, safety, t, **costs) for t in THRESHOLDS])
    return THRESHOLDS[int(curve.argmin())], curve


def summary(conf, correct, handoff_by_rule, safety, t, **costs):
    automated = ~(handoff_by_rule | (conf < t))
    return {
        "threshold": float(t),
        "cost_per_message_eur": round(float(cost_per_message(conf, correct, handoff_by_rule, safety, t, **costs)), 3),
        "automation_rate": round(float(automated.mean()), 3),
        "error_rate_of_automated": round(float((~correct[automated]).mean()), 3) if automated.any() else None,
        "wrong_automated_per_1000_messages": round(float((automated & ~correct).mean() * 1000), 1),
        "safety_misroutes": int((automated & safety).sum()),
        "must_handoff_messages": int(safety.sum()),
    }


def out_of_fold_baseline(C=10, cache=True) -> pd.DataFrame:
    """Baseline predictions for each TRAIN message from a model that did not see it (5-fold CV)."""
    cache_file = CACHE_FILE if C == 10 else CACHE_FILE.with_name(f"baseline_oof_predictions_C{C}.csv")
    if cache and cache_file.exists():
        return pd.read_csv(cache_file)
    train = load_banking77("train")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = make_model(C=C)
    proba = cross_val_predict(model, train["text"], train["intent"], cv=folds, method="predict_proba")
    classes = np.array(sorted(train["intent"].unique()))  # predict_proba columns are sorted classes
    pred = classes[proba.argmax(axis=1)]
    oof = pd.DataFrame({"true_intent": train["intent"], "predicted_intent": pred,
                        "confidence": proba.max(axis=1), "correct": pred == train["intent"]})
    cache_file.parent.mkdir(exist_ok=True)
    oof.to_csv(cache_file, index=False)
    return oof


def review_queue_cost(conf, correct, handoff_by_rule, safety, t, c_review, c2=C2, c_auto=C_AUTO):
    """Alternative deployment: a human reviews every automated reply before it is sent.
    Review costs c_review; a wrong draft is caught and handled by a human (c2 more)."""
    human = handoff_by_rule | (conf < t)
    cost = np.where(human, c2, c_review + c_auto + np.where(correct, 0.0, c2))
    return cost.mean()


def leakage_curve(conf, correct, handoff_by_rule, safety, thresholds):
    """Automation rate and must-handoff leakage for each threshold (auto-send framing)."""
    rows = []
    for t in thresholds:
        automated = ~(handoff_by_rule | (conf < t))
        leaked = int((automated & safety).sum())
        n_safety = int(safety.sum())
        # one-sided 95% upper bound for the leakage rate (exact binomial; 0 events -> "rule of three")
        from scipy.stats import beta
        upper = float(beta.ppf(0.95, leaked + 1, n_safety - leaked)) if leaked < n_safety else 1.0
        rows.append({"threshold": float(t), "automation_rate": round(float(automated.mean()), 3),
                     "leaked": leaked, "must_handoff": n_safety,
                     "leakage_rate": round(leaked / n_safety, 4), "leakage_upper95": round(upper, 4)})
    return rows


def bootstrap_llm(llm_arr, n=2000, seed=0):
    """How often does automating anything beat sending everything to a human, over resamples?"""
    rng = np.random.default_rng(seed)
    size = len(llm_arr[0])
    savings = []
    for _ in range(n):
        idx = rng.integers(0, size, size)
        _, curve = best_threshold(*(a[idx] for a in llm_arr))
        savings.append(C2 - curve.min())
    savings = np.array(savings)
    return {"share_of_resamples_where_automation_pays": round(float((savings > 1e-9).mean()), 3),
            "best_saving_median_eur": round(float(np.median(savings)), 3),
            "best_saving_90pct_range_eur": [round(float(np.percentile(savings, 5)), 3),
                                            round(float(np.percentile(savings, 95)), 3)]}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=GRAY)


def main():
    theory_t = 1 - (C2 - C_AUTO) / C1
    print(f"c1 = €{C1:.0f}, c1_safety = €{C1_SAFETY:.0f}, c2 = €{C2:.0f}  ->  "
          f"theoretical threshold 1 - (c2 - c_auto)/c1 = {theory_t:.4f}\n")

    # --- Baseline: choose on out-of-fold train predictions, evaluate on test ---
    oof = out_of_fold_baseline()
    oof_arr = arrays(oof, "confidence", "predicted_intent", "correct")
    base_t, _ = best_threshold(*oof_arr)
    base_t_old, _ = best_threshold(*oof_arr, c1_safety=C1)  # the earlier model: no separate safety cost

    test = pd.read_csv(RESULTS_DIR / "baseline_test_predictions.csv")
    base_test = arrays(test, "confidence", "predicted_intent", "correct")
    _, base_curve = best_threshold(*base_test)

    # --- LLM: 154 test messages, in-sample ---
    llm = pd.read_csv(RESULTS_DIR / "agent_test_2_per_intent.csv")
    llm_arr = arrays(llm, "confidence", "intent", "correct")
    llm_t, llm_curve = best_threshold(*llm_arr)

    # --- Same 154 messages, both systems, for a like-for-like comparison ---
    same = llm[["message", "true_intent"]].merge(
        test.drop_duplicates(["text", "true_intent"]).rename(columns={"text": "message"}),
        on=["message", "true_intent"])
    base_same = arrays(same, "confidence", "predicted_intent", "correct")

    results = {
        "assumptions": {"c1_wrong_automated_eur": C1, "c1_safety_misroute_eur": C1_SAFETY,
                        "c2_human_eur": C2, "c_auto_eur": C_AUTO, "theoretical_threshold": round(theory_t, 4),
                        "note": "routing value only: assumes a correct intent always gets a correct reply "
                                "(upper bound); averaged over a benchmark with 40 messages per intent"},
        "all_to_human_cost_per_message_eur": C2,
        "baseline_full_test_3080": {
            "chosen_on_train_out_of_fold": summary(*base_test, base_t),
            "earlier_model_without_safety_cost": summary(*base_test, base_t_old),
            "at_theoretical_threshold": summary(*base_test, theory_t),
        },
        "baseline_threshold_by_safety_cost": {
            f"c1_safety_{int(cs)}": summary(*base_test, best_threshold(*oof_arr, c1_safety=cs)[0], c1_safety=cs)
            for cs in [30, 100, 300, 1000]
        },
        "llm_test_154_in_sample": {
            "best": summary(*llm_arr, llm_t),
            "at_theoretical_threshold": summary(*llm_arr, theory_t),
            "bootstrap": bootstrap_llm(llm_arr),
        },
        "baseline_same_154": summary(*base_same, base_t),
        "framing_auto_send_leakage_curve_test": leakage_curve(*base_test, [0.545, base_t, 0.7, 0.8, 0.9, 0.95]),
        "framing_review_queue": {},
    }
    for c_review in [1.0, 2.0]:
        curve = [review_queue_cost(*oof_arr, t, c_review) for t in THRESHOLDS]
        t_rq = float(THRESHOLDS[int(np.argmin(curve))])
        automated = ~(base_test[2] | (base_test[0] < t_rq))
        results["framing_review_queue"][f"c_review_{c_review:g}"] = {
            "threshold_chosen_on_oof": t_rq,
            "test_cost_per_message_eur": round(float(review_queue_cost(*base_test, t_rq, c_review)), 3),
            "automation_rate": round(float(automated.mean()), 3),
        }
    RESULTS_DIR.joinpath("threshold_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    # --- Chart 1: cost curve -----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=SURFACE)
    style(ax)
    ax.axhline(C2, color=GRAY, linewidth=1, linestyle=":")
    ax.text(0.01, C2 + 0.12, f"Everything to a human: €{C2:.2f}", color=GRAY, fontsize=8)
    ax.axvline(theory_t, color=GRAY, linewidth=1, linestyle="--")
    ax.text(theory_t - 0.01, 0.3, f"theory: {theory_t:.2f}", color=GRAY, fontsize=8, ha="right")
    ax.plot(THRESHOLDS, base_curve, color=BLUE, linewidth=2, label="Baseline routing (TF-IDF), 3,080 test msgs")
    ax.plot(THRESHOLDS, llm_curve, color=ORANGE, linewidth=2, label="LLM (Claude Haiku), 154 test msgs")
    base_cost = cost_per_message(*base_test, base_t)
    ax.plot(base_t, base_cost, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"chosen on training data: t = {base_t:.3f}, test cost €{base_cost:.2f}",
                (base_t, base_cost), xytext=(0, -18), textcoords="offset points", ha="center",
                fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(C2 + 1, float(np.percentile(llm_curve, 95)) + 0.5))
    ax.set_xlabel("Confidence threshold (automate only above this)", color=INK)
    ax.set_ylabel("Expected cost per message (€)", color=INK)
    ax.set_title(f"Routing cost vs. threshold (c1 = €{C1:.0f}, safety misroute = €{C1_SAFETY:.0f}, "
                 f"human = €{C2:.0f})", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cost_curve.png", facecolor=SURFACE)
    plt.close(fig)

    # --- Chart 2: sensitivity of the optimal threshold to c1/c2 -------------
    ratios = np.geomspace(1.25, 50, 40)
    base_opt = [best_threshold(*oof_arr, c1=r * C2, c1_safety=max(C1_SAFETY, r * C2))[0] for r in ratios]
    base_opt_plain = [best_threshold(*oof_arr, c1=r * C2, c1_safety=r * C2)[0] for r in ratios]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=SURFACE)
    style(ax)
    ax.plot(ratios, 1 - 1 / ratios, color=GRAY, linewidth=2, linestyle="--", label="Theory: 1 − c2/c1")
    ax.plot(ratios, base_opt, color=BLUE, linewidth=2, label=f"Baseline, safety misroute €{C1_SAFETY:.0f}")
    ax.plot(ratios, base_opt_plain, color=ORANGE, linewidth=2,
            label="Baseline, no separate safety cost")
    ax.axvline(C1 / C2, color=GRAY, linewidth=1, linestyle=":")
    ax.text(C1 / C2 * 1.04, 0.05, f"our case: c1/c2 = {C1 / C2:.0f}", color=GRAY, fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([1.25, 2, 5, 10, 20, 50], labels=["1.25", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Cost ratio c1/c2 (wrong automated answer ÷ human handling)", color=INK)
    ax.set_ylabel("Cost-optimal threshold (out-of-fold)", color=INK)
    ax.set_title("How the optimal threshold moves with the cost ratio", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sensitivity.png", facecolor=SURFACE)
    plt.close(fig)
    print("\nSaved results/threshold_metrics.json, results/cost_curve.png, results/sensitivity.png")


if __name__ == "__main__":
    main()
