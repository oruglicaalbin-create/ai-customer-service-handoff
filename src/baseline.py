"""Baseline intent classifier: TF-IDF + logistic regression.

Trained on the full BANKING77 train split, evaluated on the test split.
Run: python src/baseline.py

Outputs (in results/):
  baseline_metrics.json           headline numbers
  baseline_test_predictions.csv   per-message prediction + confidence (used later
                                  for the threshold optimisation)
"""
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from load_data import load_banking77

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def make_model(C=10):
    """TF-IDF turns each message into a vector of word and word-pair weights:
    words frequent in this message but rare overall get high weight.
    Logistic regression then learns one weight vector per intent and outputs
    a probability for each of the 77 intents."""
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(C=C, max_iter=2000),
    )


def main():
    train = load_banking77("train")
    test = load_banking77("test")

    model = make_model()

    start = time.perf_counter()
    model.fit(train["text"], train["intent"])
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    probabilities = model.predict_proba(test["text"])
    predict_ms_per_message = (time.perf_counter() - start) / len(test) * 1000

    classes = model.classes_
    predicted = classes[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)  # probability of the predicted intent

    predictions = pd.DataFrame({
        "text": test["text"],
        "true_intent": test["intent"],
        "predicted_intent": predicted,
        "confidence": confidence.round(4),
        "correct": predicted == test["intent"],
    })

    accuracy = accuracy_score(test["intent"], predicted)
    macro_f1 = f1_score(test["intent"], predicted, average="macro")

    metrics = {
        "model": "TF-IDF (1-2 grams) + logistic regression (C=10)",
        "train_messages": len(train),
        "test_messages": len(test),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "train_seconds": round(train_seconds, 1),
        "predict_ms_per_message": round(predict_ms_per_message, 3),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    predictions.to_csv(RESULTS_DIR / "baseline_test_predictions.csv", index=False)
    (RESULTS_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))

    # --- A first look at errors ---
    per_intent = predictions.groupby("true_intent")["correct"].mean().sort_values()
    print("\nFive hardest intents (accuracy):")
    print(per_intent.head(5).round(3).to_string())

    errors = predictions[~predictions["correct"]]
    pairs = errors.groupby(["true_intent", "predicted_intent"]).size().sort_values(ascending=False)
    print("\nMost common confusions (true -> predicted, count):")
    print(pairs.head(8).to_string())

    # Is confidence informative? Compare accuracy of confident vs unsure predictions.
    print("\nAccuracy by confidence band:")
    bands = pd.cut(predictions["confidence"], [0, 0.3, 0.5, 0.7, 0.9, 1.0])
    print(predictions.groupby(bands, observed=True)["correct"].agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()
