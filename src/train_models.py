"""Train + evaluate all models for both targets and save artifacts.

Targets
-------
* ``class_action_sought``  (binary: Yes / No)
* ``case_type``            (multi-class: the common civil-rights types)

Models per target: Naive Bayes, Logistic Regression (GLM), Linear SVM and a
TensorFlow Bi-LSTM.  We train on the official train(+dev) split and report on
the held-out test split (overall accuracy, macro-F1 and per-class accuracy).

Artifacts written to ``models/``:
* ``clf_<target>.joblib``     - dict of fitted sklearn pipelines
* ``lstm_<target>.keras`` + ``lstm_<target>_meta.joblib`` - Keras model + tokenizer
* ``metrics.json``            - all evaluation numbers (for slides/notebooks)
* ``top_features.json``       - most important words per class per model

Run:  ``python src/train_models.py``
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

import config as C
from modeling import (
    make_pipeline, evaluate, top_features_per_class,
    train_lstm, lstm_predict,
)

TARGETS = {
    "class_action_sought": {"binary": True},
    "case_type": {"binary": False},
}
SKLEARN_KEYS = ["naive_bayes", "logreg", "linear_svm"]


def load_data():
    df = pd.read_parquet(C.SUBSET_PARQUET)
    df["text"] = df["full_text_clean"].fillna("")
    return df


def split_xy(df, target):
    d = df.copy()
    if target == "class_action_sought":
        d = d[d[target].isin(["Yes", "No"])]
    d = d[d[target].notna() & (d[target].astype(str).str.len() > 0)]
    train = d[d.split.isin(["train", "dev"])]
    test = d[d.split == "test"]
    return (train["text"].tolist(), train[target].astype(str).tolist(),
            test["text"].tolist(), test[target].astype(str).tolist())


def main():
    df = load_data()
    print(f"Loaded {len(df)} cases\n")

    metrics = {}
    top_feats = {}

    for target, cfg in TARGETS.items():
        print("=" * 70)
        print(f"TARGET: {target}")
        print("=" * 70)
        Xtr, ytr, Xte, yte = split_xy(df, target)
        labels = sorted(set(ytr) | set(yte))
        print(f"  train={len(Xtr)}  test={len(Xte)}  classes={len(labels)}")

        metrics[target] = {"labels": labels, "models": {}}
        top_feats[target] = {}
        pipelines = {}

        # ---- sklearn models ------------------------------------------------
        for key in SKLEARN_KEYS:
            pipe = make_pipeline(key)
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            res = evaluate(key, yte, pred, labels=labels)
            print("  " + res.summary_line())
            pipelines[key] = pipe
            metrics[target]["models"][key] = {
                "accuracy": res.accuracy,
                "macro_f1": res.macro_f1,
                "per_class_accuracy": res.per_class_accuracy,
                "report": res.report,
                "confusion": res.confusion.tolist(),
            }
            top_feats[target][key] = {
                cls: [w for w, _ in feats]
                for cls, feats in top_features_per_class(pipe, topn=25).items()
            }

        joblib.dump({"pipelines": pipelines, "labels": labels},
                    C.MODELS_DIR / f"clf_{target}.joblib")
        print(f"  saved -> clf_{target}.joblib")

        # ---- TensorFlow LSTM ----------------------------------------------
        print("  training LSTM (TensorFlow) ...")
        n_classes = len(labels)
        bundle = train_lstm(Xtr, ytr, Xte, yte,
                            vocab_size=20000, maxlen=400,
                            epochs=8, batch_size=32)
        lstm_pred = lstm_predict(bundle, Xte)
        res = evaluate("lstm", yte, lstm_pred, labels=labels)
        print("  " + res.summary_line())
        metrics[target]["models"]["lstm"] = {
            "accuracy": res.accuracy,
            "macro_f1": res.macro_f1,
            "per_class_accuracy": res.per_class_accuracy,
            "report": res.report,
            "confusion": res.confusion.tolist(),
            "history": {k: [float(x) for x in v] for k, v in bundle.history.items()},
        }
        bundle.model.save(C.MODELS_DIR / f"lstm_{target}.keras")
        joblib.dump({"tokenizer": bundle.tokenizer, "maxlen": bundle.maxlen,
                     "labels": bundle.labels},
                    C.MODELS_DIR / f"lstm_{target}_meta.joblib")
        print(f"  saved -> lstm_{target}.keras (+ meta)\n")

    with open(C.MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(C.MODELS_DIR / "top_features.json", "w", encoding="utf-8") as f:
        json.dump(top_feats, f, indent=2)
    print("Saved metrics.json and top_features.json")

    # quick leaderboard
    print("\n================ LEADERBOARD (test accuracy) ================")
    for target in TARGETS:
        print(f"\n{target}:")
        for m, d in sorted(metrics[target]["models"].items(),
                           key=lambda kv: -kv[1]["accuracy"]):
            print(f"  {m:14s} acc={d['accuracy']:.3f}  macro-F1={d['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
