"""Unified inference API used by the Streamlit app and the notebooks.

Given a raw (real or fake) legal case it produces:
* three-tier extractive summaries (long / short / tiny)
* a ``class_action_sought`` prediction (Yes / No) with probability
* a ``case_type`` prediction with the top-k probabilities

Models are the scikit-learn pipelines saved by ``train_models.py``.  The TF
LSTM is loaded lazily only if requested (keeps the app light to start).
"""
from __future__ import annotations

import functools

import joblib
import numpy as np

import config as C
from clean import clean_case_text
from summarize import three_tier_summary


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarize_case(text: str, method: str = "lexrank") -> dict:
    return three_tier_summary(text, method=method).as_dict()


# ---------------------------------------------------------------------------
# Classifier loading
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _load_clf(target: str):
    path = C.MODELS_DIR / f"clf_{target}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python src/train_models.py` first.")
    return joblib.load(path)


def available_models(target: str) -> list[str]:
    return list(_load_clf(target)["pipelines"].keys())


def _proba(pipe, text: str, labels):
    """Return a {label: probability} dict, handling SVM (no predict_proba)."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        p = pipe.predict_proba([text])[0]
        classes = clf.classes_
        return {str(c): float(v) for c, v in zip(classes, p)}
    # LinearSVC -> softmax over decision_function
    scores = pipe.decision_function([text])[0]
    scores = np.atleast_1d(scores)
    if scores.shape[0] == 1:  # binary
        classes = clf.classes_
        z = np.array([-scores[0], scores[0]])
    else:
        classes = clf.classes_
        z = scores
    e = np.exp(z - z.max())
    soft = e / e.sum()
    return {str(c): float(v) for c, v in zip(classes, soft)}


def predict(text: str, target: str, model: str = "logreg") -> dict:
    """Predict a single target. Returns {prediction, proba (sorted), model}."""
    bundle = _load_clf(target)
    if model not in bundle["pipelines"]:
        model = "logreg" if "logreg" in bundle["pipelines"] else list(bundle["pipelines"])[0]
    pipe = bundle["pipelines"][model]
    cleaned = clean_case_text(text)
    pred = str(pipe.predict([cleaned])[0])
    proba = _proba(pipe, cleaned, bundle["labels"])
    proba = dict(sorted(proba.items(), key=lambda kv: -kv[1]))
    return {"prediction": pred, "proba": proba, "model": model}


# ---------------------------------------------------------------------------
# Optional: TensorFlow LSTM prediction (lazy)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _load_lstm(target: str):
    import tensorflow as tf
    meta = joblib.load(C.MODELS_DIR / f"lstm_{target}_meta.joblib")
    model = tf.keras.models.load_model(C.MODELS_DIR / f"lstm_{target}.keras")
    return model, meta


def predict_lstm(text: str, target: str) -> dict:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from clean import normalize_for_model
    model, meta = _load_lstm(target)
    seq = meta["tokenizer"].texts_to_sequences([normalize_for_model(text)])
    X = pad_sequences(seq, maxlen=meta["maxlen"])
    probs = np.atleast_2d(model.predict(X, verbose=0))
    labels = meta["labels"]
    if probs.shape[-1] == 1:
        p1 = float(probs.ravel()[0])
        proba = {labels[0]: 1 - p1, labels[1]: p1}
    else:
        proba = {labels[i]: float(probs[0][i]) for i in range(len(labels))}
    proba = dict(sorted(proba.items(), key=lambda kv: -kv[1]))
    pred = next(iter(proba))
    return {"prediction": pred, "proba": proba, "model": "lstm"}


def analyze_case(text: str, model: str = "logreg", summary_method: str = "lexrank") -> dict:
    """One-call convenience: summaries + both predictions."""
    return {
        "summaries": summarize_case(text, method=summary_method),
        "class_action_sought": predict(text, "class_action_sought", model),
        "case_type": predict(text, "case_type", model),
    }


if __name__ == "__main__":
    demo = (
        "Plaintiffs, individually and on behalf of all others similarly situated, "
        "filed a class action against the state department of corrections alleging "
        "unconstitutional prison conditions including inadequate medical care. "
        "They sought class certification and injunctive relief under 42 U.S.C. 1983."
    )
    import json
    print(json.dumps(analyze_case(demo), indent=2, ensure_ascii=False)[:1500])
