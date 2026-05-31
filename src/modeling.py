"""Reusable modelling utilities for the Multi-LexSum project.

Targets
-------
* ``class_action_sought``  - binary (Yes / No)
* ``case_type``            - multi-class (the most common civil-rights types)

Feature pipeline: TF-IDF over the legal-normalised case text.  The normaliser
lives in :mod:`clean` and is wired in as the vectoriser ``preprocessor`` so a
fitted pipeline can be handed raw (or lightly cleaned) text directly - which is
exactly what the Streamlit app needs.

Models: Multinomial Naive Bayes, Logistic Regression (GLM) and Linear SVM via
scikit-learn, plus a TensorFlow/Keras LSTM built separately (see
``build_lstm`` / ``train_lstm``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

from clean import normalize_for_model

# ---------------------------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------------------------
def make_tfidf(max_features: int = 30_000, ngram=(1, 2), min_df: int = 3) -> TfidfVectorizer:
    """TF-IDF that normalises legal text internally (handles raw input)."""
    return TfidfVectorizer(
        preprocessor=normalize_for_model,   # lower/strip/stopwords inside
        token_pattern=r"[a-z]+",
        ngram_range=ngram,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )


SKLEARN_MODELS = {
    "naive_bayes": lambda: MultinomialNB(),
    "logreg":      lambda: LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
    "linear_svm":  lambda: LinearSVC(C=1.0, class_weight="balanced"),
}


def make_pipeline(model_key: str, **tfidf_kwargs) -> Pipeline:
    return Pipeline([
        ("tfidf", make_tfidf(**tfidf_kwargs)),
        ("clf", SKLEARN_MODELS[model_key]()),
    ])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    name: str
    accuracy: float
    macro_f1: float
    per_class_accuracy: dict
    report: str
    confusion: np.ndarray
    labels: list

    def summary_line(self) -> str:
        return f"{self.name:14s} acc={self.accuracy:.3f}  macro-F1={self.macro_f1:.3f}"


def evaluate(name: str, y_true, y_pred, labels=None) -> EvalResult:
    if labels is None:
        labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    y_true = list(map(str, y_true))
    y_pred = list(map(str, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # per-class accuracy = recall (diagonal / row sum)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = cm.diagonal() / cm.sum(axis=1)
    per_class_acc = {lab: (float(a) if not np.isnan(a) else 0.0)
                     for lab, a in zip(labels, per_class)}
    return EvalResult(
        name=name,
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        per_class_accuracy=per_class_acc,
        report=classification_report(y_true, y_pred, labels=labels, zero_division=0),
        confusion=cm,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Interpretability: most important words per class
# ---------------------------------------------------------------------------
def top_features_per_class(pipeline: Pipeline, topn: int = 20) -> dict:
    """Return {class_label: [(word, weight), ...]} for NB / LogReg / SVM."""
    vec: TfidfVectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feats = np.array(vec.get_feature_names_out())

    if hasattr(clf, "coef_"):
        coefs = clf.coef_
        classes = clf.classes_
        if coefs.shape[0] == 1:  # binary -> expand to both directions
            coefs = np.vstack([-coefs[0], coefs[0]])
            classes = clf.classes_
    elif hasattr(clf, "feature_log_prob_"):  # MultinomialNB
        coefs = clf.feature_log_prob_
        classes = clf.classes_
    else:
        return {}

    out = {}
    for i, cls in enumerate(classes):
        order = np.argsort(coefs[i])[::-1][:topn]
        out[str(cls)] = [(feats[j], float(coefs[i][j])) for j in order]
    return out


# ---------------------------------------------------------------------------
# TensorFlow / Keras LSTM
# ---------------------------------------------------------------------------
@dataclass
class LSTMBundle:
    model: object
    tokenizer: object
    maxlen: int
    labels: list
    history: dict = field(default_factory=dict)


def build_lstm(vocab_size: int, maxlen: int, n_classes: int, embed_dim: int = 128):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    out_units = 1 if n_classes == 2 else n_classes
    out_act = "sigmoid" if n_classes == 2 else "softmax"
    model = models.Sequential([
        layers.Input(shape=(maxlen,)),
        layers.Embedding(vocab_size, embed_dim),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(out_units, activation=out_act),
    ])
    loss = "binary_crossentropy" if n_classes == 2 else "sparse_categorical_crossentropy"
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def train_lstm(texts_train, y_train, texts_val, y_val,
               vocab_size: int = 20_000, maxlen: int = 400,
               epochs: int = 8, batch_size: int = 32) -> LSTMBundle:
    """Train a (Bi)LSTM on normalised text; returns model + tokenizer + history."""
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    norm_train = [normalize_for_model(t) for t in texts_train]
    norm_val = [normalize_for_model(t) for t in texts_val]

    labels = sorted(set(map(str, y_train)))
    lab2idx = {l: i for i, l in enumerate(labels)}
    yt = np.array([lab2idx[str(y)] for y in y_train])
    yv = np.array([lab2idx[str(y)] for y in y_val])

    tok = Tokenizer(num_words=vocab_size, oov_token="<unk>")
    tok.fit_on_texts(norm_train)
    Xt = pad_sequences(tok.texts_to_sequences(norm_train), maxlen=maxlen)
    Xv = pad_sequences(tok.texts_to_sequences(norm_val), maxlen=maxlen)

    n_classes = len(labels)
    model = build_lstm(min(vocab_size, len(tok.word_index) + 1), maxlen, n_classes)

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=2, restore_best_weights=True)
    hist = model.fit(
        Xt, yt, validation_data=(Xv, yv),
        epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=2,
    )
    return LSTMBundle(model=model, tokenizer=tok, maxlen=maxlen,
                      labels=labels, history=hist.history)


def lstm_predict(bundle: LSTMBundle, texts) -> list:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    norm = [normalize_for_model(t) for t in texts]
    X = pad_sequences(bundle.tokenizer.texts_to_sequences(norm), maxlen=bundle.maxlen)
    probs = bundle.model.predict(X, verbose=0)
    if probs.shape[-1] == 1:  # binary
        idx = (probs.ravel() > 0.5).astype(int)
    else:
        idx = probs.argmax(axis=1)
    return [bundle.labels[i] for i in idx]
