"""Multi-LexSum interactive tool (Streamlit).

Paste a real or fake civil-rights legal case and get:
  * three-tier extractive summaries (long / short / tiny)
  * a class-action-sought prediction (Yes / No)
  * a case-type prediction (multi-class)

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# make src/ importable
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import config as C                       # noqa: E402
from clean import clean_case_text        # noqa: E402
import predict as P                      # noqa: E402

st.set_page_config(page_title="Multi-LexSum Explorer", page_icon="⚖️", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_examples():
    if C.SUBSET_PARQUET.exists():
        df = pd.read_parquet(C.SUBSET_PARQUET,
                             columns=["case_id", "case_name", "case_type",
                                      "class_action_sought", "split",
                                      "full_text_raw", "summary_tiny"])
        return df[df.split == "test"].reset_index(drop=True)
    return None


@st.cache_resource(show_spinner=False)
def warm_models():
    try:
        P.available_models("case_type")
        return True
    except FileNotFoundError:
        return False


SAMPLE_FAKE = (
    "Plaintiffs, individually and on behalf of all others similarly situated, "
    "filed this action against the State Department of Corrections in the U.S. "
    "District Court alleging unconstitutional conditions of confinement. The "
    "complaint asserts that the facility provides grossly inadequate medical and "
    "mental-health care in violation of the Eighth Amendment, 42 U.S.C. 1983. "
    "Plaintiffs seek class certification, declaratory and injunctive relief, and "
    "the appointment of an independent monitor to oversee compliance. The parties "
    "engaged in extensive discovery before reaching a proposed consent decree that "
    "would require staffing increases and independent medical audits for five years."
)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("⚖️ Multi-LexSum Explorer")
st.caption("Summarise a civil-rights case and predict whether a class action was "
           "sought and the case type — Text Analytics final project (Option 2).")

models_ready = warm_models()
if not models_ready:
    st.error("Models not found. Run `python src/train_models.py` first to create "
             "the artifacts in `models/`.")

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
        "Classifier", ["logreg", "naive_bayes", "linear_svm", "lstm"],
        help="Linear models use TF-IDF features; LSTM is a TensorFlow model.")
    summary_method = st.selectbox("Summariser", ["lexrank", "textrank", "lsa"])
    st.markdown("---")
    st.subheader("Load an example")
    examples = load_examples()
    picked_text = None
    if examples is not None:
        names = ["(none)"] + [
            f"{r.case_id} · {str(r.case_name)[:40]} · {r.case_type}"
            for r in examples.itertuples()
        ]
        sel = st.selectbox("Real test-set case", names, index=0)
        if sel != "(none)":
            row = examples.iloc[names.index(sel) - 1]
            picked_text = row.full_text_raw
            st.info(f"True label → class action: **{row.class_action_sought}** · "
                    f"type: **{row.case_type}**")
    if st.button("Use synthetic sample case"):
        picked_text = SAMPLE_FAKE

default_text = picked_text if picked_text else ""
text = st.text_area("Paste a legal case (real or fake):", value=default_text,
                    height=260, key="case_text")

go = st.button("Analyze", type="primary", disabled=not models_ready)


def proba_df(proba: dict, k: int | None = None):
    items = list(proba.items())
    if k:
        items = items[:k]
    return pd.DataFrame(items, columns=["label", "probability"]).set_index("label")


if go and text.strip():
    with st.spinner("Cleaning, summarising and predicting ..."):
        cleaned = clean_case_text(text)
        summaries = P.summarize_case(text, method=summary_method)
        if model_choice == "lstm":
            cas = P.predict_lstm(text, "class_action_sought")
            ctype = P.predict_lstm(text, "case_type")
        else:
            cas = P.predict(text, "class_action_sought", model_choice)
            ctype = P.predict(text, "case_type", model_choice)

    st.subheader("Predictions")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Class action sought?", cas["prediction"],
                  help=f"model: {cas['model']}")
        st.bar_chart(proba_df(cas["proba"]))
    with c2:
        st.metric("Predicted case type", ctype["prediction"],
                  help=f"model: {ctype['model']}")
        st.bar_chart(proba_df(ctype["proba"], k=6))

    st.subheader("Summaries")
    t1, t2, t3 = st.tabs(["Tiny (1 sentence)", "Short (paragraph)", "Long (multi-paragraph)"])
    t1.write(summaries["tiny"] or "_(text too short to summarise)_")
    t2.write(summaries["short"] or "_(text too short to summarise)_")
    t3.write(summaries["long"] or "_(text too short to summarise)_")

    with st.expander("Show cleaned text (first 3,000 chars)"):
        st.text(cleaned[:3000])
elif go:
    st.warning("Please paste or load a case first.")
