# Multi-LexSum — Summarisation & Classification of Civil-Rights Lawsuits

**Text Analytics Final Project · Option 2 · Northwestern University, Spring 2025**

An end-to-end NLP pipeline over the [Multi-LexSum](https://huggingface.co/datasets/allenai/multi_lexsum)
dataset (9,280 expert-authored civil-rights case summaries). The project:

1. Loads the data and recovers the **full case text** from the 2.2 GB source file.
2. **Cleans** the noisy, OCR'd legal documents.
3. Builds an **extractive summariser** that reproduces the dataset's three
   granularities — **long / short / tiny** — and scores them with **ROUGE**.
4. Trains classifiers for two targets and evaluates them
   (overall + per-class accuracy, confusion matrices, interpretable top words):
   * `class_action_sought` — **binary** (Yes / No)
   * `case_type` — **multi-class** (12 civil-rights case types)
5. Ships an **interactive Streamlit tool**: paste any real or fake case →
   get the three summaries + both predictions.

## Results (held-out test split)

| Model | `class_action_sought` acc | `case_type` (12-class) acc |
|---|---|---|
| **Linear SVM** | **0.950** | **0.911** |
| Logistic Regression (GLM) | 0.945 | 0.895 |
| Naive Bayes | 0.882 | 0.761 |
| TensorFlow Bi-LSTM | 0.824 | 0.703 |

---

## Project layout

```
nlpproject/
├── requirements.txt
├── README.md
├── src/
│   ├── config.py          # paths + dataset/subset constants
│   ├── clean.py           # legal-text cleaning + model normalisation
│   ├── summarize.py       # extractive long/short/tiny summariser
│   ├── data_prep.py       # download + build the balanced subset  (run 1st)
│   ├── modeling.py        # TF-IDF + NB/LogReg/SVM + Keras LSTM helpers
│   ├── train_models.py    # train + evaluate + save all models    (run 2nd)
│   ├── predict.py         # unified inference API (used by app + notebooks)
│   └── build_notebooks.py # regenerates the three notebooks
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_summarization.ipynb
│   └── 03_modeling.ipynb
├── app/
│   └── streamlit_app.py   # the interactive tool                  (run 3rd)
├── data/                  # built subset (parquet) — created by data_prep
├── models/                # trained models + metrics — created by train_models
└── slides/
    └── presentation_outline.md
```

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

> Python 3.10+ recommended. The first run downloads the dataset from the
> HuggingFace Hub (the `sources.json` full-text file is ~2.2 GB and is cached
> locally, so this happens only once).

## How to run (3 steps)

```bash
# 1. Build the balanced subset (downloads data, recovers full text → data/*.parquet)
python src/data_prep.py

# 2. Train + evaluate all models, save artifacts → models/
python src/train_models.py

# 3. Launch the interactive tool
streamlit run app/streamlit_app.py
```

Then open the notebooks for the full walkthrough:

```bash
jupyter lab notebooks/        # 01_eda → 02_summarization → 03_modeling
```

(Regenerate the notebooks any time with `python src/build_notebooks.py`.)

## What each piece does

* **`data_prep.py`** — reads the train/dev/test metadata, selects a balanced
  subset over the 12 most common case types (≤180 per type, 1,915 cases),
  streams the full text out of `sources.json`, cleans and saves
  `data/multilexsum_subset.parquet`.
* **`clean.py`** — strips court page-headers (`Case … Document … Filed … Page X
  of Y PageID #: …`), form-feeds, ECF/doc stamps and lone page numbers; repairs
  the dataset's broken unicode (`§`, `–`, the `��` mojibake); a second
  `normalize_for_model` pass lower-cases and removes English + legal stop-words
  for TF-IDF.
* **`summarize.py`** — LexRank / TextRank / LSA extractive summarisation,
  exposing `three_tier_summary()` → long (~12 sentences) / short (~4) / tiny (1).
* **`modeling.py` / `train_models.py`** — TF-IDF (1–2 grams) → Naive Bayes,
  Logistic Regression, Linear SVM, plus a TensorFlow Bi-LSTM; evaluation with
  overall + per-class accuracy, macro-F1, confusion matrices and top-word
  interpretability.
* **`app/streamlit_app.py`** — the deliverable interactive tool.

## Notes & limitations

* We model a **representative balanced subset** (chosen for class balance and
  runtime) rather than all 9,280 cases; the pipeline scales to the full data by
  raising the caps in `src/config.py`.
* Summaries are **extractive** (reuse source sentences), so they recover the
  salient content but do not paraphrase like the abstractive expert summaries —
  reflected in the ROUGE scores.
* OCR typos in the scanned source documents are inherent to the dataset; the
  cleaner removes *structural* noise but not OCR character errors.
