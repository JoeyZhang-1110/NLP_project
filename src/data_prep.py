"""Build the representative Multi-LexSum subset used throughout the project.

Pipeline
--------
1. Download/locate the split metadata files (train/dev/test JSON-Lines) and the
   2.2 GB ``sources.json`` (one big ``{doc_id: {doc_text: ...}}`` object).
2. Select a balanced subset of cases restricted to the most common case types
   (so the multi-class model has enough per-class signal).
3. Stream ``sources.json`` once with ``ijson`` and pull only the documents that
   belong to the selected cases, concatenating them per case.
4. Clean the concatenated text and assemble a tidy DataFrame with the full
   case, the three reference summaries and the metadata targets.
5. Persist to ``data/multilexsum_subset.parquet`` (+ a small CSV preview).

Run:  ``python src/data_prep.py``
"""
from __future__ import annotations

import json
import random

import pandas as pd
from huggingface_hub import hf_hub_download

import config as C
from clean import clean_case_text


# ---------------------------------------------------------------------------
# Step 1 - locate raw files
# ---------------------------------------------------------------------------
def _download(path: str) -> str:
    return hf_hub_download(C.HF_REPO, path, repo_type="dataset")


def load_split_metadata() -> pd.DataFrame:
    """Read the three split files (JSON-Lines) into one DataFrame."""
    frames = []
    for split, rel in C.SPLIT_FILES.items():
        p = _download(rel)
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        df = pd.DataFrame(rows)
        df["split"] = split
        frames.append(df)
    meta = pd.concat(frames, ignore_index=True)
    # case_type in the raw data has stray trailing spaces -> normalise
    meta["case_type"] = meta["case_type"].fillna("").str.strip()
    return meta


# ---------------------------------------------------------------------------
# Step 2 - select a balanced subset
# ---------------------------------------------------------------------------
def select_subset(meta: pd.DataFrame) -> pd.DataFrame:
    top = {t.strip() for t in C.TOP_CASE_TYPES}
    pool = meta[
        meta["case_type"].isin(top)
        & meta["case_documents"].apply(lambda d: isinstance(d, list) and len(d) > 0)
        & meta["summary/long"].fillna("").str.len().gt(0)
    ].copy()

    rng = random.Random(C.RANDOM_SEED)
    chosen_idx: list[int] = []
    for ctype, grp in pool.groupby("case_type"):
        idx = list(grp.index)
        rng.shuffle(idx)
        chosen_idx.extend(idx[: C.MAX_PER_CASE_TYPE])
    rng.shuffle(chosen_idx)
    chosen_idx = chosen_idx[: C.MAX_TOTAL_CASES]

    subset = pool.loc[chosen_idx].reset_index(drop=True)
    return subset


# ---------------------------------------------------------------------------
# Step 3 - stream sources.json and pull needed documents
# ---------------------------------------------------------------------------
def fetch_documents(needed_doc_ids: set[str]) -> dict[str, str]:
    """Return {doc_id: doc_text} for the requested documents.

    ``sources.json`` is a single ~2.2 GB JSON object that contains bare ``NaN``
    literals (invalid per the JSON spec, so strict streaming parsers such as
    ijson reject it).  Python's ``json`` module accepts ``NaN`` by default, so
    we load the file once, pull only the documents we need, and immediately
    drop the large structure to release memory.
    """
    src_path = _download(C.SOURCES_FILE)
    print("      loading sources.json into memory (one-off, ~2.2 GB) ...")
    with open(src_path, "r", encoding="utf-8") as fh:
        sources = json.load(fh)  # json accepts the NaN literals; ijson does not
    print(f"      parsed {len(sources)} documents total")

    out: dict[str, str] = {}
    for doc_id in needed_doc_ids:
        payload = sources.get(doc_id)
        if isinstance(payload, dict):
            out[doc_id] = payload.get("doc_text") or ""
        else:
            out[doc_id] = ""
    del sources  # free ~2-3 GB before downstream processing
    print(f"      recovered {sum(1 for v in out.values() if v)}/{len(needed_doc_ids)} documents")
    return out


# ---------------------------------------------------------------------------
# Step 4/5 - assemble + persist
# ---------------------------------------------------------------------------
def build_subset() -> pd.DataFrame:
    print("[1/5] Loading split metadata ...")
    meta = load_split_metadata()
    print(f"      total cases with metadata: {len(meta)}")

    print("[2/5] Selecting balanced subset ...")
    subset = select_subset(meta)
    print(f"      selected {len(subset)} cases across "
          f"{subset['case_type'].nunique()} case types")
    print(subset["case_type"].value_counts().to_string())

    # map case -> its document ids
    case_docs = {row.case_id: list(row.case_documents) for row in subset.itertuples()}
    needed = {d for docs in case_docs.values() for d in docs}
    print(f"[3/5] Streaming sources.json for {len(needed)} documents ...")
    doc_text = fetch_documents(needed)

    print("[4/5] Concatenating + cleaning case text ...")
    full_raw, full_clean, n_docs, n_chars = [], [], [], []
    for cid in subset["case_id"]:
        texts = [doc_text.get(d, "") for d in case_docs[cid]]
        joined = "\n\n".join(t for t in texts if t)[: C.MAX_DOC_CHARS]
        cleaned = clean_case_text(joined)
        full_raw.append(joined)
        full_clean.append(cleaned)
        n_docs.append(sum(1 for t in texts if t))
        n_chars.append(len(cleaned))

    out = pd.DataFrame({
        "case_id": subset["case_id"],
        "case_name": subset.get("case_name"),
        "split": subset["split"],
        "case_type": subset["case_type"],
        "class_action_sought": subset["class_action_sought"],
        "court": subset.get("court"),
        "state": subset.get("state"),
        "filing_year": subset.get("filing_year"),
        "n_documents": n_docs,
        "n_clean_chars": n_chars,
        "full_text_raw": full_raw,
        "full_text_clean": full_clean,
        "summary_long": subset["summary/long"],
        "summary_short": subset["summary/short"],
        "summary_tiny": subset["summary/tiny"],
        "case_url": subset.get("case_url"),
    })
    # drop cases where we failed to recover any text
    before = len(out)
    out = out[out["n_clean_chars"] > 200].reset_index(drop=True)
    print(f"      kept {len(out)}/{before} cases with usable text")

    print("[5/5] Saving artifacts ...")
    out.to_parquet(C.SUBSET_PARQUET, index=False)
    out.drop(columns=["full_text_raw"]).head(50).to_csv(
        C.SUBSET_SAMPLE_CSV, index=False, encoding="utf-8"
    )
    print(f"      -> {C.SUBSET_PARQUET}  ({len(out)} rows)")
    print(f"      -> {C.SUBSET_SAMPLE_CSV}  (preview)")
    return out


if __name__ == "__main__":
    df = build_subset()
    print("\nDone. Split sizes:")
    print(df["split"].value_counts().to_string())
    print("\nclass_action_sought:")
    print(df["class_action_sought"].value_counts().to_string())
