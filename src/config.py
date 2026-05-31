"""Central configuration: paths and dataset constants for the Multi-LexSum project."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Built subset artifacts (produced by src/data_prep.py)
SUBSET_PARQUET = DATA_DIR / "multilexsum_subset.parquet"
SUBSET_SAMPLE_CSV = DATA_DIR / "multilexsum_subset_preview.csv"

# ---------------------------------------------------------------------------
# HuggingFace dataset
# ---------------------------------------------------------------------------
HF_REPO = "allenai/multi_lexsum"
HF_RELEASE = "v20230518"
SPLIT_FILES = {
    "train": f"releases/{HF_RELEASE}/train.json",
    "dev":   f"releases/{HF_RELEASE}/dev.json",
    "test":  f"releases/{HF_RELEASE}/test.json",
}
SOURCES_FILE = f"releases/{HF_RELEASE}/sources.json"

# ---------------------------------------------------------------------------
# Subset construction
# ---------------------------------------------------------------------------
# We keep cases whose case_type is one of the most common ones so that the
# multi-class case_type model has enough signal per class. Everything else is
# grouped into "Other" only for analysis (the model is trained on these types).
TOP_CASE_TYPES = [
    "Equal Employment",
    "Immigration and/or the Border",   # note: raw data has a trailing space
    "Prison Conditions",
    "Jail Conditions",
    "Public Benefits / Government Services",
    "Policing",
    "Speech and Religious Freedom",
    "Criminal Justice (Other)",
    "National Security",
    "Fair Housing/Lending/Insurance",
    "Disability Rights-Pub. Accom.",
    "Education",
]

# Max cases to keep per case_type (balances the subset & keeps runtime modest).
MAX_PER_CASE_TYPE = 180
# Hard cap on total cases in the subset.
MAX_TOTAL_CASES = 2000
# Cap full-text length per case (characters) to keep memory/runtime sane.
MAX_DOC_CHARS = 120_000

RANDOM_SEED = 42
