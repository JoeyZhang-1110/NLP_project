"""Text cleaning for Multi-LexSum legal case documents.

Two levels of cleaning are exposed:

* ``clean_case_text``    - light, readability-preserving cleaning used for
  summarization and for display.  It strips court page headers, footer
  artefacts, form-feeds and repairs the broken unicode that the dataset ships
  with, then collapses whitespace.

* ``normalize_for_model`` - aggressive normalisation used to build TF-IDF
  features: lower-cases, removes digits/punctuation, drops English + legal
  stop-words and short tokens.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# ---------------------------------------------------------------------------
# Regex patterns for recurring legal-document boilerplate
# ---------------------------------------------------------------------------
# e.g. "Case 1:19-cv-01075-ERK-PK Document 134 Filed 05/25/21 Page 1 of 27 PageID #: 1374"
_PAGE_HEADER = re.compile(
    r"Case\s+\d+:\d+-[a-z]{2}-\d+.*?(?:PageID\s*#?:?\s*\d+)?(?=\n|$)",
    re.IGNORECASE,
)
# "Page 12 of 27" stragglers, "PageID #: 1374" stragglers
_PAGE_OF = re.compile(r"\bPage\s+\d+\s+of\s+\d+\b", re.IGNORECASE)
_PAGEID = re.compile(r"\bPageID\s*#?:?\s*\d+\b", re.IGNORECASE)
# Bates / document stamps like "ECF No. 29", "Doc. 134", "Document 134-2"
_DOC_STAMP = re.compile(r"\b(?:ECF\s+No\.?|Doc(?:ument)?\.?)\s*[\d\-]+", re.IGNORECASE)
# Standalone line that is just a page number
_LONE_NUMBER_LINE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
# Form feed (page break) char and the literal escape sometimes left in text
_FORMFEED = re.compile(r"[\f\x0c]")

# The dataset contains a known mojibake artefact where the section sign and a
# few other glyphs were double-mangled into U+FFFD pairs ("��").
_MOJIBAKE = re.compile(r"�+")

# Generic whitespace collapsing
_MULTI_NL = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# A small set of unicode punctuation we map back to ASCII for cleaner display.
_UNICODE_MAP = {
    "–": "-", "—": "-",          # en/em dash
    "‘": "'", "’": "'",          # curly single quotes
    "“": '"', "”": '"',          # curly double quotes
    "§": "section ",                   # § section sign
    "¶": "paragraph ",                 # ¶ pilcrow
    "…": "...",                         # ellipsis
    " ": " ",                           # non-breaking space
}


def repair_unicode(text: str) -> str:
    """Normalise unicode and repair the dataset's mojibake artefacts."""
    text = unicodedata.normalize("NFKC", text)
    for bad, good in _UNICODE_MAP.items():
        text = text.replace(bad, good)
    # The "section number" sign is frequently mangled into U+FFFD pairs that
    # sit right before a statute number; turn them into the word "section".
    text = _MOJIBAKE.sub(" section ", text)
    return text


def clean_case_text(text: str | None) -> str:
    """Light cleaning that keeps the text human-readable (for summarisation)."""
    if not text:
        return ""
    text = repair_unicode(text)
    text = _FORMFEED.sub("\n", text)
    text = _PAGE_HEADER.sub(" ", text)
    text = _DOC_STAMP.sub(" ", text)
    text = _PAGEID.sub(" ", text)
    text = _PAGE_OF.sub(" ", text)
    text = _LONE_NUMBER_LINE.sub("", text)
    # collapse whitespace
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Aggressive normalisation for modelling features
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z]+")

# Legal-domain stop-words that appear in nearly every case and carry little
# discriminative signal between case types.
LEGAL_STOPWORDS = {
    "plaintiff", "plaintiffs", "defendant", "defendants", "court", "case",
    "v", "vs", "filed", "complaint", "motion", "order", "id", "et", "al",
    "no", "inc", "llc", "corp", "u.s", "us", "usc", "section", "paragraph",
    "ecf", "doc", "document", "page", "pageid", "district", "circuit",
    "honorable", "judge", "magistrate", "cv", "civ", "action", "law",
}


@lru_cache(maxsize=1)
def _english_stopwords() -> frozenset[str]:
    try:
        from nltk.corpus import stopwords
        return frozenset(stopwords.words("english"))
    except Exception:  # pragma: no cover - fallback if nltk data missing
        return frozenset(
            "the a an and or but if then of to in on at by for with as is are "
            "was were be been being this that these those it its he she they "
            "them his her their we you i not no nor so than too very can will "
            "just do does did has have had which who whom what when where why "
            "how all any both each few more most other some such".split()
        )


def normalize_for_model(text: str | None, min_len: int = 3) -> str:
    """Lower-case, strip non-alpha, remove stop-words; returns a token string."""
    if not text:
        return ""
    text = repair_unicode(text).lower()
    stop = _english_stopwords() | LEGAL_STOPWORDS
    tokens = [
        t for t in _TOKEN.findall(text)
        if len(t) >= min_len and t not in stop
    ]
    return " ".join(tokens)


def ensure_nltk() -> None:
    """Download the nltk resources used across the project (idempotent)."""
    import nltk
    for pkg, path in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


if __name__ == "__main__":
    ensure_nltk()
    sample = (
        "Case 1:19-cv-01075-ERK-PK Document 134 Filed 05/25/21 Page 1 of 27 "
        "PageID #: 1374\n\nPlaintiffs brought this “FTCA” action under "
        "42 U.S.C. �� 1983 – see ECF No. 29 at 1.\f\n12\n"
    )
    print("RAW:\n", sample)
    print("\nCLEAN:\n", clean_case_text(sample))
    print("\nNORMALIZED:\n", normalize_for_model(clean_case_text(sample)))
