"""Extractive summarisation for Multi-LexSum cases.

Produces three summaries that mirror the granularities in the dataset:

* ``long``  - multi-paragraph narration (~10-12 sentences)
* ``short`` - one paragraph (~3-4 sentences)
* ``tiny``  - a single sentence

The default engine is LexRank (graph centrality over TF-IDF sentence vectors)
which is robust on the very long, noisy legal documents.  TextRank and LSA are
also available for comparison.  Everything runs on CPU with no model download,
matching the project's "extractive" choice.
"""
from __future__ import annotations

from dataclasses import dataclass

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer

from clean import clean_case_text, ensure_nltk

_LANG = "english"

# Minimum words for a sentence to be a summary candidate. Filters out OCR /
# footnote fragments like "Plaintiff Gonzales 14." that the sentence splitter
# otherwise treats as a sentence (and which made poor "tiny" summaries).
MIN_SENTENCE_WORDS = 8


def _filter_sentences(text: str, min_words: int = MIN_SENTENCE_WORDS) -> str:
    """Keep only reasonably-formed sentences before ranking."""
    import re
    from nltk.tokenize import sent_tokenize
    kept = []
    for s in sent_tokenize(text):
        s = s.strip()
        words = s.split()
        if len(words) < min_words:
            continue
        # require some lowercase letters (drops ALL-CAPS headers / caption blocks)
        if not re.search(r"[a-z]{3,}", s):
            continue
        # drop lines that are mostly digits/punctuation
        alpha = sum(c.isalpha() for c in s)
        if alpha < 0.5 * len(s):
            continue
        kept.append(s)
    return " ".join(kept)

# Default sentence budgets for the three tiers.
LONG_SENTENCES = 12
SHORT_SENTENCES = 4
TINY_SENTENCES = 1

_SUMMARIZERS = {
    "lexrank": LexRankSummarizer,
    "textrank": TextRankSummarizer,
    "lsa": LsaSummarizer,
}


def _build_summarizer(method: str):
    cls = _SUMMARIZERS.get(method.lower())
    if cls is None:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(_SUMMARIZERS)}")
    summ = cls(Stemmer(_LANG))
    summ.stop_words = get_stop_words(_LANG)
    return summ


def summarize(text: str, n_sentences: int, method: str = "lexrank",
              pre_clean: bool = True) -> str:
    """Return an extractive summary of ``text`` with ``n_sentences`` sentences."""
    ensure_nltk()
    if pre_clean:
        text = clean_case_text(text)
    text = _filter_sentences(text)
    if not text.strip():
        return ""
    parser = PlaintextParser.from_string(text, Tokenizer(_LANG))
    summarizer = _build_summarizer(method)
    sentences = summarizer(parser.document, n_sentences)
    return " ".join(str(s) for s in sentences)


@dataclass
class TieredSummary:
    long: str
    short: str
    tiny: str

    def as_dict(self) -> dict:
        return {"long": self.long, "short": self.short, "tiny": self.tiny}


def three_tier_summary(
    text: str,
    method: str = "lexrank",
    long_n: int = LONG_SENTENCES,
    short_n: int = SHORT_SENTENCES,
    tiny_n: int = TINY_SENTENCES,
) -> TieredSummary:
    """Produce long / short / tiny summaries in a single clean+parse pass.

    To keep the three tiers consistent (short is a subset-style condensation of
    the document, tiny the single most central sentence) each tier is ranked
    independently by the summariser, which naturally yields nested-importance
    sentences.
    """
    ensure_nltk()
    cleaned = _filter_sentences(clean_case_text(text))
    if not cleaned.strip():
        return TieredSummary("", "", "")

    parser = PlaintextParser.from_string(cleaned, Tokenizer(_LANG))
    summarizer = _build_summarizer(method)
    doc = parser.document

    def top(n: int) -> str:
        return " ".join(str(s) for s in summarizer(doc, n))

    return TieredSummary(long=top(long_n), short=top(short_n), tiny=top(tiny_n))


if __name__ == "__main__":
    demo = (
        "The plaintiffs filed a class action lawsuit against the state prison "
        "system alleging unconstitutional conditions of confinement. "
        "They claimed the facility lacked adequate heating during winter. "
        "The court certified the class in 2019. "
        "After two years of litigation the parties reached a settlement. "
        "The settlement required the prison to install new heating units. "
        "It also mandated independent monitoring for three years. "
        "The judge approved the consent decree in 2021. "
        "Attorneys' fees were awarded to the plaintiffs' counsel. "
        "The case was administratively closed in 2022."
    )
    s = three_tier_summary(demo)
    print("LONG :", s.long)
    print("SHORT:", s.short)
    print("TINY :", s.tiny)
