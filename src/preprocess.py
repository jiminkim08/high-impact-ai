"""Text preprocessing.

Each patent is decomposed into weighted parts before embedding: the title as a
single part, and the abstract as one part per sentence. Preprocessing is kept
deliberately light — whitespace normalization only — because the embedding
model is trained on natural text and benefits little from stemming or
stop-word removal.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

WHITESPACE_RE = re.compile(r"\s+")

Part = tuple[str, str]  # (field tag, text)


def clean_text(value: Any) -> str:
    """Collapse whitespace and coerce missing values to an empty string."""
    if value is None or pd.isna(value):
        return ""
    return WHITESPACE_RE.sub(" ", str(value)).strip()


def _split_sentences(text: str) -> list[str]:
    """Split an abstract into sentences.

    blingfire is used when available (it is what the study used, and it is
    considerably faster on this volume); otherwise a regex fallback keeps the
    pipeline runnable without the optional dependency.
    """
    try:
        from blingfire import text_to_sentences

        return text_to_sentences(text).split("\n")
    except ImportError:
        return re.split(r"(?<=[.!?])\s+", text)


def document_parts(
    title: str,
    abstract: str,
    min_sentence_words: int = 3,
    max_sentences: int | None = None,
    sentence_split: bool = True,
) -> list[Part]:
    """Decompose one patent into tagged parts."""
    parts: list[Part] = []

    title_clean = clean_text(title)
    if title_clean:
        parts.append(("TITLE", title_clean))

    abstract_clean = clean_text(abstract)
    if abstract_clean:
        sentences = _split_sentences(abstract_clean) if sentence_split else [abstract_clean]
        if max_sentences:
            sentences = sentences[:max_sentences]
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) >= min_sentence_words:
                parts.append(("ABS", sentence))

    return parts


def build_parts_frame(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add a `parts` column and drop documents that yield no usable text."""
    min_words = cfg.get("min_sentence_words", 3)
    max_sentences = cfg.get("max_sentences_per_doc")
    sentence_split = cfg.get("sentence_split", True)

    tqdm.pandas(desc="  preprocessing")
    out = df.copy()
    out["parts"] = out.progress_apply(
        lambda row: document_parts(
            row.get("title", ""),
            row.get("abstract", ""),
            min_sentence_words=min_words,
            max_sentences=max_sentences,
            sentence_split=sentence_split,
        ),
        axis=1,
    )

    n_before = len(out)
    out = out[out["parts"].map(len) > 0].reset_index(drop=True)
    if len(out) < n_before:
        print(f"  dropped {n_before - len(out)} documents with no usable text")

    return out
