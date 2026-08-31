"""Stage 5: topic labeling.

Each topic is labeled with the n-grams most similar to the topic as a whole, a
KeyBERT-style procedure implemented directly on the same embedder used for the
documents, so that candidates and topic are compared in one vector space:

1. sample up to `max_docs_per_topic` documents from the topic;
2. count candidate n-grams over that sample with a CountVectorizer;
3. embed the topic text and every candidate;
4. rank candidates by cosine similarity to the topic text.

The displayed label is the top `label_top_k` keyphrases joined by a bullet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .embedding import Qwen3Embedder

NOISE_LABEL = "(noise)"


def document_text(row: pd.Series) -> str:
    title = row.get("title") or ""
    abstract = row.get("abstract") or ""
    return f"{title}. {abstract}".strip()


def extract_keyphrases(
    text: str,
    embedder: Qwen3Embedder,
    cfg: dict,
) -> list[tuple[str, float]]:
    vectorizer = CountVectorizer(
        ngram_range=tuple(cfg.get("ngram_range", (1, 3))),
        stop_words=cfg.get("stop_words", "english"),
        max_features=cfg.get("max_features", 500),
    )

    try:
        vectorizer.fit_transform([text.lower()])
        candidates = list(vectorizer.get_feature_names_out())
    except ValueError:
        # raised when the text contains only stop words
        return []

    if not candidates:
        return []

    topic_vector = embedder.encode(
        text,
        batch_size=1,
        max_length=cfg.get("topic_max_length", 512),
        return_single_vector=True,
    )
    candidate_vectors = embedder.encode(
        candidates,
        batch_size=64,
        max_length=cfg.get("candidate_max_length", 128),
    )

    similarities = cosine_similarity([topic_vector], candidate_vectors)[0]
    top_n = cfg.get("top_n", 10)
    top_idx = np.argsort(similarities)[-top_n:][::-1]
    return [(candidates[i], float(similarities[i])) for i in top_idx]


def label_topic(
    df: pd.DataFrame,
    texts: list[str],
    topic_id: int,
    embedder: Qwen3Embedder,
    cfg: dict,
) -> dict:
    mask = df["topic_id"] == topic_id
    if not mask.any():
        return {"topic_id": int(topic_id), "label": "(empty)", "keyphrases": []}

    max_docs = cfg.get("max_docs_per_topic", 200)
    indices = df.index[mask].tolist()[:max_docs]
    topic_text = " ".join(texts[i] for i in indices)

    keyphrases = [phrase for phrase, _ in extract_keyphrases(topic_text, embedder, cfg)]
    label_top_k = cfg.get("label_top_k", 3)
    label = " • ".join(keyphrases[:label_top_k]) if keyphrases else "(unlabeled)"

    return {
        "topic_id": int(topic_id),
        "n_docs": int(mask.sum()),
        "label": label,
        "keyphrases": keyphrases,
    }


def label_topics(
    df: pd.DataFrame,
    embedder: Qwen3Embedder,
    cfg: dict,
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Label every non-noise topic; returns the table and an id -> label map."""
    texts = df.apply(document_text, axis=1).tolist()
    topic_ids = sorted(t for t in df["topic_id"].unique() if t != -1)

    print(f"  labeling {len(topic_ids)} topics")
    infos = [label_topic(df, texts, t, embedder, cfg) for t in topic_ids]

    label_map = {info["topic_id"]: info["label"] for info in infos}
    label_map[-1] = NOISE_LABEL
    return pd.DataFrame(infos), label_map
