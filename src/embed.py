"""Stage 1-2: preprocess a domain corpus and embed it into document vectors.

For each domain this writes

    <embedding_dir>/<domain>.npy      float32 (n_docs, dim), L2-normalised
    <embedding_dir>/<domain>_meta.csv one row per document, aligned by position

The two files are positionally aligned; downstream stages depend on that, so
the metadata is written from the same frame that produced the vectors and
includes appl_date, avoiding any need to re-join against the source later.

Usage
-----
    python -m src.embed --config config/pipeline.yaml
    python -m src.embed --config config/pipeline.yaml --domains energy water
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import Config
from .data import load_domain_corpus
from .embedding import Qwen3Embedder
from .preprocess import build_parts_frame

META_COLUMNS = ["pub_number", "title", "abstract", "domain", "year", "pub_date", "appl_date"]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(embeddings: np.ndarray, processed: int, path: Path) -> None:
    np.savez_compressed(path, embeddings=embeddings, processed_count=processed)


def load_checkpoint(path: Path) -> tuple[np.ndarray | None, int]:
    if not path.exists():
        return None, 0
    try:
        data = np.load(path)
        return data["embeddings"], int(data["processed_count"])
    except Exception as exc:
        print(f"  could not read checkpoint {path} ({exc}); starting from scratch")
        return None, 0


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def dynamic_batch_size(mean_length: float, base: int, max_length: int) -> int:
    """Scale the batch size to the texts at hand.

    Throughput optimisation only: batching affects padding, not the resulting
    vectors, since every text is pooled independently.
    """
    if mean_length < max_length / 4:
        return base * 2
    if mean_length < max_length / 2:
        return int(base * 1.5)
    if mean_length < max_length:
        return base
    return max(base // 2, 32)


def embed_parts(
    embedder: Qwen3Embedder,
    texts: np.ndarray,
    lengths: np.ndarray,
    cfg: dict,
    checkpoint_path: Path | None = None,
) -> np.ndarray:
    base_batch = cfg["batch_size"]
    max_length = cfg["max_length"]
    use_dynamic = cfg.get("dynamic_batching", True)
    interval = cfg.get("checkpoint_interval", 5000)

    chunks: list[np.ndarray] = []
    start_idx = 0
    if checkpoint_path:
        loaded, start_idx = load_checkpoint(checkpoint_path)
        if loaded is not None:
            chunks = [loaded]
            print(f"  resuming from checkpoint at {start_idx:,} texts")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if start_idx == 0:  # warm up kernels so the ETA is not skewed
            embedder.encode(list(texts[: min(16, len(texts))]), batch_size=16, max_length=max_length)
            torch.cuda.empty_cache()

    processed = start_idx
    started = time.time()

    with tqdm(total=len(texts) - start_idx, desc="  embedding") as bar:
        i = start_idx
        while i < len(texts):
            if use_dynamic:
                window = lengths[i : i + base_batch * 2]
                batch_size = dynamic_batch_size(float(np.mean(window)), base_batch, max_length)
            else:
                batch_size = base_batch

            batch = list(texts[i : i + batch_size])
            try:
                vectors = embedder.encode(batch, batch_size=batch_size, max_length=max_length)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                batch_size = max(batch_size // 2, 16)
                print(f"\n  CUDA OOM; retrying with batch size {batch_size}")
                batch = list(texts[i : i + batch_size])
                vectors = embedder.encode(batch, batch_size=batch_size, max_length=max_length)

            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            chunks.append(vectors)

            processed += len(batch)
            i += batch_size
            bar.update(len(batch))

            elapsed = time.time() - started
            speed = (processed - start_idx) / elapsed if elapsed > 0 else 0.0
            bar.set_postfix({"batch": batch_size, "texts/s": f"{speed:.0f}"})

            if checkpoint_path and processed % interval < batch_size:
                save_checkpoint(np.vstack(chunks), processed, checkpoint_path)

    return np.vstack(chunks)


def aggregate_to_documents(
    part_vectors: np.ndarray,
    weights: np.ndarray,
    doc_ids: np.ndarray,
    n_docs: int,
) -> np.ndarray:
    """Weighted mean of part vectors per document, then L2-normalised."""
    dim = part_vectors.shape[1]
    weighted_sum = np.zeros((n_docs, dim), dtype=np.float32)
    weight_sum = np.zeros(n_docs, dtype=np.float32)

    np.add.at(weighted_sum, doc_ids, weights[:, None] * part_vectors)
    np.add.at(weight_sum, doc_ids, weights)

    doc_vectors = weighted_sum / np.clip(weight_sum[:, None], 1e-9, None)
    norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    return doc_vectors / np.clip(norms, 1e-9, None)


# ---------------------------------------------------------------------------
# Per-domain driver
# ---------------------------------------------------------------------------
def embed_domain(domain: str, config: Config, embedder: Qwen3Embedder) -> None:
    out_dir = config.path("embedding_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / f"{domain}.npy"
    meta_path = out_dir / f"{domain}_meta.csv"
    checkpoint_path = out_dir / f"{domain}_checkpoint.npz"

    print(f"\n=== {domain} ===")
    df = load_domain_corpus(
        domain,
        config.path("corpus_dir"),
        year_field=config.get("loadings.year_field", "appl_date"),
    )
    print(f"  {len(df):,} documents")

    df = build_parts_frame(df, config["preprocess"])

    field_weights = config["field_weights"]
    texts: list[str] = []
    weights: list[float] = []
    doc_ids: list[int] = []
    for doc_id, parts in enumerate(df["parts"]):
        for tag, text in parts:
            texts.append(text)
            weights.append(field_weights.get(tag, 1.0))
            doc_ids.append(doc_id)

    texts_arr = np.array(texts, dtype=object)
    weights_arr = np.asarray(weights, dtype=np.float32)
    doc_ids_arr = np.asarray(doc_ids, dtype=np.int32)
    print(f"  {len(texts_arr):,} text parts")

    # Sort by length so each batch pads to a similar width. The inverse
    # permutation is applied before aggregation, so document order is preserved.
    lengths = np.array([len(t) for t in texts_arr])
    order = np.argsort(lengths, kind="stable")
    part_vectors_sorted = embed_parts(
        embedder,
        texts_arr[order],
        lengths[order],
        config["embedding"],
        checkpoint_path=checkpoint_path,
    )

    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    part_vectors = part_vectors_sorted[inverse]

    doc_vectors = aggregate_to_documents(part_vectors, weights_arr, doc_ids_arr, len(df))

    np.save(emb_path, doc_vectors.astype(np.float32))
    df[META_COLUMNS].to_csv(meta_path, index=False)
    checkpoint_path.unlink(missing_ok=True)

    print(f"  wrote {emb_path.name} {doc_vectors.shape} and {meta_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed the retrieved patent corpus.")
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--domains", nargs="*", help="defaults to every domain in the config")
    args = parser.parse_args()

    config = Config.load(args.config)
    domains = args.domains or config.domains

    embedder = Qwen3Embedder.from_config(config["embedding"])
    for domain in domains:
        embed_domain(domain, config, embedder)
    embedder.clear_cache()


if __name__ == "__main__":
    main()
