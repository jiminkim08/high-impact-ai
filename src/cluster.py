"""Stage 3-5: dimensionality reduction, topic clustering, and labeling.

Each domain is clustered independently: UMAP and OPTICS are fitted on that
domain's document vectors alone, so a `topic_id` is only meaningful within its
domain. Both steps are seeded (`umap.random_state`) or deterministic (OPTICS),
so repeated runs on the same embeddings reproduce the same topics.

Outputs, per domain, under <output_dir>/<domain>/:
    doc_topics.csv    document -> topic_id, topic_label
    topics.json       topic_id -> label, keyphrases, document count
    umap_5d.npy       the reduced space the clustering was performed in

Usage
-----
    python -m src.cluster --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS

from .config import Config
from .embedding import Qwen3Embedder
from .label import label_topics

DOC_TOPIC_COLUMNS = ["pub_number", "title", "domain", "year", "topic_id", "topic_label"]


def load_domain_embeddings(domain: str, embedding_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    vectors = np.load(embedding_dir / f"{domain}.npy")
    meta = pd.read_csv(embedding_dir / f"{domain}_meta.csv", dtype={"pub_number": str})

    if len(vectors) != len(meta):
        raise ValueError(
            f"[{domain}] embeddings ({len(vectors)}) and metadata ({len(meta)}) "
            "have different lengths; re-run src.embed for this domain."
        )
    return vectors, meta.reset_index(drop=True)


def reduce_dimensions(vectors: np.ndarray, cfg: dict) -> np.ndarray:
    import umap  # imported lazily: heavy dependency

    reducer = umap.UMAP(
        n_neighbors=cfg["n_neighbors"],
        n_components=cfg["n_components"],
        min_dist=cfg["min_dist"],
        metric=cfg["metric"],
        random_state=cfg.get("random_state"),
    )
    return reducer.fit_transform(vectors)


def cluster(reduced: np.ndarray, cfg: dict) -> np.ndarray:
    clusterer = OPTICS(
        min_samples=cfg["min_samples"],
        min_cluster_size=cfg["min_cluster_size"],
        metric=cfg["metric"],
        cluster_method=cfg["cluster_method"],
        xi=cfg["xi"],
        n_jobs=cfg.get("n_jobs", -1),
    )
    return clusterer.fit_predict(reduced)


def cluster_domain(domain: str, config: Config, embedder: Qwen3Embedder) -> pd.DataFrame:
    out_dir = config.path("output_dir") / domain
    out_dir.mkdir(parents=True, exist_ok=True)

    vectors, meta = load_domain_embeddings(domain, config.path("embedding_dir"))
    print(f"\n=== {domain} === {len(meta):,} documents")

    reduced = reduce_dimensions(vectors, config["umap"])
    np.save(out_dir / "umap_5d.npy", reduced.astype(np.float32))

    meta["topic_id"] = cluster(reduced, config["clustering"])
    topic_table, label_map = label_topics(meta, embedder, config["labeling"])
    meta["topic_label"] = meta["topic_id"].map(label_map)

    topic_table.to_json(out_dir / "topics.json", orient="records", force_ascii=False, indent=2)
    meta[DOC_TOPIC_COLUMNS].to_csv(out_dir / "doc_topics.csv", index=False)

    n_topics = len(topic_table)
    n_noise = int((meta["topic_id"] == -1).sum())
    print(f"  {n_topics} topics, {n_noise:,} noise ({n_noise / len(meta):.1%})")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster and label patent topics per domain.")
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--domains", nargs="*")
    args = parser.parse_args()

    config = Config.load(args.config)
    domains = args.domains or config.domains
    out_root = config.path("output_dir")
    out_root.mkdir(parents=True, exist_ok=True)

    embedder = Qwen3Embedder.from_config(config["embedding"])

    summary = []
    frames = []
    for domain in domains:
        meta = cluster_domain(domain, config, embedder)
        frames.append(meta[DOC_TOPIC_COLUMNS])
        n_noise = int((meta["topic_id"] == -1).sum())
        summary.append(
            {
                "domain": domain,
                "n_docs": len(meta),
                "n_topics": int(meta.loc[meta["topic_id"] != -1, "topic_id"].nunique()),
                "n_noise": n_noise,
                "noise_ratio": round(n_noise / len(meta), 3) if len(meta) else 0.0,
            }
        )

    embedder.clear_cache()

    pd.concat(frames, ignore_index=True).to_csv(out_root / "doc_topics_all_domains.csv", index=False)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_root / "domain_cluster_summary.csv", index=False)

    print("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
