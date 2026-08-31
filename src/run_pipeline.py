"""Run the full pipeline end to end.

    python -m src.run_pipeline --config config/pipeline.yaml

Stages can also be run individually; see the module docstrings. The embedding
stage is the expensive one and checkpoints itself, so a run interrupted there
resumes rather than restarting.

    --from / --to   restrict the run to a range of stages
    --domains       restrict the run to particular domains
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import aggregate, changepoint, cluster, embed, figures
from .config import Config
from .embedding import Qwen3Embedder

STAGES = ["embed", "cluster", "changepoint", "aggregate", "figures"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full analytical pipeline.")
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--domains", nargs="*")
    parser.add_argument("--from", dest="from_stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--to", dest="to_stage", choices=STAGES, default=STAGES[-1])
    args = parser.parse_args()

    config = Config.load(args.config)
    domains = args.domains or config.domains
    selected = STAGES[STAGES.index(args.from_stage) : STAGES.index(args.to_stage) + 1]
    output_dir = config.path("output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"domains: {', '.join(domains)}")
    print(f"stages:  {' -> '.join(selected)}\n")

    embedder: Qwen3Embedder | None = None
    if {"embed", "cluster"} & set(selected):
        embedder = Qwen3Embedder.from_config(config["embedding"])

    if "embed" in selected:
        print("--- embed ---")
        for domain in domains:
            embed.embed_domain(domain, config, embedder)

    if "cluster" in selected:
        print("\n--- cluster ---")
        frames, summary = [], []
        for domain in domains:
            meta = cluster.cluster_domain(domain, config, embedder)
            frames.append(meta[cluster.DOC_TOPIC_COLUMNS])
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
        pd.concat(frames, ignore_index=True).to_csv(
            output_dir / "doc_topics_all_domains.csv", index=False
        )
        pd.DataFrame(summary).to_csv(output_dir / "domain_cluster_summary.csv", index=False)

    if embedder is not None:
        embedder.clear_cache()

    if "changepoint" in selected:
        print("\n--- changepoint ---")
        for domain in domains:
            changepoint.analyse_domain(domain, output_dir, config)

    if "aggregate" in selected:
        print("\n--- aggregate ---")
        # Always aggregates over every domain in the config, since the heatmap
        # is a cross-domain comparison and a partial one would be misleading.
        aggregate.run(config)

    if "figures" in selected:
        print("\n--- figures ---")
        for domain in domains:
            if (output_dir / domain / "doc_topics.csv").exists():
                figures.figures_for_domain(domain, config)
        heatmap_path = output_dir / "domain_year_change_ratio_heatmap.csv"
        if heatmap_path.exists():
            figures.plot_change_ratio_heatmap(
                pd.read_csv(heatmap_path, index_col=0),
                output_dir / "domain_year_change_ratio_heatmap.png",
                config.get("figures.dpi", 180),
                config.get("figures.heatmap_cmap", "YlOrRd"),
            )

    print(f"\ndone; outputs under {output_dir}")


if __name__ == "__main__":
    main()
