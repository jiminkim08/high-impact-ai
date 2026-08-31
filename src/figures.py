"""Figure generation.

Styling here reproduces the figures as published: matplotlib's `tab20` family
for topic identity and `YlOrRd` for the change-ratio heatmap. Because this is a
replication package, the specification is deliberately fixed rather than
restyled — a reviewer should be able to place the output beside the figures in
the paper.

Two notes on how identity is encoded, since a domain can have more than twenty
topics and colors then repeat:

* in the scatter plots each topic's centroid is annotated with its `topic_id`
  in white-outlined text, so a topic is identifiable without relying on color;
* the stacked area chart is a composition overview; individual topics are read
  from topic_loadings_matrix.csv rather than off the fill colors.

Usage
-----
    python -m src.figures --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # write files without requiring a display
import matplotlib.patheffects as path_effects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def topic_palette(n_topics: int) -> list:
    """Qualitative colors, widened as the topic count grows."""
    if n_topics <= 10:
        return list(plt.cm.tab10.colors)
    if n_topics <= 20:
        return list(plt.cm.tab20.colors)
    return list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)


def plot_topic_loadings(matrix: pd.DataFrame, domain: str, out_path: Path, dpi: int) -> None:
    """Stacked area chart of topic shares by application year."""
    if matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    matrix.plot.area(ax=ax, color=topic_palette(matrix.shape[1])[: matrix.shape[1]])
    ax.set_title(f"[{domain}] Topic loadings over time (share)")
    ax.set_xlabel("Application year")
    ax.set_ylabel("Share")
    ax.legend_.remove() if ax.legend_ else None
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_umap_scatter(
    coords: np.ndarray,
    topic_ids: np.ndarray,
    domain: str,
    out_path: Path,
    dpi: int,
    show_noise: bool,
    max_topics_in_legend: int = 30,
) -> None:
    """2-D UMAP scatter, colored by topic, with topic ids drawn at centroids."""
    unique_topics = sorted(t for t in np.unique(topic_ids) if t != -1)
    palette = topic_palette(len(unique_topics))
    n_noise = int((topic_ids == -1).sum())

    fig, ax = plt.subplots(figsize=(16, 11))

    if show_noise and n_noise:
        noise_mask = topic_ids == -1
        ax.scatter(
            coords[noise_mask, 0], coords[noise_mask, 1],
            c="lightgray", s=15, alpha=0.3, label="Noise", zorder=1,
        )

    for i, topic_id in enumerate(unique_topics):
        mask = topic_ids == topic_id
        points = coords[mask]
        color = palette[i % len(palette)]
        ax.scatter(
            points[:, 0], points[:, 1],
            c=[color], s=25, alpha=0.6 if show_noise else 0.5,
            label=f"Topic {topic_id}", zorder=2,
        )
        if not show_noise:
            # Direct label at the centroid so identity does not rest on color.
            cx, cy = float(points[:, 0].mean()), float(points[:, 1].mean())
            text = ax.text(
                cx, cy, str(topic_id),
                fontsize=15, fontweight="bold", color="black",
                ha="center", va="center", zorder=100,
            )
            text.set_path_effects(
                [path_effects.Stroke(linewidth=3, foreground="white"), path_effects.Normal()]
            )

    suffix = f" + {n_noise:,} noise" if show_noise and n_noise else ""
    ax.set_title(
        f"[{domain}] UMAP: {len(unique_topics)} topics{suffix}, {len(coords):,} documents",
        fontsize=16, fontweight="bold", pad=20,
    )
    ax.set_xlabel("UMAP-1", fontsize=13)
    ax.set_ylabel("UMAP-2", fontsize=13)
    ax.grid(True, alpha=0.2)

    if len(unique_topics) <= max_topics_in_legend:
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True,
            fontsize=8, title="Topics", title_fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_change_ratio_heatmap(
    heatmap: pd.DataFrame, out_path: Path, dpi: int, cmap: str = "YlOrRd"
) -> None:
    """Domain x year change-ratio heatmap. Years that are all zero are dropped."""
    table = heatmap.loc[:, (heatmap != 0).any(axis=0)]
    if table.empty:
        print("  no non-zero years; heatmap skipped")
        return

    n_rows, n_cols = table.shape
    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.1), max(6, n_rows * 0.55)))
    image = ax.imshow(table.values, cmap=cmap, aspect="auto", vmin=0)

    ax.set_xticks(range(n_cols), labels=table.columns, fontsize=10)
    ax.set_yticks(range(n_rows), labels=table.index, fontsize=10)

    # White gridlines on the minor ticks separate the cells.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    vmax = table.values.max()
    for i in range(n_rows):
        for j in range(n_cols):
            value = table.values[i, j]
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color="white" if value > vmax * 0.6 else "black",
            )

    fig.colorbar(image, ax=ax, label="Change ratio", shrink=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Domain")
    ax.set_title("Domain × Year change ratio")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def figures_for_domain(domain: str, config) -> None:
    domain_dir = config.path("output_dir") / domain
    dpi = config.get("figures.dpi", 180)

    matrix_path = domain_dir / "topic_loadings_matrix.csv"
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path, index_col=0)
        matrix.columns = matrix.columns.astype(int)
        plot_topic_loadings(matrix, domain, domain_dir / "topic_loadings_area.png", dpi)

    doc_topics = pd.read_csv(domain_dir / "doc_topics.csv", dtype={"pub_number": str})
    vectors = np.load(config.path("embedding_dir") / f"{domain}.npy")

    # The scatter uses its own 2-component projection; clustering was performed
    # in the 5-component space written by src/cluster.py.
    import umap

    cfg2d = config["umap_2d"]
    coords = umap.UMAP(
        n_neighbors=cfg2d["n_neighbors"],
        n_components=cfg2d["n_components"],
        min_dist=cfg2d["min_dist"],
        metric=cfg2d["metric"],
        random_state=cfg2d.get("random_state"),
    ).fit_transform(vectors)

    topic_ids = doc_topics["topic_id"].to_numpy()
    max_legend = config.get("figures.max_topics_in_legend", 30)
    for show_noise, name in ((False, "umap_topics.png"), (True, "umap_topics_with_noise.png")):
        plot_umap_scatter(
            coords, topic_ids, domain, domain_dir / name, dpi,
            show_noise=show_noise, max_topics_in_legend=max_legend,
        )

    print(f"  {domain}: figures written to {domain_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the paper's figures.")
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--domains", nargs="*")
    parser.add_argument("--skip-domain-figures", action="store_true")
    args = parser.parse_args()

    from .config import Config

    config = Config.load(args.config)
    output_dir = config.path("output_dir")

    if not args.skip_domain_figures:
        for domain in args.domains or config.domains:
            if (output_dir / domain / "doc_topics.csv").exists():
                figures_for_domain(domain, config)
            else:
                print(f"  skipping {domain}: no results yet")

    heatmap_path = output_dir / "domain_year_change_ratio_heatmap.csv"
    if heatmap_path.exists():
        heatmap = pd.read_csv(heatmap_path, index_col=0)
        plot_change_ratio_heatmap(
            heatmap,
            output_dir / "domain_year_change_ratio_heatmap.png",
            config.get("figures.dpi", 180),
            config.get("figures.heatmap_cmap", "YlOrRd"),
        )
        print(f"  heatmap written to {output_dir}")


if __name__ == "__main__":
    main()
