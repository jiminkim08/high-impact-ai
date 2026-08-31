"""Stage 6: topic loadings over time and change-point detection.

For each domain, the yearly share of every topic is computed from the
application year, and a PELT search with an RBF cost detects years at which a
topic's share changes regime.

Note on the share denominator. `loadings.include_noise_in_denominator`
controls whether documents OPTICS assigned to noise count towards the yearly
total. With the paper's setting (`true`) the denominator is all documents in
the year, so a domain's topic shares sum to one minus its noise share; the
alternative (`false`) restricts the denominator to clustered documents so the
shares sum to one. The choice changes the level of every share, and can shift
detected change points where the noise rate itself moves over time, so it is
made explicit here rather than left implicit in the code.

Outputs, per domain, under <output_dir>/<domain>/:
    topic_loadings.csv         long form: year, topic_id, count, N, share
    topic_loadings_matrix.csv  wide form: years x topics
    change_points.csv          detected change points

Usage
-----
    python -m src.changepoint --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compute_loadings(df: pd.DataFrame, include_noise_in_denominator: bool) -> pd.DataFrame:
    """Yearly share of each topic."""
    clustered = df[df["topic_id"] != -1]

    counts = clustered.groupby(["year", "topic_id"]).size().rename("count").reset_index()
    denominator_source = df if include_noise_in_denominator else clustered
    totals = denominator_source.groupby("year").size().rename("N").reset_index()

    loadings = counts.merge(totals, on="year", how="left")
    loadings["share"] = loadings["count"] / loadings["N"]
    return loadings


def loadings_matrix(loadings: pd.DataFrame) -> pd.DataFrame:
    matrix = loadings.pivot(index="year", columns="topic_id", values="share").fillna(0.0)
    return matrix.sort_index()


def detect_change_points(matrix: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """PELT change-point detection on each topic's share series."""
    import ruptures as rpt  # imported lazily: optional dependency

    min_years = cfg.get("min_years", 4)
    events = []
    years = matrix.index.tolist()

    for topic_id in matrix.columns:
        series = matrix[topic_id].values
        if len(series) < min_years or series.max() == 0:
            continue

        algo = rpt.Pelt(
            model=cfg.get("model", "rbf"),
            min_size=cfg.get("min_size", 2),
            jump=cfg.get("jump", 1),
        ).fit(series)
        # predict() returns segment end indices and always includes len(series)
        # as the final element; that sentinel is not a change point.
        breakpoints = algo.predict(pen=cfg["penalty"])[:-1]

        for idx in breakpoints:
            events.append(
                {
                    "topic_id": int(topic_id),
                    "year": int(years[idx - 1]),
                    "type": "change-point",
                }
            )

    return pd.DataFrame(events, columns=["topic_id", "year", "type"])


def analyse_domain(domain: str, output_dir: Path, config) -> pd.DataFrame:
    domain_dir = output_dir / domain
    doc_topics = pd.read_csv(domain_dir / "doc_topics.csv", dtype={"pub_number": str})

    include_noise = config.get("loadings.include_noise_in_denominator", True)
    loadings = compute_loadings(doc_topics, include_noise)
    loadings.to_csv(domain_dir / "topic_loadings.csv", index=False)

    matrix = loadings_matrix(loadings)
    matrix.to_csv(domain_dir / "topic_loadings_matrix.csv")

    events = detect_change_points(matrix, config["changepoint"])
    events.to_csv(domain_dir / "change_points.csv", index=False)

    print(
        f"  {domain}: {matrix.shape[1]} topics over {matrix.shape[0]} years, "
        f"{len(events)} change points"
    )
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Topic loadings and change-point detection.")
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--domains", nargs="*")
    args = parser.parse_args()

    from .config import Config

    config = Config.load(args.config)
    output_dir = config.path("output_dir")

    for domain in args.domains or config.domains:
        analyse_domain(domain, output_dir, config)


if __name__ == "__main__":
    main()
