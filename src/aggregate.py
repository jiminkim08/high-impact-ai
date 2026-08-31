"""Cross-domain aggregation: the domain x year change ratio.

For each domain and year, the change ratio is the number of distinct topics
with a detected change point in that year, divided by the number of topics
eligible for detection in that domain (those that passed the `min_years` and
non-zero filters in src/changepoint.py). Normalising by the eligible count is
what makes domains with different topic counts comparable.

This stage reads the per-domain outputs only, so it can be re-run without
repeating clustering or detection.

Outputs, under <output_dir>/:
    domain_year_change_ratio.csv          long form
    domain_year_change_ratio_heatmap.csv  wide form: domains x years

Usage
-----
    python -m src.aggregate --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def eligible_topics(matrix: pd.DataFrame, min_years: int) -> list[int]:
    """Topics that change-point detection was actually run on."""
    return [t for t in matrix.columns if len(matrix[t]) >= min_years and matrix[t].max() > 0]


def domain_change_ratio(domain: str, output_dir: Path, min_years: int) -> pd.DataFrame:
    domain_dir = output_dir / domain

    matrix = pd.read_csv(domain_dir / "topic_loadings_matrix.csv", index_col=0)
    matrix.columns = matrix.columns.astype(int)
    events = pd.read_csv(domain_dir / "change_points.csv")

    years = matrix.index.tolist()
    n_eligible = len(eligible_topics(matrix, min_years))

    if events.empty:
        changed = pd.Series(0, index=years)
    else:
        changed = events.groupby("year")["topic_id"].nunique().reindex(years, fill_value=0)

    ratio = changed / n_eligible if n_eligible else changed.astype(float) * 0.0

    return pd.DataFrame(
        {
            "domain": domain,
            "year": years,
            "changed_topics": changed.values,
            "total_topics": n_eligible,
            "change_ratio": ratio.values,
        }
    )


def run(config) -> pd.DataFrame:
    """Aggregate every domain's change points; returns the wide heatmap table."""
    output_dir = config.path("output_dir")
    min_years = config.get("changepoint.min_years", 4)

    frames = []
    for domain in config.domains:
        matrix_path = output_dir / domain / "topic_loadings_matrix.csv"
        if not matrix_path.exists():
            print(f"  skipping {domain}: no results under {matrix_path.parent}")
            continue
        frames.append(domain_change_ratio(domain, output_dir, min_years))

    if not frames:
        raise SystemExit("No per-domain results found; run src.changepoint first.")

    results = pd.concat(frames, ignore_index=True)

    duplicates = results.duplicated(subset=["domain", "year"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate domain-year rows found, which usually means a "
            "topic_loadings_matrix.csv was appended to rather than overwritten:\n"
            f"{results.loc[duplicates, ['domain', 'year']].drop_duplicates().to_string(index=False)}"
        )

    results.to_csv(output_dir / "domain_year_change_ratio.csv", index=False)

    heatmap = results.pivot(index="domain", columns="year", values="change_ratio").fillna(0.0)
    heatmap.to_csv(output_dir / "domain_year_change_ratio_heatmap.csv")

    print(f"\n{len(frames)} domains aggregated")
    print(heatmap.round(2).to_string())
    return heatmap


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate change ratios across domains.")
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()

    from .config import Config

    run(Config.load(args.config))


if __name__ == "__main__":
    main()
