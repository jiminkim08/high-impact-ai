"""Corpus loading.

The pipeline starts from an already-retrieved corpus: one file per domain,
holding the patents that satisfy the criteria in `queries/`. The retrieval
itself is not part of this repository (see queries/README.md).

Expected columns
----------------
    pub_number   publication number, treated as a string throughout
    pub_date     publication date
    appl_date    application date — the year used in the temporal analysis
    title        patent title
    abstract     patent abstract

A PostgreSQL reader is also provided for the setup used in the study, where
each domain lived in its own `<domain>_class` table. It is optional and is
enabled only when the PATENT_DB_DSN environment variable is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["pub_number", "pub_date", "appl_date", "title", "abstract"]


def _finalize(df: pd.DataFrame, domain: str, year_field: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[{domain}] corpus is missing columns: {missing}")

    df = df.copy()
    # pub_number is the join key across every stage. The source data mixes
    # integer-looking and alphanumeric numbers, and pandas infers the former as
    # int64, which then fails to merge against the string form. Pin it to str.
    df["pub_number"] = df["pub_number"].astype(str)
    df["domain"] = domain
    df["year"] = pd.to_datetime(df[year_field], errors="coerce").dt.year

    n_before = len(df)
    df = df[df["year"].notna()].reset_index(drop=True)
    if len(df) < n_before:
        print(f"  [{domain}] dropped {n_before - len(df)} rows with an unparseable {year_field}")
    df["year"] = df["year"].astype(int)

    return df[["pub_number", "title", "abstract", "domain", "year", "pub_date", "appl_date"]]


def load_domain_corpus(
    domain: str,
    corpus_dir: str | Path,
    year_field: str = "appl_date",
) -> pd.DataFrame:
    """Load one domain's corpus from `<corpus_dir>/<domain>.{parquet,csv}`."""
    corpus_dir = Path(corpus_dir)
    for suffix in (".parquet", ".csv"):
        candidate = corpus_dir / f"{domain}{suffix}"
        if candidate.exists():
            df = (
                pd.read_parquet(candidate)
                if suffix == ".parquet"
                else pd.read_csv(candidate, dtype={"pub_number": str})
            )
            return _finalize(df, domain, year_field)

    raise FileNotFoundError(
        f"No corpus file for domain '{domain}' in {corpus_dir}. "
        f"Expected {domain}.parquet or {domain}.csv with columns {REQUIRED_COLUMNS}."
    )


def load_domain_corpus_from_db(
    domain: str,
    year_field: str = "appl_date",
    table_pattern: str = "{domain}_class",
    schema: str = "public",
) -> pd.DataFrame:
    """Load one domain from PostgreSQL, as done in the study.

    Requires PATENT_DB_DSN, e.g.
        postgresql+psycopg2://user:password@localhost:5432/uspto
    Credentials are never read from the config file or the source.
    """
    dsn = os.environ.get("PATENT_DB_DSN")
    if not dsn:
        raise RuntimeError("PATENT_DB_DSN is not set; cannot read the corpus from the database.")

    from sqlalchemy import create_engine  # imported lazily: optional dependency

    engine = create_engine(dsn)
    table = table_pattern.format(domain=domain)
    columns = ", ".join(REQUIRED_COLUMNS)
    df = pd.read_sql(f"SELECT {columns} FROM {schema}.{table};", engine)
    return _finalize(df, domain, year_field)
