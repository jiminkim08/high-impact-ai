"""Analytical pipeline for the study "From Regulation to Innovation: Mining
High-Impact AI Patents Grounded in the EU AI Act and Korea's AI Basic Act."

Stages, in order:

    src.embed        preprocess and embed each domain corpus
    src.cluster      UMAP reduction, OPTICS clustering, topic labeling
    src.changepoint  yearly topic loadings and PELT change-point detection
    src.aggregate    cross-domain change-ratio table
    src.figures      the paper's figures

`src.run_pipeline` runs all of them. Parameters live in config/pipeline.yaml.
"""

__all__ = [
    "aggregate",
    "changepoint",
    "cluster",
    "config",
    "data",
    "embed",
    "embedding",
    "figures",
    "label",
    "preprocess",
]
