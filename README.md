# High-Impact AI Patents

Replication package for the study **"From Regulation to Innovation: Mining High-Impact AI Patents Grounded in the EU AI Act and Korea's AI Basic Act."**

This repository provides the retrieval queries, CPC-based selection criteria, analysis code, and parameter settings used to identify and characterize high-impact AI patents.

---

## Overview

The EU AI Act and Korea's AI Basic Act both designate a set of domains in which AI systems may materially affect human life, safety, or fundamental rights — termed *high-risk* and *high-impact*, respectively. This study operationalizes these regulatory definitions as a reproducible patent-retrieval procedure, and applies an embedding-based topic pipeline to the resulting corpus to characterize the technological composition and temporal dynamics of patenting in regulated AI domains.

The repository contains everything needed to reconstruct the analytical dataset and re-run the analysis; it does not redistribute the underlying patent records (see [Data Availability](#data-availability)).

---

## Repository Structure

```
high-impact-ai-patents/
├── queries/          # Retrieval criteria: CPC codes, keywords, worked SQL example
├── src/              # Analysis pipeline
├── config/           # Model specifications and analysis parameters
├── requirements.txt  # Python dependencies
├── LICENSE
└── README.md
```

| Directory | Contents |
|---|---|
| `queries/` | Domain-level and AI-level CPC code lists and keyword sets, published as CSV, with the retrieval procedure rendered as SQL. See `queries/README.md`. |
| `src/` | The analytical pipeline, one module per stage. |
| `config/` | `pipeline.yaml` — every model specification and hyperparameter used, in one file. |

---

## Requirements

* Python 3.10 or later
* Dependencies listed in `requirements.txt`

```bash
git clone https://github.com/<ORG>/high-impact-ai-patents.git
cd high-impact-ai-patents

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

The embedding stage was run on a single 12 GB consumer GPU (RTX 3060); `embedding.batch_size` and `embedding.max_length` in `config/pipeline.yaml` are set for that budget and should be lowered if CUDA reports out of memory. The pipeline runs on CPU as well, with substantially longer runtimes.

---

## Patent Retrieval

The patent dataset used in this study was constructed from publicly available U.S. patent publication data. High-impact AI patents were identified through a **two-stage retrieval procedure**:

**Stage 1 — Domain identification.**
Patents belonging to regulated domains were retrieved using domain-specific CPC codes together with keyword-based queries derived from the high-risk categories of the EU AI Act and the high-impact domains of Korea's AI Basic Act.

**Stage 2 — AI identification.**
Within the Stage 1 results, AI-related patents were identified using AI-related CPC codes and keyword-based criteria.

The complete set of retrieval queries and CPC codes is provided in `queries/`, together with the mapping from each regulatory domain category to its corresponding CPC codes and keywords. Applying these criteria to the data sources described in the paper reproduces the analytical corpus.

---

## Analysis

Each of the fifteen domains is analysed **independently**: embedding, reduction, and clustering are fitted on that domain's corpus alone, so a `topic_id` is meaningful only within its domain. The pipeline stages are:

1. **Text preprocessing** (`src/preprocess.py`) — whitespace normalization, then decomposition of each patent into weighted parts: the title as one part, the abstract as one part per sentence. Only title and abstract are used.
2. **Patent embedding** (`src/embed.py`, `src/embedding.py`) — each part is encoded with **Qwen3-Embedding-0.6B**. As a decoder model it is pooled at the last non-padding token, which requires left padding. A document vector is the L2-normalised weighted mean of its parts, with the title weighted three times an abstract sentence.
3. **Dimensionality reduction** (`src/cluster.py`) — UMAP to five components, cosine metric, seeded.
4. **Topic clustering** (`src/cluster.py`) — OPTICS with the ξ cluster method; documents not assigned to a cluster are retained as noise.
5. **Topic labeling** (`src/label.py`) — candidate n-grams are counted over a sample of each topic's documents and ranked by cosine similarity to the topic text in the same embedding space, a KeyBERT-style procedure implemented on the same embedder.
6. **Temporal change-point analysis** (`src/changepoint.py`, `src/aggregate.py`) — yearly topic shares by **application year**, PELT change-point detection with an RBF cost, then a domain × year change-ratio table for cross-domain comparison.

Every parameter for these stages lives in `config/pipeline.yaml`; nothing is hard-coded in `src/`. UMAP is seeded and OPTICS is deterministic, so repeated runs on the same embeddings reproduce the same topics.

### Input

The pipeline begins from an already-retrieved corpus — one file per domain at `data/corpus/<domain>.csv`, with columns:

```
pub_number, pub_date, appl_date, title, abstract
```

These are the patents satisfying the criteria in `queries/`. Retrieval itself is not part of this repository; `src/data.py` can also read the corpus from PostgreSQL when `PATENT_DB_DSN` is set, as in the original setup.

### Running the pipeline

```bash
# All stages, all domains
python -m src.run_pipeline --config config/pipeline.yaml

# A subset of domains, or a range of stages
python -m src.run_pipeline --domains energy water
python -m src.run_pipeline --from changepoint --to figures
```

Stages can also be run on their own:

```bash
python -m src.embed        --config config/pipeline.yaml
python -m src.cluster      --config config/pipeline.yaml
python -m src.changepoint  --config config/pipeline.yaml
python -m src.aggregate    --config config/pipeline.yaml
python -m src.figures      --config config/pipeline.yaml
```

Embedding is the expensive stage and checkpoints itself every 5,000 texts, so an interrupted run resumes rather than restarting.

### Outputs

```
outputs/<domain>/doc_topics.csv                document → topic_id, topic_label
outputs/<domain>/topics.json                   topic labels and keyphrases
outputs/<domain>/topic_loadings.csv            yearly topic shares
outputs/<domain>/topic_loadings_matrix.csv     years × topics
outputs/<domain>/change_points.csv             detected change points
outputs/domain_cluster_summary.csv             topic and noise counts per domain
outputs/domain_year_change_ratio_heatmap.csv   cross-domain comparison
```

---

## Data Availability

This repository contains **source code, retrieval queries, and methodological configurations only.** The underlying patent records are not redistributed.

Users can reconstruct the analytical dataset by applying the retrieval criteria in `queries/` to the data sources described in the paper. Because patent databases are updated continuously, corpora retrieved at a later date may differ slightly from the one used in the study; the reproducibility tag below fixes the criteria as applied.

---

## Reproducibility

The version of this repository corresponding to the results reported in the paper is preserved under the tag:

```
v1.0-trust26
```

Later versions may incorporate updated CPC classifications, revised retrieval criteria, or extended analytical methods. Such updates do not affect the tagged version used to produce the published results.

---

## Citation

If you use this repository in academic work, please cite:

> [Authors]. "From Regulation to Innovation: Mining High-Impact AI Patents Grounded in the EU AI Act and Korea's AI Basic Act." In *Proceedings of the 1st International Workshop on Trustworthy and Responsible aUtonomous SysTems (TRUST 2026)*, co-located with the 41st IEEE/ACM International Conference on Automated Software Engineering (ASE 2026), Munich, Germany, 2026.

```bibtex
@inproceedings{authors2026regulation,
  title     = {From Regulation to Innovation: Mining High-Impact AI Patents
               Grounded in the {EU} {AI} Act and Korea's {AI} Basic Act},
  author    = {[Authors]},
  booktitle = {Proceedings of the 1st International Workshop on Trustworthy and
               Responsible aUtonomous SysTems (TRUST 2026), co-located with the
               41st IEEE/ACM International Conference on Automated Software
               Engineering (ASE 2026)},
  address   = {Munich, Germany},
  month     = oct,
  year      = {2026}
}
```

---

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.

## Contact

Questions and issues are welcome via the repository's [issue tracker](../../issues).
