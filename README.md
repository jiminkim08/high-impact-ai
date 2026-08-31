# High-Impact AI Patents

Replication package for the study **"From Regulation to Innovation: Mining High-Impact AI Patents Grounded in the EU AI Act and Korea's AI Basic Act."**

**Status: This repository is currently being prepared as a replication package. The code, queries, and documentation are subject to change until the first stable release.**

This repository provides the retrieval queries, CPC-based selection criteria, analysis code, and parameter settings used to identify and characterize high-impact AI patents.

---

## Overview

The EU AI Act and Korea's AI Basic Act both designate a set of domains in which AI systems may materially affect human life, safety, or fundamental rights — termed *high-risk* and *high-impact*, respectively. This study operationalizes these regulatory definitions as a reproducible patent-retrieval procedure, and applies an embedding-based topic pipeline to the resulting corpus to characterize the technological composition and temporal dynamics of patenting in regulated AI domains.

The repository contains everything needed to reconstruct the analytical dataset and re-run the analysis; it does not redistribute the underlying patent records (see [Data Availability](#data-availability)).

---

## Repository Structure

```
high-impact-ai-patents/
├── queries/          # Patent retrieval queries and CPC code lists
├── src/              # Analysis scripts (preprocessing → clustering → change-point analysis)
├── config/           # Model specifications and analysis parameters
├── requirements.txt  # Python dependencies
├── LICENSE
└── README.md
```

| Directory | Contents |
|---|---|
| `queries/` | Domain-level and AI-level CPC code lists, and the keyword-based search queries used in each retrieval stage. |
| `src/` | Scripts implementing the analytical pipeline, organized by stage. |
| `config/` | Configuration files specifying embedding models, dimensionality-reduction and clustering hyperparameters, and change-point detection settings. |

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

> A GPU is recommended for the embedding stage but is not required; all steps run on CPU with longer runtimes.

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

The `src/` directory implements the main analytical pipeline:

1. **Text preprocessing** — normalization and cleaning of patent titles, abstracts, and claims.
2. **Patent embedding** — dense vector representation of each patent document.
3. **Dimensionality reduction** — projection of embeddings into a lower-dimensional space for clustering.
4. **Topic clustering** — density-based grouping of patents into technological topics.
5. **Topic labeling** — generation of interpretable labels for each topic.
6. **Temporal change-point analysis** — detection of structural breaks in topic-level filing trends.

Model specifications and hyperparameters for each stage are documented in `config/` and in the corresponding source files, so that reported results can be reproduced exactly.

### Running the pipeline

```bash
# Adjust paths and stage names to match the scripts in src/
python -m src.preprocess   --config config/preprocess.yaml
python -m src.embed        --config config/embed.yaml
python -m src.cluster      --config config/cluster.yaml
python -m src.changepoint  --config config/changepoint.yaml
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
