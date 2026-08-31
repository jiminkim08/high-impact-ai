# Retrieval Criteria

This directory contains the complete retrieval criteria used to construct the analytical corpus: the CPC codes and keyword sets that operationalize the high-risk categories of the EU AI Act and the high-impact domains of Korea's AI Basic Act as patent search conditions.

The criteria are published as data rather than as code so that they can be inspected, audited, and reused independently of any particular database or implementation.

```
queries/
├── cpc_codes.csv      # CPC codes per domain and for the AI stage
├── keywords.csv       # Keyword terms per domain and for the AI stage
├── example_query.sql  # The retrieval procedure rendered in full for one domain
└── README.md
```

The mapping from each domain to the EU AI Act and AI Basic Act provisions it derives from is given in the paper and is not duplicated here.

---

## Retrieval procedure

A patent enters the corpus if it satisfies **both** stages:

```
   ( domain CPC  OR  domain keyword )
AND
   (   AI CPC    OR    AI keyword   )
```

**Stage 1 — Domain identification.** A patent matches a domain if any of its CPC classifications appears in that domain's rows in `cpc_codes.csv`, **or** if its abstract matches any of that domain's terms in `keywords.csv`.

**Stage 2 — AI identification.** A patent is AI-related if any of its CPC classifications falls under `G06N`, **or** if its abstract matches any AI term in `keywords.csv` (`stage = ai`).

Both stages are evaluated over patents with `appl_date >= 2018-01-01`. Each domain is retrieved independently, so a patent may belong to more than one domain.

`example_query.sql` shows the procedure rendered in full for the `rail` domain, including every literal value; the same template applies to all other domains.

---

## File schemas

### Domain identifiers

The `domain` column is the join key between the two CSV files. Fifteen values are used:

`energy` · `water` · `healthcare` · `nuclear` · `crime` · `hr` · `loan` · `vehicle` · `ship` · `aviation` · `rail` · `public` · `education` · `immigration` · `judicial`

The four transport domains (`vehicle`, `ship`, `aviation`, `rail`) are retrieved separately although they correspond to a single category in both statutes, as do `hr` and `loan` in the Korean statute; they are kept apart because their CPC and keyword profiles differ substantially. `immigration` and `judicial` derive from the EU AI Act only.

### `cpc_codes.csv`

| Column | Description |
|---|---|
| `stage` | `domain` or `ai`. |
| `domain` | Domain identifier, or `ai` for Stage 2 rows. |
| `cpc_code` | The CPC symbol in conventional notation, e.g. `G06Q50/06`. |
| `section`, `class`, `subclass`, `main_group`, `subgroup` | The symbol decomposed into its levels. Empty means *unconstrained at that level and below*. |
| `match_level` | The most specific level constrained: `class`, `subclass`, or `group`. |
| `description` | The CPC definition, abbreviated. |

A row matches a classification when every non-empty field is equal. `("B", "61", "", "", "")` therefore matches all of `B61`, including `B61L`; `("G", "06", "Q", "50", "06")` matches only `G06Q50/06`.

### `keywords.csv`

| Column | Description |
|---|---|
| `stage` | `domain` or `ai`. |
| `domain` | Domain identifier, or `ai` for Stage 2 rows. |
| `keyword` | The search term. |
| `match_type` | `substring` or `word_boundary` — see below. |
| `note` | Why a term uses word-boundary matching, where applicable. |

All keyword matching is applied to the **abstract** field and is case-insensitive.

---

## Matching rules

Two rules must be reproduced exactly for the retrieval to yield the same corpus.

**1. Word-boundary matching.** Terms marked `word_boundary` are matched as whole words (PostgreSQL: `abstract ~* '\yterm\y'`), not as substrings. Plain substring matching produces false positives that are severe enough to distort the corpus: `AI` matches *contain*, *detail*, *maintain*, *said*; `ship` matches *relationship*, *membership*, *ownership*; `rail` matches *trail*, *trailer*; `election` matches *selection*, which is common in algorithm patents. All other terms are matched as substrings (`ILIKE '%term%'`).

**2. CPC field normalization.** The CPC values in `cpc_codes.csv` are given as they must appear after normalization. In the source data used for this study, `main_group` is stored unpadded (`A61B5` → `5`, `G06Q50` → `50`) while `subgroup` is zero-filled to a minimum of two digits (`G06Q50/6` → `06`). Any reconstruction must apply the same normalization to both the criteria and the classification table, or the group-level conditions will silently match nothing.

---

## Notes on the criteria

**Class-level codes are deliberately broad.** `B60`, `B61`, `B63`, `B64`, `F01`, `F02`, and `F03` are matched at class level, which favours recall over precision at Stage 1. Precision is recovered at Stage 2, since a patent must also be AI-related to enter the corpus.

**Some codes are nested.** Where a subclass and one of its groups both appear (`A61B` and `A61B5`; `G16H` and `G16H50/20`; `B60` and `B60W`; `B64` and `B64C39/024`), the narrower code is redundant under OR matching. These rows are retained as executed rather than pruned, so that the published criteria correspond exactly to what produced the reported results.

**Shared codes are intentional.** `G06Q50/06` appears under `energy`, `water`, and `nuclear`, and `G06Q50/26` under `crime`, `public`, `immigration`, and `judicial`. Domains are retrieved independently and overlap is expected.

**Keyword terms were adjusted to avoid contamination by AI vocabulary.** Generic terms that collide with the AI stage were replaced with more specific phrases: `learning` (education) was narrowed to `student learning`, `personalized learning`, and `learning outcome`; `train`, `track`, and `signaling` (rail) were narrowed to `railway train`, `train track`, `railway signaling`, and similar. Without these substitutions, the education and rail domains absorb large numbers of unrelated machine-learning patents.

---

## Reusing the criteria

Both CSV files are self-contained and can be loaded into any implementation. To retrieve a single domain, select the rows where `domain` equals that identifier plus the rows where `stage = 'ai'`, then apply the procedure above. To add a domain, append rows to `cpc_codes.csv` and `keywords.csv` under a new identifier; no other changes are required.
