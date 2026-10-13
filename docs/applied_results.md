# Applied results: inputs, definitions and reproduction

This layer adds user-facing examples, customer activity comparisons and catalog exposure to the existing H&M analysis. It uses the original validation-selected policy and final temporal test. It does not fit models, change recommendations, select a new policy or overwrite the technical benchmarks.

## Reproduce

From the repository root with the project's Python environment:

```bash
make applied
```

Equivalent CLI:

```bash
PYTHONPATH=src MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m ranking.cli applied --config configs/hm_sample.json
```

The Python function also accepts a configurable `project_root`; the CLI supports `--project-root`. All data/report paths come from the project configuration, resolved against that root. No machine-specific absolute paths are embedded in the new implementation.

If starting from the official CSVs without saved outputs, run `make prepare`, `make run`, `make rolling`, `make publish`, then `make applied`. The extra `make eda` is optional because `make run` already generates EDA.

## Inputs reused

- `data/processed/hm_sample/transactions.parquet` and `articles.parquet`.
- `data/raw/articles.csv`: product names, types and colour labels for example cards.
- `reports/runs/hm_sample/recommendations.csv`: saved final policy's exact rankings, preserving article ID strings/leading zeros.
- `reports/runs/hm_sample/snapshots/2020-09-16/candidates_features.parquet`: recover the original popularity and recent-purchase baselines using saved source ranks and flags.
- Original `metrics.csv`, `segments.csv`, `catalog_diagnostics.csv`, `selection.json`, run manifest, and candidate-recall summary.

The implementation recomputes the five ranking metrics for all three policies and activity-group MAP/Recall/NDCG. Values must match the original saved results to an absolute tolerance of 1e-12; catalog coverage is also checked. A mismatch stops generation rather than silently mixing runs.

## Generated outputs

Under `reports/hm_sample/applied/`:

- `figures/recommendation_examples.png`: five anonymous cases, recent history → all top-12 items → future purchases.
- `figures/customer_segments.png`: MAP@12 and Recall@12 for personalization, recent purchases and popularity.
- `figures/catalog_exposure.png`: catalog coverage and exposure-weighted item-popularity mix.
- `figures/system_overview.png`: pipeline with verified sample, retrieval and test statistics.
- `tables/segment_performance.csv`: customer counts, metrics and any-hit rate by activity.
- `tables/catalog_exposure.csv`: unique items, coverage, popularity bins, novelty/diversity diagnostics.
- `tables/category_mix.csv`: aggregate product-group exposure by policy.
- `summary.json`: source, verified statistics, image availability and case-selection rules.
- `findings.md`: concise interpretation and measurement limits.

Real customer IDs and detailed case-item mappings are kept only in ignored `reports/runs/hm_sample/applied/examples_private.json`. The public figures use aliases A–E; article IDs displayed on cards are product identifiers, not customer identifiers. No private mappings or model/candidate matrices are exported to the public report folder.

## Example selection

Cases are deliberately selected after observing test outcomes:

| Alias | Diagnostic case | Selection rule |
|---|---|---|
| A | Strong match | At least two hits, at most eight distinct future articles; order by hit count, then MAP |
| B | Partial match | One hit among 3–8 distinct future articles; first hit in the first six ranks |
| C | Active-customer miss | At least 50 prior transactions, at most six future articles, zero hits |
| D | Low-history match | One or two lifetime prior transactions, at least one hit; order by MAP |
| E | True cold-start miss | No recorded prior transactions, at most six future articles, zero hits |

Customer ID is used locally only to break ties deterministically, and never printed as a public case label. History shows up to four latest distinct articles before the cutoff; future cards are deduplicated by article ID. Green borders mark exact article-ID matches. Product-name/colour similarity alone does not count as a hit.

The five examples are not random or estimates of population performance. Across the full test cohort, 268/2,452 customers (10.9%) have any top-12 hit. The examples include failures so the visual does not imply uniformly successful personalization.

## Local product photos

No H&M photo directory was present in the downloaded data. Current examples therefore use actual product names, categories, colour labels and illustrative swatches. No images are generated or fetched from unrelated products.

If you separately obtain the H&M images locally, the script auto-detects `data/raw/images/` or `data/images/`, or accepts:

```bash
PYTHONPATH=src MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m ranking.cli applied --image-dir data/raw/images
```

Supported layout: `<image-root>/<first-three-article-ID-digits>/<article-ID>.jpg`; a flat `<image-root>/<article-ID>.jpg` is also accepted. Missing photos fall back to metadata cards. Local source photos stay ignored. Check the dataset terms before sharing derived figures containing competition photos.

## Metric definitions and product limits

- Activity bins use all recorded transactions strictly before the cutoff: 0, 1–2, 3–10 and >10. They are not equal-sized bins; the 1–2 group has 76 customers, versus 1,896 in >10.
- Catalog eligibility uses the original 120-day lookback: 23,120 articles. It is a purchase-history heuristic, not confirmed inventory.
- Popularity rank uses transaction counts in that same lookback, breaking ties by article ID. Bins are top 1%, next 9%, and remaining 90%; the last is this report's explicitly defined long tail.
- Exposure shares count recommendation slots across all test users, rather than counting each distinct product once. Unique-item coverage is reported separately.
- Novelty and content diversity are reused from the original diagnostics. Coverage and tail share are not substitutes for either metric.
- Larger coverage than popularity is accompanied by lower coverage than recent purchases. No-history performance is worse than the popularity fallback. These observations motivate an online policy experiment, not a claim of conversion lift.
- Case success/failure and segment comparisons are descriptive on a fixed held-out week. No subgroup significance, causal effect or model-training uncertainty is inferred.

The README A/B proposal now uses the stronger recent-purchase baseline as control. No online experiment or production deployment has been performed.
