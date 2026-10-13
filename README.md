# Personalized Recommendation & Learning-to-Rank

A two-stage H&M fashion recommender: retrieve relevant articles, then learn a personalized top-12 ordering. The project compares six retrieval baselines and seven ML scorers using weekly temporal holdouts, candidate recall, cold-start/segment analysis, and candidate/feature ablations.

**Completed development experiment:** 50,000 customers, 1,156,381 actual transactions, 105,542 article metadata records. The raw CSV contains 31,788,324 transactions; model experiments use the fixed customer sample. These are offline development results, not full-dataset Kaggle scores or online business lift.

## Key Results

The validation-selected **XGBoost classifier scorer** improves held-out **MAP@12 by 17.3% relative to recent-purchase ranking**. Test: September 16–22, 2020; 2,452 purchasing customers, including zero-history users. Every policy uses the same users and future-purchase targets.

| Metric | Personalized XGBoost | Recent purchases | Decayed popularity |
|---|---:|---:|---:|
| MAP@12 | **0.02607** | 0.02224 | 0.00505 |
| Recall@12 | **0.05530** | 0.04544 | 0.02132 |
| NDCG@12 | **0.03862** | 0.03239 | 0.01105 |
| Eligible-catalog coverage | **14.06%** | 34.87% | 0.052% |

Six candidate generators produce **146.6 distinct candidates/customer on average**, with **12.50% candidate recall**. Broader exposure than popularity comes with less coverage than recent-purchase ranking.

![End-to-end recommendation system and verified results](reports/hm_sample/applied/figures/system_overview.png)

## Applied Recommendation Results

### What does a customer receive?

![Recent history, personalized top-12 and actual future purchases for five anonymous cases](reports/hm_sample/applied/figures/recommendation_examples.png)

Five curated cases show strong and partial matches, an active-customer miss, a low-history match, and a true cold-start miss. Green borders mark **exact article-ID matches**; similar names/colours can refer to distinct articles. All 12 ranked items are shown. Local product photos are unavailable, so the cards use actual product names, categories and colour metadata; swatches are illustrative.

These cases were selected after observing test outcomes. They illustrate behavior rather than population success: **268/2,452 customers (10.9%)** receive at least one exact match.

### For which customers does personalization help?

![MAP and Recall by prior customer activity, including baseline comparisons](reports/hm_sample/applied/figures/customer_segments.png)

The 1–2-purchase group has the largest observed absolute MAP gain (**+0.01022**, only **76 users**). The >10 group improves by **+0.00494** across **1,896 users**. The 3–10 group is approximately tied on MAP, while zero-history MAP falls from **0.01055 to 0.00736**. These are descriptive results for this week; a cold-start fallback deserves a separate policy evaluation.

### Does the model only expose popular products?

![Eligible-catalog coverage and recommendation mix by historical item popularity](reports/hm_sample/applied/figures/catalog_exposure.png)

The personalized scorer exposes **3,251 distinct articles**, versus **12** for popularity and **8,061** for recent purchases. Long-tail articles account for **13.23%** of personalized recommendation slots, versus **0%** for popularity and **28.08%** for recent purchases. Long tail here means outside the top 10% of the 23,120-item eligible catalog, ranked by purchases in the preceding 120 days.

Personalization creates broader exposure than popularity, while prioritizing relevance over the broader recent-purchase mix. Coverage is a separate metric from within-list diversity and business conversion.

See [generated applied findings and tables](reports/hm_sample/applied/findings.md) and [reproduction/selection details](docs/applied_results.md). The original model comparisons and ablations follow below.

## Technical Model Comparison

| Model | MAP@12 | Recall@12 | NDCG@12 |
|---|---:|---:|---:|
| Decayed popularity | 0.00505 | 0.02132 | 0.01105 |
| Age-segment popularity | 0.00732 | 0.02203 | 0.01305 |
| Recent purchases + popularity fallback | 0.02224 | 0.04544 | 0.03239 |
| Implicit ALS + popularity fallback | 0.01203 | 0.02832 | 0.01917 |
| Logistic scorer | 0.02462 | 0.05252 | 0.03615 |
| Random forest scorer | 0.02507 | 0.05287 | 0.03707 |
| LightGBM classifier scorer | 0.02469 | 0.05381 | 0.03710 |
| **XGBoost classifier scorer** | **0.02607** | **0.05530** | **0.03862** |
| XGBoost pairwise ranker | 0.02432 | 0.05223 | 0.03623 |
| XGBoost NDCG LambdaMART | 0.02447 | 0.05288 | 0.03641 |
| LightGBM LambdaMART | 0.02495 | 0.05159 | 0.03676 |

![Model comparison and retrieval recall](reports/hm_sample/model_comparison.png)

The hybrid union reaches **12.50% macro candidate recall**, versus 5.34% for 40 popularity candidates, at about 147 candidates per customer. Retrieval remains the main limitation. Removing recency features reduces the selected scorer's MAP@12 to 0.02418; removing ALS features slightly improves it to 0.02641. An algorithm being more complex does not guarantee a better result.

Full comparisons, tuning, segments, diversity, ablations and the exact run configuration are in [the experiment report](reports/hm_sample/report.md). Two expanding temporal validation folds are in [rolling_validation.csv](reports/hm_sample/rolling_validation.csv). Bootstrap intervals and the sample/measurement limits accompany the report.

## Pipeline

```text
H&M transactions / customers / articles
  → weekly cutoffs with past-only behavioral features
  → popularity + age segment + recency + co-occurrence + implicit ALS + content retrieval
  → candidate union, source flags, candidate recall
  → user / article / affinity / interaction / retrieval features
  → candidate hard-negative sampling for training only
  → pointwise and grouped learning-to-rank models
  → distinct top-12 recommendations
  → temporal test, segments, cold start, diversity, ablations
```

Two training snapshots predict Aug 26–Sep 1 and Sep 2–8, 2020. Validation predicts Sep 9–15; final test predicts Sep 16–22. Each snapshot uses a 120-day lookback. A query is one customer-week. Settings are chosen on validation, then rankers are refitted with train plus validation rows before final test scoring. Actual target items absent from retrieval still count against recall and ranking metrics.

## Run locally

Run commands from this `ranking/` folder. Python 3.11+ is required. The models use three competition CSVs; optional local product photos are used only for the applied example figures.

**Saved run already available?** Generate the new applied outputs without retraining:

```bash
make applied
```

For a full reproduction from the official CSVs:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# Place the three official CSVs in data/raw/, then:
make prepare
make eda
make run
make rolling
make publish
make applied
```

`make run` took about 2.4 minutes on the local machine for this sample, excluding preparation and rolling-fold analysis. `make applied` reuses saved recommendations, candidate features and aggregate metrics; it verifies recomputed values against the original run and writes four figures, three aggregate tables, and findings. Private case mappings remain under ignored `reports/runs/`; public examples use anonymous A–E labels. `make publish` retains the original technical report.

`configs/hm_sample.json` controls this experiment. A larger customer cohort requires rerunning preparation and rebuilding all weekly features/retrievers; do not label these 50,000-user scores as full-dataset results. Final test data should not become an iterative tuning target.

## Repository guide

- `src/ranking/data.py`: H&M ingestion, stable customer sample, temporal windows.
- `candidates.py`: six retrieval sources, sparse implicit ALS, TF-IDF/SVD content model.
- `features.py`: historical features, customer-week labels, hard-negative sampling.
- `models.py`: seven pointwise / learning-to-rank scorers.
- `pipeline.py`: training, validation selection, held-out test and diagnostics.
- `analysis.py`, `rolling.py`, `reporting.py`: EDA, segments, temporal folds, aggregate reports.
- `applied.py`, `applied_plots.py`: saved-prediction examples, activity evaluation, exposure analysis and applied figures.
- `notebooks/`: walkthroughs for data, baselines and model interpretation.
- `docs/`: original proposal, experiment design, execution checklist, GitHub steps, A/B proposal and three résumé bullets.

Start with [the Chinese execution guide](docs/roadmap_zh.md) and [the incremental GitHub guide](docs/github_steps_zh.md).

## Product and Deployment Interpretation

**Proposed online A/B test — not executed.** Randomize customers persistently 1:1. Control: recent-purchase ranking with a popularity fallback. Treatment: hybrid candidate generation plus the final personalized scorer. Keep recommendation slots, layout, eligibility and tracking identical.

- **Primary:** seven-day purchase conversion per assigned eligible customer.
- **Secondary:** recommendation CTR, add-to-cart rate, and revenue per assigned customer.
- **Guardrails:** p95/p99 serving latency, recommendation failures, unavailable items, returns, catalog coverage and diversity.

A deployment could batch-compute retrieval artifacts and historical features, then retrieve, score, apply real inventory rules and return 12 distinct articles per request. Log policy version, exposure, item positions, latency and outcomes. The observed cold-start weakness motivates testing a popularity fallback for zero-history users; that modification is not part of the reported offline scores. The [detailed experiment proposal](docs/online_ab_test.md) covers assignment, follow-up and analysis. Offline purchase ranking does not establish online conversion or revenue lift.

## Interpretation

H&M records purchases, without logged impressions or inventory. A nonpurchase is not a verified dislike; classifier outputs after negative sampling are ranking scores, not calibrated conversion probabilities. Article eligibility is defined by purchases before the cutoff, and age/product attributes come from a static metadata snapshot. Cold-item analysis reports unseen future articles as unretrievable rather than assuming they were available. Coverage is measured against the 23,120-item eligible test catalog.

XGBRanker is an API supporting ranking objectives. `rank:ndcg` and LightGBM `lambdarank` are LambdaMART-style methods with metric-weighted pairwise gradients; this project does not call them distinct listwise/neural architectures. The optional two-tower extension has not been implemented. [The online A/B test](docs/online_ab_test.md) is a proposed next step, not a conducted experiment.

## Sources

- [Official H&M competition data](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)
- [XGBoost learning-to-rank documentation](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)
- [LightGBM ranker documentation](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html)
- [Hu, Koren & Volinsky: Collaborative Filtering for Implicit Feedback Datasets](https://doi.org/10.1109/ICDM.2008.22)

Raw competition data is not redistributed. Follow the original competition terms when obtaining or using it.
