# Experiment design

## Task

Recommend 12 distinct H&M articles for the next seven days from a customer's prior transactions. Repeated purchases are allowed. Evaluate macro MAP@12, Recall@12, Precision@12, NDCG@12, and MRR@12 on customers with a purchase in the target week. All methods use the same evaluation cohort, including customers without prior purchases. A missing recommendation counts as zero; a short list uses K=12 for the precision denominator.

## Retrieval

1. **Popularity:** 28-day global transaction counts with exponential decay, scale 14 days.
2. **Age-segment popularity:** the same popularity calculation within <25 / 25–34 / 35–44 / 45+ / unknown. Missing segment history falls back to global popularity. No gender is inferred.
3. **Personal recency:** customer–article purchase counts with exponential decay, scale 30 days.
4. **Co-occurrence:** binary user–article incidence, cosine-normalized shared-user counts, at most 50 neighbors per article. Self-neighbors are removed.
5. **Implicit ALS:** sparse confidence-weighted factorization with `p_ui=1` for purchases and `c_ui=1+15*r_ui`. Exact alternating ridge updates include the contribution of all missing pairs through the factor Gram matrix. This is implicit ALS, not truncated SVD renamed as ALS.
6. **Content:** TF-IDF on descriptions and article attributes, truncated SVD, normalized article vectors and recency-weighted user profiles. Vocabulary and components are fitted separately on the eligible catalog for each cutoff.

Each source supplies up to 40 items. The union deduplicates articles while retaining source indicators and retrieval scores. Reciprocal-rank fusion orders the pool if a cap is used; the current 240-item cap includes all six source quotas. No test/training label is injected to boost candidate recall. Source-level recall uses the original unfilled retrieval list; baseline top-12 lists use historical popularity to fill a short personal list.

## Features and labels

Features include customer activity/RFM proxies, age, channel preference, category diversity; article popularity over 7/14/28 days and trends; prior customer–article purchases and recency; category/color/department affinity and price gap; ALS/content/co-occurrence scores; and source flags/ranks. H&M `price` is treated as a dataset monetary proxy, not interpreted as USD revenue.

One query means `(customer_id, cutoff)`, not customer alone across multiple weeks. Only retrieved positives are retained. Candidate hard negatives are sampled at up to 10 per positive for training. Zero-positive or all-positive queries cannot supply effective ranking pairs and are dropped from model training, while all such users remain in evaluation. Validation/test score the full candidate pool without negative sampling. Candidate recall includes target items missing from retrieval.

## Models

| Model | Objective |
|---|---|
| Standardized logistic regression | Binary log loss |
| Random forest | Classification |
| LightGBM classifier | Binary log loss |
| XGBoost classifier | Binary log loss |
| XGBRanker pairwise | Pairwise logistic ranking (`rank:pairwise`) |
| XGBRanker NDCG | NDCG-weighted LambdaMART (`rank:ndcg`) |
| LightGBM LambdaMART | NDCG-weighted lambda gradients (`lambdarank`) |

LambdaMART is not a separate neural/listwise-loss model. It uses metric-weighted pairwise gradients. XGBRanker is the XGBoost interface supporting several objectives. See [XGBoost's official LTR explanation](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html) and [LightGBM's ranker API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html).

The initial tuning grid compares LightGBM LambdaMART depths 3 and 5 using validation MAP@12. Other models use fixed settings. The selected complete policy can be a strong heuristic baseline if it wins validation. The best ML ranker is also recorded for feature diagnostics; its selection is independent of final test results. All models refit on train plus validation weeks with settings frozen.

## Analysis

- Compare all baselines and rankers on identical users and target sets.
- Source recall and union recall measure retrieval separately from ordering.
- Candidate ablation removes articles unique to a source with the ranker and remaining article features fixed.
- Feature ablation retrains the chosen ranker after removing a feature family. Composite source-count/RRF features are removed when ablating a retrieval family to avoid indirect proxies. Retrieval membership stays fixed, so these are feature-value ablations, not deletion of an entire source's information.
- Customer segments use prior lifetime purchase counts (0, 1–2, 3–10, >10), age and product-type diversity. Model features still use the 120-day window.
- Target item age distinguishes never observed, first observed in the last seven days, and established items. These conditional metrics overlap across users and do not sum to overall MAP.
- Coverage uses the eligible catalog denominator; novelty uses Laplace-smoothed prior purchase probabilities; within-list diversity uses content cosine distance.
- Paired customer bootstrap compares the validation-selected policy with popularity. Its interval is conditional on this fixed split/model/cohort and does not capture model-training or temporal uncertainty.

Test ablation and segment analyses are descriptive, after policy settings are frozen. They should not be used to make a new policy claim on the same test.
