# Longitudinal purchase propensity: design and reproduction

## Two related decisions

The original ranker predicts which articles are relevant to a customer. The new landmark model predicts whether an already observed customer will return within a specified horizon. Its customer-level score controls prioritization, not ordering within an article list. Multiplying every item's score by the same customer probability would not change that customer's ranking.

This is horizon-based binary prediction informed by longitudinal timing features. It is not a fitted continuous-time survival model, an estimate of exact next-purchase time, or causal uplift modeling.

## Cohort eligibility and labels

Each landmark includes every sampled customer with at least one transaction strictly before its cutoff. Long-inactive customers remain eligible negatives. Customers are never selected because they purchase in the outcome window. The final customer metadata table has no registration timestamps; using its final membership to infer enrollment earlier in time would require an unsupported assumption. Therefore, customers first observed purchasing in the future are excluded from the return-purchase classifier and counted explicitly in `cohorts.csv`.

At the 7-day test cutoff there are 49,421 eligible customers, 2,232 return purchasers and 220 additional first-time buyers in the original recommendation test. Those 220 retain their original recommendation metrics as a separate unmodeled group. A clearly labeled prior-only placeholder is recorded privately for joining their recommendations, without calling it a learned or validated cold-start probability.

Labels indicate at least one recorded purchase in `[T, T+horizon)`. Every target window must finish by 2020-09-23 (exclusive), the dataset's observed endpoint. No truncated window is labeled negative.

## Exact windows

| Horizon | Training cutoffs | Validation cutoff | Test cutoff | Test outcome window |
|---|---|---|---|---|
| 7 days | Aug 26; Sep 2 | Sep 9 | Sep 16 | Sep 16–22 |
| 14 days | Jul 29; Aug 12 | Aug 26 | Sep 9 | Sep 9–22 |
| 28 days | Jun 3; Jul 1 | Jul 29 | Aug 26 | Aug 26–Sep 22 |

All dates are 2020. Consecutive outcome windows are nonoverlapping. Longer-horizon tests move earlier to observe full follow-up; they do not estimate a monotone 7/14/28-day probability curve at the same cutoff. Two 7-day rolling folds predict Sep 2–8 from the Aug 26 training week and Sep 9–15 from both training weeks. They use fixed defaults rather than a parameter selected in a later week.

## The 43 features

- Recency and observed tenure from first/last recorded purchase.
- Historical transaction rows, unique articles, shopping days, active weeks, items per shopping day and transactions per observed/active week.
- Transactions and distinct shopping days during the prior 7, 14, 28 and 56 days.
- Mean/median price, total normalized spend proxy, recent average price and change versus the preceding 28 days.
- Mean/median/std/min/max gaps between **distinct shopping days**, completed-gap count and coefficient of variation.
- Recency divided by the typical gap, with a seven-day denominator floor. Missing gaps and insufficient variation have explicit flags.
- Recent-versus-earlier 28-day transaction/activity momentum and relative frequency change.
- Product-type diversity/entropy/dominant share, online-channel share and repeated article-event share.

Multiple items in one shopping basket do not create artificial zero-day timing gaps. Repeated article events use distinct day/article combinations; repeated units within the same day are not treated as repurchases. Missing gap statistics are filled with zero plus flags in model inputs, and shown as **not estimable** in the examples. Missing recent prices fall back to historical means; price-change missingness is explicit. `price` is a dataset spend proxy rather than currency revenue. Tenure/gaps are observed within the dataset's history, with unknown pre-dataset activity.

Product type comes from the supplied static metadata, assumed immutable. No customer age, Active, membership, postal code, or future transaction summary is used. Customer ID and cutoff are identifiers, never predictors.

## Models, tuning and calibration

Seven comparison entries: population prior, recency/frequency heuristic, raw-feature logistic, signed-log-feature logistic, Random Forest, XGBoost and LightGBM. The heuristic is an exponential recency score scaled by observed training prevalence; its raw numbers are an approximate baseline, not inherently calibrated probabilities.

Signed `log1p(abs(x))` preserves negative momentum while reducing count/gap/spend tails. Logistic C is compared at 0.1/1 for the log variant. Boosting depths are compared at 3/5; other settings are fixed. All positives and negatives remain in training, preserving class prevalence. Selection uses validation average precision, not accuracy; PR-AUC is reported as **average precision**, not trapezoidal PR integration.

After choosing settings on validation AP, sigmoid calibration fits a logistic map from the validation score logit to the validation outcome. Base models remain trained only on training weeks, avoiding a scoring-distribution change from refitting them after calibrator training. Validation serves both tuning and calibration, so validation metrics are not a fully independent generalization estimate; final test labels remain untouched for those decisions.

Evaluate ROC-AUC, AP, log loss, Brier and quantile-bin ECE. Compare raw/calibrated test results; calibration does not guarantee perfect individual probabilities or transport across weeks. Random Forest wins 7-day validation; LightGBM is slightly stronger by 7-day test AP. That descriptive difference does not change the selected model after inspecting test outcomes.

Capacity operating points are the top 10% and 20% of **all eligible customers**, sorted by predicted score with outcome-independent ID tie-breaking. Deciles and cumulative gains use the same ranks. Purchase-rate lift compares observed precision to population purchase rate; it is not incremental treatment-effect uplift.

## Integrating the two layers

High = top 20% of eligible customers; medium = next 30%; low = remaining 50%. Segments are defined before observing test outcomes. For all original recommendation-test buyers, join customer propensity with their unchanged top-12 lists. The original metrics, selection and predictions are hash-checked to confirm they remain unchanged.

MAP/Recall/NDCG by propensity segment condition on actual purchasers, with identical users across product-ranking baselines. They are distinct from return-purchase prediction metrics. Prioritization diagnostics also use all eligible customers as the capacity denominator and count observed exact-match customers. Nonbuyers have no future target items and thus no observed match, independent of what products they would receive. Their ranking lists are not generated for this aggregate capacity analysis, which cannot assess engagement, latency, exposure or causal impact.

Two selected nonbuyers need actual article lists for the integrated examples. Only one past-only test retriever is rebuilt, using the original configuration and original saved ranker; one cached buyer's list is checked for an exact match. No recommender retraining or entire-population recommendation scoring occurs. The five example cases are curated after outcomes and include true positives, high-score nonbuyers and low-score returning customers. Public case labels are anonymous; IDs and complete lists remain local/ignored. Product photos were not downloaded; the existing metadata-card fallback is used.

## Business use and future experiments

Propensity can allocate limited recommendation-serving or outreach capacity, and relevance can choose the products. It should not automatically exclude low-score customers: the low group still contains 191 purchasers, and retention, exploration and long-term value are not modeled. Cold-start customers need a separately evaluated acquisition policy.

A randomized comparison of uniform versus propensity-based prioritization should hold the product ranker and total budget fixed and measure customer-level conversion/engagement plus retention, coverage and latency guardrails. Comparing naturally high- and low-propensity outcomes is not such an experiment. An uplift-aware extension needs randomized recommendation assignment/exposure, unexposed controls and outcome follow-up; the current data does not supply them.

## Reproduce

From the repository root, with the existing `.venv` and prepared H&M tables:

```bash
# Inspect/produce landmark features and horizon labels only:
make propensity-features
# Full training, evaluation, frozen-ranker integration and six figures:
make propensity
```

`make propensity` includes feature generation. It took about 0.7 minutes in the measured local run. It reuses the original recommendation predictions, candidate features and saved selected ranker. If those ignored artifacts are absent, first reproduce the original run (`make prepare`, `make run`, `make rolling`, `make publish`, `make applied`). Configurations and paths are project-relative in `configs/propensity.json`; the CLI also accepts `--project-root`.

Public aggregate results live in `reports/hm_sample/propensity/`. Feature/label tables live in ignored `data/processed/propensity_hm_sample/`; individual scores and example mappings in ignored `reports/runs/hm_sample/propensity/`; serialized models/calibrators in ignored `models/hm_sample/propensity/`. No raw data, customer-level probabilities or large matrices belong in the public result folder.
