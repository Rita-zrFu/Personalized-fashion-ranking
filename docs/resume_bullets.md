# Résumé — three verified bullets

**Personalized Recommendation & Learning-to-Rank | Independent Project | GitHub link pending**

- Built a two-stage fashion recommender on 1.16M H&M transactions from 50K customers, combining popularity, recency, item co-occurrence, implicit ALS, and TF-IDF/SVD content retrieval.
- Compared logistic regression, random forests, XGBoost/LightGBM classifiers, pairwise ranking, and LambdaMART with temporal validation, MAP@12, NDCG@12, and Recall@12.
- Achieved MAP@12 of 0.0261 on a held-out week with 2,452 purchasing customers, improving 17.3% over a recent-purchase baseline; analyzed cold start, customer segments, and candidate/feature ablations.

Evidence: reports/hm_sample/metrics.csv, run_manifest.json, snapshots, segments.csv and ablations.csv. The quantitative result is a fixed 50K-user development experiment. The improvement is relative MAP, not a 17.3-percentage-point change or online conversion lift. No Two-Tower, production deployment or online A/B experiment is claimed.

For a two-bullet version, combine the first two bullets and keep the quantitative result. Replace “GitHub link pending” only after a real repository exists.

## Combined system: three strongest measured results

1. **Return-purchase forecasting:** the validation-selected 7-day Random Forest achieves ROC-AUC 0.8135 and AP/PR-AUC 0.2089 on 49,421 previously observed customers.
2. **Customer prioritization:** the highest-scored 10% have an 18.94% observed purchase rate (4.19× the 4.52% population rate) and capture 41.94% of returning purchasers. This is predictive lift, not causal uplift.
3. **Product ranking:** the unchanged ranker achieves MAP@12 0.02607, 17.3% above recent purchases, on its original 2,452-purchaser test cohort. High-propensity purchasers have conditional MAP@12 0.03385; this is a subgroup description, not a new overall ranking improvement.

Evidence: reports/hm_sample/propensity/selected_models.csv, operating_points.csv, recommendation_segments.csv and the original ranking metrics. The propensity cohort excludes 220 future-first-time buyers; the original ranking cohort includes them, so denominators must be stated correctly.

Optional combined résumé version, still three bullets:

- Built a time-aware fashion recommendation system on 1.16M H&M transactions from 50K customers, combining hybrid retrieval, collaborative filtering, content features, and personalized product ranking.
- Engineered longitudinal shopping-rhythm features and compared logistic regression, random forests, XGBoost and LightGBM across temporal purchase horizons; achieved 7-day ROC-AUC 0.813 and PR-AUC 0.209, with 4.19× purchase-rate lift in the top propensity decile.
- Compared pointwise and learning-to-rank models, improving held-out MAP@12 by 17.3% over a recent-purchase baseline; integrated customer prioritization with recommendation, cold-start, segment and ablation analysis.
