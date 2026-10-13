# Résumé — three verified bullets

**Personalized Recommendation & Learning-to-Rank | Independent Project | GitHub link pending**

- Built a two-stage fashion recommender on 1.16M H&M transactions from 50K customers, combining popularity, recency, item co-occurrence, implicit ALS, and TF-IDF/SVD content retrieval.
- Compared logistic regression, random forests, XGBoost/LightGBM classifiers, pairwise ranking, and LambdaMART with temporal validation, MAP@12, NDCG@12, and Recall@12.
- Achieved MAP@12 of 0.0261 on a held-out week with 2,452 purchasing customers, improving 17.3% over a recent-purchase baseline; analyzed cold start, customer segments, and candidate/feature ablations.

Evidence: reports/hm_sample/metrics.csv, run_manifest.json, snapshots, segments.csv and ablations.csv. The quantitative result is a fixed 50K-user development experiment. The improvement is relative MAP, not a 17.3-percentage-point change or online conversion lift. No Two-Tower, production deployment or online A/B experiment is claimed.

For a two-bullet version, combine the first two bullets and keep the quantitative result. Replace “GitHub link pending” only after a real repository exists.
