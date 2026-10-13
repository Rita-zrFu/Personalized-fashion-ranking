# H&M data

Source: [H&M Personalized Fashion Recommendations — official competition data](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data).

Local `raw/` contains `transactions_train.csv`, `customers.csv`, and `articles.csv`.
The CSVs and downloaded ZIPs are excluded from Git. Use the competition's access process and terms; this repository does not redistribute the data.

`make prepare` reads the full transaction CSV in chunks and selects 50,000 customers using a seeded hash of customer ID. It retains all transactions for those users; it never selects users because of future purchases. IDs remain strings so article leading zeros survive. Repeated transaction rows are kept because they may represent multiple purchased units.

Processed Parquet tables and the exact selection manifest live in `processed/hm_sample/` and are excluded from Git.
The development sample contains 1,156,381 transactions from 2018-09-20 through 2020-09-22. The original source CSV contains 31,788,324 transactions. All 105,542 article records are retained as metadata.

No images, private customer information, credentials, raw customer IDs, row-level predictions, or transaction extracts belong in the public repository. Publish aggregate tables/figures from the report only.

## Temporal experiment

| Snapshot cutoff | Label period | Purpose |
|---|---|---|
| 2020-08-26 | Aug 26–Sep 1 | Ranker training |
| 2020-09-02 | Sep 2–8 | Ranker training |
| 2020-09-09 | Sep 9–15 | Validation and policy/depth selection |
| 2020-09-16 | Sep 16–22 | Final held-out test |

Each retrieval model and behavioral feature uses purchases **strictly earlier** than its own cutoff, with a 120-day lookback. After selecting settings on validation, train and validation candidate rows are combined to refit the rankers before evaluating the final test.

## Known data limits

The data records purchases rather than impressions, clicks, live inventory, or item release times. Unpurchased candidates are unlabeled implicit feedback, not confirmed dislikes. Product metadata and age are supplied in a single snapshot. Customer fields such as Active, membership, newsletter status, and postal_code are deliberately unused.

Catalog eligibility means observed in the last 120 days before the cutoff. It is a documented heuristic, not an inventory guarantee. Cold-item analysis distinguishes first observed in the preceding week from never observed before the cutoff; the latter cannot be retrieved under this catalog rule. Sample-based first observation is not a reliable launch date.
