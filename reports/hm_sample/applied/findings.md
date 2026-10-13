# Applied recommendation findings

Generated from the saved, validation-selected policy and the original temporal test. No retraining or test-driven policy change.

MAP@12 is 0.02607 vs 0.02224 for recent purchases (17.3% relative improvement).
268 of 2,452 test purchasers (10.9%) receive at least one exact article match.

## Activity segments

| Prior lifetime purchases | Customers | Personalized MAP@12 | Recent MAP@12 | Absolute difference |
|---|---:|---:|---:|---:|
| No history | 220 | 0.00736 | 0.01055 | -0.00319 |
| 1–2 purchases | 76 | 0.05009 | 0.03988 | +0.01022 |
| 3–10 purchases | 260 | 0.02626 | 0.02634 | -0.00008 |
| >10 purchases | 1,896 | 0.02726 | 0.02232 | +0.00494 |

The largest observed absolute MAP gain is in the 1-2-purchase group (n=76). The >10 group has 1,896 users and MAP changes by +0.00494. The 3–10 group changes by -0.00008 in MAP. Zero-history MAP is 0.00736 vs 0.01055 for the baseline. These are descriptive comparisons on one test week, not segment-specific causal or significance claims.

## Catalog exposure

Personalization exposes 3,251 distinct items (14.06% of the eligible catalog), versus 12 (0.052%) for popularity and 8,061 (34.87%) for recent purchases.
Long-tail recommendation-slot share is 13.23% for personalization, 0.00% for popularity and 28.08% for recent purchases. Long tail means outside the top 10% of the prior 120-day eligible catalog, not an industry-standard exposure threshold.
The selected scorer creates broader exposure than popularity, while concentrating recommendations more than the strong recent-purchase baseline. Coverage alone does not establish better relevance, novelty, or within-list diversity.

## Examples and images

Five cases curated after observing test outcomes; not a random sample or population success estimate.
Five cases illustrate a strong match, partial match, an active-customer miss, a low-history match and a true cold-start miss. Green borders mark exact article-ID matches, not merely similar clothing. All 12 ranked items are shown; history shows up to four recent distinct articles. Anonymous aliases are presentation labels; real IDs and detailed case mappings remain in ignored local outputs.
Actual local photos when available; otherwise category/name/colour metadata cards, not product photos.

## Proposed online experiment

Control: recent purchases with a popularity fallback. Treatment: hybrid candidates plus the frozen selected scorer. Randomize persistently at customer level. Primary: seven-day purchase conversion per assigned eligible customer. Secondary: recommendation CTR, add-to-cart, and revenue per assigned customer. Guardrails: p95/p99 latency, failures, unavailable items, returns, catalog coverage and diversity. This is a proposal, not an H&M experiment performed by this project.

Observed cold-start weakness suggests keeping a popularity fallback for zero-history customers in a separately specified treatment policy. That modification has not been included in the reported offline metrics.

## Reproduce

Run `make applied` from the project root after generating or restoring the saved run. See docs/applied_results.md for inputs and optional local photo support.
