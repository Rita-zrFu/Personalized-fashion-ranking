# Proposed online A/B test — not executed

## Decision

Can a personalized two-stage policy improve purchase conversion relative to a recent-popularity policy while meeting product guardrails? Only advance an ML policy if offline relevance and serving cost justify an experiment. A stronger heuristic that wins validation remains a valid candidate for treatment.

## Assignment and exposure

Randomize eligible customers 1:1 with a persistent customer-level assignment before their first eligible recommendation request. Control shows the same 12-slot component with recent-purchase ranking and a decayed-popularity fallback. Treatment uses hybrid candidates plus the validation-selected personalized scorer. Keep layout, prices, eligibility rules, and logging identical. The offline analysis shows worse zero-history performance for the personalized scorer, so separately specify and validate a segment/global-popularity fallback for that group before using it in a treatment policy. This fallback modification is not included in the current reported personalized offline scores.

Log assignment, exposure, request timestamp, policy/model version, article IDs, positions, availability, latency, clicks, add-to-cart, purchases, revenue, and returns. Keep users in their assigned arm throughout the experiment. Use intention-to-treat analysis; report exposure and logging failures separately.

## Metrics

- Primary: fraction of assigned eligible customers with a purchase within a prespecified seven-day observation window.
- Secondary: recommendation CTR, add-to-cart rate, and revenue per assigned eligible customer. Sessions may be reported descriptively; customer is the randomization/analysis unit.
- Guardrails: p95/p99 serving latency, recommendation failure rate, unavailable-item rate, return rate measured after sufficient follow-up, and catalog coverage/diversity.

Verify sample-ratio balance and tracking before interpreting outcomes. Prespecify the baseline rate, minimum detectable absolute effect, alpha, power, and sample size from real product telemetry; this purchase-only dataset cannot supply a reliable online denominator or baseline conversion rate. Include at least complete weekly cycles and the full outcome follow-up. Do not stop early based on repeatedly inspected unadjusted p-values.

Analyze the primary difference with a customer-level confidence interval. Treat age/activity segment comparisons as exploratory unless a multiplicity plan was prespecified. A launch decision requires primary evidence plus acceptable guardrails, not an offline MAP improvement alone.

## Operational policy

Batch-compute retrieval factors and features. Fetch eligible candidates, apply the trained scorer, deduplicate, and return 12 slots. Apply inventory exclusions only with a real inventory feed. Preserve repeat purchases for the evaluated policy. Use global popularity for unknown customers or retrieval failures, and version both candidate and ranker artifacts together.

This repository runs offline experiments. Production serving, live monitoring, CTR/revenue lift, and an online A/B result are not claimed.
