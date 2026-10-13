"""Macro user-level metrics, with explicit empty-target and duplicate behavior."""
from __future__ import annotations

import numpy as np
import pandas as pd


def user_metrics(predictions, relevant, k=12):
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant)
    # Duplicate slots receive no extra credit and are not compacted away.
    seen, hits = set(), []
    for item in list(predictions)[:k]:
        hits.append(int(item in relevant and item not in seen))
        seen.add(item)
    hits = np.asarray(hits, dtype=float)
    m = len(relevant)
    if not m:
        return dict(map=0., recall=0., precision=0., ndcg=0., mrr=0.)
    ranks = np.arange(1, len(hits) + 1)
    ap = float(np.sum(np.cumsum(hits) / ranks * hits) / min(m, k))
    dcg = float(np.sum(hits / np.log2(ranks + 1)))
    ideal = float(np.sum(1 / np.log2(np.arange(1, min(m, k) + 1) + 1)))
    positive_ranks = ranks[hits.astype(bool)]
    return dict(map=ap, recall=float(hits.sum() / m), precision=float(hits.sum() / k),
                ndcg=dcg / ideal, mrr=float(1 / positive_ranks[0]) if len(positive_ranks) else 0.)


def per_user_metrics(predictions, truth, k=12):
    if not truth:
        raise ValueError("Cannot evaluate an empty ground-truth cohort")
    return pd.DataFrame([dict(customer_id=u, target_items=len(y),
                              **user_metrics(predictions.get(u, []), y, k))
                         for u, y in sorted(truth.items())])


def evaluate(predictions, truth, k=12):
    rows = per_user_metrics(predictions, truth, k)
    return {f"{name}@{k}": float(rows[name].mean())
            for name in ["map", "recall", "precision", "ndcg", "mrr"]} | {"users": len(rows)}


def candidate_recall(candidates, truth):
    values = [len(set(candidates.get(u, [])) & y) / len(y)
              for u, y in truth.items() if y]
    return float(np.mean(values)) if values else 0.


def ranked_lists(frame, scores, k=12):
    ordered = frame[["customer_id", "article_id"]].assign(score=np.asarray(scores))
    if not np.isfinite(ordered.score).all():
        raise ValueError("Non-finite ranking scores")
    ordered = ordered.sort_values(["customer_id", "score", "article_id"],
                                  ascending=[True, False, True], kind="stable")
    ordered = ordered.drop_duplicates(["customer_id", "article_id"])
    return ordered.groupby("customer_id").head(k).groupby("customer_id").article_id.agg(list).to_dict()


def paired_bootstrap(first, second, replicates=1000, seed=42):
    aligned = first[["customer_id", "map"]].merge(second[["customer_id", "map"]],
                                                 on="customer_id", suffixes=("_a", "_b"), validate="one_to_one")
    if not len(aligned) or len(aligned) != len(first) or len(aligned) != len(second):
        raise ValueError("Bootstrap requires the same nonempty customer cohort")
    delta = (aligned.map_a - aligned.map_b).to_numpy()
    rng = np.random.default_rng(seed)
    means = [float(delta[rng.integers(0, len(delta), len(delta))].mean()) for _ in range(replicates)]
    return dict(delta_map=float(delta.mean()), ci_low=float(np.quantile(means, .025)),
                ci_high=float(np.quantile(means, .975)), bootstrap_replicates=replicates)
