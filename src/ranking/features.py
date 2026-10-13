"""H&M user, article, affinity, and retrieval features from one past-only snapshot."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .candidates import SOURCES

FEATURE_GROUPS = {
    "user": ["age", "user_count", "user_unique", "user_mean_price", "user_median_price",
             "user_money", "user_recency", "user_tenure", "user_online_share", "user_diversity"],
    "item": ["item_count", "item_unique_users", "item_mean_price", "item_age", "sales_7d", "sales_14d",
             "sales_28d", "sales_trend"],
    "recency": ["pair_count", "pair_recency", "bought_before", "score_recent", "rank_recent", "from_recent"],
    "affinity": ["affinity_category", "affinity_colour", "affinity_department", "price_gap"],
    "als": ["score_als", "rank_als", "from_als"],
    "content": ["score_content", "rank_content", "from_content"],
    "cooccurrence": ["score_cooccurrence", "rank_cooccurrence", "from_cooccurrence"],
    "retrieval": ["retrieval_rrf", "source_count", "score_popularity", "score_segment",
                  "rank_popularity", "rank_segment", "from_popularity", "from_segment"],
}


def build_features(candidates, history, customers, articles, cutoff):
    cutoff = pd.Timestamp(cutoff)
    users = history.groupby("customer_id").agg(
        user_count=("article_id", "size"), user_unique=("article_id", "nunique"),
        user_mean_price=("price", "mean"), user_median_price=("price", "median"),
        user_money=("price", "sum"), last=("t_dat", "max"), first=("t_dat", "min"),
        user_online_share=("sales_channel_id", lambda v: v.eq(2).mean()))
    users["user_recency"] = (cutoff - users.pop("last")).dt.days
    users["user_tenure"] = (cutoff - users.pop("first")).dt.days
    items = history.groupby("article_id").agg(item_count=("customer_id", "size"),
        item_unique_users=("customer_id", "nunique"), item_mean_price=("price", "mean"), first=("t_dat", "min"))
    items["item_age"] = (cutoff - items.pop("first")).dt.days
    for days in [7, 14, 28]:
        items[f"sales_{days}d"] = history[history.t_dat.ge(cutoff - pd.Timedelta(days=days))].groupby("article_id").size()
    items = items.fillna(0)
    items["sales_trend"] = items.sales_7d - (items.sales_14d - items.sales_7d)
    pairs = history.groupby(["customer_id", "article_id"]).agg(pair_count=("price", "size"), last=("t_dat", "max"))
    pairs["pair_recency"] = (cutoff - pairs.pop("last")).dt.days
    frame = candidates.merge(users, on="customer_id", how="left").merge(items, on="article_id", how="left")
    frame = frame.merge(customers[["customer_id", "age"]], on="customer_id", how="left")
    frame = frame.merge(pairs, on=["customer_id", "article_id"], how="left")
    frame["bought_before"] = frame.pair_count.notna().astype(int)
    frame["pair_count"] = frame.pair_count.fillna(0)
    frame["pair_recency"] = frame.pair_recency.fillna(999)
    meta_cols = ["product_type_name", "colour_group_name", "department_name"]
    enriched = history.merge(articles[["article_id"] + meta_cols], on="article_id", how="left")
    item_meta = articles.set_index("article_id")
    for name, col in zip(["category", "colour", "department"], meta_cols):
        counts = enriched.groupby(["customer_id", col]).size().rename("affinity")
        totals = enriched.groupby("customer_id").size()
        frame[col] = frame.article_id.map(item_meta[col])
        mapped = frame[["customer_id", col]].merge(counts.reset_index(), on=["customer_id", col], how="left")
        frame[f"affinity_{name}"] = mapped.affinity.to_numpy() / frame.customer_id.map(totals).to_numpy()
        frame = frame.drop(columns=col)
    diversity = enriched.groupby("customer_id").product_type_name.nunique()
    frame["user_diversity"] = frame.customer_id.map(diversity)
    frame["price_gap"] = (frame.item_mean_price - frame.user_mean_price).abs()
    frame["source_count"] = frame[[f"from_{s}" for s in SOURCES]].sum(axis=1)
    frame["user_recency"] = frame.user_recency.fillna(999)
    frame["age"] = frame.age.fillna(-1)
    feature_cols = list(dict.fromkeys(c for group in FEATURE_GROUPS.values() for c in group))
    frame[feature_cols] = frame[feature_cols].fillna(0).astype("float32")
    return frame


def label_candidates(frame, target, cutoff):
    labels = target[["customer_id", "article_id"]].drop_duplicates().assign(label=1)
    frame = frame.merge(labels, on=["customer_id", "article_id"], how="left")
    frame["label"] = frame.label.fillna(0).astype("int8")
    frame["query_id"] = frame.customer_id + ":" + str(pd.Timestamp(cutoff).date())
    return frame


def sample_negatives(frame, ratio=10, seed=42):
    frame = frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    groups = []
    for _, group in frame.groupby("query_id", sort=True):
        positives = group[group.label.eq(1)]
        negatives = group[group.label.eq(0)]
        # No effective positive-negative ranking pair; leave out of ranker training.
        if positives.empty or negatives.empty:
            continue
        n = min(len(negatives), ratio * len(positives))
        selected = rng.choice(negatives.index.to_numpy(), n, replace=False)
        groups.extend([positives, frame.loc[selected]])
    if not groups:
        raise ValueError("No trainable candidate query groups; inspect candidate recall")
    return pd.concat(groups, ignore_index=True).sort_values(["query_id", "article_id"]).reset_index(drop=True)
