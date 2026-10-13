"""Longitudinal customer features from purchases strictly before a landmark."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_customer_features(transactions, articles, cutoff):
    cutoff = pd.Timestamp(cutoff)
    history = transactions[transactions.t_dat.lt(cutoff)].copy()
    history["online"] = history.sales_channel_id.eq(2).astype("float32")
    # Enrollment times are unavailable: prior purchase establishes observability.
    # Do not select a customer's eligibility using their future purchase.
    group = history.groupby("customer_id", sort=True)
    features = group.agg(prior_transactions=("article_id", "size"),
        unique_articles=("article_id", "nunique"), active_days=("t_dat", "nunique"),
        mean_price=("price", "mean"), median_price=("price", "median"), spend_proxy=("price", "sum"),
        first_purchase=("t_dat", "min"), last_purchase=("t_dat", "max"),
        online_share=("online", "mean"))
    features["recency_days"] = (cutoff - features.pop("last_purchase")).dt.days
    features["tenure_days"] = (cutoff - features.pop("first_purchase")).dt.days
    features["active_weeks"] = history.assign(week=history.t_dat.dt.to_period("W")).groupby("customer_id").week.nunique()
    features["purchases_per_active_week"] = features.prior_transactions / features.active_weeks
    features["purchases_per_observed_week"] = features.prior_transactions / np.maximum(features.tenure_days / 7, 1)
    features["items_per_shopping_day"] = features.prior_transactions / features.active_days
    for days in [7, 14, 28, 56]:
        recent = history[history.t_dat.ge(cutoff - pd.Timedelta(days=days))]
        counts = recent.groupby("customer_id")
        features[f"transactions_{days}d"] = counts.size()
        features[f"active_days_{days}d"] = counts.t_dat.nunique()
    features = features.fillna(0)
    recent = history[history.t_dat.ge(cutoff - pd.Timedelta(days=28))]
    earlier = history[history.t_dat.ge(cutoff - pd.Timedelta(days=56)) &
                      history.t_dat.lt(cutoff - pd.Timedelta(days=28))]
    features["recent_mean_price_28d"] = recent.groupby("customer_id").price.mean()
    features["earlier_mean_price_28d"] = earlier.groupby("customer_id").price.mean()
    features["recent_price_missing"] = features.recent_mean_price_28d.isna().astype(int)
    features["price_change_missing"] = (features.recent_mean_price_28d.isna() | features.earlier_mean_price_28d.isna()).astype(int)
    features["price_change_28d"] = features.recent_mean_price_28d - features.earlier_mean_price_28d
    features["recent_mean_price_28d"] = features.recent_mean_price_28d.fillna(features.mean_price)
    features = features.drop(columns="earlier_mean_price_28d")
    features["previous_transactions_28d"] = features.transactions_56d - features.transactions_28d
    features["transaction_momentum_28d"] = features.transactions_28d - features.previous_transactions_28d
    features["activity_momentum_28d"] = features.active_days_28d - (features.active_days_56d - features.active_days_28d)
    features["relative_frequency_change"] = features.transaction_momentum_28d / (features.previous_transactions_28d + 1)
    shopping_days = history[["customer_id", "t_dat"]].drop_duplicates().sort_values(["customer_id", "t_dat"])
    shopping_days["gap"] = shopping_days.groupby("customer_id").t_dat.diff().dt.days
    intervals = shopping_days.groupby("customer_id").gap.agg(["count", "mean", "median", "std", "min", "max"])
    intervals.columns = ["gap_count", "gap_mean", "gap_median", "gap_std", "gap_min", "gap_max"]
    features = features.join(intervals)
    features["gap_missing"] = features.gap_count.eq(0).astype(int)
    features["gap_std_missing"] = features.gap_count.le(1).astype(int)
    features["recency_to_typical_gap"] = features.recency_days / features.gap_mean.fillna(7).clip(lower=7)
    features["gap_coefficient_variation"] = features.gap_std.fillna(0) / features.gap_mean.fillna(7).clip(lower=1)
    category_map = articles.set_index("article_id").product_type_name
    history["category"] = history.article_id.map(category_map).fillna("Unknown")
    category_counts = history.groupby(["customer_id", "category"]).size()
    category_shares = category_counts / category_counts.groupby(level=0).transform("sum")
    features["unique_categories"] = category_counts.groupby(level=0).size()
    features["category_entropy"] = (-(category_shares * np.log(category_shares))).groupby(level=0).sum()
    features["dominant_category_share"] = category_shares.groupby(level=0).max()
    day_articles = history[["customer_id", "t_dat", "article_id"]].drop_duplicates().groupby("customer_id").size()
    features["repeat_article_event_share"] = 1 - features.unique_articles / day_articles
    features = features.fillna(0).astype("float32")
    assert np.isfinite(features.to_numpy()).all()
    assert features.index.is_unique
    features = features.reset_index()
    features["cutoff"] = cutoff
    audit = dict(cutoff=str(cutoff.date()), eligible_customers=len(features), history_rows=len(history),
                 latest_feature_transaction=str(history.t_dat.max().date()),
                 cohort="all sampled customers with a purchase strictly before the cutoff; includes inactive customers",
                 feature_count=len(features.columns) - 2,
                 timing="gaps between distinct shopping days; the current unfinished gap is represented by recency",
                 metadata="static product type only; no customer age, membership or post-cutoff activity fields")
    return features, audit


def model_feature_columns(frame):
    return [c for c in frame.columns if c not in ["customer_id", "cutoff", "target"]]
