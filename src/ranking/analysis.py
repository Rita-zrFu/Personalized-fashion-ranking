"""EDA and product diagnostics for the actual experiment cohort."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import age_segment
from .metrics import per_user_metrics


def eda(transactions, customers, articles, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    weekly = transactions.assign(week=transactions.t_dat.dt.to_period("W").dt.start_time).groupby("week").agg(
        transactions=("article_id", "size"), active_customers=("customer_id", "nunique"),
        active_items=("article_id", "nunique"))
    weekly.to_csv(output / "weekly_activity.csv")
    user = transactions.groupby("customer_id").agg(transactions=("article_id", "size"),
        unique_items=("article_id", "nunique"), first=("t_dat", "min"), last=("t_dat", "max"),
        average_price=("price", "mean"))
    user["days_since_last_purchase"] = (transactions.t_dat.max() - user["last"]).dt.days
    user["purchases_per_observed_month"] = user.transactions / np.maximum(1, (user["last"] - user["first"]).dt.days / 30)
    user.describe().to_csv(output / "user_summary.csv")
    pop = transactions.groupby("article_id").size().sort_values(ascending=False)
    pop.rename("transactions").reset_index().assign(popularity_rank=lambda f: np.arange(1, len(f)+1)).to_csv(output / "item_popularity.csv", index=False)
    item = transactions.groupby("article_id").agg(first=("t_dat", "min"), last=("t_dat", "max"))
    item["observed_lifespan_days"] = (item["last"] - item["first"]).dt.days
    item.describe().to_csv(output / "item_lifespan_summary.csv")
    introductions = item.assign(week=item["first"].dt.to_period("W").dt.start_time).groupby("week").size()
    introductions.rename("first_observed_items").to_csv(output / "item_introductions.csv")
    for col in ["product_type_name", "colour_group_name", "product_group_name"]:
        articles[col].value_counts().rename("articles").to_csv(output / f"catalog_{col}.csv")
    summary = dict(transactions=len(transactions), customers_in_metadata=len(customers),
        customers_with_transactions=transactions.customer_id.nunique(), items_with_transactions=transactions.article_id.nunique(),
        catalog_items=len(articles), start=str(transactions.t_dat.min().date()), end=str(transactions.t_dat.max().date()),
        repeat_transaction_fraction=float(1 - len(transactions[["customer_id", "article_id"]].drop_duplicates()) / len(transactions)),
        user_transaction_quantiles=user.transactions.quantile([.25, .5, .75, .9, .99]).to_dict())
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=int) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    weekly.transactions.plot(ax=axes[0], title="Weekly transactions (experiment cohort)")
    axes[1].plot(np.arange(1, len(pop)+1), pop.to_numpy()); axes[1].set(xscale="log", yscale="log", title="Item popularity long tail", xlabel="Popularity rank", ylabel="Transactions")
    axes[2].hist(user.transactions, bins=np.geomspace(1, user.transactions.max()+1, 35)); axes[2].set(xscale="log", title="Purchases per customer", xlabel="Transactions", ylabel="Customers")
    fig.tight_layout(); fig.savefig(output / "eda.png", dpi=160); plt.close(fig)
    return summary


def segment_analysis(predictions_by_model, truth, full_history, customers, articles, cutoff, k):
    count = full_history.groupby("customer_id").size().to_dict()
    diversity = full_history.merge(articles[["article_id", "product_type_name"]], on="article_id").groupby("customer_id").product_type_name.nunique().to_dict()
    ages = customers.set_index("customer_id").age.to_dict()
    first = full_history.groupby("article_id").t_dat.min().to_dict()
    new_boundary = pd.Timestamp(cutoff) - pd.Timedelta(days=7)
    rows = []
    for name, predictions in predictions_by_model.items():
        per_user = per_user_metrics(predictions, truth, k)
        per_user["activity"] = per_user.customer_id.map(lambda u: "zero_history" if count.get(u, 0) == 0 else "1-2" if count[u] <= 2 else "3-10" if count[u] <= 10 else ">10")
        per_user["age"] = per_user.customer_id.map(lambda u: age_segment(ages.get(u, np.nan)))
        per_user["diversity"] = per_user.customer_id.map(lambda u: "unknown" if count.get(u, 0) == 0 else "specialist" if diversity.get(u, 0) <= 3 else "generalist")
        for axis in ["activity", "age", "diversity"]:
            for segment, group in per_user.groupby(axis):
                rows.append(dict(model=name, axis=axis, segment=segment, users=len(group),
                    **{f"{m}@{k}": float(group[m].mean()) for m in ["map", "recall", "ndcg"]}))
        for bucket, predicate in [
            ("unseen_before_cutoff", lambda i: i not in first),
            ("first_seen_last_7d", lambda i: i in first and first[i] >= new_boundary),
            ("established", lambda i: i in first and first[i] < new_boundary)]:
            filtered = {u: {i for i in y if predicate(i)} for u, y in truth.items()}
            filtered = {u: y for u, y in filtered.items() if y}
            if filtered:
                metrics = per_user_metrics(predictions, filtered, k)
                rows.append(dict(model=name, axis="target_item_age", segment=bucket, users=len(metrics),
                    **{f"{m}@{k}": float(metrics[m].mean()) for m in ["map", "recall", "ndcg"]}))
    return pd.DataFrame(rows)


def catalog_diagnostics(predictions, retrieval, k):
    all_items = {i for items in predictions.values() for i in items}
    counts = retrieval.history.groupby("article_id").size().reindex(retrieval.items, fill_value=0).to_numpy()
    probability = (counts + 1) / (counts.sum() + len(counts))
    novelty, diversity = [], []
    for items in predictions.values():
        indices = [retrieval.item_index[i] for i in items[:k] if i in retrieval.item_index]
        if indices:
            novelty.append(float((-np.log2(probability[indices])).mean()))
        if len(indices) > 1:
            embedding = retrieval.content[indices]
            similarities = embedding @ embedding.T
            diversity.append(float((1 - similarities[np.triu_indices(len(indices), 1)]).mean()))
    return dict(catalog_coverage=float(len(all_items) / len(retrieval.items)),
                mean_novelty_bits=float(np.mean(novelty)),
                mean_content_diversity=float(np.mean(diversity)), recommended_items=len(all_items))
