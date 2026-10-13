"""Strict schemas, stable user sampling, and half-open temporal windows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TX_COLUMNS = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
ARTICLE_COLUMNS = ["article_id", "product_type_name", "product_group_name",
                   "colour_group_name", "department_name", "section_name",
                   "garment_group_name", "detail_desc"]


def validate_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(TX_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing transaction columns: {sorted(missing)}")
    frame = frame[TX_COLUMNS].copy()
    for col in ["customer_id", "article_id"]:
        if frame[col].isna().any():
            raise ValueError(f"Null {col}")
        frame[col] = frame[col].astype(str)
        if frame[col].str.strip().eq("").any():
            raise ValueError(f"Empty {col}")
    frame["t_dat"] = pd.to_datetime(frame["t_dat"], errors="raise")
    frame["price"] = pd.to_numeric(frame["price"], errors="raise")
    if frame.t_dat.isna().any() or not np.isfinite(frame.price).all() or frame.price.lt(0).any():
        raise ValueError("Invalid dates or prices")
    frame["sales_channel_id"] = pd.to_numeric(frame["sales_channel_id"], errors="raise")
    if not frame.sales_channel_id.isin([1, 2]).all():
        raise ValueError("Expected H&M sales_channel_id in {1, 2}")
    return frame


def prepare(raw_dir: Path, output: Path, max_users: int | None, seed: int) -> dict:
    """Choose users independently of outcomes; keep all their transaction rows.

    Does not take the first N transactions, or sample future purchasers. Chunked
    reading avoids loading the full original CSV into RAM for development runs.
    """
    for name in ["transactions_train.csv", "customers.csv", "articles.csv"]:
        if not (raw_dir / name).is_file():
            raise FileNotFoundError(f"Missing {raw_dir / name}. See data/README.md.")
    customers = pd.read_csv(raw_dir / "customers.csv", dtype={"customer_id": str},
                            usecols=["customer_id", "age"])
    if customers.customer_id.isna().any() or customers.customer_id.duplicated().any():
        raise ValueError("customers requires unique, non-null customer_id")
    customers["age"] = pd.to_numeric(customers.age, errors="coerce")
    if max_users is not None:
        if max_users <= 0:
            raise ValueError("max_users must be positive")
        hashes = customers.customer_id.map(
            lambda u: hashlib.blake2b(f"{seed}:{u}".encode(), digest_size=8).hexdigest())
        customers = customers.loc[hashes.sort_values().index[:max_users]].copy()
    user_ids = set(customers.customer_id)
    parts, source_rows = [], 0
    for chunk in pd.read_csv(raw_dir / "transactions_train.csv", usecols=TX_COLUMNS,
                             dtype={"customer_id": str, "article_id": str}, chunksize=500_000):
        source_rows += len(chunk)
        chunk = chunk[chunk.customer_id.isin(user_ids)]
        if len(chunk):
            parts.append(validate_transactions(chunk))
    if not parts:
        raise ValueError("No transactions for selected users")
    transactions = pd.concat(parts, ignore_index=True).sort_values("t_dat")
    articles = pd.read_csv(raw_dir / "articles.csv", dtype={"article_id": str},
                           usecols=lambda c: c in ARTICLE_COLUMNS)
    if articles.article_id.isna().any() or articles.article_id.duplicated().any():
        raise ValueError("articles requires unique, non-null article_id")
    if not set(transactions.article_id).issubset(set(articles.article_id)):
        raise ValueError("Transactions reference unknown articles")
    for col in ARTICLE_COLUMNS[1:]:
        if col not in articles:
            articles[col] = "unknown"
        articles[col] = articles[col].fillna("unknown").astype(str)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in [("transactions", transactions), ("customers", customers), ("articles", articles)]:
        frame.to_parquet(output / f"{name}.parquet", index=False)
    manifest = dict(dataset="hm", source=str(raw_dir.resolve()), seed=seed,
                    source_transaction_rows=source_rows, selected_customers=len(customers),
                    selected_transactions=len(transactions), articles=len(articles),
                    start=str(transactions.t_dat.min().date()), end=str(transactions.t_dat.max().date()),
                    sampling="stable blake2b hash of customer_id; all selected-user history retained",
                    duplicate_transactions="preserved: repeated rows may represent multiple units")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_tables(directory: Path):
    return tuple(pd.read_parquet(directory / f"{n}.parquet")
                 for n in ["transactions", "customers", "articles"])


def temporal_snapshot(transactions, cutoff, horizon_days=7, history_days=None):
    cutoff = pd.Timestamp(cutoff)
    history = transactions[transactions.t_dat.lt(cutoff)].copy()
    if history_days is not None:
        history = history[history.t_dat.ge(cutoff - pd.Timedelta(days=history_days))]
    target = transactions[transactions.t_dat.ge(cutoff) &
                          transactions.t_dat.lt(cutoff + pd.Timedelta(days=horizon_days))].copy()
    if history.empty or target.empty:
        raise ValueError(f"Empty history or target at {cutoff.date()}")
    return history, target


def ground_truth(target):
    return target.groupby("customer_id").article_id.agg(set).to_dict()
