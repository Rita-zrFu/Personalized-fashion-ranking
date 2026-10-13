"""Fully observed future-horizon targets for historical customer cohorts."""
import pandas as pd


def build_targets(features, transactions, cutoff, horizon_days):
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.Timedelta(days=horizon_days)
    observed_end = transactions.t_dat.max().normalize() + pd.Timedelta(days=1)
    if end > observed_end:
        raise ValueError(f"Incomplete follow-up: {cutoff.date()} + {horizon_days} days exceeds {observed_end.date()}")
    future = transactions[transactions.t_dat.ge(cutoff) & transactions.t_dat.lt(end)]
    buyers = set(future.customer_id)
    result = features.copy()
    result["target"] = result.customer_id.isin(buyers).astype("int8")
    audit = dict(cutoff=str(cutoff.date()), horizon_days=horizon_days,
                 outcome_start=str(cutoff.date()), outcome_end_inclusive=str((end - pd.Timedelta(days=1)).date()),
                 follow_up_complete=True, eligible_customers=len(result), buyers=int(result.target.sum()),
                 prevalence=float(result.target.mean()), excluded_first_time_buyers=len(buyers - set(result.customer_id)))
    return result, audit


def check_temporal_windows(windows, horizon_days):
    cutoffs = windows["train_cutoffs"] + [windows["validation_cutoff"], windows["test_cutoff"]]
    for first, second in zip(cutoffs, cutoffs[1:]):
        assert pd.Timestamp(first) + pd.Timedelta(days=horizon_days) <= pd.Timestamp(second)
