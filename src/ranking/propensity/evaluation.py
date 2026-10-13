"""Probability quality and prospectively defined customer-priority operating points."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def calibration_bins(target, probabilities, bins=10):
    probabilities = np.asarray(probabilities)
    if len(np.unique(probabilities)) == 1:
        groups = np.zeros(len(probabilities), dtype=int)
    else:
        groups = pd.qcut(pd.Series(probabilities), bins, labels=False, duplicates="drop").to_numpy()
    frame = pd.DataFrame(dict(target=np.asarray(target), probability=probabilities, bin=groups))
    return frame.groupby("bin").agg(customers=("target", "size"), observed_rate=("target", "mean"),
                                     mean_probability=("probability", "mean")).reset_index()


def evaluate_probabilities(target, probabilities):
    probabilities = np.asarray(probabilities)
    assert np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1))
    bins = calibration_bins(target, probabilities)
    ece = np.average(np.abs(bins.observed_rate - bins.mean_probability), weights=bins.customers)
    return dict(customers=len(target), buyers=int(np.sum(target)), prevalence=float(np.mean(target)),
                roc_auc=float(roc_auc_score(target, probabilities)),
                pr_auc_ap=float(average_precision_score(target, probabilities)),
                log_loss=float(log_loss(target, probabilities, labels=[0, 1])),
                brier=float(brier_score_loss(target, probabilities)), calibration_ece=float(ece),
                mean_probability=float(np.mean(probabilities)))


def ranked_customers(customer_ids, target, probabilities):
    return pd.DataFrame(dict(customer_id=np.asarray(customer_ids), target=np.asarray(target),
                             propensity=np.asarray(probabilities))).sort_values(
        ["propensity", "customer_id"], ascending=[False, True], kind="stable").reset_index(drop=True)


def operating_points(customer_ids, target, probabilities, fractions=(.1, .2)):
    ordered = ranked_customers(customer_ids, target, probabilities)
    population_rate = ordered.target.mean()
    rows = []
    for fraction in fractions:
        n = max(1, int(np.ceil(len(ordered) * fraction)))
        selected = ordered.head(n)
        buyers = int(selected.target.sum())
        rows.append(dict(top_fraction=fraction, customers=n, buyers=buyers,
            precision=float(selected.target.mean()), recall=float(buyers / ordered.target.sum()),
            lift=float(selected.target.mean() / population_rate), population_rate=float(population_rate),
            minimum_score=float(selected.propensity.min())))
    return pd.DataFrame(rows)


def propensity_deciles(customer_ids, target, probabilities):
    ordered = ranked_customers(customer_ids, target, probabilities)
    # Capacity bins are exactly ranked counts; score ties break by customer ID, never outcome.
    ordered["decile"] = np.minimum(9, np.arange(len(ordered)) * 10 // len(ordered)) + 1
    return ordered.groupby("decile").agg(customers=("target", "size"), buyers=("target", "sum"),
        observed_purchase_rate=("target", "mean"), mean_propensity=("propensity", "mean")).reset_index()


def gains_curve(customer_ids, target, probabilities):
    ordered = ranked_customers(customer_ids, target, probabilities)
    captured = ordered.target.cumsum().to_numpy() / ordered.target.sum()
    positions = np.unique(np.r_[1, np.ceil(np.linspace(.01, 1, 100) * len(ordered)).astype(int)])
    fraction = positions / len(ordered)
    return pd.DataFrame(dict(customer_fraction=fraction, buyer_fraction=captured[positions-1],
                             cumulative_lift=captured[positions-1] / fraction))
