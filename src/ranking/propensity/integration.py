"""Prospective customer capacity ranks with unchanged personalized article ordering."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..applied import build_examples
from ..candidates import Retrieval
from ..data import ground_truth
from ..features import build_features
from ..metrics import per_user_metrics, ranked_lists
from ..models import predict as ranking_predict
from ..rolling import source_predictions


def propensity_segments(scores, config):
    ordered = scores.sort_values(["propensity", "customer_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    high_n = int(np.ceil(len(ordered) * config["high_propensity_fraction"]))
    middle_end = high_n + int(np.ceil(len(ordered) * config["medium_propensity_fraction"]))
    ordered["propensity_segment"] = np.where(np.arange(len(ordered)) < high_n, "high",
        np.where(np.arange(len(ordered)) < middle_end, "medium", "low"))
    return ordered


def choose_cases(scores, per_user):
    examples = scores.merge(per_user[["customer_id", "map", "hits"]], on="customer_id", how="left").fillna({"map":0, "hits":0})
    specs = [
        ("A", "High propensity · purchase", examples[(examples.propensity_segment == "high") & (examples.target == 1) & (examples.hits >= 1)], ["propensity"], [False]),
        ("B", "High propensity · no purchase", examples[(examples.propensity_segment == "high") & (examples.target == 0)], ["propensity"], [False]),
        ("C", "Medium propensity · purchase", examples[(examples.propensity_segment == "medium") & (examples.target == 1) & (examples.hits >= 1)], ["hits", "propensity"], [False, False]),
        ("D", "Low propensity · no purchase", examples[(examples.propensity_segment == "low") & (examples.target == 0)], ["propensity"], [True]),
        ("E", "Low propensity · purchase", examples[(examples.propensity_segment == "low") & (examples.target == 1)], ["propensity"], [True]),
    ]
    selected = []
    for alias, title, candidates, columns, ascending in specs:
        if candidates.empty:
            raise ValueError(f"No diagnostic case: {title}")
        row = candidates.sort_values(columns + ["customer_id"], ascending=ascending + [True]).iloc[0]
        selected.append(dict(alias=alias, title=title, customer_id=row.customer_id, propensity=float(row.propensity),
                             propensity_segment=row.propensity_segment, outcome=int(row.target)))
    return selected


def integrate(root, config, frames, transactions, articles):
    from .plots import plot_customer_decisions, plot_recommendation_segments, plot_integrated_system

    root = Path(root)
    private = root / config["private_dir"]
    output = root / config["output_dir"]
    figures = output / "figures"; figures.mkdir(exist_ok=True)
    recommendation_config = json.loads((root / config["recommendation_config"]).read_text())
    run_dir = root / recommendation_config["output_dir"]
    cutoff = config["horizons"]["7"]["test_cutoff"]
    assert cutoff == recommendation_config["test_cutoff"] and recommendation_config["horizon_days"] == 7
    k = recommendation_config["k"]
    scores = propensity_segments(pd.read_parquet(private / "scores_h7.parquet"), config)
    features = frames[cutoff]
    scores = scores.merge(features[["customer_id", "prior_transactions", "recency_days", "gap_mean"]], on="customer_id", validate="one_to_one")
    scores.to_parquet(private / "customer_prioritization.parquet", index=False)
    history = transactions[transactions.t_dat.lt(pd.Timestamp(cutoff))]
    target = transactions[transactions.t_dat.ge(pd.Timestamp(cutoff)) &
                          transactions.t_dat.lt(pd.Timestamp(cutoff) + pd.Timedelta(days=7))]
    truth = ground_truth(target)
    selected_ranker = json.loads((run_dir / "selection.json").read_text())["selected_policy"]
    recommendation_rows = pd.read_csv(run_dir / "recommendations.csv", dtype={"customer_id":str, "article_id":str})
    predictions = recommendation_rows.sort_values(["customer_id", "rank"]).groupby("customer_id").article_id.agg(list).to_dict()
    candidate_features = pd.read_parquet(run_dir / "snapshots" / cutoff / "candidates_features.parquet")
    policies = {selected_ranker:predictions, "recent":source_predictions(candidate_features, "recent", k),
                "popularity":source_predictions(candidate_features, "popularity", k)}
    group_lookup = scores.set_index("customer_id").propensity_segment.to_dict()
    segment_rows, user_metrics = [], {}
    for name, lists in policies.items():
        rows = per_user_metrics(lists, truth, k)
        rows["hits"] = [len(set(lists[u]) & truth[u]) for u in rows.customer_id]
        rows["propensity_segment"] = rows.customer_id.map(group_lookup).fillna("no_history_unmodeled")
        user_metrics[name] = rows
        for segment, part in rows.groupby("propensity_segment"):
            population = scores[scores.propensity_segment.eq(segment)]
            segment_rows.append(dict(model=name, propensity_segment=segment,
                purchasing_customers=len(part), eligible_population=len(population) if len(population) else np.nan,
                population_purchase_rate=float(population.target.mean()) if len(population) else np.nan,
                any_hit_rate_among_buyers=float(part.hits.gt(0).mean()),
                **{f"{metric}@{k}":float(part[metric].mean()) for metric in ["map", "recall", "ndcg"]}))
    segment_table = pd.DataFrame(segment_rows)
    segment_table.to_csv(output / "recommendation_segments.csv", index=False)
    buyer_metrics = user_metrics[selected_ranker].set_index("customer_id")
    priority_rows = []
    for fraction in [.1, .2, .5, 1.]:
        n = int(np.ceil(len(scores)*fraction))
        selected = scores.head(n)
        buyers = selected[selected.target.eq(1)]
        metrics = buyer_metrics.loc[buyers.customer_id]
        matches = int(metrics.hits.gt(0).sum())
        priority_rows.append(dict(top_fraction=fraction, prioritized_customers=n,
            purchasers=int(selected.target.sum()), purchaser_precision=float(selected.target.mean()),
            purchaser_recall=float(selected.target.sum()/scores.target.sum()),
            observed_exact_match_customers=matches, observed_match_rate_per_prioritized_customer=matches/n,
            conditional_map_among_purchasers=float(metrics["map"].mean()),
            conditional_recall_among_purchasers=float(metrics["recall"].mean()),
            conditional_ndcg_among_purchasers=float(metrics["ndcg"].mean())))
    pd.DataFrame(priority_rows).to_csv(output / "prioritization_diagnostics.csv", index=False)
    # Preserve all legacy buyer scores, with an explicitly unmodeled cold-start group.
    joined = recommendation_rows.merge(scores[["customer_id", "propensity", "propensity_segment"]], on="customer_id", how="left")
    fallback_prior = float(pd.read_parquet(root / config["feature_dir"] / f"h7_{config['horizons']['7']['validation_cutoff']}.parquet").target.mean())
    joined["score_origin"] = np.where(joined.propensity.isna(), "unvalidated_prior_only_placeholder", "learned_temporal_model")
    joined["propensity"] = joined.propensity.fillna(fallback_prior)
    joined["propensity_segment"] = joined.propensity_segment.fillna("no_history_unmodeled")
    joined.to_parquet(private / "ranked_products_with_propensity.parquet", index=False)
    cases = choose_cases(scores, user_metrics[selected_ranker])
    # Buyer recommendations are already cached. New lists are needed only for
    # the two illustrated nonbuyers, not the entire propensity population.
    new_users = [case["customer_id"] for case in cases if case["customer_id"] not in predictions]
    if new_users:
        print(f"  Rebuilding one past-only test retriever for {len(new_users)} nonbuyer examples; ranker is not retrained.", flush=True)
        lookback = history[history.t_dat.ge(pd.Timestamp(cutoff)-pd.Timedelta(days=recommendation_config["history_days"]))]
        customers = pd.read_parquet(root / config["data_dir"] / "customers.parquet")
        retrieval = Retrieval(lookback, customers, articles, cutoff, recommendation_config)
        # One cached buyer cross-check confirms the regenerated retrieval/scoring recipe.
        reference = next(case["customer_id"] for case in cases if case["customer_id"] in predictions)
        candidates, _ = retrieval.generate(new_users + [reference])
        features_for_items = build_features(candidates, lookback, customers, articles, cutoff)
        model = joblib.load(root / "models" / run_dir.name / "selected_ranker.joblib")
        assert model["name"] == selected_ranker
        new_lists = ranked_lists(features_for_items, ranking_predict(model, features_for_items), k)
        assert new_lists[reference] == predictions[reference]
        predictions = predictions | {u:new_lists[u] for u in new_users}
    lookup = scores.set_index("customer_id")
    for case in cases:
        row = lookup.loc[case["customer_id"]]
        case["history_purchases"] = int(row.prior_transactions)
        case["recency_days"] = float(row.recency_days)
        case["typical_gap"] = float(row.gap_mean) if row.gap_mean > 0 else None
        case["future_items"] = len(truth.get(case["customer_id"], set()))
        case["hits"] = len(set(predictions[case["customer_id"]]) & truth.get(case["customer_id"], set()))
        case["map_at_k"] = float(buyer_metrics.loc[case["customer_id"], "map"]) if case["customer_id"] in buyer_metrics.index else 0.
    truth_for_cases = truth | {u:set() for u in new_users}
    metadata = pd.read_csv(root / "data/raw/articles.csv", dtype={"article_id":str},
        usecols=["article_id", "prod_name", "product_type_name", "colour_group_name"]).fillna("Unknown")
    panels = build_examples(cases, history, target, predictions, truth_for_cases, metadata)
    (private / "decision_examples_private.json").write_text(json.dumps(panels, indent=2, ensure_ascii=False)+"\n")
    public_cases = [{key:case[key] for key in ["alias", "title", "propensity", "propensity_segment", "outcome",
        "history_purchases", "recency_days", "typical_gap", "future_items", "hits"]} for case in cases]
    (output / "decision_example_summary.json").write_text(json.dumps(public_cases, indent=2, ensure_ascii=False)+"\n")
    image_root = next((p for p in [root / "data/raw/images", root / "data/images"] if p.is_dir()), None)
    plot_customer_decisions(panels, figures / "customer_decisions.png", image_root)
    plot_recommendation_segments(segment_table, selected_ranker, figures / "recommendation_segments.png")
    summary = dict(known_population=len(scores), known_recommendation_buyers=int(scores.target.sum()),
        new_history_recommendation_buyers=len(set(truth)-set(scores.customer_id)),
        propensity_groups="ranked on all known customers before outcomes: top 20%, next 30%, remaining 50%",
        cold_start="No-history buyers remain a separate unmodeled group; private scores contain a labeled prior-only placeholder",
        recommendation_order_unchanged=True, personalized_policy=selected_ranker,
        selected_nonbuyers_with_new_recommendations=len(new_users), examples_curated_after_test=True)
    (output / "integration_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    plot_integrated_system(root, output, config, summary, figures / "integrated_system.png")
    return summary
