"""An H&M experiment from weekly retrieval through held-out test analysis."""
from __future__ import annotations

import gc
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .analysis import catalog_diagnostics, eda, segment_analysis
from .candidates import Retrieval, SOURCES
from .data import ground_truth, load_tables, temporal_snapshot
from .features import FEATURE_GROUPS, build_features, label_candidates, sample_negatives
from .metrics import candidate_recall, evaluate, paired_bootstrap, per_user_metrics, ranked_lists
from .models import ALL_FEATURES, fit_model, importance, predict


def json_write(path, content):
    Path(path).write_text(json.dumps(content, indent=2, default=str) + "\n")


def snapshot(transactions, customers, articles, cutoff, config, output, keep_retrieval=False):
    print(f"\nPreparing weekly snapshot at {cutoff}", flush=True)
    history, target = temporal_snapshot(transactions, cutoff, config["horizon_days"], config["history_days"])
    truth = ground_truth(target)
    retrieval = Retrieval(history, customers, articles, cutoff, config)
    candidates, sources = retrieval.generate(truth)
    features = label_candidates(build_features(candidates, history, customers, articles, cutoff), target, cutoff)
    baselines = retrieval.baseline_predictions(sources, truth, config["k"])
    folder = output / "snapshots" / cutoff
    folder.mkdir(parents=True, exist_ok=True)
    features.to_parquet(folder / "candidates_features.parquet", index=False)
    recall_rows = [dict(source=s, candidate_recall=candidate_recall(sources[s], truth),
                        mean_candidates=float(np.mean([len(v) for v in sources[s].values()]))) for s in SOURCES]
    union = features.groupby("customer_id").article_id.agg(list).to_dict()
    recall_rows.append(dict(source="union", candidate_recall=candidate_recall(union, truth),
                            mean_candidates=float(np.mean([len(v) for v in union.values()]))))
    recall_table = pd.DataFrame(recall_rows)
    recall_table.to_csv(folder / "candidate_recall.csv", index=False)
    full_history = transactions[transactions.t_dat.lt(pd.Timestamp(cutoff))]
    prior_pairs = set(zip(full_history.customer_id, full_history.article_id))
    relevant_pairs = set(zip(target.customer_id, target.article_id))
    past_items = set(full_history.article_id)
    eligible_items = set(retrieval.items)
    metadata = dict(cutoff=cutoff, history_rows=len(history), target_rows=len(target),
        target_users=len(truth), candidates=len(features), positives_in_candidates=int(features.label.sum()),
        distinct_target_pairs=len(relevant_pairs),
        repeat_target_pair_fraction=len(relevant_pairs & prior_pairs) / len(relevant_pairs),
        truly_unseen_target_item_fraction=sum(i not in past_items for _, i in relevant_pairs) / len(relevant_pairs),
        ineligible_target_item_fraction=sum(i not in eligible_items for _, i in relevant_pairs) / len(relevant_pairs),
        history_window_days=config["history_days"], candidate_catalog="items purchased in history lookback",
        training_positives="only positives found by retrieval; no future-positive injection")
    json_write(folder / "summary.json", metadata)
    del prior_pairs, relevant_pairs
    print(f"  Snapshot complete: {len(truth):,} target users; {len(features):,} candidates; "
          f"union recall {recall_rows[-1]['candidate_recall']:.3f}", flush=True)
    result = dict(frame=features, truth=truth, baselines=baselines, recall=recall_table, summary=metadata)
    if keep_retrieval:
        result["retrieval"] = retrieval
    return result


def run(config_path):
    started = time.perf_counter()
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    json_write(output / "config.json", config)
    manifest = json.loads((Path(config["data_dir"]) / "manifest.json").read_text())
    if manifest["dataset"] != "hm":
        raise ValueError("This project runs the H&M dataset; use the real prepared tables")
    tx, customers, articles = load_tables(Path(config["data_dir"]))
    cutoffs = list(map(pd.Timestamp, config["train_cutoffs"]))
    validation_cutoff, test_cutoff = pd.Timestamp(config["validation_cutoff"]), pd.Timestamp(config["test_cutoff"])
    horizon = pd.Timedelta(days=config["horizon_days"])
    assert all(a + horizon <= b for a, b in zip(cutoffs, cutoffs[1:] + [validation_cutoff]))
    assert validation_cutoff + horizon <= test_cutoff
    assert tx.t_dat.max() >= test_cutoff + horizon - pd.Timedelta(days=1)
    print("PHASE 1: EDA", flush=True)
    eda_summary = eda(tx, customers, articles, output / "eda")
    print("PHASE 2: TRAIN / VALIDATION CANDIDATES AND FEATURES", flush=True)
    train_frames = []
    for cutoff in config["train_cutoffs"]:
        built = snapshot(tx, customers, articles, cutoff, config, output)
        train_frames.append(built["frame"])
        del built; gc.collect()
    validation = snapshot(tx, customers, articles, config["validation_cutoff"], config, output)
    train = pd.concat(train_frames, ignore_index=True)
    del train_frames; gc.collect()
    sampled = sample_negatives(train, config["negative_ratio"], config["seed"])
    sampled.to_parquet(output / "train_sampled.parquet", index=False)
    print(f"PHASE 3: FIT MODELS ({len(sampled):,} candidate rows after negative sampling)", flush=True)
    rows, trial_rows, selected_depths, fitted_models = [], [], {}, {}
    k = config["k"]
    for name, predictions in validation["baselines"].items():
        rows.append(dict(split="validation", model=name, **evaluate(predictions, validation["truth"], k)))
    for name in config["models"]:
        depths = config["tree_depths"] if name in config["tune_models"] else [5]
        best_metric, best = -np.inf, None
        for depth in depths:
            fitted = fit_model(name, sampled, config, depth)
            predictions = ranked_lists(validation["frame"], predict(fitted, validation["frame"]), k)
            metric = evaluate(predictions, validation["truth"], k)
            trial_rows.append(dict(model=name, depth=depth, train_seconds=fitted["train_seconds"], **metric))
            print(f"  {name} depth={depth}: validation MAP@{k}={metric[f'map@{k}']:.5f}", flush=True)
            if metric[f"map@{k}"] > best_metric:
                best_metric, best = metric[f"map@{k}"], (fitted, metric)
        fitted_models[name] = best[0]
        selected_depths[name] = best[0]["depth"]
        rows.append(dict(split="validation", model=name, **best[1]))
        pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
        pd.DataFrame(trial_rows).to_csv(output / "tuning.csv", index=False)
    validation_results = pd.DataFrame(rows)
    # Choose the complete policy before opening the final test target.
    selection = validation_results.sort_values([f"map@{k}", "model"], ascending=[False, True]).iloc[0].model
    best_ranker = validation_results[validation_results.model.isin(config["models"])].sort_values(
        [f"map@{k}", "model"], ascending=[False, True]).iloc[0].model
    json_write(output / "selection.json", dict(selected_policy=selection, selected_ranker=best_ranker,
        selected_depths=selected_depths, criterion=f"validation MAP@{k}", test_used_for_selection=False))
    print(f"PHASE 4: SETTINGS FROZEN. Selected policy={selection}; ranker={best_ranker}", flush=True)
    combined = pd.concat([train, validation["frame"]], ignore_index=True)
    final_training = sample_negatives(combined, config["negative_ratio"], config["seed"])
    json_write(output / "training_summary.json", dict(raw_rows=len(combined), sampled_rows=len(final_training),
        positive_rows=int(final_training.label.sum()), query_groups=final_training.query_id.nunique(),
        negative_ratio=config["negative_ratio"], zero_positive_queries="excluded only from model training; kept in evaluation"))
    del train, sampled, combined, validation; gc.collect()
    final_models = {name: fit_model(name, final_training, config, selected_depths[name]) for name in config["models"]}
    fitted_models.clear(); gc.collect()
    print("PHASE 5: FINAL TEMPORAL TEST", flush=True)
    test = snapshot(tx, customers, articles, config["test_cutoff"], config, output, keep_retrieval=True)
    predictions_by_model = dict(test["baselines"])
    for name, fitted in final_models.items():
        predictions_by_model[name] = ranked_lists(test["frame"], predict(fitted, test["frame"]), k)
        importance(fitted).to_csv(output / f"importance_{name}.csv", index=False)
    for name, predictions in predictions_by_model.items():
        metric = evaluate(predictions, test["truth"], k)
        rows.append(dict(split="test", model=name, **metric))
        print(f"  {name}: test MAP@{k}={metric[f'map@{k}']:.5f}", flush=True)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    model_dir = Path("models") / output.name; model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_models[best_ranker], model_dir / "selected_ranker.joblib", compress=3)
    selected_predictions = predictions_by_model[selection]
    prediction_rows = [dict(customer_id=u, rank=rank, article_id=i)
                       for u, items in selected_predictions.items() for rank, i in enumerate(items, 1)]
    pd.DataFrame(prediction_rows).to_csv(output / "recommendations.csv", index=False)
    a = per_user_metrics(selected_predictions, test["truth"], k)
    b = per_user_metrics(predictions_by_model["popularity"], test["truth"], k)
    a.to_csv(output / "per_user_metrics.csv", index=False)
    json_write(output / "selected_vs_popularity_bootstrap.json", paired_bootstrap(
        a, b, config["bootstrap_replicates"], config["seed"]))
    print("PHASE 6: SEGMENTS / COLD START / CATALOG DIAGNOSTICS", flush=True)
    full_history = tx[tx.t_dat.lt(test_cutoff)]
    segment_analysis(predictions_by_model, test["truth"], full_history, customers, articles,
                     test_cutoff, k).to_csv(output / "segments.csv", index=False)
    pd.DataFrame([dict(model=name, **catalog_diagnostics(preds, test["retrieval"], k))
                  for name, preds in predictions_by_model.items()]).to_csv(output / "catalog_diagnostics.csv", index=False)
    if config["ablation"]:
        print("PHASE 7: CANDIDATE / FEATURE ABLATIONS", flush=True)
        ablation_rows = []
        frozen = final_models[best_ranker]
        full_metric = evaluate(predictions_by_model[best_ranker], test["truth"], k)
        ablation_rows.append(dict(kind="reference", removed="none", **full_metric))
        for source in SOURCES:
            # Candidate-only ablation: freeze the ranker and retained-item features.
            # cap >= six source quotas ensures no extra candidates appear on removal.
            assert config["max_candidates"] >= len(SOURCES) * config["per_source"]
            flags = [f"from_{s}" for s in SOURCES if s != source]
            reduced = test["frame"][test["frame"][flags].sum(axis=1).gt(0)]
            preds = ranked_lists(reduced, predict(frozen, reduced), k)
            candidate_sets = reduced.groupby("customer_id").article_id.agg(list).to_dict()
            ablation_rows.append(dict(kind="candidate_fixed_ranker", removed=source,
                candidate_recall=candidate_recall(candidate_sets, test["truth"]), **evaluate(preds, test["truth"], k)))
        for group in ["als", "recency", "affinity", "cooccurrence", "content"]:
            features = [f for f in ALL_FEATURES if f not in FEATURE_GROUPS[group]]
            # Aggregate RRF and source count encode all sources; drop to avoid proxy leakage.
            if group in ["als", "recency", "cooccurrence", "content"]:
                features = [f for f in features if f not in ["retrieval_rrf", "source_count"]]
            ablated = fit_model(best_ranker, final_training, config, selected_depths[best_ranker], features)
            preds = ranked_lists(test["frame"], predict(ablated, test["frame"]), k)
            metric = evaluate(preds, test["truth"], k)
            ablation_rows.append(dict(kind="feature_retrained", removed=group, **metric))
            print(f"  Without {group}: test MAP@{k}={metric[f'map@{k}']:.5f}", flush=True)
        ablations = pd.DataFrame(ablation_rows)
        ablations["delta_map"] = ablations[f"map@{k}"] - full_metric[f"map@{k}"]
        ablations.to_csv(output / "ablations.csv", index=False)
    runtime = time.perf_counter() - started
    versions = {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm", "pyarrow"]}
    json_write(output / "run_manifest.json", dict(dataset=manifest, config=str(config_path),
        python=platform.python_version(), packages=versions, seconds=runtime,
        completed=True, optional_two_tower=False, cohort="users with a next-week purchase", all_users_ctr_not_estimated=True))
    write_report(output, config, eda_summary, rows, selection, best_ranker, test, runtime)
    print(f"\nDONE in {runtime/60:.1f} minutes. Report: {output / 'report.md'}", flush=True)
    return output


def write_report(output, config, eda_summary, rows, selection, best_ranker, test, runtime):
    metrics = pd.DataFrame(rows)
    k = config["k"]
    table = metrics[metrics.split.eq("test")].sort_values(f"map@{k}", ascending=False)
    lines = ["# H&M Recommendation & Learning-to-Rank — Experiment Report", "",
        f"Actual H&M transactions; {eda_summary['customers_in_metadata']:,} hash-sampled customers; "
        f"{eda_summary['transactions']:,} transactions across the complete observed date range.", "",
        f"Train label weeks: {', '.join(config['train_cutoffs'])}; validation: {config['validation_cutoff']}; "
        f"test: {config['test_cutoff']} (each {config['horizon_days']} days).", "",
        f"History lookback: {config['history_days']} days; model selection uses validation MAP@{k} only. "
        f"Selected policy: **{selection}**. Selected ML ranker: **{best_ranker}**.", "",
        f"Test cohort: {len(test['truth']):,} customers with at least one next-week purchase, including zero-history customers. "
        "Every model uses the same users and ground truth. Missing recommendations score zero.", "",
        "## Final test comparison", "", f"| Model | MAP@{k} | Recall@{k} | NDCG@{k} |", "|---|---:|---:|---:|"]
    for _, row in table.iterrows():
        lines.append(f"| {row.model} | {row[f'map@{k}']:.5f} | {row[f'recall@{k}']:.5f} | {row[f'ndcg@{k}']:.5f} |")
    lines.extend(["", "## Candidate recall", "", "| Source | Candidate recall | Average pool size |", "|---|---:|---:|"])
    for _, row in test["recall"].iterrows():
        lines.append(f"| {row.source} | {row.candidate_recall:.4f} | {row.mean_candidates:.1f} |")
    lines.extend(["", "## Interpretation and limits", "",
        "- This is a sampled-user development experiment, not a full-dataset leaderboard result. "
        "Popularity, co-occurrence, and ALS are fitted on the selected users; a larger cohort may change their quality.",
        "- Zero labels mean no observed purchase, not a logged negative impression. Sampled classifier probabilities are relevance scores, not calibrated conversion probabilities.",
        "- Static age and product descriptions come from the supplied metadata snapshot. No point-in-time membership or demographic snapshots are available; dynamic customer fields are excluded.",
        "- Candidate eligibility is a prior-purchase lookback heuristic; the data has no live inventory feed. "
        "Future-unseen items remain in the ground-truth denominator but cannot be retrieved. 'First seen' is not a verified launch date.",
        "- Repeated purchases remain eligible. A discovery-only policy would require a separate experiment.",
        "- Candidate ablations freeze ranker and retained features. Feature ablations retrain on the same sampled training rows; retrieval remains fixed.",
        "- Test ablations and segment findings are descriptive diagnostics after settings are frozen; they do not select the reported policy.",
        "- LambdaMART uses metric-weighted pairwise gradients; XGBRanker is an interface, not an independent model family. "
        "The experiment separates XGBoost pairwise, XGBoost NDCG LambdaMART, and LightGBM LambdaMART.",
        "- Offline purchase ranking cannot establish CTR, conversion, revenue lift, or a causal effect. The A/B plan is proposed only.", "",
        f"Runtime: {runtime/60:.1f} minutes. See `run_manifest.json` for exact package versions.", "",
        "Generated analysis: `tuning.csv`, `segments.csv`, `catalog_diagnostics.csv`, `ablations.csv`, "
        "`selected_vs_popularity_bootstrap.json`, and `importance_*.csv`.", "",
        "![EDA](eda/eda.png)", ""])
    (output / "report.md").write_text("\n".join(lines))
