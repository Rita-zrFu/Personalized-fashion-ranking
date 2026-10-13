"""Train and evaluate temporal propensity layers without changing the H&M ranker."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..data import load_tables
from .customer_features import build_customer_features, model_feature_columns
from .evaluation import calibration_bins, evaluate_probabilities, gains_curve, operating_points, propensity_deciles
from .models import fit_calibrator, fit_model, predict_calibrated, predict_raw, trial_settings
from .targets import build_targets, check_temporal_windows


def write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, default=str) + "\n")


def project_paths(config_path, project_root=None):
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[3]
    path = Path(config_path)
    path = path if path.is_absolute() else root / path
    config = json.loads(path.read_text())
    return root, config


def prepare_features(root, config, transactions, articles):
    directory = root / config["feature_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    cutoffs = sorted({date for window in config["horizons"].values()
                      for date in window["train_cutoffs"] + [window["validation_cutoff"], window["test_cutoff"]]})
    frames, audits, cohorts = {}, [], []
    for cutoff in cutoffs:
        frame, audit = build_customer_features(transactions, articles, cutoff)
        frame.to_parquet(directory / f"features_{cutoff}.parquet", index=False)
        frames[cutoff] = frame
        audits.append(audit)
        print(f"  Features {cutoff}: {len(frame):,} known customers; {audit['feature_count']} behavioral features", flush=True)
    for days, windows in config["horizons"].items():
        horizon = int(days)
        check_temporal_windows(windows, horizon)
        for split, dates in [("train", windows["train_cutoffs"]), ("validation", [windows["validation_cutoff"]]),
                             ("test", [windows["test_cutoff"]])]:
            for cutoff in dates:
                frame, audit = build_targets(frames[cutoff], transactions, cutoff, horizon)
                frame.to_parquet(directory / f"h{horizon}_{cutoff}.parquet", index=False)
                cohorts.append(dict(split=split, **audit))
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audits).to_csv(output / "feature_audit.csv", index=False)
    pd.DataFrame(cohorts).to_csv(output / "cohorts.csv", index=False)
    write_json(output / "feature_columns.json", model_feature_columns(next(iter(frames.values()))))
    return frames


def train_horizon(root, config, horizon, windows, frames, transactions, metric_rows, tuning_rows, op_rows):
    train = pd.concat([build_targets(frames[t], transactions, t, horizon)[0] for t in windows["train_cutoffs"]], ignore_index=True)
    validation, _ = build_targets(frames[windows["validation_cutoff"]], transactions, windows["validation_cutoff"], horizon)
    test, _ = build_targets(frames[windows["test_cutoff"]], transactions, windows["test_cutoff"], horizon)
    models, validation_predictions = {}, {}
    for name in config["models"]:
        best_ap, best_model, best_prob = -np.inf, None, None
        for settings in trial_settings(name):
            fitted = fit_model(name, train, config, settings)
            probabilities = predict_raw(fitted, validation)
            metrics = evaluate_probabilities(validation.target, probabilities)
            tuning_rows.append(dict(horizon_days=horizon, model=name, settings=json.dumps(settings, sort_keys=True), **metrics))
            if metrics["pr_auc_ap"] > best_ap:
                best_ap, best_model, best_prob = metrics["pr_auc_ap"], fitted, probabilities
        models[name], validation_predictions[name] = best_model, best_prob
        metric_rows.append(dict(horizon_days=horizon, split="validation", variant="raw", model=name,
                                cutoff=windows["validation_cutoff"], **evaluate_probabilities(validation.target, best_prob)))
        print(f"  {horizon}d {name}: validation AP={best_ap:.4f}", flush=True)
    chosen = sorted(models, key=lambda name: (-evaluate_probabilities(validation.target, validation_predictions[name])["pr_auc_ap"], name))[0]
    print(f"  {horizon}d selection frozen: {chosen}; test labels do not select models/settings.", flush=True)
    private = root / config["private_dir"]
    model_dir = root / config["model_dir"]
    private.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)
    score_table = test[["customer_id", "cutoff", "target"]].copy()
    selected_raw, selected_calibrated = None, None
    for name, fitted in models.items():
        raw = predict_raw(fitted, test)
        calibrator = fit_calibrator(validation_predictions[name], validation.target)
        calibrated = predict_calibrated(calibrator, raw)
        for variant, probabilities in [("raw", raw), ("calibrated", calibrated)]:
            metrics = evaluate_probabilities(test.target, probabilities)
            metric_rows.append(dict(horizon_days=horizon, split="test", variant=variant, model=name,
                                    cutoff=windows["test_cutoff"], **metrics))
            points = operating_points(test.customer_id, test.target, probabilities, config["operating_fractions"])
            op_rows.extend(dict(horizon_days=horizon, model=name, variant=variant, **row) for row in points.to_dict("records"))
        score_table[f"{name}_raw"] = raw
        score_table[f"{name}_calibrated"] = calibrated
        if name == chosen:
            selected_raw, selected_calibrated = raw, calibrated
            fitted["calibrator"] = calibrator
            joblib.dump(fitted, model_dir / f"selected_h{horizon}.joblib", compress=3)
            write_json(private / f"selected_h{horizon}.json", dict(model=name, settings=fitted["settings"],
                selection_criterion="validation average precision", base_training_cutoffs=windows["train_cutoffs"],
                calibration_cutoff=windows["validation_cutoff"],
                base_refitted_with_validation=False, test_used_for_selection=False))
    score_table["propensity"] = selected_calibrated
    score_table.to_parquet(private / f"scores_h{horizon}.parquet", index=False)
    output = root / config["output_dir"]
    deciles = propensity_deciles(test.customer_id, test.target, selected_calibrated)
    deciles.to_csv(output / f"deciles_h{horizon}.csv", index=False)
    gains = gains_curve(test.customer_id, test.target, selected_calibrated)
    gains.to_csv(output / f"gains_h{horizon}.csv", index=False)
    for variant, probabilities in [("raw", selected_raw), ("calibrated", selected_calibrated)]:
        calibration_bins(test.target, probabilities).to_csv(output / f"calibration_h{horizon}_{variant}.csv", index=False)
    metrics = evaluate_probabilities(test.target, selected_calibrated)
    print(f"  {horizon}d test: ROC-AUC={metrics['roc_auc']:.4f}, AP={metrics['pr_auc_ap']:.4f}, Brier={metrics['brier']:.4f}", flush=True)
    return dict(horizon_days=horizon, selected_model=chosen, **metrics)


def rolling_evaluation(config, frames, transactions):
    windows = config["horizons"]["7"]
    train_cutoffs = windows["train_cutoffs"]
    rows = []
    for cutoff in [train_cutoffs[1], windows["validation_cutoff"]]:
        dates = [t for t in train_cutoffs if pd.Timestamp(t) + pd.Timedelta(days=7) <= pd.Timestamp(cutoff)]
        train = pd.concat([build_targets(frames[t], transactions, t, 7)[0] for t in dates], ignore_index=True)
        validation, _ = build_targets(frames[cutoff], transactions, cutoff, 7)
        for name in config["models"]:
            # Prespecified parameters, not a later week's tuned setting.
            fitted = fit_model(name, train, config, {})
            probabilities = predict_raw(fitted, validation)
            rows.append(dict(cutoff=cutoff, model=name, parameters="fixed defaults", horizon_days=7,
                             **evaluate_probabilities(validation.target, probabilities)))
        print(f"  Fixed-parameter rolling fold {cutoff} complete", flush=True)
    return pd.DataFrame(rows)


def run(config_path="configs/propensity.json", project_root=None, features_only=False):
    started = time.perf_counter()
    root, config = project_paths(config_path, project_root)
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    # Keep original recommender outputs fixed throughout the additional experiment.
    original = [root / "reports/hm_sample/metrics.csv", root / "reports/hm_sample/selection.json",
                root / "reports/runs/hm_sample/recommendations.csv"]
    original_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in original}
    tx, _, articles = load_tables(root / config["data_dir"])
    print("PROPENSITY PHASE 1: landmark features and complete future labels", flush=True)
    frames = prepare_features(root, config, tx, articles)
    if features_only:
        return output
    metrics, tuning, operating, summary = [], [], [], []
    print("PROPENSITY PHASE 2: validation selection and frozen temporal tests", flush=True)
    for days, windows in config["horizons"].items():
        summary.append(train_horizon(root, config, int(days), windows, frames, tx, metrics, tuning, operating))
        pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
        pd.DataFrame(tuning).to_csv(output / "tuning.csv", index=False)
        pd.DataFrame(operating).to_csv(output / "operating_points.csv", index=False)
        pd.DataFrame(summary).to_csv(output / "selected_models.csv", index=False)
    if config["rolling_7d"]:
        print("PROPENSITY PHASE 3: two fixed-parameter rolling folds", flush=True)
        rolling_evaluation(config, frames, tx).to_csv(output / "rolling_validation_7d.csv", index=False)
    print("PROPENSITY PHASE 4: frozen recommendation integration and customer decision examples", flush=True)
    from .integration import integrate
    integration_summary = integrate(root, config, frames, tx, articles)
    from .plots import plot_propensity_results
    plot_propensity_results(output)
    assert original_hashes == {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in original}
    manifest = dict(dataset="H&M fixed 50K-customer sample", python=platform.python_version(),
        packages={p:importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm"]},
        original_recommender_hashes=original_hashes, original_recommender_unchanged=True,
        selection="validation AP only; sigmoid calibration uses validation; no train+validation refit",
        cohort="known historical buyers, selected without future outcomes; future-first-time buyers excluded from propensity metrics",
        horizons=[int(h) for h in config["horizons"]], seconds=time.perf_counter()-started,
        primary_model=summary[0], integration=integration_summary,
        exact_purchase_time_estimated=False, uplift_estimated=False, completed=True)
    write_json(output / "run_manifest.json", manifest)
    write_report(output, manifest, pd.DataFrame(summary), pd.DataFrame(operating), integration_summary)
    print(f"PROPENSITY DONE in {manifest['seconds']/60:.1f} minutes: {output.relative_to(root) / 'report.md'}", flush=True)
    return output


def write_report(output, manifest, models, operating, integration):
    primary = models[models.horizon_days.eq(7)].iloc[0]
    top = operating[(operating.horizon_days == 7) & (operating.model == primary.selected_model) &
                    (operating.variant == "calibrated") & (operating.top_fraction == .1)].iloc[0]
    lines = ["# Time-aware purchase propensity — actual H&M results", "",
        "Landmark prediction estimates a customer's probability of returning within 7, 14 or 28 days. It does not estimate exact event times or causal recommendation effects.", "",
        "## Selected models", "", "| Horizon | Validation-selected model | Test customers | Prevalence | ROC-AUC | PR-AUC (AP) | Brier | Calibration ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in models.itertuples():
        lines.append(f"| {row.horizon_days}d | {row.selected_model} | {row.customers:,} | {row.prevalence:.2%} | {row.roc_auc:.4f} | {row.pr_auc_ap:.4f} | {row.brier:.4f} | {row.calibration_ece:.4f} |")
    lines.extend(["", f"7-day top-decile precision is {top.precision:.2%}, recall {top.recall:.2%}, lift {top.lift:.2f}× the eligible-population rate ({top.population_rate:.2%}).",
        "", "## Design", "",
        "Customers are eligible only after a purchase has been recorded before each cutoff; long-inactive users remain in the denominator. Without enrollment dates, using the final metadata roster to include future first-time customers would rely on retrospectively known membership.",
        "All predictors use purchases strictly before each cutoff. Timing gaps use distinct shopping days; multiple basket items are not zero-day interpurchase gaps. Missing interval statistics have explicit flags. Product-type labels are static metadata; no post-cutoff customer fields are used.",
        "Only fully matured future windows are labeled. The 7-day cutoffs match the ranker. The longer tasks shift earlier because the dataset ends on September 22, 2020. See cohorts.csv for exact dates, prevalence and excluded first-time purchasers.",
        "All training negatives are retained. Model selection prioritizes average precision rather than accuracy. Logistic raw/log transforms are compared; C and boosting depth grids are small. Validation also fits sigmoid calibration. Base models remain trained on training weeks only so the calibrator's scoring distribution stays consistent; final test labels do not tune either component.",
        "Calibration ECE uses up to 10 quantile bins and is a descriptive metric. Calibration is not guaranteed outside these periods. Overlapping customers across landmark dates are expected; there is no random row split. Rolling folds use fixed defaults independent of later tuning.", "",
        "## Connection to recommendations", "",
        f"The primary test includes {integration['known_recommendation_buyers']:,} returning buyers with saved top-12 lists; {integration['new_history_recommendation_buyers']} additional first-time buyers are shown as an unmodeled history group. The original recommendation scores and article ordering stay fixed.",
        "High/medium/low segments are capacity ranks of all eligible customers before inspecting outcomes: top 20%, next 30%, remaining 50%. Recommendation MAP/Recall/NDCG by segment condition on customers who actually buy; they are not classification metrics or overall population hit rates.",
        "Prioritization tables use all eligible customers as the budget denominator and report captured purchasers and observed exact-match customers. Nonbuyers have no observed future match whatever products would be shown. Their ranking lists were not generated for this capacity analysis; these diagnostics do not measure engagement, catalog exposure or treatment effects.",
        "New-history buyers have no learned propensity estimate in the historical-return task. A labeled prior-only placeholder is recorded privately, and this group is not assigned to the learned high/medium/low cohorts.", "",
        "## Three measured résumé-ready results", "",
        f"1. 7-day propensity: ROC-AUC {primary.roc_auc:.4f}, PR-AUC/AP {primary.pr_auc_ap:.4f} on {primary.customers:,} known customers.",
        f"2. Top propensity decile: {top.lift:.2f}× purchase-rate lift, {top.precision:.2%} precision, {top.recall:.2%} of returning purchasers captured.",
        "3. The unchanged recommender's held-out MAP@12 is 0.02607, 17.3% above the recent-purchase baseline on its original 2,452-purchaser cohort. This differs from the propensity population denominator.", "",
        "## Reproduce", "", "Run `make propensity` after preparing the existing sample and restoring/reproducing the frozen recommendation run. All feature tables, probabilities, IDs and serialized models remain ignored locally; public outputs are aggregate diagnostics plus anonymous examples.", "",
        "![Propensity deciles and gains](figures/propensity_prioritization.png)", "",
        "![Probability calibration](figures/calibration.png)", "",
        "![Recommendation by propensity group](figures/recommendation_segments.png)", "",
        "![Integrated decision examples](figures/customer_decisions.png)", ""])
    (output / "report.md").write_text("\n".join(lines))
