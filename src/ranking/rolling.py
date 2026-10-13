"""Two expanding temporal validation folds using the already-built weekly features."""
import json
from pathlib import Path

import pandas as pd

from .data import ground_truth, load_tables, temporal_snapshot
from .features import sample_negatives
from .metrics import evaluate, paired_bootstrap, per_user_metrics, ranked_lists
from .models import fit_model, predict


def source_predictions(frame, source, k):
    # Current config includes all source quotas in the union, so source lists can
    # be recovered exactly from stored indicators and ranks.
    retained = frame[frame[f"from_{source}"].eq(1)].sort_values(["customer_id", f"rank_{source}"])
    personal = retained.groupby("customer_id").article_id.agg(list).to_dict()
    popular = frame[frame.from_popularity.eq(1)].sort_values("rank_popularity")
    fallback = popular.article_id.drop_duplicates().head(k).tolist()
    return {u: list(dict.fromkeys(personal.get(u, []) + fallback))[:k] for u in frame.customer_id.unique()}


def rolling(config_path):
    config = json.loads(Path(config_path).read_text())
    output = Path(config["output_dir"])
    tx, _, _ = load_tables(Path(config["data_dir"]))
    rows, prior_frames = [], []
    cutoffs = config["train_cutoffs"] + [config["validation_cutoff"]]
    for i, cutoff in enumerate(cutoffs):
        frame = pd.read_parquet(output / "snapshots" / cutoff / "candidates_features.parquet")
        if i:
            _, target = temporal_snapshot(tx, cutoff, config["horizon_days"])
            truth = ground_truth(target)
            train = sample_negatives(pd.concat(prior_frames, ignore_index=True), config["negative_ratio"], config["seed"])
            for source in ["popularity", "recent", "als"]:
                rows.append(dict(cutoff=cutoff, model=source, depth="", **evaluate(source_predictions(frame, source, config["k"]), truth, config["k"])))
            for name in config["models"]:
                # Fixed depth=5 across folds; do not transfer a depth selected
                # by a later validation week into an earlier fold.
                model = fit_model(name, train, config, depth=5)
                predictions = ranked_lists(frame, predict(model, frame), config["k"])
                metrics = evaluate(predictions, truth, config["k"])
                rows.append(dict(cutoff=cutoff, model=name, depth=5, **metrics))
                print(f"  Fold {cutoff} {name}: MAP@12={metrics['map@12']:.5f}", flush=True)
        prior_frames.append(frame)
    pd.DataFrame(rows).to_csv(output / "rolling_validation.csv", index=False)
    # Compare the unchanged validation-selected policy with the stronger
    # recent-purchase baseline, using saved test recommendations.
    selected = pd.read_csv(output / "recommendations.csv", dtype={"customer_id": str, "article_id": str}).sort_values(["customer_id", "rank"])
    predictions = selected.groupby("customer_id").article_id.agg(list).to_dict()
    test_frame = pd.read_parquet(output / "snapshots" / config["test_cutoff"] / "candidates_features.parquet")
    _, test_target = temporal_snapshot(tx, config["test_cutoff"], config["horizon_days"])
    truth = ground_truth(test_target)
    recent = source_predictions(test_frame, "recent", config["k"])
    comparison = paired_bootstrap(per_user_metrics(predictions, truth, config["k"]),
                                 per_user_metrics(recent, truth, config["k"]),
                                 config["bootstrap_replicates"], config["seed"])
    (output / "selected_vs_recent_bootstrap.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return rows
