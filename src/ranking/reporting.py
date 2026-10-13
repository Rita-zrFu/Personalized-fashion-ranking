"""Publish aggregate H&M experiment outputs; keep customer-level files local."""
import json
import shutil
from pathlib import Path

import pandas as pd


def publish(run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = ["report.md", "metrics.csv", "tuning.csv", "selection.json", "training_summary.json",
             "segments.csv", "catalog_diagnostics.csv", "ablations.csv", "config.json",
             "selected_vs_popularity_bootstrap.json", "selected_vs_recent_bootstrap.json",
             "rolling_validation.csv"]
    names.extend(p.name for p in run_dir.glob("importance_*.csv"))
    for name in names:
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, output / name)
    for directory in (run_dir / "snapshots").iterdir():
        target = output / "snapshots" / directory.name
        target.mkdir(parents=True, exist_ok=True)
        for name in ["candidate_recall.csv", "summary.json"]:
            shutil.copy2(directory / name, target / name)
    (output / "eda").mkdir(exist_ok=True)
    for source in (run_dir / "eda").iterdir():
        if source.name != "item_popularity.csv":
            shutil.copy2(source, output / "eda" / source.name)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    manifest["dataset"]["source"] = "data/raw"
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot_results(output)
    print(f"Aggregate report ready: {output / 'report.md'}", flush=True)


def plot_results(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    directory = Path(directory)
    metrics = pd.read_csv(directory / "metrics.csv")
    test = metrics[metrics.split.eq("test")].sort_values("map@12")
    config = json.loads((directory / "config.json").read_text())
    manifest = json.loads((directory / "run_manifest.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colours = ["#156f63" if name == "xgboost_classifier" else "#84ada6" for name in test.model]
    axes[0].barh(test.model, test["map@12"], color=colours)
    axes[0].set(xlabel="MAP@12", title=f"Final temporal test — {int(test.users.iloc[0]):,} purchasers")
    recall = pd.read_csv(directory / "snapshots" / config["test_cutoff"] / "candidate_recall.csv")
    axes[1].barh(recall.source, recall.candidate_recall, color=["#156f63" if s=="union" else "#84ada6" for s in recall.source])
    axes[1].set(xlabel="Macro candidate recall", title=f"Retrieval: up to {config['per_source']} items/source")
    counts = manifest["dataset"]
    fig.suptitle(f"H&M development cohort: {counts['selected_customers']:,} customers / {counts['selected_transactions']/1e6:.2f}M transactions", fontsize=12)
    fig.tight_layout(); fig.savefig(directory / "model_comparison.png", dpi=180); plt.close(fig)
    if (directory / "ablations.csv").exists():
        data = pd.read_csv(directory / "ablations.csv")
        data = data[data.kind.eq("feature_retrained")].sort_values("delta_map")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(data.removed, data.delta_map, color=np.where(data.delta_map < 0, "#bc6258", "#156f63"))
        ax.axvline(0, color="black", linewidth=.7)
        ax.set(xlabel="Change in MAP@12 after removing feature family", title="Feature ablation — retrained XGBoost classifier")
        fig.tight_layout(); fig.savefig(directory / "feature_ablation.png", dpi=180); plt.close(fig)
