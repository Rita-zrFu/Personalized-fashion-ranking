"""Applied H&M results from saved predictions; no retrieval or model refitting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ground_truth, load_tables
from .metrics import per_user_metrics
from .rolling import source_predictions
from .candidates import SOURCES

POLICY_LABELS = {"popularity": "Popularity", "recent": "Recent purchases"}
ACTIVITY_ORDER = ["zero_history", "1-2", "3-10", ">10"]
ACTIVITY_LABELS = ["No history", "1–2 purchases", "3–10 purchases", ">10 purchases"]


def activity_group(count):
    return "zero_history" if count == 0 else "1-2" if count <= 2 else "3-10" if count <= 10 else ">10"


def evaluate_segments(predictions, truth, lifetime_history, saved_segments, k):
    counts = lifetime_history.groupby("customer_id").size()
    private_rows, aggregate = {}, []
    for policy, lists in predictions.items():
        rows = per_user_metrics(lists, truth, k)
        rows["history_purchases"] = rows.customer_id.map(counts).fillna(0).astype(int)
        rows["activity"] = rows.history_purchases.map(activity_group)
        rows["hits"] = [len(set(lists.get(u, [])) & truth[u]) for u in rows.customer_id]
        private_rows[policy] = rows
        for group, part in rows.groupby("activity"):
            record = dict(model=policy, segment=group, users=len(part),
                          any_hit_rate=float(part.hits.gt(0).mean()))
            for metric in ["map", "recall", "ndcg"]:
                value = float(part[metric].mean())
                expected = saved_segments[(saved_segments.model == policy) &
                    (saved_segments.axis == "activity") & (saved_segments.segment == group)]
                np.testing.assert_allclose(value, expected.iloc[0][f"{metric}@{k}"], rtol=0, atol=1e-12)
                record[f"{metric}@{k}"] = value
            aggregate.append(record)
    table = pd.DataFrame(aggregate)
    table["segment"] = pd.Categorical(table.segment, categories=ACTIVITY_ORDER, ordered=True)
    return table.sort_values(["segment", "model"]), private_rows


def catalog_exposure(predictions, lookback_history, articles, saved_diagnostics):
    counts = lookback_history.groupby("article_id").size().rename("prior_transactions").reset_index()
    counts = counts.sort_values(["prior_transactions", "article_id"], ascending=[False, True])
    counts["popularity_rank"] = np.arange(1, len(counts) + 1)
    rank = counts.set_index("article_id").popularity_rank.to_dict()
    popularity = counts.set_index("article_id").prior_transactions.to_dict()
    catalog_size = len(counts)
    category = articles.set_index("article_id").product_group_name.to_dict()
    records, categories = [], []
    for policy, lists in predictions.items():
        exposed = [item for items in lists.values() for item in items]
        percentiles = np.asarray([rank[item] / catalog_size for item in exposed])
        saved = saved_diagnostics[saved_diagnostics.model.eq(policy)].iloc[0]
        coverage = len(set(exposed)) / catalog_size
        np.testing.assert_allclose(coverage, saved.catalog_coverage, rtol=0, atol=1e-12)
        records.append(dict(model=policy, users=len(lists), recommendation_slots=len(exposed),
            eligible_catalog_items=catalog_size, distinct_recommended_items=len(set(exposed)),
            catalog_coverage=coverage, head_top_1pct_share=float(np.mean(percentiles <= .01)),
            next_9pct_share=float(np.mean((percentiles > .01) & (percentiles <= .1))),
            tail_bottom_90pct_share=float(np.mean(percentiles > .1)),
            median_prior_item_transactions=float(np.median([popularity[i] for i in exposed])),
            mean_novelty_bits=float(saved.mean_novelty_bits),
            mean_content_diversity=float(saved.mean_content_diversity)))
        groups = pd.Series([category.get(i, "Unknown") for i in exposed]).value_counts()
        categories.extend(dict(model=policy, product_group=str(group), recommendation_slots=int(n),
                               share=float(n / len(exposed))) for group, n in groups.items())
    return pd.DataFrame(records), pd.DataFrame(categories)


def select_examples(per_user, predictions, truth):
    """Curate five diagnostic cases explicitly using test outcomes, not randomly."""
    rows = per_user.copy()
    rows["first_hit_rank"] = rows.customer_id.map(
        lambda u: next((i for i, item in enumerate(predictions[u], 1) if item in truth[u]), 99))
    specifications = [
        ("A", "Strong match", "At least two hits; at most eight future items; highest hit count then MAP.",
         rows[(rows.hits >= 2) & (rows.target_items <= 8)], ["hits", "map"], [False, False]),
        ("B", "Partial match", "One hit among at least three future items, with a match in the first six ranks.",
         rows[(rows.hits == 1) & (rows.target_items.between(3, 8)) & (rows.first_hit_rank <= 6)],
         ["map", "history_purchases"], [False, False]),
        ("C", "Active customer · miss", "At least 50 prior purchases, at most six future items, and no exact match.",
         rows[(rows.history_purchases >= 50) & (rows.hits == 0) & (rows.target_items <= 6)],
         ["history_purchases"], [False]),
        ("D", "Low-history match", "One or two prior transactions; an exact match; highest MAP.",
         rows[(rows.history_purchases.between(1, 2)) & (rows.hits >= 1) & (rows.target_items <= 6)],
         ["map", "hits"], [False, False]),
        ("E", "New customer · miss", "No recorded prior transactions and no exact match; at most six future items.",
         rows[(rows.history_purchases == 0) & (rows.hits == 0) & (rows.target_items <= 6)],
         ["target_items"], [True]),
    ]
    examples, used = [], set()
    for alias, title, rule, subset, columns, ascending in specifications:
        subset = subset[~subset.customer_id.isin(used)].sort_values(
            columns + ["customer_id"], ascending=ascending + [True], kind="stable")
        if subset.empty:
            raise ValueError(f"No qualifying H&M example: {title}")
        row = subset.iloc[0]
        examples.append(dict(alias=alias, title=title, selection_rule=rule,
            customer_id=row.customer_id, history_purchases=int(row.history_purchases),
            future_items=int(row.target_items), hits=int(row.hits), map_at_k=float(row["map"])))
        used.add(row.customer_id)
    return examples


def build_examples(examples, history, target, predictions, truth, metadata):
    lookup = metadata.set_index("article_id").to_dict("index")

    def item_card(item, **extra):
        row = lookup[item]
        return dict(article_id=item, name=row.get("prod_name", row["product_type_name"]),
                    category=row["product_type_name"], colour=row["colour_group_name"], **extra)

    panels = []
    for example in examples:
        user = example["customer_id"]
        recent = history[history.customer_id.eq(user)].sort_values(
            ["t_dat", "article_id"], ascending=[False, True]).drop_duplicates("article_id").head(4)
        future = target[target.customer_id.eq(user)].sort_values(["t_dat", "article_id"]).drop_duplicates("article_id")
        recommended = predictions[user]
        panel = example | {
            "history": [item_card(row.article_id, date=str(row.t_dat.date())) for row in recent.itertuples()],
            "recommendations": [item_card(i, rank=r, hit=i in truth[user]) for r, i in enumerate(recommended, 1)],
            "future": [item_card(row.article_id, date=str(row.t_dat.date()), hit=row.article_id in recommended)
                       for row in future.itertuples()],
        }
        panels.append(panel)
    return panels


def generate_applied(config_path="configs/hm_sample.json", project_root=None, image_dir=None):
    from .applied_plots import plot_catalog, plot_examples, plot_segments, plot_system

    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    config_path = Path(config_path)
    config_path = config_path if config_path.is_absolute() else root / config_path
    config = json.loads(config_path.read_text())
    run_dir = root / config["output_dir"]
    report = root / "reports" / run_dir.name
    output = report / "applied"
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    selection = json.loads((run_dir / "selection.json").read_text())
    selected = selection["selected_policy"]
    labels = POLICY_LABELS | {selected: "Personalized XGBoost" if selected == "xgboost_classifier" else f"Personalized ({selected})"}
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    metrics = pd.read_csv(run_dir / "metrics.csv")
    pd.testing.assert_frame_equal(pd.read_csv(report / "metrics.csv"), metrics)
    assert json.loads((report / "selection.json").read_text()) == selection
    saved_segments = pd.read_csv(run_dir / "segments.csv")
    saved_catalog = pd.read_csv(run_dir / "catalog_diagnostics.csv")
    tx, _, articles = load_tables(root / config["data_dir"])
    cutoff = pd.Timestamp(config["test_cutoff"])
    history = tx[tx.t_dat.lt(cutoff)]
    lookback = history[history.t_dat.ge(cutoff - pd.Timedelta(days=config["history_days"]))]
    target = tx[tx.t_dat.ge(cutoff) & tx.t_dat.lt(cutoff + pd.Timedelta(days=config["horizon_days"]))]
    truth = ground_truth(target)
    frame = pd.read_parquet(run_dir / "snapshots" / config["test_cutoff"] / "candidates_features.parquet")
    selected_rows = pd.read_csv(run_dir / "recommendations.csv", dtype={"customer_id": str, "article_id": str})
    predictions = {selected: selected_rows.sort_values(["customer_id", "rank"]).groupby("customer_id").article_id.agg(list).to_dict(),
                   "recent": source_predictions(frame, "recent", config["k"]),
                   "popularity": source_predictions(frame, "popularity", config["k"])}
    segments, per_user = evaluate_segments(predictions, truth, history, saved_segments, config["k"])
    for policy, rows in per_user.items():
        expected = metrics[(metrics.split == "test") & (metrics.model == policy)].iloc[0]
        for metric in ["map", "recall", "precision", "ndcg", "mrr"]:
            np.testing.assert_allclose(rows[metric].mean(), expected[f"{metric}@{config['k']}"], rtol=0, atol=1e-12)
        assert len(rows) == int(expected.users)
    print("Verified saved predictions and baselines against original test/segment metrics.", flush=True)
    exposure, category = catalog_exposure(predictions, lookback, articles, saved_catalog)
    examples = select_examples(per_user[selected], predictions[selected], truth)
    metadata = pd.read_csv(root / "data/raw/articles.csv", dtype={"article_id": str},
        usecols=["article_id", "prod_name", "product_type_name", "colour_group_name"])
    metadata = metadata.fillna("Unknown")
    panels = build_examples(examples, history, target, predictions[selected], truth, metadata)
    private = run_dir / "applied"
    private.mkdir(exist_ok=True)
    (private / "examples_private.json").write_text(json.dumps(panels, indent=2, ensure_ascii=False) + "\n")
    segments.to_csv(tables / "segment_performance.csv", index=False)
    exposure.to_csv(tables / "catalog_exposure.csv", index=False)
    category.to_csv(tables / "category_mix.csv", index=False)
    if image_dir:
        image_root = Path(image_dir)
        image_root = image_root if image_root.is_absolute() else root / image_root
        if not image_root.is_dir():
            raise FileNotFoundError(image_root)
    else:
        image_root = next((p for p in [root / "data/raw/images", root / "data/images"] if p.is_dir()), None)
    image_counts = plot_examples(panels, figures / "recommendation_examples.png", cutoff,
                                config["horizon_days"], config["k"], image_root)
    plot_segments(segments, selected, labels, config["k"], figures / "customer_segments.png")
    plot_catalog(exposure, selected, labels, figures / "catalog_exposure.png")
    snapshot = json.loads((run_dir / "snapshots" / config["test_cutoff"] / "summary.json").read_text())
    recall = pd.read_csv(run_dir / "snapshots" / config["test_cutoff"] / "candidate_recall.csv")
    selected_metric = metrics[(metrics.split == "test") & metrics.model.eq(selected)].iloc[0]
    recent_metric = metrics[(metrics.split == "test") & metrics.model.eq("recent")].iloc[0]
    improvement = selected_metric[f"map@{config['k']}"] / recent_metric[f"map@{config['k']}"] - 1
    statistics = dict(selected_policy=selected, test_cutoff=config["test_cutoff"], k=config["k"],
        test_users=len(truth), modeled_customers=manifest["dataset"]["selected_customers"],
        transactions=manifest["dataset"]["selected_transactions"], article_metadata=manifest["dataset"]["articles"],
        eligible_catalog_items=lookback.article_id.nunique(), candidate_generators=len(SOURCES),
        mean_candidates=float(recall[recall.source.eq("union")].iloc[0].mean_candidates),
        candidate_recall=float(recall[recall.source.eq("union")].iloc[0].candidate_recall),
        final_map=float(selected_metric[f"map@{config['k']}"]),
        recent_map=float(recent_metric[f"map@{config['k']}"]), relative_map_improvement=float(improvement),
        final_recall=float(selected_metric[f"recall@{config['k']}"]),
        final_ndcg=float(selected_metric[f"ndcg@{config['k']}"]),
        users_with_any_hit=int(per_user[selected].hits.gt(0).sum()),
        any_hit_rate=float(per_user[selected].hits.gt(0).mean()),
        matched_recommendation_slots=int(per_user[selected].hits.sum()),
        source_run="reports/runs/" + run_dir.name,
        verification="Recomputed all three policies and activity metrics; matched saved results to 1e-12.",
        selection_disclosure="Five cases curated after observing test outcomes; not a random sample or population success estimate.",
        product_images_used=image_counts["photos"], metadata_cards_used=image_counts["metadata"],
        product_image_note="Actual local photos when available; otherwise category/name/colour metadata cards, not product photos.",
        popularity_definition="Rank articles by transaction count in the prior 120-day eligible catalog; ties by article ID. Exposure is weighted by recommendation slots.",
        long_tail_definition="Items ranked below the top 10% of the eligible catalog by prior transactions.",
        examples=[{key: example[key] for key in ["alias", "title", "selection_rule", "history_purchases", "future_items", "hits", "map_at_k"]} for example in examples])
    (output / "summary.json").write_text(json.dumps(statistics, indent=2, ensure_ascii=False) + "\n")
    plot_system(statistics, snapshot, config, labels[selected], figures / "system_overview.png")
    write_findings(output, statistics, segments, exposure, selected, config["k"])
    print(f"Applied results generated: {output.relative_to(root)} (4 figures; no model refitting).", flush=True)
    return statistics


def write_findings(output, summary, segments, exposure, selected, k):
    selected_exposure = exposure[exposure.model.eq(selected)].iloc[0]
    recent_exposure = exposure[exposure.model.eq("recent")].iloc[0]
    pop_exposure = exposure[exposure.model.eq("popularity")].iloc[0]
    model_segments = segments[segments.model.eq(selected)].set_index("segment")
    baseline_segments = segments[segments.model.eq("recent")].set_index("segment")
    gains = model_segments[f"map@{k}"] - baseline_segments[f"map@{k}"]
    strongest_group = str(gains.idxmax())
    strongest_n = int(model_segments.loc[strongest_group, "users"])
    lines = ["# Applied recommendation findings", "",
        "Generated from the saved, validation-selected policy and the original temporal test. No retraining or test-driven policy change.", "",
        f"MAP@{k} is {summary['final_map']:.5f} vs {summary['recent_map']:.5f} for recent purchases ({summary['relative_map_improvement']:.1%} relative improvement).",
        f"{summary['users_with_any_hit']:,} of {summary['test_users']:,} test purchasers ({summary['any_hit_rate']:.1%}) receive at least one exact article match.", "",
        "## Activity segments", "", f"| Prior lifetime purchases | Customers | Personalized MAP@{k} | Recent MAP@{k} | Absolute difference |", "|---|---:|---:|---:|---:|"]
    for group, label in zip(ACTIVITY_ORDER, ACTIVITY_LABELS):
        chosen = segments[(segments.model == selected) & (segments.segment == group)].iloc[0]
        baseline = segments[(segments.model == "recent") & (segments.segment == group)].iloc[0]
        lines.append(f"| {label} | {int(chosen.users):,} | {chosen[f'map@{k}']:.5f} | {baseline[f'map@{k}']:.5f} | {chosen[f'map@{k}']-baseline[f'map@{k}']:+.5f} |")
    lines.extend(["", f"The largest observed absolute MAP gain is in the {strongest_group}-purchase group (n={strongest_n:,}). "
        f"The >10 group has {int(model_segments.loc['>10', 'users']):,} users and MAP changes by {gains.loc['>10']:+.5f}. "
        f"The 3–10 group changes by {gains.loc['3-10']:+.5f} in MAP. "
        f"Zero-history MAP is {model_segments.loc['zero_history', f'map@{k}']:.5f} vs "
        f"{baseline_segments.loc['zero_history', f'map@{k}']:.5f} for the baseline. "
        "These are descriptive comparisons on one test week, not segment-specific causal or significance claims.", "",
        "## Catalog exposure", "",
        f"Personalization exposes {int(selected_exposure.distinct_recommended_items):,} distinct items ({selected_exposure.catalog_coverage:.2%} of the eligible catalog), versus {int(pop_exposure.distinct_recommended_items)} ({pop_exposure.catalog_coverage:.3%}) for popularity and {int(recent_exposure.distinct_recommended_items):,} ({recent_exposure.catalog_coverage:.2%}) for recent purchases.",
        f"Long-tail recommendation-slot share is {selected_exposure.tail_bottom_90pct_share:.2%} for personalization, {pop_exposure.tail_bottom_90pct_share:.2%} for popularity and {recent_exposure.tail_bottom_90pct_share:.2%} for recent purchases. Long tail means outside the top 10% of the prior 120-day eligible catalog, not an industry-standard exposure threshold.",
        "The selected scorer creates broader exposure than popularity, while concentrating recommendations more than the strong recent-purchase baseline. Coverage alone does not establish better relevance, novelty, or within-list diversity.", "",
        "## Examples and images", "", summary["selection_disclosure"],
        "Five cases illustrate a strong match, partial match, an active-customer miss, a low-history match and a true cold-start miss. Green borders mark exact article-ID matches, not merely similar clothing. All 12 ranked items are shown; history shows up to four recent distinct articles. Anonymous aliases are presentation labels; real IDs and detailed case mappings remain in ignored local outputs.",
        summary["product_image_note"], "",
        "## Proposed online experiment", "",
        "Control: recent purchases with a popularity fallback. Treatment: hybrid candidates plus the frozen selected scorer. Randomize persistently at customer level. Primary: seven-day purchase conversion per assigned eligible customer. Secondary: recommendation CTR, add-to-cart, and revenue per assigned customer. Guardrails: p95/p99 latency, failures, unavailable items, returns, catalog coverage and diversity. This is a proposal, not an H&M experiment performed by this project.", "",
        "Observed cold-start weakness suggests keeping a popularity fallback for zero-history customers in a separately specified treatment policy. That modification has not been included in the reported offline metrics.", "",
        "## Reproduce", "", "Run `make applied` from the project root after generating or restoring the saved run. See docs/applied_results.md for inputs and optional local photo support.", ""])
    (output / "findings.md").write_text("\n".join(lines))
