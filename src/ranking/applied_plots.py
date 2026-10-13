"""Static, shareable applied-results figures built from verified H&M outputs."""
from __future__ import annotations

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

from .applied import ACTIVITY_LABELS, ACTIVITY_ORDER

TEAL, BLUE, GOLD = "#146f63", "#4c78a8", "#bb8a45"
INK, MUTED, PAPER = "#223743", "#64747f", "#f5f8f9"
POLICY_COLOURS = [GOLD, BLUE, TEAL]
SWATCHES = {
    "Black": "#34383b", "White": "#f4f3ed", "Off White": "#eeeadf", "Grey": "#9b9da0",
    "Dark Grey": "#66676c", "Light Grey": "#cbcbd0", "Greyish Beige": "#b4ada0",
    "Blue": "#557da4", "Dark Blue": "#30475b", "Light Blue": "#a4c6df",
    "Beige": "#cfbda2", "Light Beige": "#e1d2bd", "Dark Beige": "#a38c6d",
    "Brown": "#8e6b52", "Dark Brown": "#654938", "Yellow": "#ddc369",
    "Red": "#b96260", "Dark Red": "#8c4749", "Pink": "#d99fae", "Light Pink": "#e5c0ca",
    "Green": "#819a79", "Dark Green": "#4c6a5c", "Light Green": "#afc6a5",
    "Orange": "#d79860", "Purple": "#9980ab", "Turquoise": "#7caeb1",
}


def save_figure(fig, path):
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def product_photo(image_root, article_id):
    if image_root is None:
        return None
    for path in [image_root / article_id[:3] / f"{article_id}.jpg", image_root / f"{article_id}.jpg"]:
        if path.is_file():
            return path
    return None


def draw_card(ax, item, x, y, width, height, counts, image_root=None, small=False):
    hit = bool(item.get("hit", False))
    border = TEAL if hit else "#d3dde1"
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.006,rounding_size=0.009",
                              facecolor="#eef8f3" if hit else "white", edgecolor=border,
                              linewidth=1.8 if hit else .7, transform=ax.transAxes))
    photo = product_photo(image_root, item["article_id"])
    if photo is not None:
        image_ax = ax.inset_axes([x + .012, y + height * .42, width - .024, height * .5])
        image_ax.imshow(plt.imread(photo)); image_ax.axis("off")
        counts["photos"] += 1
    else:
        colour = SWATCHES.get(item["colour"], "#b5bbbf")
        ax.add_patch(Rectangle((x + .012, y + height - .045), width - .024, .017,
                               facecolor=colour, edgecolor="#adb7bc", linewidth=.3, transform=ax.transAxes))
        counts["metadata"] += 1
    header = f"#{item['rank']}" if "rank" in item else item.get("date", "")
    if hit:
        header += " · MATCH"
    ax.text(x + .012, y + height - .066, header, fontsize=6.3 if small else 7.2,
            weight="bold", color=TEAL if hit else MUTED, transform=ax.transAxes, va="top")
    ax.text(x + width - .012, y + height - .066, item["article_id"], fontsize=4.7 if small else 5.2,
            ha="right", color=MUTED, transform=ax.transAxes, va="top")
    # Names and colour labels are dataset metadata; no inferred product photos.
    name_width = 21 if small else 25
    wrapped = textwrap.wrap(str(item["name"]), width=name_width, max_lines=1 if small else 2, placeholder="…")
    text_y = y + height * (.31 if photo else .54)
    ax.text(x + .012, text_y, "\n".join(wrapped), fontsize=6.5 if small else 7.3,
            color=INK, transform=ax.transAxes, va="top", linespacing=1.1)
    ax.text(x + .012, y + .033, f"{item['category']} · {item['colour']}", fontsize=5.9 if small else 6.3,
            color=MUTED, transform=ax.transAxes, va="bottom", wrap=True)


def draw_grid(ax, items, columns, counts, image_root, maximum=None):
    shown = items if maximum is None else items[:maximum]
    rows = max(1, int(np.ceil(len(shown) / columns)))
    gap = .018
    width = (1 - gap * (columns + 1)) / columns
    height = min(.38, (.82 - gap * (rows - 1)) / rows)
    for n, item in enumerate(shown):
        col, row = n % columns, n // columns
        x, y = gap + col * (width + gap), .84 - (row + 1) * height - row * gap
        draw_card(ax, item, x, y, width, height, counts, image_root, small=columns >= 4)
    if maximum and len(items) > maximum:
        ax.text(.02, .012, f"{len(items) - maximum} further items not pictured", fontsize=7, color=MUTED,
                transform=ax.transAxes)


def plot_examples(panels, path, cutoff, horizon_days, k, image_root=None):
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(19, 16.4))
    grid = fig.add_gridspec(len(panels), 3, width_ratios=[1.25, 2.8, 1.25],
                           left=.055, right=.98, bottom=.04, top=.85, hspace=.27, wspace=.065)
    end = cutoff + pd.Timedelta(days=horizon_days - 1)
    fig.text(.055, .96, "What does personalization recommend?", fontsize=23, color=INK, weight="bold")
    fig.text(.055, .936, f"Five anonymous diagnostic cases · actual test purchases {cutoff:%b %d}–{end:%b %d, %Y}",
             fontsize=11.5, color=MUTED)
    fig.text(.055, .908, f"Recent history → all {k} ranked articles → actual future purchases  |  Green borders = exact article-ID matches",
             fontsize=11, color=TEAL)
    fig.text(.055, .011, "Curated after observing test outcomes; these are not random customers or a success-rate estimate. "
             "Cards show real product metadata; colour swatches are illustrative, not product photos.", fontsize=8.8, color=MUTED)
    counts = dict(photos=0, metadata=0)
    for row, panel in enumerate(panels):
        axes = [fig.add_subplot(grid[row, c]) for c in range(3)]
        for ax in axes:
            ax.axis("off"); ax.set(xlim=(0, 1), ylim=(0, 1))
        title = f"{panel['alias']}  {panel['title']}"
        history_word = "purchase" if panel["history_purchases"] == 1 else "purchases"
        future_word = "item" if panel["future_items"] == 1 else "items"
        title += f"  |  {panel['history_purchases']:,} prior {history_word} · {panel['hits']}/{k} recommended hits · {panel['future_items']} future {future_word}"
        axes[0].text(0, 1.18, title, transform=axes[0].transAxes, fontsize=10.8, color=INK, weight="bold")
        for ax, label in zip(axes, ["RECENT HISTORY", f"PERSONALIZED TOP-{k}", "ACTUAL FUTURE"]):
            ax.text(.018, .94, label, transform=ax.transAxes, fontsize=8.5, color=MUTED, weight="bold")
        if panel["history"]:
            draw_grid(axes[0], panel["history"], 2, counts, image_root)
        else:
            axes[0].text(.1, .52, "No recorded\npast purchases", fontsize=12, color=MUTED,
                         transform=axes[0].transAxes, va="center")
        draw_grid(axes[1], panel["recommendations"], 4, counts, image_root)
        draw_grid(axes[2], panel["future"], 2, counts, image_root, maximum=8)
    save_figure(fig, path)
    return counts


def plot_segments(table, selected, labels, k, path):
    policies = ["popularity", "recent", selected]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3))
    x = np.arange(len(ACTIVITY_ORDER)); width = .24
    for ax, metric in zip(axes, ["map", "recall"]):
        for i, (policy, colour) in enumerate(zip(policies, POLICY_COLOURS)):
            data = table[table.model.eq(policy)].set_index("segment").reindex(ACTIVITY_ORDER)
            bars = ax.bar(x + (i - 1) * width, data[f"{metric}@{k}"], width=width, color=colour,
                          label=labels[policy])
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=3)
        counts = table[table.model.eq(selected)].set_index("segment").reindex(ACTIVITY_ORDER).users
        ax.set_xticks(x, [f"{label}\nn={int(n):,}" for label, n in zip(ACTIVITY_LABELS, counts)], fontsize=8.5)
        ax.set(ylabel=f"{metric.upper()}@{k}", ylim=(0, table[f"{metric}@{k}"].max() * 1.28))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
    axes[0].set_title(f"Ranking quality: MAP@{k}", fontsize=12, color=INK)
    axes[1].set_title(f"Target-item recovery: Recall@{k}", fontsize=12, color=INK)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=3, loc="upper center", bbox_to_anchor=(.5, .84), frameon=False)
    fig.suptitle("Where does personalization help?", fontsize=18, color=INK, weight="bold", y=.972, va="top")
    fig.text(.5, .895, "Groups use purchases strictly before the cutoff; identical test purchasers within each policy comparison.",
             ha="center", fontsize=9, color=MUTED)
    low_n = int(table[(table.model == selected) & (table.segment == "1-2")].iloc[0].users)
    fig.text(.05, .012, f"The 1–2 group is small ({low_n} users). Zero-history MAP is lower than popularity. "
             "These are descriptive results for one held-out week.", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=.06, right=.98, bottom=.19, top=.72, wspace=.2)
    save_figure(fig, path)


def plot_catalog(table, selected, labels, path):
    policies = ["popularity", "recent", selected]
    data = table.set_index("model").loc[policies]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), gridspec_kw={"width_ratios": [1, 1.3]})
    y = np.arange(3)
    bars = axes[0].barh(y, data.catalog_coverage * 100, color=POLICY_COLOURS)
    for bar, (_, row) in zip(bars, data.iterrows()):
        axes[0].text(bar.get_width() + .8, bar.get_y() + bar.get_height()/2,
                     f"{row.catalog_coverage:.2%} · {int(row.distinct_recommended_items):,} items", fontsize=8.5,
                     color=INK, va="center")
    axes[0].set_yticks(y, [labels[p] for p in policies], fontsize=9)
    axes[0].set(xlabel="Eligible catalog exposed (%)", xlim=(0, data.catalog_coverage.max() * 100 * 1.6),
                title="Distinct items across top-12 lists")
    axes[0].invert_yaxis()
    left = np.zeros(3)
    bins = [("head_top_1pct_share", "Top 1%", "#758a99"),
            ("next_9pct_share", "Next 9%", "#c3d0d7"),
            ("tail_bottom_90pct_share", "Long tail: remaining 90%", "#4b9b87")]
    for column, label, colour in bins:
        values = data[column].to_numpy() * 100
        axes[1].barh(y, values, left=left, color=colour, label=label)
        for row, (start, value) in enumerate(zip(left, values)):
            if value > 5:
                axes[1].text(start + value / 2, row, f"{value:.1f}%", ha="center", va="center",
                             fontsize=9, color="white" if column != "next_9pct_share" else INK)
        left += values
    axes[1].set_yticks(y, [labels[p] for p in policies], fontsize=9)
    axes[1].set(xlim=(0, 100), xlabel="Share of recommendation slots (%)",
                title="Exposure by historical item-popularity rank")
    axes[1].invert_yaxis()
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=3, loc="lower center", bbox_to_anchor=(.67, .065), frameon=False, fontsize=8)
    fig.suptitle("Broader exposure than popularity — with a relevance/coverage tradeoff", fontsize=15,
                 color=INK, weight="bold", y=.98)
    fig.text(.5, .915, f"Denominator: {int(data.eligible_catalog_items.iloc[0]):,} prior-purchase eligible articles. "
             "Popularity ranks use 120-day transaction counts; ties use article ID.", ha="center", fontsize=8.8, color=MUTED)
    fig.text(.04, .013, "The personalized scorer exposes fewer distinct items and a smaller tail share than recent purchases. "
             "Coverage is not conversion lift or within-list diversity.", fontsize=8.4, color=MUTED)
    fig.subplots_adjust(left=.14, right=.98, top=.79, bottom=.23, wspace=.42)
    save_figure(fig, path)


def plot_system(stats, snapshot, config, model_label, path):
    fig, ax = plt.subplots(figsize=(12.8, 9.2))
    ax.axis("off"); ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.text(.055, .965, "From transaction history to personalized recommendations", fontsize=18,
            color=INK, weight="bold", va="top")
    ax.text(.055, .922, "Verified offline H&M development experiment · weekly temporal validation", fontsize=10, color=MUTED)
    boxes = [
        ("Transaction history + product metadata",
         f"{stats['modeled_customers']:,} sampled customers · {stats['transactions']:,} transactions\n"
         f"{stats['article_metadata']:,} metadata articles; {stats['eligible_catalog_items']:,} eligible at test cutoff"),
        (f"{stats['candidate_generators']} candidate generators",
         "Popularity · age segment · personal recency\nCo-occurrence · implicit ALS · TF-IDF/SVD content"),
        ("Candidate union + historical features",
         f"{stats['mean_candidates']:.1f} candidates/customer on average (cap {config['max_candidates']})\n"
         f"Candidate recall {stats['candidate_recall']:.2%} · no future-positive injection"),
        ("Validation-selected personalized scorer",
         f"{model_label}\nSeven ML scorers compared; settings frozen before final test"),
        (f"Top-{stats['k']} personalized recommendations",
         f"Distinct article IDs for {stats['test_users']:,} test purchasers\n"
         f"MAP@{stats['k']} {stats['final_map']:.5f} vs recent-purchase baseline {stats['recent_map']:.5f}"),
    ]
    x, width, height, gap = .055, .64, .127, .032
    start = .866
    for i, (title, detail) in enumerate(boxes):
        y = start - i * (height + gap) - height
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.009,rounding_size=.012",
            facecolor="#eaf5f1" if i in [2, 4] else PAPER, edgecolor=TEAL if i in [2, 4] else "#d3dfe3", linewidth=1.1))
        ax.text(x + .02, y + height - .023, title, fontsize=12, color=INK, weight="bold", va="top")
        ax.text(x + .02, y + .045, detail, fontsize=9.3, color=MUTED, va="center", linespacing=1.5)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + width/2, y - gap + .006), xytext=(x + width/2, y - .006),
                        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.6))
    ax.text(.75, .8, "POINT-IN-TIME", color=TEAL, fontsize=11, weight="bold")
    ax.text(.75, .755, f"{config['history_days']}-day lookback\n{snapshot['history_rows']:,} test-history rows\nPurchases before {stats['test_cutoff']}",
            color=MUTED, fontsize=9.5, va="top", linespacing=1.6)
    ax.text(.75, .59, "OFFLINE RESULT", color=TEAL, fontsize=11, weight="bold")
    ax.text(.75, .54, f"+{stats['relative_map_improvement']:.1%}", fontsize=30, color=TEAL, weight="bold", va="top")
    ax.text(.75, .47, "relative MAP vs recent purchases", fontsize=9, color=MUTED)
    ax.text(.75, .41, f"Recall@{stats['k']}  {stats['final_recall']:.4f}\nNDCG@{stats['k']}  {stats['final_ndcg']:.4f}\nAny exact hit  {stats['any_hit_rate']:.1%}",
            fontsize=10, color=INK, va="top", linespacing=1.8)
    ax.text(.055, .055, "Observed purchases are offline relevance labels. This experiment does not measure clicks, conversion lift or revenue lift.",
            fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save_figure(fig, path)
