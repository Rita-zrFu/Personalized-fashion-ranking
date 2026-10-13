"""Probability quality, customer prioritization and integrated decision-support figures."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from ..applied_plots import BLUE, GOLD, INK, MUTED, TEAL, draw_grid, save_figure

MODEL_LABELS = {"population_prior":"Population prior", "recency_frequency":"Recency / frequency",
                "logistic_raw":"Logistic (raw)", "logistic_log":"Logistic (log features)",
                "random_forest":"Random Forest", "xgboost":"XGBoost", "lightgbm":"LightGBM"}


def plot_propensity_results(output):
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    selected = pd.read_csv(output / "selected_models.csv")
    primary = selected[selected.horizon_days.eq(7)].iloc[0]
    metrics = pd.read_csv(output / "metrics.csv")
    comparison = metrics[(metrics.horizon_days==7) & (metrics.split=="test") & (metrics.variant=="calibrated")].sort_values("pr_auc_ap")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    for ax, column, label in zip(axes, ["roc_auc", "pr_auc_ap"], ["ROC-AUC", "PR-AUC (average precision)"]):
        bars = ax.barh([MODEL_LABELS[m] for m in comparison.model], comparison[column], color=[TEAL if m==primary.selected_model else "#9cb8b2" for m in comparison.model])
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
        ax.set(xlabel=label, xlim=(0, min(1.05, comparison[column].max()*1.2)))
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].axvline(primary.prevalence, color=GOLD, ls="--", label=f"Population purchase rate {primary.prevalence:.2%}")
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Predicting a return purchase within 7 days", fontsize=17, color=INK, weight="bold", y=.98)
    fig.text(.5, .89, f"{int(primary.customers):,} previously observed customers · selected by validation AP · test Sep 16–22, 2020",
             ha="center", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=.15, right=.97, top=.79, bottom=.14, wspace=.38)
    save_figure(fig, figures / "model_comparison.png")
    deciles = pd.read_csv(output / "deciles_h7.csv")
    gains = pd.read_csv(output / "gains_h7.csv")
    operating = pd.read_csv(output / "operating_points.csv")
    top = operating[(operating.horizon_days==7)&(operating.model==primary.selected_model)&
        (operating.variant=="calibrated")&(operating.top_fraction==.1)].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.))
    axes[0].bar(deciles.decile, deciles.observed_purchase_rate*100, color=TEAL, label="Observed purchase rate")
    axes[0].plot(deciles.decile, deciles.mean_propensity*100, marker="o", color=BLUE, label="Mean predicted probability")
    axes[0].axhline(primary.prevalence*100, color=GOLD, ls="--", label="Population rate")
    axes[0].set(xticks=np.arange(1,11), xlabel="Propensity decile (1 = highest)", ylabel="Purchase probability / rate (%)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(gains.customer_fraction*100, gains.buyer_fraction*100, color=TEAL, lw=2.4, label="Propensity prioritization")
    axes[1].plot([0,100],[0,100], color=GOLD, ls="--", label="Random prioritization")
    axes[1].scatter([10],[top.recall*100], color=TEAL, s=35)
    axes[1].annotate(f"Top 10% captures {top.recall:.1%} of buyers\n{top.lift:.2f}× purchase-rate lift",
                      xy=(10, top.recall*100), xytext=(35, 25), fontsize=9, color=INK,
                      arrowprops=dict(arrowstyle="->", color=MUTED))
    axes[1].set(xlabel="Customers prioritized (%)", ylabel="Returning purchasers captured (%)", xlim=(0,100), ylim=(0,105))
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Who is likely to shop soon?", fontsize=18, color=INK, weight="bold", y=.98)
    fig.text(.5, .89, "Deciles and capacity ranks use predicted scores only; future purchases are used solely for evaluation.",
             fontsize=9, color=MUTED, ha="center")
    fig.text(.045, .018, "Purchase-rate lift is observational prioritization; it is not incremental treatment-effect uplift.", color=MUTED, fontsize=9)
    fig.subplots_adjust(left=.065, right=.98, top=.8, bottom=.16, wspace=.22)
    save_figure(fig, figures / "propensity_prioritization.png")
    raw = pd.read_csv(output / "calibration_h7_raw.csv")
    calibrated = pd.read_csv(output / "calibration_h7_calibrated.csv")
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    limit = max(raw.mean_probability.max(), raw.observed_rate.max(), calibrated.mean_probability.max(), calibrated.observed_rate.max())*1.16
    ax.plot([0,limit],[0,limit],ls="--",color="#9aa8ad",label="Perfect calibration")
    for data, colour, label in [(raw,GOLD,"Raw model"),(calibrated,TEAL,"Validation-sigmoid calibration")]:
        ax.plot(data.mean_probability, data.observed_rate, marker="o", color=colour, label=label)
    raw_metric = metrics[(metrics.horizon_days==7)&(metrics.model==primary.selected_model)&(metrics.split=="test")&(metrics.variant=="raw")].iloc[0]
    ax.set(xlabel="Mean predicted probability", ylabel="Observed purchase rate", xlim=(0,limit), ylim=(0,limit))
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Do the probabilities match observed rates?", fontsize=15, color=INK, weight="bold", y=.98)
    fig.text(.5, .89, f"Brier: raw {raw_metric.brier:.4f} → calibrated {primary.brier:.4f}  |  ECE {primary.calibration_ece:.4f}",
             fontsize=9, color=MUTED, ha="center")
    fig.text(.09, .018, "Quantile bins on the final test; calibration was fitted on validation only.", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=.13,right=.97,bottom=.14,top=.8)
    save_figure(fig, figures / "calibration.png")


def plot_recommendation_segments(table, selected_ranker, path):
    segments = ["high","medium","low","no_history_unmodeled"]
    labels = ["High\ntop 20%", "Medium\nnext 30%", "Low\nremaining 50%", "No history\nunmodeled"]
    fig, axes = plt.subplots(1,3,figsize=(14,4.8))
    for ax, metric in zip(axes,["map@12","recall@12","ndcg@12"]):
        x=np.arange(4)
        for i,(policy,colour,name) in enumerate([(selected_ranker,TEAL,"Personalized scorer"),("recent",BLUE,"Recent purchases")]):
            data=table[table.model.eq(policy)].set_index("propensity_segment").reindex(segments)
            bars=ax.bar(x+(i-.5)*.3,data[metric],width=.3,color=colour,label=name)
            ax.bar_label(bars,fmt="%.3f",fontsize=7,padding=3)
        counts=table[table.model.eq(selected_ranker)].set_index("propensity_segment").reindex(segments).purchasing_customers
        ax.set_xticks(x,[f"{label}\nbuyers n={int(n)}" for label,n in zip(labels,counts)],fontsize=7.3)
        ax.set(ylabel=metric.upper(),ylim=(0,table[metric].max()*1.3))
        ax.spines[["top","right"]].set_visible(False)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,ncol=2,loc="upper center",bbox_to_anchor=(.5,.85),frameon=False,fontsize=9)
    fig.suptitle("When customers buy, how useful is the product ranking?", fontsize=16,color=INK,weight="bold",y=.98)
    fig.text(.5,.89,"Propensity groups are assigned across all known customers; ranking metrics condition on observed purchasers.",ha="center",fontsize=9,color=MUTED)
    fig.text(.045,.012,"The 220 no-history buyers are a separate group without a learned return-purchase propensity. Product ordering stays frozen.",fontsize=8.5,color=MUTED)
    fig.subplots_adjust(left=.045,right=.985,bottom=.21,top=.71,wspace=.24)
    save_figure(fig,path)


def plot_customer_decisions(panels,path,image_root=None):
    fig=plt.figure(figsize=(16,13.4))
    grid=fig.add_gridspec(len(panels),3,width_ratios=[1.2,2.2,1.25],left=.045,right=.98,bottom=.04,top=.86,hspace=.3,wspace=.1)
    fig.text(.045,.965,"Two decisions: who to prioritize, and what to recommend",fontsize=20,color=INK,weight="bold")
    fig.text(.045,.932,"Actual 7-day forecasts and future purchases · anonymous cases with both correct predictions and errors",fontsize=10,color=MUTED)
    fig.text(.045,.9,"Green borders = exact product match; the four displayed products are the first four of the retained top-12 list.",fontsize=9.5,color=TEAL)
    counts=dict(photos=0,metadata=0)
    for row,panel in enumerate(panels):
        axes=[fig.add_subplot(grid[row,column]) for column in range(3)]
        for ax in axes:
            ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        axes[0].text(0,1.1,f"{panel['alias']}  {panel['title']}",transform=axes[0].transAxes,color=INK,fontsize=11,weight="bold")
        axes[0].text(.01,.76,f"{panel['propensity']:.1%}",fontsize=27,color=TEAL,weight="bold",transform=axes[0].transAxes,va="center")
        axes[0].text(.01,.59,f"predicted purchase within 7 days\n{panel['propensity_segment'].upper()} propensity segment",fontsize=8.5,color=MUTED,transform=axes[0].transAxes,linespacing=1.5,va="top")
        gap_text=f"{panel['typical_gap']:.1f} days" if panel['typical_gap'] is not None else "not estimable"
        day_word="day" if panel['recency_days']==1 else "days"
        axes[0].text(.01,.34,f"{panel['history_purchases']:,} prior transactions\nLast purchase: {panel['recency_days']:.0f} {day_word} ago\nObserved mean shopping gap: {gap_text}",fontsize=8.3,color=INK,transform=axes[0].transAxes,linespacing=1.5,va="top")
        axes[1].text(.018,.94,"PERSONALIZED TOP-4 PREVIEW",fontsize=8.3,color=MUTED,weight="bold",transform=axes[1].transAxes)
        draw_grid(axes[1],panel["recommendations"][:4],2,counts,image_root)
        axes[2].text(.018,.94,f"PURCHASE WITHIN 7 DAYS: {'YES' if panel['outcome'] else 'NO'}",fontsize=8.3,color=TEAL if panel['outcome'] else MUTED,weight="bold",transform=axes[2].transAxes)
        if panel["future"]:
            draw_grid(axes[2],panel["future"],2,counts,image_root,maximum=4)
        else:
            axes[2].text(.1,.48,"No recorded purchases\nin the full test week",fontsize=10.5,color=MUTED,transform=axes[2].transAxes)
        axes[1].text(.02,.005,f"All top-12: {panel['hits']} exact item matches among {panel['future_items']} future items",fontsize=7.5,color=MUTED,transform=axes[1].transAxes)
    fig.text(.045,.01,"Cases curated after seeing outcomes, not a population estimate. Metadata cards are not product photos. No causal effect is estimated.",fontsize=8.3,color=MUTED)
    save_figure(fig,path)


def plot_integrated_system(root,output,config,integration,path):
    primary=pd.read_csv(output/"selected_models.csv").query("horizon_days == 7").iloc[0]
    ops=pd.read_csv(output/"operating_points.csv")
    top=ops[(ops.horizon_days==7)&(ops.model==primary.selected_model)&(ops.variant=="calibrated")&(ops.top_fraction==.1)].iloc[0]
    ranking=json.loads((root/"reports/hm_sample/applied/summary.json").read_text())
    fig,ax=plt.subplots(figsize=(13,10))
    ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
    ax.text(.04,.965,"Time-aware customer prioritization + personalized product ranking",fontsize=17,color=INK,weight="bold",va="top")
    boxes=[
        ("Customer transaction history",f"{ranking['transactions']:,} transactions · {ranking['modeled_customers']:,} sampled customers\nPast-only shopping rhythm, frequency, price and category preferences"),
        ("Purchase propensity: when is a customer likely to return?",f"7-day {MODEL_LABELS[primary.selected_model]} + validation calibration\nROC-AUC {primary.roc_auc:.3f} · AP {primary.pr_auc_ap:.3f}; also evaluate 14/28 days"),
        ("Customer prioritization",f"Scores for {int(primary.customers):,} previously observed customers\nTop decile: {top.precision:.1%} purchase rate · {top.lift:.2f}× population lift"),
        ("Hybrid candidate generation",f"Six sources · {ranking['mean_candidates']:.1f} candidates/customer\n{ranking['candidate_recall']:.2%} candidate recall in the original ranking test"),
        ("Personalized scorer → Top-12 products",f"Frozen XGBoost product scorer; existing item order retained\nMAP@12 {ranking['final_map']:.5f}; +{ranking['relative_map_improvement']:.1%} vs recent purchases"),
    ]
    start=.875;height=.13;gap=.028;x=.04;width=.66
    for i,(title,detail) in enumerate(boxes):
        y=start-i*(height+gap)-height
        ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle="round,pad=.008,rounding_size=.01",
            facecolor="#eaf5f1" if i in [1,2] else "#f5f8f9",edgecolor=TEAL if i in [1,2] else "#d3dfe3",lw=1.1))
        ax.text(x+.017,y+height-.023,title,fontsize=11.5,color=INK,weight="bold",va="top")
        ax.text(x+.017,y+.042,detail,fontsize=9,color=MUTED,va="center",linespacing=1.5)
        if i<len(boxes)-1:
            ax.annotate("",xy=(x+width/2,y-gap+.003),xytext=(x+width/2,y-.008),arrowprops=dict(arrowstyle="->",color=TEAL,lw=1.5))
    ax.text(.74,.79,"WHEN?",fontsize=14,color=TEAL,weight="bold")
    ax.text(.74,.745,"Purchase probability\nfor all known customers\nincluding nonbuyers",fontsize=10,color=MUTED,va="top",linespacing=1.5)
    ax.text(.74,.56,"WHAT?",fontsize=14,color=TEAL,weight="bold")
    ax.text(.74,.515,"Product relevance\nconditional on a\nrecommendation request",fontsize=10,color=MUTED,va="top",linespacing=1.5)
    ax.text(.74,.31,"INCREMENTAL EFFECT?",fontsize=11,color=TEAL,weight="bold")
    ax.text(.74,.265,"Not identified here.\nRequires randomized\nrecommendation exposure.",fontsize=10,color=MUTED,va="top",linespacing=1.5)
    ax.text(.04,.04,"Architecture is decision support; propensity does not reorder items or prove that recommendation exposure causes a purchase.",fontsize=9,color=MUTED)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    save_figure(fig,path)
