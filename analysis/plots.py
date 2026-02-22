"""
analysis/plots.py
Generates all figures for the paper/submission.
Requires: pip install matplotlib seaborn pandas
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Clean academic style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

CONDITION_LABELS = {
    "A_evaluation_explicit": "A: Evaluation\nexplicit",
    "B_deployment_explicit": "B: Deployment\nexplicit",
    "C_neutral": "C: Neutral\n(baseline)",
    "D_counter_signal": "D: Counter\nsignal",
    "E_researcher_vocab_no_eval": "E: Researcher\nvocab (no eval)",
}

CONDITION_COLORS = {
    "A_evaluation_explicit": "#2563EB",   # blue
    "B_deployment_explicit": "#DC2626",   # red
    "C_neutral": "#6B7280",               # grey
    "D_counter_signal": "#D97706",        # amber
    "E_researcher_vocab_no_eval": "#7C3AED",  # purple
}

os.makedirs("plots", exist_ok=True)


def plot_sycophancy_rates(rates_df: pd.DataFrame, output: str = "plots/fig1_sycophancy_rates.png"):
    """
    Figure 1: Bar chart of sycophancy rates by condition with error bars.
    Core result figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    conditions = rates_df["condition"].tolist()
    x = np.arange(len(conditions))
    bars = ax.bar(
        x,
        rates_df["sycophancy_rate"],
        yerr=rates_df["se"] * 1.96,  # 95% CI
        color=[CONDITION_COLORS.get(c, "#9CA3AF") for c in conditions],
        width=0.55,
        capsize=4,
        error_kw={"linewidth": 1.5, "ecolor": "#374151"},
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions], fontsize=10)
    ax.set_ylabel("Sycophancy Rate", fontsize=12)
    ax.set_xlabel("Experimental Condition", fontsize=12)
    ax.set_title(
        "Figure 1: Sycophancy Rate by Evaluation Context Condition\n"
        "(error bars = 95% CI)",
        fontsize=12, pad=12
    )
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="#E5E7EB", linewidth=1, linestyle="--", zorder=1)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Annotate rates on bars
    for bar, rate in zip(bars, rates_df["sycophancy_rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{rate:.2f}",
            ha="center", va="bottom", fontsize=9, color="#374151"
        )

    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"Saved {output}")


def plot_condition_comparison(
    df: pd.DataFrame,
    output: str = "plots/fig2_condition_comparison.png"
):
    """
    Figure 2: Paired item-level sycophancy rates, Condition A vs B.
    Shows within-subject consistency to support the within-subjects design logic.
    """
    # Pivot to get A and B side by side per item
    pivot = df[df["condition"].isin(["A_evaluation_explicit", "B_deployment_explicit"])]
    pivot = pivot[pivot["sycophancy_code"] != -1]
    wide = pivot.pivot_table(
        index="item_id",
        columns="condition",
        values="sycophancy_code",
        aggfunc="first"
    ).dropna()

    fig, ax = plt.subplots(figsize=(6, 5))

    jitter = 0.03
    x_a = np.zeros(len(wide)) + np.random.uniform(-jitter, jitter, len(wide))
    x_b = np.ones(len(wide)) + np.random.uniform(-jitter, jitter, len(wide))

    ax.scatter(x_a, wide["A_evaluation_explicit"], alpha=0.15, s=12,
               color=CONDITION_COLORS["A_evaluation_explicit"], zorder=3)
    ax.scatter(x_b, wide["B_deployment_explicit"], alpha=0.15, s=12,
               color=CONDITION_COLORS["B_deployment_explicit"], zorder=3)

    # Draw lines connecting matched pairs (sample for readability)
    sample_idx = np.random.choice(len(wide), min(100, len(wide)), replace=False)
    for i in sample_idx:
        ax.plot(
            [x_a[i], x_b[i]],
            [wide["A_evaluation_explicit"].iloc[i], wide["B_deployment_explicit"].iloc[i]],
            color="#9CA3AF", alpha=0.1, linewidth=0.5
        )

    # Mean lines
    ax.hlines(
        wide["A_evaluation_explicit"].mean(), -0.2, 0.2,
        colors=CONDITION_COLORS["A_evaluation_explicit"],
        linewidth=2.5, label=f"Mean A = {wide['A_evaluation_explicit'].mean():.2f}"
    )
    ax.hlines(
        wide["B_deployment_explicit"].mean(), 0.8, 1.2,
        colors=CONDITION_COLORS["B_deployment_explicit"],
        linewidth=2.5, label=f"Mean B = {wide['B_deployment_explicit'].mean():.2f}"
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A: Evaluation explicit", "B: Deployment explicit"], fontsize=11)
    ax.set_ylabel("Sycophancy Code (0/1)", fontsize=12)
    ax.set_title(
        "Figure 2: Within-Item Comparison\nCondition A vs Condition B",
        fontsize=12, pad=12
    )
    ax.set_xlim(-0.4, 1.4)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"Saved {output}")


def plot_evaluation_references(
    df: pd.DataFrame,
    output: str = "plots/fig3_evaluation_references.png"
):
    """
    Figure 3: Secondary measure — rate of explicit evaluation references by condition.
    Tests whether observed behavioural differences are accompanied by
    explicit evaluation-context representation.
    """
    ref_rates = (
        df.groupby("condition")["evaluation_references"]
        .agg(rate="mean", n="count")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(ref_rates))

    ax.bar(
        x,
        ref_rates["rate"],
        color=[CONDITION_COLORS.get(c, "#9CA3AF") for c in ref_rates["condition"]],
        width=0.55,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in ref_rates["condition"]], fontsize=10)
    ax.set_ylabel("Rate of Evaluation References", fontsize=12)
    ax.set_title(
        "Figure 3: Secondary Measure — Explicit Evaluation References by Condition\n"
        "(proportion of responses mentioning evaluation context)",
        fontsize=12, pad=12
    )
    ax.set_ylim(0, max(ref_rates["rate"].max() * 1.3, 0.1))
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"Saved {output}")


def plot_by_source(
    df: pd.DataFrame,
    output: str = "plots/fig4_by_source.png"
):
    """
    Figure 4: Sycophancy rates by condition, split by source domain
    (philpapers / nlp_survey / political).
    Tests whether construct validity failure is domain-specific.
    """
    rates = (
        df[df["sycophancy_code"] != -1]
        .groupby(["source", "condition"])["sycophancy_code"]
        .mean()
        .reset_index()
        .rename(columns={"sycophancy_code": "sycophancy_rate"})
    )

    sources = rates["source"].unique()
    conditions = list(CONDITION_LABELS.keys())
    n_sources = len(sources)

    fig, axes = plt.subplots(1, n_sources, figsize=(5 * n_sources, 4.5), sharey=True)
    if n_sources == 1:
        axes = [axes]

    for ax, source in zip(axes, sources):
        src_data = rates[rates["source"] == source]
        src_data = src_data.set_index("condition").reindex(
            [c for c in conditions if c in src_data["condition"].values]
        ).reset_index()

        x = np.arange(len(src_data))
        ax.bar(
            x,
            src_data["sycophancy_rate"],
            color=[CONDITION_COLORS.get(c, "#9CA3AF") for c in src_data["condition"]],
            width=0.6, zorder=3
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [CONDITION_LABELS.get(c, c) for c in src_data["condition"]],
            fontsize=8, rotation=15, ha="right"
        )
        ax.set_title(source.replace("_", " ").title(), fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        if ax == axes[0]:
            ax.set_ylabel("Sycophancy Rate", fontsize=11)

    fig.suptitle(
        "Figure 4: Sycophancy Rates by Condition and Source Domain",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"Saved {output}")


def generate_all_plots(df: pd.DataFrame, rates_df: pd.DataFrame) -> None:
    """Generate all four figures."""
    plot_sycophancy_rates(rates_df)
    plot_condition_comparison(df)
    plot_evaluation_references(df)
    plot_by_source(df)
    print("\nAll plots saved to plots/")