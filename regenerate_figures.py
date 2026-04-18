"""
Regenerates figures for Experiments 3, 4 and 5 with the correct experiment
numbers as used in the article.

Article → Repository mapping:
  Article Experiment 1 → Repository Experiment 3 (exp3_*.csv)
  Article Experiment 2 → Repository Experiment 4 (exp4_*.csv)
  Article Experiment 3 → Repository Experiment 5 (exp5_*.csv)

The script reads the existing CSV result files and overwrites the corresponding
PNG files with updated figure titles that match the article numbering.

Usage:
    python regenerate_figures.py
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = "results"
STEPS = 200

# Capital levels used in Experiment 5 (Article Experiment 3)
CAPITAL_LEVELS = [
    {"name": "Bootstrapped", "runway": 30},
    {"name": "Seed Stage", "runway": 50},
    {"name": "Series A Low", "runway": 75},
    {"name": "Series A Standard", "runway": 100},
    {"name": "Series A High", "runway": 150},
    {"name": "Series B", "runway": 200},
    {"name": "Well Funded", "runway": 300},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_timestamp(filepath):
    """Returns the YYYYMMDD_HHMMSS timestamp embedded in a result filename."""
    name = os.path.basename(filepath).replace(".csv", "").replace(".png", "")
    parts = name.split("_")
    # Filenames: exp3_summary_20260109_083536  →  parts[-2:]
    return "_".join(parts[-2:])


def find_latest_files(repo_prefix):
    """
    Returns (summary_path, timeseries_path, timestamp) for the most recent
    result files whose names start with *repo_prefix* (e.g. 'exp3').
    timeseries_path may be None if no timeseries file exists.
    """
    summary_pattern = f"{OUTPUT_DIR}/{repo_prefix}_summary_*.csv"
    ts_pattern = f"{OUTPUT_DIR}/{repo_prefix}_timeseries_*.csv"

    summary_files = sorted(glob.glob(summary_pattern))
    ts_files = sorted(glob.glob(ts_pattern))

    if not summary_files:
        raise FileNotFoundError(f"No summary CSV found matching {summary_pattern}")

    summary_path = summary_files[-1]
    ts_path = ts_files[-1] if ts_files else None
    timestamp = _extract_timestamp(summary_path)
    return summary_path, ts_path, timestamp


# ---------------------------------------------------------------------------
# Article Experiment 1  (Repository Experiment 3: Unbalanced Teams)
# ---------------------------------------------------------------------------

def plot_exp1_unbalanced_teams(df_results, df_time_series, timestamp):
    """Regenerates all figures for Article Experiment 1 (repo exp3)."""

    print("\n" + "=" * 60)
    print("REGENERATING FIGURES – Article Experiment 1 (repo: exp3)")
    print("=" * 60)

    plt.style.use("seaborn-v0_8-darkgrid")
    configs = df_results["config_name"].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(configs)))
    config_colors = dict(zip(configs, colors))

    # ------------------------------------------------------------------
    # Figure 1: Overview
    # ------------------------------------------------------------------
    fig1, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle("Experiment 1: Team Configuration Overview", fontsize=16, fontweight="bold")

    survival_rates = df_results.groupby("config_name")["survived"].mean()
    axes[0, 0].bar(range(len(survival_rates)), survival_rates.values,
                   color=[config_colors[c] for c in survival_rates.index])
    axes[0, 0].set_xticks(range(len(survival_rates)))
    axes[0, 0].set_xticklabels(survival_rates.index, rotation=45, ha="right")
    axes[0, 0].set_ylabel("Survival Rate")
    axes[0, 0].set_title("Survival Rate by Configuration")
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis="y", alpha=0.3)

    market_share_avg = df_results.groupby("config_name")["market_share"].mean()
    market_share_std = df_results.groupby("config_name")["market_share"].std()
    axes[0, 1].bar(range(len(market_share_avg)), market_share_avg.values,
                   yerr=market_share_std.values,
                   color=[config_colors[c] for c in market_share_avg.index], capsize=5)
    axes[0, 1].set_xticks(range(len(market_share_avg)))
    axes[0, 1].set_xticklabels(market_share_avg.index, rotation=45, ha="right")
    axes[0, 1].set_ylabel("Market Share (%)")
    axes[0, 1].set_title("Average Market Share")
    axes[0, 1].grid(axis="y", alpha=0.3)

    revenue_avg = df_results.groupby("config_name")["revenue"].mean()
    revenue_std = df_results.groupby("config_name")["revenue"].std()
    axes[1, 0].bar(range(len(revenue_avg)), revenue_avg.values,
                   yerr=revenue_std.values,
                   color=[config_colors[c] for c in revenue_avg.index], capsize=5)
    axes[1, 0].set_xticks(range(len(revenue_avg)))
    axes[1, 0].set_xticklabels(revenue_avg.index, rotation=45, ha="right")
    axes[1, 0].set_ylabel("Revenue")
    axes[1, 0].set_title("Average Revenue")
    axes[1, 0].grid(axis="y", alpha=0.3)

    for config in configs:
        config_data = df_results[df_results["config_name"] == config]
        survival_data = config_data["survival_time"].dropna()
        if len(survival_data) > 0:
            axes[1, 1].hist(survival_data, alpha=0.5,
                            label=config, color=config_colors[config], bins=20)
    axes[1, 1].set_xlabel("Survival Time (steps)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Survival Time Distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    f1 = f"{OUTPUT_DIR}/exp3_overview_{timestamp}.png"
    plt.savefig(f1, dpi=300, bbox_inches="tight")
    print(f"  Saved: {f1}")
    plt.close()

    # ------------------------------------------------------------------
    # Figure 2: Product Quality Metrics
    # ------------------------------------------------------------------
    fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig2.suptitle("Experiment 1: Product Quality Metrics", fontsize=16, fontweight="bold")

    metrics = [
        ("feature_completeness", "Feature Completeness"),
        ("technical_debt", "Technical Debt"),
        ("bug_count", "Bug Count"),
        ("code_quality", "Code Quality"),
    ]
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        metric_avg = df_results.groupby("config_name")[metric].mean()
        metric_std = df_results.groupby("config_name")[metric].std()
        ax.bar(range(len(metric_avg)), metric_avg.values,
               yerr=metric_std.values,
               color=[config_colors[c] for c in metric_avg.index], capsize=5)
        ax.set_xticks(range(len(metric_avg)))
        ax.set_xticklabels(metric_avg.index, rotation=45, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"Average {title}")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    f2 = f"{OUTPUT_DIR}/exp3_product_quality_{timestamp}.png"
    plt.savefig(f2, dpi=300, bbox_inches="tight")
    print(f"  Saved: {f2}")
    plt.close()

    # ------------------------------------------------------------------
    # Figure 3: Time Series Evolution
    # ------------------------------------------------------------------
    fig3, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig3.suptitle("Experiment 1: Time Series Evolution", fontsize=16, fontweight="bold")

    time_metrics = [
        ("Market Fit", "Market Fit"),
        ("Market Share", "Market Share (%)"),
        ("Feature Completeness", "Feature Completeness"),
        ("Technical Debt", "Technical Debt"),
        ("Bug Count", "Bug Count"),
        ("Cash Runway", "Cash Runway (months)"),
    ]
    for idx, (metric, ylabel) in enumerate(time_metrics):
        ax = axes[idx // 2, idx % 2]
        for config in configs:
            config_data = df_time_series[df_time_series["config_name"] == config]
            grouped = config_data.groupby(config_data.index % STEPS)[metric]
            mean_vals = grouped.mean()
            std_vals = grouped.std()
            ax.plot(mean_vals.index, mean_vals.values,
                    label=config, color=config_colors[config], linewidth=2)
            ax.fill_between(mean_vals.index,
                            mean_vals.values - std_vals.values,
                            mean_vals.values + std_vals.values,
                            alpha=0.2, color=config_colors[config])
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    f3 = f"{OUTPUT_DIR}/exp3_time_series_{timestamp}.png"
    plt.savefig(f3, dpi=300, bbox_inches="tight")
    print(f"  Saved: {f3}")
    plt.close()

    # ------------------------------------------------------------------
    # Figure 4: Organizational Health
    # ------------------------------------------------------------------
    fig4, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig4.suptitle("Experiment 1: Organizational Health", fontsize=16, fontweight="bold")

    alignment_avg = df_results.groupby("config_name")["organizational_alignment"].mean()
    alignment_std = df_results.groupby("config_name")["organizational_alignment"].std()
    axes[0].bar(range(len(alignment_avg)), alignment_avg.values,
                yerr=alignment_std.values,
                color=[config_colors[c] for c in alignment_avg.index], capsize=5)
    axes[0].set_xticks(range(len(alignment_avg)))
    axes[0].set_xticklabels(alignment_avg.index, rotation=45, ha="right")
    axes[0].set_ylabel("Organizational Alignment")
    axes[0].set_title("Average Organizational Alignment")
    axes[0].grid(axis="y", alpha=0.3)

    conflict_avg = df_results.groupby("config_name")["organizational_conflict"].mean()
    conflict_std = df_results.groupby("config_name")["organizational_conflict"].std()
    axes[1].bar(range(len(conflict_avg)), conflict_avg.values,
                yerr=conflict_std.values,
                color=[config_colors[c] for c in conflict_avg.index], capsize=5)
    axes[1].set_xticks(range(len(conflict_avg)))
    axes[1].set_xticklabels(conflict_avg.index, rotation=45, ha="right")
    axes[1].set_ylabel("Organizational Conflict")
    axes[1].set_title("Average Organizational Conflict")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    f4 = f"{OUTPUT_DIR}/exp3_organizational_{timestamp}.png"
    plt.savefig(f4, dpi=300, bbox_inches="tight")
    print(f"  Saved: {f4}")
    plt.close()

    # ------------------------------------------------------------------
    # Figure 5: Key Metrics Relationships
    # ------------------------------------------------------------------
    fig5, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig5.suptitle("Experiment 1: Key Metrics Relationships", fontsize=16, fontweight="bold")

    scatter_pairs = [
        ("market_share", "revenue", "Market Share (%)", "Revenue"),
        ("feature_completeness", "market_fit", "Feature Completeness", "Market Fit"),
        ("technical_debt", "bug_count", "Technical Debt", "Bug Count"),
        ("code_quality", "organizational_alignment", "Code Quality", "Org. Alignment"),
    ]
    for idx, (x_metric, y_metric, x_label, y_label) in enumerate(scatter_pairs):
        ax = axes[idx // 2, idx % 2]
        for config in configs:
            config_data = df_results[df_results["config_name"] == config]
            ax.scatter(config_data[x_metric], config_data[y_metric],
                       label=config, color=config_colors[config], alpha=0.6, s=50)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{y_label} vs {x_label}")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    f5 = f"{OUTPUT_DIR}/exp3_relationships_{timestamp}.png"
    plt.savefig(f5, dpi=300, bbox_inches="tight")
    print(f"  Saved: {f5}")
    plt.close()


# ---------------------------------------------------------------------------
# Article Experiment 2  (Repository Experiment 4: Quality vs Speed)
# ---------------------------------------------------------------------------

def plot_exp2_quality_vs_speed(df_results, df_time_series, timestamp):
    """Regenerates all figures for Article Experiment 2 (repo exp4)."""

    print("\n" + "=" * 60)
    print("REGENERATING FIGURES – Article Experiment 2 (repo: exp4)")
    print("=" * 60)

    plt.style.use("seaborn-v0_8-darkgrid")
    strategies = df_results["strategy_name"].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))
    strategy_colors = dict(zip(strategies, colors))

    # ------------------------------------------------------------------
    # Figure 1: Core Trade-offs Overview
    # ------------------------------------------------------------------
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle("Experiment 2: Quality vs Speed - Core Trade-offs", fontsize=16, fontweight="bold")

    feature_avg = df_results.groupby("strategy_name")["feature_completeness"].mean()
    feature_std = df_results.groupby("strategy_name")["feature_completeness"].std()
    axes[0, 0].bar(range(len(feature_avg)), feature_avg.values, yerr=feature_std.values,
                   color=[strategy_colors[s] for s in feature_avg.index], capsize=5)
    axes[0, 0].set_xticks(range(len(feature_avg)))
    axes[0, 0].set_xticklabels(feature_avg.index, rotation=45, ha="right", fontsize=8)
    axes[0, 0].set_ylabel("Feature Completeness")
    axes[0, 0].set_title("Feature Development")
    axes[0, 0].grid(axis="y", alpha=0.3)

    debt_avg = df_results.groupby("strategy_name")["technical_debt"].mean()
    debt_std = df_results.groupby("strategy_name")["technical_debt"].std()
    axes[0, 1].bar(range(len(debt_avg)), debt_avg.values, yerr=debt_std.values,
                   color=[strategy_colors[s] for s in debt_avg.index], capsize=5)
    axes[0, 1].set_xticks(range(len(debt_avg)))
    axes[0, 1].set_xticklabels(debt_avg.index, rotation=45, ha="right", fontsize=8)
    axes[0, 1].set_ylabel("Technical Debt")
    axes[0, 1].set_title("Technical Debt Accumulation")
    axes[0, 1].grid(axis="y", alpha=0.3)

    bug_avg = df_results.groupby("strategy_name")["bug_count"].mean()
    bug_std = df_results.groupby("strategy_name")["bug_count"].std()
    axes[0, 2].bar(range(len(bug_avg)), bug_avg.values, yerr=bug_std.values,
                   color=[strategy_colors[s] for s in bug_avg.index], capsize=5)
    axes[0, 2].set_xticks(range(len(bug_avg)))
    axes[0, 2].set_xticklabels(bug_avg.index, rotation=45, ha="right", fontsize=8)
    axes[0, 2].set_ylabel("Bug Count")
    axes[0, 2].set_title("Bug Accumulation")
    axes[0, 2].grid(axis="y", alpha=0.3)

    quality_avg = df_results.groupby("strategy_name")["code_quality"].mean()
    quality_std = df_results.groupby("strategy_name")["code_quality"].std()
    axes[1, 0].bar(range(len(quality_avg)), quality_avg.values, yerr=quality_std.values,
                   color=[strategy_colors[s] for s in quality_avg.index], capsize=5)
    axes[1, 0].set_xticks(range(len(quality_avg)))
    axes[1, 0].set_xticklabels(quality_avg.index, rotation=45, ha="right", fontsize=8)
    axes[1, 0].set_ylabel("Code Quality")
    axes[1, 0].set_title("Code Quality Score")
    axes[1, 0].grid(axis="y", alpha=0.3)

    sustain_avg = df_results.groupby("strategy_name")["sustainability_index"].mean()
    sustain_std = df_results.groupby("strategy_name")["sustainability_index"].std()
    axes[1, 1].bar(range(len(sustain_avg)), sustain_avg.values, yerr=sustain_std.values,
                   color=[strategy_colors[s] for s in sustain_avg.index], capsize=5)
    axes[1, 1].set_xticks(range(len(sustain_avg)))
    axes[1, 1].set_xticklabels(sustain_avg.index, rotation=45, ha="right", fontsize=8)
    axes[1, 1].set_ylabel("Sustainability Index")
    axes[1, 1].set_title("Overall Sustainability")
    axes[1, 1].grid(axis="y", alpha=0.3)

    survival_rate = df_results.groupby("strategy_name")["survived"].mean()
    axes[1, 2].bar(range(len(survival_rate)), survival_rate.values,
                   color=[strategy_colors[s] for s in survival_rate.index])
    axes[1, 2].set_xticks(range(len(survival_rate)))
    axes[1, 2].set_xticklabels(survival_rate.index, rotation=45, ha="right", fontsize=8)
    axes[1, 2].set_ylabel("Survival Rate")
    axes[1, 2].set_title("Company Survival Rate")
    axes[1, 2].set_ylim([0, 1])
    axes[1, 2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    f1 = f"{OUTPUT_DIR}/exp4_core_tradeoffs_{timestamp}.png"
    plt.savefig(f1, dpi=300, bbox_inches="tight")
    print(f"  Saved: {f1}")
    plt.close()

    # ------------------------------------------------------------------
    # Figure 2: Quality Metrics Evolution Over Time
    # ------------------------------------------------------------------
    fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig2.suptitle("Experiment 2: Quality Metrics Evolution Over Time", fontsize=16, fontweight="bold")

    quality_metrics = [
        ("Feature Completeness", "Features"),
        ("Technical Debt", "Technical Debt"),
        ("Bug Count", "Bugs"),
        ("Code Quality", "Quality Score"),
    ]
    for idx, (metric, ylabel) in enumerate(quality_metrics):
        ax = axes[idx // 2, idx % 2]
        for strategy in strategies:
            strategy_data = df_time_series[df_time_series["strategy_name"] == strategy]
            grouped = strategy_data.groupby(strategy_data.index % STEPS)[metric]
            mean_vals = grouped.mean()
            std_vals = grouped.std()
            ax.plot(mean_vals.index, mean_vals.values,
                    label=strategy, color=strategy_colors[strategy], linewidth=2)
            ax.fill_between(mean_vals.index,
                            mean_vals.values - std_vals.values,
                            mean_vals.values + std_vals.values,
                            alpha=0.2, color=strategy_colors[strategy])
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    f2 = f"{OUTPUT_DIR}/exp4_quality_evolution_{timestamp}.png"
    plt.savefig(f2, dpi=300, bbox_inches="tight")
    print(f"  Saved: {f2}")
    plt.close()

    # ------------------------------------------------------------------
    # Figure 3: Business Metrics Evolution Over Time
    # ------------------------------------------------------------------
    fig3, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig3.suptitle("Experiment 2: Business Metrics Evolution Over Time", fontsize=16, fontweight="bold")

    business_metrics = [
        ("Market Fit", "Market Fit"),
        ("Market Share", "Market Share (%)"),
        ("Revenue", "Revenue"),
        ("Cash Runway", "Cash Runway (months)"),
    ]
    for idx, (metric, ylabel) in enumerate(business_metrics):
        ax = axes[idx // 2, idx % 2]
        for strategy in strategies:
            strategy_data = df_time_series[df_time_series["strategy_name"] == strategy]
            grouped = strategy_data.groupby(strategy_data.index % STEPS)[metric]
            mean_vals = grouped.mean()
            std_vals = grouped.std()
            ax.plot(mean_vals.index, mean_vals.values,
                    label=strategy, color=strategy_colors[strategy], linewidth=2)
            ax.fill_between(mean_vals.index,
                            mean_vals.values - std_vals.values,
                            mean_vals.values + std_vals.values,
                            alpha=0.2, color=strategy_colors[strategy])
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    f3 = f"{OUTPUT_DIR}/exp4_business_evolution_{timestamp}.png"
    plt.savefig(f3, dpi=300, bbox_inches="tight")
    print(f"  Saved: {f3}")
    plt.close()

    # ------------------------------------------------------------------
    # Figure 4: Strategy Trade-offs Scatter
    # ------------------------------------------------------------------
    fig4, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig4.suptitle("Experiment 2: Strategy Trade-offs Analysis", fontsize=16, fontweight="bold")

    scatter_pairs = [
        ("feature_completeness", "technical_debt", "Features", "Technical Debt"),
        ("velocity", "sustainability_index", "Velocity (features/step)", "Sustainability"),
        ("technical_debt", "market_share", "Technical Debt", "Market Share (%)"),
        ("bug_count", "revenue", "Bug Count", "Revenue"),
    ]
    for idx, (x_metric, y_metric, x_label, y_label) in enumerate(scatter_pairs):
        ax = axes[idx // 2, idx % 2]
        for strategy in strategies:
            strategy_data = df_results[df_results["strategy_name"] == strategy]
            ax.scatter(strategy_data[x_metric], strategy_data[y_metric],
                       label=strategy, color=strategy_colors[strategy], alpha=0.6, s=50)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{y_label} vs {x_label}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    f4 = f"{OUTPUT_DIR}/exp4_tradeoffs_{timestamp}.png"
    plt.savefig(f4, dpi=300, bbox_inches="tight")
    print(f"  Saved: {f4}")
    plt.close()

    # ------------------------------------------------------------------
    # Figure 5: Distribution Box Plots
    # ------------------------------------------------------------------
    fig5, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig5.suptitle("Experiment 2: Distribution of Key Metrics", fontsize=16, fontweight="bold")

    box_metrics = [
        ("feature_completeness", "Feature Completeness"),
        ("technical_debt", "Technical Debt"),
        ("bug_count", "Bug Count"),
        ("market_share", "Market Share (%)"),
        ("revenue", "Revenue"),
        ("sustainability_index", "Sustainability Index"),
    ]
    for idx, (metric, title) in enumerate(box_metrics):
        ax = axes[idx // 3, idx % 3]
        data_to_plot = [df_results[df_results["strategy_name"] == s][metric].values
                        for s in strategies]
        bp = ax.boxplot(data_to_plot, tick_labels=strategies, patch_artist=True)
        for patch, strategy in zip(bp["boxes"], strategies):
            patch.set_facecolor(strategy_colors[strategy])
            patch.set_alpha(0.6)
        ax.set_xticklabels(strategies, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(f"{title} Distribution")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    f5 = f"{OUTPUT_DIR}/exp4_distributions_{timestamp}.png"
    plt.savefig(f5, dpi=300, bbox_inches="tight")
    print(f"  Saved: {f5}")
    plt.close()

    # ------------------------------------------------------------------
    # Figure 6: Velocity vs Quality Scatter
    # ------------------------------------------------------------------
    fig6, ax = plt.subplots(figsize=(12, 8))
    fig6.suptitle("Experiment 2: The Velocity-Quality Trade-off", fontsize=16, fontweight="bold")

    for strategy in strategies:
        strategy_data = df_results[df_results["strategy_name"] == strategy]
        ax.scatter(strategy_data["velocity"], strategy_data["code_quality"],
                   label=strategy, color=strategy_colors[strategy],
                   alpha=0.6, s=100, edgecolors="black", linewidth=1)

    for strategy in strategies:
        strategy_data = df_results[df_results["strategy_name"] == strategy]
        ax.scatter(strategy_data["velocity"].mean(), strategy_data["code_quality"].mean(),
                   color=strategy_colors[strategy], s=300, marker="*",
                   edgecolors="black", linewidth=2, zorder=10)

    ax.set_xlabel("Development Velocity (features per step)", fontsize=12)
    ax.set_ylabel("Code Quality Score", fontsize=12)
    ax.set_title("Trade-off between Speed and Quality\n(Stars indicate strategy means)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    f6 = f"{OUTPUT_DIR}/exp4_velocity_quality_{timestamp}.png"
    plt.savefig(f6, dpi=300, bbox_inches="tight")
    print(f"  Saved: {f6}")
    plt.close()


# ---------------------------------------------------------------------------
# Article Experiment 3  (Repository Experiment 5: Initial Capital)
# ---------------------------------------------------------------------------

def plot_exp3_initial_capital(df_results, timestamp):
    """Regenerates all figures for Article Experiment 3 (repo exp5)."""

    print("\n" + "=" * 60)
    print("REGENERATING FIGURES – Article Experiment 3 (repo: exp5)")
    print("=" * 60)

    plt.style.use("seaborn-v0_8-darkgrid")
    capital_levels = df_results["capital_level"].unique()
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(capital_levels)))
    capital_colors = dict(zip(capital_levels, colors))
    runways = [c["runway"] for c in CAPITAL_LEVELS]

    # ------------------------------------------------------------------
    # Figure 1: Capital Impact Overview
    # ------------------------------------------------------------------
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle("Experiment 3: Initial Capital Impact Overview", fontsize=16, fontweight="bold")

    survival_rate = df_results.groupby("capital_level")["survived"].mean()
    axes[0, 0].plot(runways, [survival_rate[c["name"]] for c in CAPITAL_LEVELS],
                    marker="o", linewidth=2, markersize=8, color="darkblue")
    axes[0, 0].set_xlabel("Initial Capital (months)")
    axes[0, 0].set_ylabel("Survival Rate")
    axes[0, 0].set_title("Survival Rate vs Initial Capital")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_ylim([0, 1])

    pmf_rate = df_results.groupby("capital_level")["market_fit_achieved"].mean()
    axes[0, 1].plot(runways, [pmf_rate[c["name"]] for c in CAPITAL_LEVELS],
                    marker="s", linewidth=2, markersize=8, color="darkgreen")
    axes[0, 1].set_xlabel("Initial Capital (months)")
    axes[0, 1].set_ylabel("PMF Achievement Rate")
    axes[0, 1].set_title("Product-Market Fit Achievement")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    market_share = df_results.groupby("capital_level")["market_share"].mean()
    market_share_std = df_results.groupby("capital_level")["market_share"].std()
    axes[0, 2].errorbar(runways, [market_share[c["name"]] for c in CAPITAL_LEVELS],
                        yerr=[market_share_std[c["name"]] for c in CAPITAL_LEVELS],
                        marker="D", linewidth=2, markersize=8, capsize=5, color="darkred")
    axes[0, 2].set_xlabel("Initial Capital (months)")
    axes[0, 2].set_ylabel("Market Share (%)")
    axes[0, 2].set_title("Average Market Share Achieved")
    axes[0, 2].grid(alpha=0.3)

    time_to_pmf = df_results[df_results["market_fit_achieved"]].groupby("capital_level")["time_to_pmf"].mean()
    available_runways = [c["runway"] for c in CAPITAL_LEVELS if c["name"] in time_to_pmf.index]
    axes[1, 0].plot(available_runways,
                    [time_to_pmf[c["name"]] for c in CAPITAL_LEVELS if c["name"] in time_to_pmf.index],
                    marker="^", linewidth=2, markersize=8, color="purple")
    axes[1, 0].set_xlabel("Initial Capital (months)")
    axes[1, 0].set_ylabel("Time to PMF (steps)")
    axes[1, 0].set_title("Time to Product-Market Fit")
    axes[1, 0].grid(alpha=0.3)

    efficiency = df_results.groupby("capital_level")["capital_efficiency"].mean()
    axes[1, 1].bar(range(len(efficiency)), efficiency.values,
                   color=[capital_colors[c] for c in efficiency.index])
    axes[1, 1].set_xticks(range(len(efficiency)))
    axes[1, 1].set_xticklabels(efficiency.index, rotation=45, ha="right", fontsize=8)
    axes[1, 1].set_ylabel("Market Share / Capital")
    axes[1, 1].set_title("Capital Efficiency")
    axes[1, 1].grid(axis="y", alpha=0.3)

    revenue = df_results.groupby("capital_level")["revenue"].mean()
    revenue_std = df_results.groupby("capital_level")["revenue"].std()
    axes[1, 2].errorbar(runways, [revenue[c["name"]] for c in CAPITAL_LEVELS],
                        yerr=[revenue_std[c["name"]] for c in CAPITAL_LEVELS],
                        marker="o", linewidth=2, markersize=8, capsize=5, color="darkgoldenrod")
    axes[1, 2].set_xlabel("Initial Capital (months)")
    axes[1, 2].set_ylabel("Final Revenue")
    axes[1, 2].set_title("Average Final Revenue")
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    f1 = f"{OUTPUT_DIR}/exp5_overview_{timestamp}.png"
    plt.savefig(f1, dpi=300, bbox_inches="tight")
    print(f"  Saved: {f1}")
    plt.close()

    # ------------------------------------------------------------------
    # Figure 2: Survival and PMF Distribution
    # ------------------------------------------------------------------
    fig2, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig2.suptitle("Experiment 3: Success Metrics Distribution", fontsize=16, fontweight="bold")

    survival_data = []
    labels = []
    for capital in CAPITAL_LEVELS:
        capital_data = df_results[df_results["capital_level"] == capital["name"]]
        survived_count = capital_data["survived"].sum()
        failed_count = len(capital_data) - survived_count
        survival_data.append([survived_count, failed_count])
        labels.append(f"{capital['name']}\n({capital['runway']}m)")
    survival_data = np.array(survival_data)
    x = np.arange(len(labels))
    width = 0.6

    axes[0].bar(x, survival_data[:, 0], width, label="Survived", color="green", alpha=0.7)
    axes[0].bar(x, survival_data[:, 1], width, bottom=survival_data[:, 0],
                label="Failed", color="red", alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("Number of Companies")
    axes[0].set_title("Survival Distribution by Capital Level")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    pmf_data = []
    for capital in CAPITAL_LEVELS:
        capital_data = df_results[df_results["capital_level"] == capital["name"]]
        achieved_count = capital_data["market_fit_achieved"].sum()
        not_achieved_count = len(capital_data) - achieved_count
        pmf_data.append([achieved_count, not_achieved_count])
    pmf_data = np.array(pmf_data)

    axes[1].bar(x, pmf_data[:, 0], width, label="Achieved PMF", color="blue", alpha=0.7)
    axes[1].bar(x, pmf_data[:, 1], width, bottom=pmf_data[:, 0],
                label="Did Not Achieve", color="orange", alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Number of Companies")
    axes[1].set_title("PMF Achievement by Capital Level")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    f2 = f"{OUTPUT_DIR}/exp5_distributions_{timestamp}.png"
    plt.savefig(f2, dpi=300, bbox_inches="tight")
    print(f"  Saved: {f2}")
    plt.close()

    # ------------------------------------------------------------------
    # Figure 3: Scatter Analysis
    # ------------------------------------------------------------------
    fig3, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig3.suptitle("Experiment 3: Capital vs Performance Metrics", fontsize=16, fontweight="bold")

    scatter_metrics = [
        ("initial_runway", "market_share", "Initial Capital (months)", "Market Share (%)"),
        ("initial_runway", "revenue", "Initial Capital (months)", "Revenue"),
        ("capital_spent", "market_share", "Capital Spent", "Market Share (%)"),
        ("initial_runway", "feature_completeness", "Initial Capital (months)", "Feature Completeness"),
    ]
    for idx, (x_metric, y_metric, x_label, y_label) in enumerate(scatter_metrics):
        ax = axes[idx // 2, idx % 2]
        for capital in capital_levels:
            capital_data = df_results[df_results["capital_level"] == capital]
            ax.scatter(capital_data[x_metric], capital_data[y_metric],
                       label=capital, color=capital_colors[capital], alpha=0.6, s=50)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{y_label} vs {x_label}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    f3 = f"{OUTPUT_DIR}/exp5_scatter_{timestamp}.png"
    plt.savefig(f3, dpi=300, bbox_inches="tight")
    print(f"  Saved: {f3}")
    plt.close()

    # ------------------------------------------------------------------
    # Figure 4: ROI and Diminishing Returns
    # ------------------------------------------------------------------
    fig4, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig4.suptitle("Experiment 3: Return on Investment Analysis", fontsize=16, fontweight="bold")

    marginal_returns_share = []
    marginal_returns_survival = []
    capital_increases = []
    for i in range(len(CAPITAL_LEVELS) - 1):
        current = CAPITAL_LEVELS[i]
        next_level = CAPITAL_LEVELS[i + 1]
        current_data = df_results[df_results["capital_level"] == current["name"]]
        next_data = df_results[df_results["capital_level"] == next_level["name"]]
        share_increase = next_data["market_share"].mean() - current_data["market_share"].mean()
        survival_increase = next_data["survived"].mean() - current_data["survived"].mean()
        capital_increase = next_level["runway"] - current["runway"]
        marginal_returns_share.append(share_increase / capital_increase)
        marginal_returns_survival.append(survival_increase / capital_increase)
        capital_increases.append(f"{current['runway']}-{next_level['runway']}")

    axes[0].bar(range(len(marginal_returns_share)), marginal_returns_share, color="steelblue")
    axes[0].set_xticks(range(len(marginal_returns_share)))
    axes[0].set_xticklabels(capital_increases, rotation=45, ha="right")
    axes[0].set_ylabel("Market Share Gain / Capital Increase")
    axes[0].set_title("Marginal Returns: Market Share")
    axes[0].axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(range(len(marginal_returns_survival)), marginal_returns_survival, color="forestgreen")
    axes[1].set_xticks(range(len(marginal_returns_survival)))
    axes[1].set_xticklabels(capital_increases, rotation=45, ha="right")
    axes[1].set_ylabel("Survival Rate Gain / Capital Increase")
    axes[1].set_title("Marginal Returns: Survival Rate")
    axes[1].axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    f4 = f"{OUTPUT_DIR}/exp5_roi_{timestamp}.png"
    plt.savefig(f4, dpi=300, bbox_inches="tight")
    print(f"  Saved: {f4}")
    plt.close()

    # ------------------------------------------------------------------
    # Figure 5: Box Plots by Capital Level
    # ------------------------------------------------------------------
    fig5, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig5.suptitle("Experiment 3: Metric Distributions by Capital Level", fontsize=16, fontweight="bold")

    box_metrics = [
        ("market_share", "Market Share (%)"),
        ("revenue", "Revenue"),
        ("feature_completeness", "Feature Completeness"),
        ("technical_debt", "Technical Debt"),
        ("capital_efficiency", "Capital Efficiency"),
        ("survival_time", "Survival Time (steps)"),
    ]
    for idx, (metric, title) in enumerate(box_metrics):
        ax = axes[idx // 3, idx % 3]
        data_to_plot = [df_results[df_results["capital_level"] == c["name"]][metric].values
                        for c in CAPITAL_LEVELS]
        bp = ax.boxplot(data_to_plot, tick_labels=[c["name"] for c in CAPITAL_LEVELS], patch_artist=True)
        for patch, capital in zip(bp["boxes"], CAPITAL_LEVELS):
            patch.set_facecolor(capital_colors[capital["name"]])
            patch.set_alpha(0.6)
        ax.set_xticklabels([c["name"] for c in CAPITAL_LEVELS], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(f"{title} Distribution")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    f5 = f"{OUTPUT_DIR}/exp5_boxplots_{timestamp}.png"
    plt.savefig(f5, dpi=300, bbox_inches="tight")
    print(f"  Saved: {f5}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("REGENERATING FIGURES WITH ARTICLE EXPERIMENT NUMBERS")
    print("Article Exp 1 ← repo exp3 | Exp 2 ← exp4 | Exp 3 ← exp5")
    print("=" * 60)

    # --- Article Experiment 1 (repo: exp3) ---
    summary3, ts3, ts3_stamp = find_latest_files("exp3")
    print(f"\nLoading exp3 files (timestamp: {ts3_stamp})")
    df3_results = pd.read_csv(summary3)
    df3_ts = pd.read_csv(ts3) if ts3 else None
    plot_exp1_unbalanced_teams(df3_results, df3_ts, ts3_stamp)

    # --- Article Experiment 2 (repo: exp4) ---
    summary4, ts4, ts4_stamp = find_latest_files("exp4")
    print(f"\nLoading exp4 files (timestamp: {ts4_stamp})")
    df4_results = pd.read_csv(summary4)
    df4_ts = pd.read_csv(ts4) if ts4 else None
    plot_exp2_quality_vs_speed(df4_results, df4_ts, ts4_stamp)

    # --- Article Experiment 3 (repo: exp5) ---
    summary5, _, ts5_stamp = find_latest_files("exp5")
    print(f"\nLoading exp5 files (timestamp: {ts5_stamp})")
    df5_results = pd.read_csv(summary5)
    plot_exp3_initial_capital(df5_results, ts5_stamp)

    print("\n" + "=" * 60)
    print("Done. All figures saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
