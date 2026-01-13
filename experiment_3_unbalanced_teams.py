"""
Experiment 3: Unbalanced Teams Analysis

Tests different team composition strategies to understand trade-offs.
Compares Tech-Heavy, Sales-Heavy, and Balanced team configurations.
Analyzes survival time, market conquest, and overall company health.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import run_simulation, get_default_params
import os
from datetime import datetime

# Configuration
NUM_RUNS_PER_CONFIG = 20  # Number of simulations per team configuration
STEPS = 200
OUTPUT_DIR = "results"

# Team configurations to test
TEAM_CONFIGS = [
    {
        'name': 'Tech-Heavy',
        'description': 'Focus on product development',
        'eng': 20, 'sales': 2, 'mkt': 1, 'hr': 1, 'mgmt': 1
    },
    {
        'name': 'Sales-Heavy',
        'description': 'Focus on market conquest',
        'eng': 5, 'sales': 15, 'mkt': 5, 'hr': 1, 'mgmt': 1
    },
    {
        'name': 'Balanced',
        'description': 'Balanced approach',
        'eng': 10, 'sales': 10, 'mkt': 5, 'hr': 1, 'mgmt': 1
    },
    {
        'name': 'Marketing-Heavy',
        'description': 'Focus on brand and awareness',
        'eng': 8, 'sales': 5, 'mkt': 12, 'hr': 1, 'mgmt': 1
    },
    {
        'name': 'Minimal',
        'description': 'Lean startup approach',
        'eng': 5, 'sales': 3, 'mkt': 2, 'hr': 1, 'mgmt': 1
    },
]

def ensure_output_dir():
    """Creates output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def calculate_total_headcount(config):
    """Calculates total headcount for a team configuration."""
    return config['eng'] + config['sales'] + config['mkt'] + config['hr'] + config['mgmt']

def calculate_team_ratio(config):
    """Calculates the ratio of team composition."""
    total = calculate_total_headcount(config)
    return {
        'eng_ratio': config['eng'] / total,
        'sales_ratio': config['sales'] / total,
        'mkt_ratio': config['mkt'] / total
    }

def run_unbalanced_teams_experiment():
    """Runs unbalanced teams experiment."""
    
    print("="*60)
    print("EXPERIMENT 3: UNBALANCED TEAMS ANALYSIS")
    print("="*60)
    print(f"Runs per configuration: {NUM_RUNS_PER_CONFIG}")
    print(f"Simulation steps: {STEPS}\n")
    
    all_results = []
    all_time_series = []
    
    for config in TEAM_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']} - {config['description']}")
        print(f"Team: Eng={config['eng']}, Sales={config['sales']}, "
              f"Mkt={config['mkt']}, HR={config['hr']}, Mgmt={config['mgmt']}")
        print(f"Total headcount: {calculate_total_headcount(config)}")
        ratios = calculate_team_ratio(config)
        print(f"Ratios: Eng={ratios['eng_ratio']:.1%}, "
              f"Sales={ratios['sales_ratio']:.1%}, Mkt={ratios['mkt_ratio']:.1%}")
        print(f"{'='*60}")
        
        config_results = []
        
        for run in range(NUM_RUNS_PER_CONFIG):
            print(f"  Run {run+1}/{NUM_RUNS_PER_CONFIG}...", end=" ")
            
            # Run simulation with default parameters
            result = run_simulation(
                num_engineers=config['eng'],
                num_sales=config['sales'],
                num_marketing=config['mkt'],
                num_hr=config['hr'],
                num_mgmt=config['mgmt'],
                steps=STEPS,
                verbose=False
            )
            
            # Store summary results
            final_state = result['final_state']
            config_results.append({
                'config_name': config['name'],
                'run': run + 1,
                'survived': not result['game_over'],
                'survival_time': result.get('game_over_step', STEPS),
                'market_fit': final_state['market_fit'],
                'market_share': final_state['market_share'],
                'feature_completeness': final_state['feature_completeness'],
                'technical_debt': final_state['technical_debt'],
                'bug_count': final_state['bug_count'],
                'code_quality': final_state['code_quality'],
                'brand_awareness': final_state['brand_awareness'],
                'organizational_alignment': final_state['organizational_alignment'],
                'organizational_conflict': final_state['organizational_conflict'],
                'cash_runway': final_state['cash_runway'],
                'revenue': final_state['revenue'],
                'funding_rounds': final_state['funding_rounds'],
                'pivots_used': final_state['pivots_used'],
                'total_headcount': calculate_total_headcount(config),
                'eng_ratio': ratios['eng_ratio'],
                'sales_ratio': ratios['sales_ratio'],
                'mkt_ratio': ratios['mkt_ratio'],
            })
            
            # Store time series data
            time_series_data = result['data'].copy()
            time_series_data['config_name'] = config['name']
            time_series_data['run'] = run + 1
            all_time_series.append(time_series_data)
            
            print(f"✓ (Survived: {not result['game_over']}, "
                  f"Market Share: {final_state['market_share']:.1f}%)")
        
        all_results.extend(config_results)
        
        # Print summary for this configuration
        df_config = pd.DataFrame(config_results)
        print(f"\n  Summary for {config['name']}:")
        print(f"    Survival rate: {df_config['survived'].mean():.1%}")
        print(f"    Avg survival time: {df_config['survival_time'].mean():.1f} steps")
        print(f"    Avg market share: {df_config['market_share'].mean():.1f}%")
        print(f"    Avg market fit: {df_config['market_fit'].mean():.1f}")
        print(f"    Avg revenue: {df_config['revenue'].mean():.1f}")
    
    return pd.DataFrame(all_results), pd.concat(all_time_series, ignore_index=True)

def analyze_results(df_results, df_time_series):
    """Analyzes and visualizes experiment results."""
    
    print("\n" + "="*60)
    print("ANALYSIS: UNBALANCED TEAMS COMPARISON")
    print("="*60)
    
    # Group by configuration
    grouped = df_results.groupby('config_name')
    
    # 1. Survival Analysis
    print("\n1. SURVIVAL ANALYSIS")
    print("-" * 60)
    survival_stats = grouped.agg({
        'survived': ['mean', 'sum'],
        'survival_time': ['mean', 'std', 'min', 'max']
    })
    print(survival_stats)
    
    # 2. Market Performance
    print("\n2. MARKET PERFORMANCE")
    print("-" * 60)
    market_stats = grouped.agg({
        'market_share': ['mean', 'std', 'max'],
        'market_fit': ['mean', 'std'],
        'brand_awareness': ['mean', 'std']
    })
    print(market_stats)
    
    # 3. Product Quality
    print("\n3. PRODUCT QUALITY")
    print("-" * 60)
    quality_stats = grouped.agg({
        'feature_completeness': ['mean', 'std'],
        'technical_debt': ['mean', 'std'],
        'bug_count': ['mean', 'std'],
        'code_quality': ['mean', 'std']
    })
    print(quality_stats)
    
    # 4. Financial Health
    print("\n4. FINANCIAL HEALTH")
    print("-" * 60)
    financial_stats = grouped.agg({
        'cash_runway': ['mean', 'std'],
        'revenue': ['mean', 'std', 'max'],
        'funding_rounds': ['mean', 'max']
    })
    print(financial_stats)
    
    # 5. Organizational Health
    print("\n5. ORGANIZATIONAL HEALTH")
    print("-" * 60)
    org_stats = grouped.agg({
        'organizational_alignment': ['mean', 'std'],
        'organizational_conflict': ['mean', 'std']
    })
    print(org_stats)
    
    # 6. Key Insights
    print("\n6. KEY INSIGHTS")
    print("-" * 60)
    
    # Best configuration for survival
    best_survival = df_results.groupby('config_name')['survived'].mean().idxmax()
    best_survival_rate = df_results.groupby('config_name')['survived'].mean().max()
    print(f"Best for survival: {best_survival} ({best_survival_rate:.1%})")
    
    # Best configuration for market share
    best_market = df_results.groupby('config_name')['market_share'].mean().idxmax()
    best_market_share = df_results.groupby('config_name')['market_share'].mean().max()
    print(f"Best for market share: {best_market} ({best_market_share:.1f}%)")
    
    # Best configuration for revenue
    best_revenue = df_results.groupby('config_name')['revenue'].mean().idxmax()
    best_revenue_val = df_results.groupby('config_name')['revenue'].mean().max()
    print(f"Best for revenue: {best_revenue} ({best_revenue_val:.1f})")
    
    # Best configuration for quality
    best_quality = df_results.groupby('config_name')['code_quality'].mean().idxmax()
    best_quality_val = df_results.groupby('config_name')['code_quality'].mean().max()
    print(f"Best for code quality: {best_quality} ({best_quality_val:.1f})")
    
    # Configuration with least technical debt
    least_debt = df_results.groupby('config_name')['technical_debt'].mean().idxmin()
    least_debt_val = df_results.groupby('config_name')['technical_debt'].mean().min()
    print(f"Least technical debt: {least_debt} ({least_debt_val:.1f})")

def create_visualizations(df_results, df_time_series, timestamp):
    """Creates comprehensive visualizations."""
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    ensure_output_dir()
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    configs = df_results['config_name'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(configs)))
    config_colors = dict(zip(configs, colors))
    
    # Figure 1: Survival and Market Performance Overview
    fig1, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle('Experiment 3: Team Configuration Overview', fontsize=16, fontweight='bold')
    
    # Survival Rate
    survival_rates = df_results.groupby('config_name')['survived'].mean()
    axes[0, 0].bar(range(len(survival_rates)), survival_rates.values, 
                   color=[config_colors[c] for c in survival_rates.index])
    axes[0, 0].set_xticks(range(len(survival_rates)))
    axes[0, 0].set_xticklabels(survival_rates.index, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Survival Rate')
    axes[0, 0].set_title('Survival Rate by Configuration')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Average Market Share
    market_share_avg = df_results.groupby('config_name')['market_share'].mean()
    market_share_std = df_results.groupby('config_name')['market_share'].std()
    axes[0, 1].bar(range(len(market_share_avg)), market_share_avg.values,
                   yerr=market_share_std.values,
                   color=[config_colors[c] for c in market_share_avg.index],
                   capsize=5)
    axes[0, 1].set_xticks(range(len(market_share_avg)))
    axes[0, 1].set_xticklabels(market_share_avg.index, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Market Share (%)')
    axes[0, 1].set_title('Average Market Share')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Average Revenue
    revenue_avg = df_results.groupby('config_name')['revenue'].mean()
    revenue_std = df_results.groupby('config_name')['revenue'].std()
    axes[1, 0].bar(range(len(revenue_avg)), revenue_avg.values,
                   yerr=revenue_std.values,
                   color=[config_colors[c] for c in revenue_avg.index],
                   capsize=5)
    axes[1, 0].set_xticks(range(len(revenue_avg)))
    axes[1, 0].set_xticklabels(revenue_avg.index, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Revenue')
    axes[1, 0].set_title('Average Revenue')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Survival Time Distribution
    for config in configs:
        config_data = df_results[df_results['config_name'] == config]
        survival_data = config_data['survival_time'].dropna()
        if len(survival_data) > 0:
            axes[1, 1].hist(survival_data, alpha=0.5, 
                           label=config, color=config_colors[config], bins=20)
    axes[1, 1].set_xlabel('Survival Time (steps)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Survival Time Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename1 = f"{OUTPUT_DIR}/exp3_overview_{timestamp}.png"
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename1}")
    plt.close()
    
    # Figure 2: Product Quality Metrics
    fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig2.suptitle('Experiment 3: Product Quality Metrics', fontsize=16, fontweight='bold')
    
    metrics = [
        ('feature_completeness', 'Feature Completeness'),
        ('technical_debt', 'Technical Debt'),
        ('bug_count', 'Bug Count'),
        ('code_quality', 'Code Quality')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        metric_avg = df_results.groupby('config_name')[metric].mean()
        metric_std = df_results.groupby('config_name')[metric].std()
        ax.bar(range(len(metric_avg)), metric_avg.values,
               yerr=metric_std.values,
               color=[config_colors[c] for c in metric_avg.index],
               capsize=5)
        ax.set_xticks(range(len(metric_avg)))
        ax.set_xticklabels(metric_avg.index, rotation=45, ha='right')
        ax.set_ylabel(title)
        ax.set_title(f'Average {title}')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename2 = f"{OUTPUT_DIR}/exp3_product_quality_{timestamp}.png"
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename2}")
    plt.close()
    
    # Figure 3: Time Series Evolution
    fig3, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig3.suptitle('Experiment 3: Time Series Evolution', fontsize=16, fontweight='bold')
    
    time_metrics = [
        ('Market Fit', 'Market Fit'),
        ('Market Share', 'Market Share (%)'),
        ('Feature Completeness', 'Feature Completeness'),
        ('Technical Debt', 'Technical Debt'),
        ('Bug Count', 'Bug Count'),
        ('Cash Runway', 'Cash Runway (months)')
    ]
    
    for idx, (metric, ylabel) in enumerate(time_metrics):
        ax = axes[idx // 2, idx % 2]
        
        for config in configs:
            config_data = df_time_series[df_time_series['config_name'] == config]
            grouped = config_data.groupby(config_data.index % STEPS)[metric]
            mean_vals = grouped.mean()
            std_vals = grouped.std()
            
            ax.plot(mean_vals.index, mean_vals.values, 
                   label=config, color=config_colors[config], linewidth=2)
            ax.fill_between(mean_vals.index,
                           mean_vals.values - std_vals.values,
                           mean_vals.values + std_vals.values,
                           alpha=0.2, color=config_colors[config])
        
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filename3 = f"{OUTPUT_DIR}/exp3_time_series_{timestamp}.png"
    plt.savefig(filename3, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename3}")
    plt.close()
    
    # Figure 4: Organizational Health
    fig4, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig4.suptitle('Experiment 3: Organizational Health', fontsize=16, fontweight='bold')
    
    # Organizational Alignment
    alignment_avg = df_results.groupby('config_name')['organizational_alignment'].mean()
    alignment_std = df_results.groupby('config_name')['organizational_alignment'].std()
    axes[0].bar(range(len(alignment_avg)), alignment_avg.values,
               yerr=alignment_std.values,
               color=[config_colors[c] for c in alignment_avg.index],
               capsize=5)
    axes[0].set_xticks(range(len(alignment_avg)))
    axes[0].set_xticklabels(alignment_avg.index, rotation=45, ha='right')
    axes[0].set_ylabel('Organizational Alignment')
    axes[0].set_title('Average Organizational Alignment')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Organizational Conflict
    conflict_avg = df_results.groupby('config_name')['organizational_conflict'].mean()
    conflict_std = df_results.groupby('config_name')['organizational_conflict'].std()
    axes[1].bar(range(len(conflict_avg)), conflict_avg.values,
               yerr=conflict_std.values,
               color=[config_colors[c] for c in conflict_avg.index],
               capsize=5)
    axes[1].set_xticks(range(len(conflict_avg)))
    axes[1].set_xticklabels(conflict_avg.index, rotation=45, ha='right')
    axes[1].set_ylabel('Organizational Conflict')
    axes[1].set_title('Average Organizational Conflict')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename4 = f"{OUTPUT_DIR}/exp3_organizational_{timestamp}.png"
    plt.savefig(filename4, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename4}")
    plt.close()
    
    # Figure 5: Scatter Matrix - Key Metrics Relationships
    fig5, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig5.suptitle('Experiment 3: Key Metrics Relationships', fontsize=16, fontweight='bold')
    
    scatter_pairs = [
        ('market_share', 'revenue', 'Market Share (%)', 'Revenue'),
        ('feature_completeness', 'market_fit', 'Feature Completeness', 'Market Fit'),
        ('technical_debt', 'bug_count', 'Technical Debt', 'Bug Count'),
        ('code_quality', 'organizational_alignment', 'Code Quality', 'Org. Alignment')
    ]
    
    for idx, (x_metric, y_metric, x_label, y_label) in enumerate(scatter_pairs):
        ax = axes[idx // 2, idx % 2]
        
        for config in configs:
            config_data = df_results[df_results['config_name'] == config]
            ax.scatter(config_data[x_metric], config_data[y_metric],
                      label=config, color=config_colors[config], alpha=0.6, s=50)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'{y_label} vs {x_label}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filename5 = f"{OUTPUT_DIR}/exp3_relationships_{timestamp}.png"
    plt.savefig(filename5, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename5}")
    plt.close()

def save_results(df_results, df_time_series, timestamp):
    """Saves results to CSV files."""
    
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    ensure_output_dir()
    
    # Save summary results
    summary_file = f"{OUTPUT_DIR}/exp3_summary_{timestamp}.csv"
    df_results.to_csv(summary_file, index=False)
    print(f"  Saved summary: {summary_file}")
    
    # Save time series data
    timeseries_file = f"{OUTPUT_DIR}/exp3_timeseries_{timestamp}.csv"
    df_time_series.to_csv(timeseries_file, index=False)
    print(f"  Saved time series: {timeseries_file}")
    
    # Save aggregated statistics
    stats_file = f"{OUTPUT_DIR}/exp3_statistics_{timestamp}.csv"
    stats = df_results.groupby('config_name').agg({
        'survived': ['mean', 'sum'],
        'survival_time': ['mean', 'std'],
        'market_fit': ['mean', 'std'],
        'market_share': ['mean', 'std', 'max'],
        'feature_completeness': ['mean', 'std'],
        'technical_debt': ['mean', 'std'],
        'bug_count': ['mean', 'std'],
        'code_quality': ['mean', 'std'],
        'revenue': ['mean', 'std', 'max'],
        'cash_runway': ['mean', 'std'],
        'organizational_alignment': ['mean', 'std'],
        'organizational_conflict': ['mean', 'std']
    })
    stats.to_csv(stats_file)
    print(f"  Saved statistics: {stats_file}")

def main():
    """Main execution function."""
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run experiment
    df_results, df_time_series = run_unbalanced_teams_experiment()
    
    # Analyze results
    analyze_results(df_results, df_time_series)
    
    # Create visualizations
    create_visualizations(df_results, df_time_series, timestamp)
    
    # Save results
    save_results(df_results, df_time_series, timestamp)
    
    print("\n" + "="*60)
    print("EXPERIMENT 3 COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Results saved with timestamp: {timestamp}")
    print(f"Check the '{OUTPUT_DIR}' directory for outputs.")

if __name__ == "__main__":
    main()
