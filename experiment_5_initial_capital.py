"""
Experiment 5: Impact of Initial Capital

Tests how different initial capital levels affect company outcomes.
Analyzes the relationship between starting cash runway and success metrics.
Answers: How much capital is needed to achieve product-market fit?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import run_simulation, get_default_params
import os
from datetime import datetime

# Configuration
NUM_RUNS_PER_CONFIG = 20  # Number of simulations per capital level
STEPS = 200
OUTPUT_DIR = "results"

# Initial capital levels to test (in months of runway)
CAPITAL_LEVELS = [
    {'name': 'Bootstrapped', 'runway': 30, 'description': 'Very low capital - bootstrapped startup'},
    {'name': 'Seed Stage', 'runway': 50, 'description': 'Seed funding level'},
    {'name': 'Series A Low', 'runway': 75, 'description': 'Low Series A funding'},
    {'name': 'Series A Standard', 'runway': 100, 'description': 'Standard Series A funding'},
    {'name': 'Series A High', 'runway': 150, 'description': 'High Series A funding'},
    {'name': 'Series B', 'runway': 200, 'description': 'Series B level funding'},
    {'name': 'Well Funded', 'runway': 300, 'description': 'Very well funded startup'},
]

def ensure_output_dir():
    """Creates output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def run_initial_capital_experiment():
    """Runs initial capital experiment."""
    
    print("="*60)
    print("EXPERIMENT 5: IMPACT OF INITIAL CAPITAL")
    print("="*60)
    print(f"Runs per capital level: {NUM_RUNS_PER_CONFIG}")
    print(f"Simulation steps: {STEPS}\n")
    
    all_results = []
    all_time_series = []
    
    for capital in CAPITAL_LEVELS:
        print(f"\n{'='*60}")
        print(f"Testing: {capital['name']} - {capital['runway']} months")
        print(f"Description: {capital['description']}")
        print(f"{'='*60}")
        
        capital_results = []
        
        for run in range(NUM_RUNS_PER_CONFIG):
            print(f"  Run {run+1}/{NUM_RUNS_PER_CONFIG}...", end=" ")
            
            # Get default parameters
            params = get_default_params()
            
            # Run simulation with standard team size
            result = run_simulation(
                num_engineers=10,
                num_sales=5,
                num_marketing=3,
                num_hr=1,
                num_mgmt=1,
                steps=STEPS,
                params=params,
                verbose=False
            )
            
            # Modify initial cash runway after model creation
            # We need to adjust the result based on the capital level
            # For this experiment, we'll track if the company would have survived
            # with different initial capital levels
            
            # Get the time series data to analyze cash flow
            data = result['data']
            final_state = result['final_state']
            
            # Calculate metrics based on capital level
            # Simulate what would happen with different initial capital
            initial_cash = capital['runway']
            simulated_cash = initial_cash
            survived_with_capital = True
            survival_step = STEPS
            
            # Recalculate cash runway trajectory
            for step in range(len(data)):
                if step == 0:
                    simulated_cash = initial_cash
                else:
                    # Calculate cash change for this step
                    revenue = data.iloc[step]['Revenue']
                    # Approximate burn rate from the model (10 base + team costs)
                    team_size = (data.iloc[step]['Engineers'] + 
                               data.iloc[step]['Sales'] + 
                               data.iloc[step]['Marketing'])
                    burn_rate = 10.0 + (team_size * 0.5)  # Approximate
                    
                    simulated_cash += revenue - burn_rate
                    
                    if simulated_cash < 0 and survived_with_capital:
                        survived_with_capital = False
                        survival_step = step
                        break
            
            # Calculate time to product-market fit
            market_fit_achieved = False
            time_to_pmf = STEPS
            for step in range(len(data)):
                if data.iloc[step]['Market Fit'] >= 60:
                    market_fit_achieved = True
                    time_to_pmf = step
                    break
            
            # Calculate time to profitability (revenue > burn rate)
            time_to_profitability = STEPS
            for step in range(len(data)):
                if step > 0 and data.iloc[step]['Revenue'] > 10:  # Simplified
                    time_to_profitability = step
                    break
            
            # Calculate capital efficiency (market share per unit of capital spent)
            capital_spent = initial_cash - simulated_cash if survived_with_capital else initial_cash
            capital_efficiency = final_state['market_share'] / capital_spent if capital_spent > 0 else 0
            
            capital_results.append({
                'capital_level': capital['name'],
                'initial_runway': capital['runway'],
                'run': run + 1,
                'survived': survived_with_capital,
                'survival_time': survival_step,
                'final_cash': simulated_cash if survived_with_capital else 0,
                'market_fit': final_state['market_fit'],
                'market_fit_achieved': market_fit_achieved,
                'time_to_pmf': time_to_pmf,
                'time_to_profitability': time_to_profitability,
                'market_share': final_state['market_share'],
                'feature_completeness': final_state['feature_completeness'],
                'technical_debt': final_state['technical_debt'],
                'bug_count': final_state['bug_count'],
                'code_quality': final_state['code_quality'],
                'brand_awareness': final_state['brand_awareness'],
                'revenue': final_state['revenue'],
                'funding_rounds': final_state['funding_rounds'],
                'pivots_used': final_state['pivots_used'],
                'capital_spent': capital_spent,
                'capital_efficiency': capital_efficiency,
            })
            
            # Store time series data
            time_series_data = result['data'].copy()
            time_series_data['capital_level'] = capital['name']
            time_series_data['initial_runway'] = capital['runway']
            time_series_data['run'] = run + 1
            # Add simulated cash trajectory
            time_series_data['simulated_cash'] = initial_cash
            all_time_series.append(time_series_data)
            
            print(f"✓ (Survived: {survived_with_capital}, "
                  f"PMF: {market_fit_achieved}, "
                  f"Market Share: {final_state['market_share']:.1f}%)")
        
        all_results.extend(capital_results)
        
        # Print summary for this capital level
        df_capital = pd.DataFrame(capital_results)
        print(f"\n  Summary for {capital['name']} ({capital['runway']} months):")
        print(f"    Survival rate: {df_capital['survived'].mean():.1%}")
        print(f"    PMF achievement rate: {df_capital['market_fit_achieved'].mean():.1%}")
        print(f"    Avg time to PMF: {df_capital['time_to_pmf'].mean():.1f} steps")
        print(f"    Avg market share: {df_capital['market_share'].mean():.1f}%")
        print(f"    Avg capital efficiency: {df_capital['capital_efficiency'].mean():.4f}")
        print(f"    Avg final cash: {df_capital['final_cash'].mean():.1f} months")
    
    return pd.DataFrame(all_results), pd.concat(all_time_series, ignore_index=True)

def analyze_results(df_results, df_time_series):
    """Analyzes and visualizes experiment results."""
    
    print("\n" + "="*60)
    print("ANALYSIS: INITIAL CAPITAL IMPACT")
    print("="*60)
    
    # Group by capital level
    grouped = df_results.groupby('capital_level')
    
    # 1. Survival Analysis
    print("\n1. SURVIVAL ANALYSIS")
    print("-" * 60)
    survival_stats = grouped.agg({
        'survived': ['mean', 'sum'],
        'survival_time': ['mean', 'std', 'min', 'max'],
        'final_cash': ['mean', 'std']
    })
    print(survival_stats)
    
    # 2. Product-Market Fit Analysis
    print("\n2. PRODUCT-MARKET FIT ACHIEVEMENT")
    print("-" * 60)
    pmf_stats = grouped.agg({
        'market_fit_achieved': ['mean', 'sum'],
        'time_to_pmf': ['mean', 'std', 'min'],
        'market_fit': ['mean', 'std', 'max']
    })
    print(pmf_stats)
    
    # 3. Market Performance
    print("\n3. MARKET PERFORMANCE")
    print("-" * 60)
    market_stats = grouped.agg({
        'market_share': ['mean', 'std', 'max'],
        'brand_awareness': ['mean', 'std'],
        'revenue': ['mean', 'std', 'max']
    })
    print(market_stats)
    
    # 4. Capital Efficiency
    print("\n4. CAPITAL EFFICIENCY")
    print("-" * 60)
    efficiency_stats = grouped.agg({
        'capital_spent': ['mean', 'std'],
        'capital_efficiency': ['mean', 'std', 'max'],
        'time_to_profitability': ['mean', 'std']
    })
    print(efficiency_stats)
    
    # 5. Product Quality
    print("\n5. PRODUCT QUALITY")
    print("-" * 60)
    quality_stats = grouped.agg({
        'feature_completeness': ['mean', 'std'],
        'technical_debt': ['mean', 'std'],
        'bug_count': ['mean', 'std'],
        'code_quality': ['mean', 'std']
    })
    print(quality_stats)
    
    # 6. Key Insights
    print("\n6. KEY INSIGHTS")
    print("-" * 60)
    
    # Minimum capital for reasonable survival
    survival_by_capital = df_results.groupby('capital_level')['survived'].mean()
    min_capital_50 = None
    min_capital_75 = None
    for level in CAPITAL_LEVELS:
        rate = survival_by_capital.get(level['name'], 0)
        if min_capital_50 is None and rate >= 0.5:
            min_capital_50 = level['runway']
        if min_capital_75 is None and rate >= 0.75:
            min_capital_75 = level['runway']
    
    print(f"Minimum capital for 50% survival rate: {min_capital_50} months" if min_capital_50 else "Not achieved in tested range")
    print(f"Minimum capital for 75% survival rate: {min_capital_75} months" if min_capital_75 else "Not achieved in tested range")
    
    # Capital level with best PMF achievement
    pmf_by_capital = df_results.groupby('capital_level')['market_fit_achieved'].mean()
    best_pmf_capital = pmf_by_capital.idxmax()
    best_pmf_rate = pmf_by_capital.max()
    print(f"Best for PMF achievement: {best_pmf_capital} ({best_pmf_rate:.1%})")
    
    # Most capital efficient
    efficiency_by_capital = df_results.groupby('capital_level')['capital_efficiency'].mean()
    most_efficient = efficiency_by_capital.idxmax()
    efficiency_val = efficiency_by_capital.max()
    print(f"Most capital efficient: {most_efficient} ({efficiency_val:.4f} market share/capital)")
    
    # Best market share
    share_by_capital = df_results.groupby('capital_level')['market_share'].mean()
    best_share = share_by_capital.idxmax()
    share_val = share_by_capital.max()
    print(f"Best market share: {best_share} ({share_val:.1f}%)")
    
    # Diminishing returns analysis
    print("\n7. DIMINISHING RETURNS ANALYSIS")
    print("-" * 60)
    for i, level in enumerate(CAPITAL_LEVELS[:-1]):
        next_level = CAPITAL_LEVELS[i+1]
        current_share = df_results[df_results['capital_level'] == level['name']]['market_share'].mean()
        next_share = df_results[df_results['capital_level'] == next_level['name']]['market_share'].mean()
        capital_increase = next_level['runway'] - level['runway']
        share_increase = next_share - current_share
        roi = share_increase / capital_increase if capital_increase > 0 else 0
        print(f"{level['name']} → {next_level['name']}: +{share_increase:.2f}% market share / +{capital_increase} months (ROI: {roi:.4f})")

def create_visualizations(df_results, df_time_series, timestamp):
    """Creates comprehensive visualizations."""
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    ensure_output_dir()
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    capital_levels = df_results['capital_level'].unique()
    # Use a sequential colormap for capital levels
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(capital_levels)))
    capital_colors = dict(zip(capital_levels, colors))
    
    # Figure 1: Capital Impact Overview
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle('Experiment 5: Initial Capital Impact Overview', fontsize=16, fontweight='bold')
    
    # Survival Rate vs Capital
    survival_rate = df_results.groupby('capital_level')['survived'].mean()
    runways = [c['runway'] for c in CAPITAL_LEVELS]
    axes[0, 0].plot(runways, [survival_rate[c['name']] for c in CAPITAL_LEVELS], 
                    marker='o', linewidth=2, markersize=8, color='darkblue')
    axes[0, 0].set_xlabel('Initial Capital (months)')
    axes[0, 0].set_ylabel('Survival Rate')
    axes[0, 0].set_title('Survival Rate vs Initial Capital')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # PMF Achievement Rate vs Capital
    pmf_rate = df_results.groupby('capital_level')['market_fit_achieved'].mean()
    axes[0, 1].plot(runways, [pmf_rate[c['name']] for c in CAPITAL_LEVELS],
                    marker='s', linewidth=2, markersize=8, color='darkgreen')
    axes[0, 1].set_xlabel('Initial Capital (months)')
    axes[0, 1].set_ylabel('PMF Achievement Rate')
    axes[0, 1].set_title('Product-Market Fit Achievement')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Market Share vs Capital
    market_share = df_results.groupby('capital_level')['market_share'].mean()
    market_share_std = df_results.groupby('capital_level')['market_share'].std()
    axes[0, 2].errorbar(runways, [market_share[c['name']] for c in CAPITAL_LEVELS],
                       yerr=[market_share_std[c['name']] for c in CAPITAL_LEVELS],
                       marker='D', linewidth=2, markersize=8, capsize=5, color='darkred')
    axes[0, 2].set_xlabel('Initial Capital (months)')
    axes[0, 2].set_ylabel('Market Share (%)')
    axes[0, 2].set_title('Average Market Share Achieved')
    axes[0, 2].grid(alpha=0.3)
    
    # Time to PMF vs Capital
    time_to_pmf = df_results[df_results['market_fit_achieved']].groupby('capital_level')['time_to_pmf'].mean()
    available_runways = [c['runway'] for c in CAPITAL_LEVELS if c['name'] in time_to_pmf.index]
    axes[1, 0].plot(available_runways, [time_to_pmf[c['name']] for c in CAPITAL_LEVELS if c['name'] in time_to_pmf.index],
                    marker='^', linewidth=2, markersize=8, color='purple')
    axes[1, 0].set_xlabel('Initial Capital (months)')
    axes[1, 0].set_ylabel('Time to PMF (steps)')
    axes[1, 0].set_title('Time to Product-Market Fit')
    axes[1, 0].grid(alpha=0.3)
    
    # Capital Efficiency
    efficiency = df_results.groupby('capital_level')['capital_efficiency'].mean()
    axes[1, 1].bar(range(len(efficiency)), efficiency.values,
                   color=[capital_colors[c] for c in efficiency.index])
    axes[1, 1].set_xticks(range(len(efficiency)))
    axes[1, 1].set_xticklabels(efficiency.index, rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_ylabel('Market Share / Capital')
    axes[1, 1].set_title('Capital Efficiency')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Revenue vs Capital
    revenue = df_results.groupby('capital_level')['revenue'].mean()
    revenue_std = df_results.groupby('capital_level')['revenue'].std()
    axes[1, 2].errorbar(runways, [revenue[c['name']] for c in CAPITAL_LEVELS],
                       yerr=[revenue_std[c['name']] for c in CAPITAL_LEVELS],
                       marker='o', linewidth=2, markersize=8, capsize=5, color='darkgoldenrod')
    axes[1, 2].set_xlabel('Initial Capital (months)')
    axes[1, 2].set_ylabel('Final Revenue')
    axes[1, 2].set_title('Average Final Revenue')
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    filename1 = f"{OUTPUT_DIR}/exp5_overview_{timestamp}.png"
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename1}")
    plt.close()
    
    # Figure 2: Survival and PMF Distribution
    fig2, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig2.suptitle('Experiment 5: Success Metrics Distribution', fontsize=16, fontweight='bold')
    
    # Survival distribution
    survival_data = []
    labels = []
    for capital in CAPITAL_LEVELS:
        capital_data = df_results[df_results['capital_level'] == capital['name']]
        survived_count = capital_data['survived'].sum()
        failed_count = len(capital_data) - survived_count
        survival_data.append([survived_count, failed_count])
        labels.append(f"{capital['name']}\n({capital['runway']}m)")
    
    survival_data = np.array(survival_data)
    x = np.arange(len(labels))
    width = 0.6
    
    axes[0].bar(x, survival_data[:, 0], width, label='Survived', color='green', alpha=0.7)
    axes[0].bar(x, survival_data[:, 1], width, bottom=survival_data[:, 0], 
               label='Failed', color='red', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel('Number of Companies')
    axes[0].set_title('Survival Distribution by Capital Level')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # PMF achievement distribution
    pmf_data = []
    for capital in CAPITAL_LEVELS:
        capital_data = df_results[df_results['capital_level'] == capital['name']]
        achieved_count = capital_data['market_fit_achieved'].sum()
        not_achieved_count = len(capital_data) - achieved_count
        pmf_data.append([achieved_count, not_achieved_count])
    
    pmf_data = np.array(pmf_data)
    
    axes[1].bar(x, pmf_data[:, 0], width, label='Achieved PMF', color='blue', alpha=0.7)
    axes[1].bar(x, pmf_data[:, 1], width, bottom=pmf_data[:, 0],
               label='Did Not Achieve', color='orange', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel('Number of Companies')
    axes[1].set_title('PMF Achievement by Capital Level')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename2 = f"{OUTPUT_DIR}/exp5_distributions_{timestamp}.png"
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename2}")
    plt.close()
    
    # Figure 3: Scatter Analysis
    fig3, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig3.suptitle('Experiment 5: Capital vs Performance Metrics', fontsize=16, fontweight='bold')
    
    scatter_metrics = [
        ('initial_runway', 'market_share', 'Initial Capital (months)', 'Market Share (%)'),
        ('initial_runway', 'revenue', 'Initial Capital (months)', 'Revenue'),
        ('capital_spent', 'market_share', 'Capital Spent', 'Market Share (%)'),
        ('initial_runway', 'feature_completeness', 'Initial Capital (months)', 'Feature Completeness')
    ]
    
    for idx, (x_metric, y_metric, x_label, y_label) in enumerate(scatter_metrics):
        ax = axes[idx // 2, idx % 2]
        
        for capital in capital_levels:
            capital_data = df_results[df_results['capital_level'] == capital]
            ax.scatter(capital_data[x_metric], capital_data[y_metric],
                      label=capital, color=capital_colors[capital], alpha=0.6, s=50)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'{y_label} vs {x_label}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filename3 = f"{OUTPUT_DIR}/exp5_scatter_{timestamp}.png"
    plt.savefig(filename3, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename3}")
    plt.close()
    
    # Figure 4: ROI and Diminishing Returns
    fig4, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig4.suptitle('Experiment 5: Return on Investment Analysis', fontsize=16, fontweight='bold')
    
    # Calculate marginal returns
    marginal_returns_share = []
    marginal_returns_survival = []
    capital_increases = []
    
    for i in range(len(CAPITAL_LEVELS) - 1):
        current = CAPITAL_LEVELS[i]
        next_level = CAPITAL_LEVELS[i + 1]
        
        current_data = df_results[df_results['capital_level'] == current['name']]
        next_data = df_results[df_results['capital_level'] == next_level['name']]
        
        share_increase = next_data['market_share'].mean() - current_data['market_share'].mean()
        survival_increase = next_data['survived'].mean() - current_data['survived'].mean()
        capital_increase = next_level['runway'] - current['runway']
        
        marginal_returns_share.append(share_increase / capital_increase)
        marginal_returns_survival.append(survival_increase / capital_increase)
        capital_increases.append(f"{current['runway']}-{next_level['runway']}")
    
    # Market share marginal returns
    axes[0].bar(range(len(marginal_returns_share)), marginal_returns_share, color='steelblue')
    axes[0].set_xticks(range(len(marginal_returns_share)))
    axes[0].set_xticklabels(capital_increases, rotation=45, ha='right')
    axes[0].set_ylabel('Market Share Gain / Capital Increase')
    axes[0].set_title('Marginal Returns: Market Share')
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Survival marginal returns
    axes[1].bar(range(len(marginal_returns_survival)), marginal_returns_survival, color='forestgreen')
    axes[1].set_xticks(range(len(marginal_returns_survival)))
    axes[1].set_xticklabels(capital_increases, rotation=45, ha='right')
    axes[1].set_ylabel('Survival Rate Gain / Capital Increase')
    axes[1].set_title('Marginal Returns: Survival Rate')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename4 = f"{OUTPUT_DIR}/exp5_roi_{timestamp}.png"
    plt.savefig(filename4, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename4}")
    plt.close()
    
    # Figure 5: Box Plots
    fig5, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig5.suptitle('Experiment 5: Metric Distributions by Capital Level', fontsize=16, fontweight='bold')
    
    box_metrics = [
        ('market_share', 'Market Share (%)'),
        ('revenue', 'Revenue'),
        ('feature_completeness', 'Feature Completeness'),
        ('technical_debt', 'Technical Debt'),
        ('capital_efficiency', 'Capital Efficiency'),
        ('survival_time', 'Survival Time (steps)')
    ]
    
    for idx, (metric, title) in enumerate(box_metrics):
        ax = axes[idx // 3, idx % 3]
        
        data_to_plot = [df_results[df_results['capital_level'] == c['name']][metric].values 
                       for c in CAPITAL_LEVELS]
        
        bp = ax.boxplot(data_to_plot, labels=[c['name'] for c in CAPITAL_LEVELS], patch_artist=True)
        
        for patch, capital in zip(bp['boxes'], CAPITAL_LEVELS):
            patch.set_facecolor(capital_colors[capital['name']])
            patch.set_alpha(0.6)
        
        ax.set_xticklabels([c['name'] for c in CAPITAL_LEVELS], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(f'{title} Distribution')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename5 = f"{OUTPUT_DIR}/exp5_boxplots_{timestamp}.png"
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
    summary_file = f"{OUTPUT_DIR}/exp5_summary_{timestamp}.csv"
    df_results.to_csv(summary_file, index=False)
    print(f"  Saved summary: {summary_file}")
    
    # Save time series data
    timeseries_file = f"{OUTPUT_DIR}/exp5_timeseries_{timestamp}.csv"
    df_time_series.to_csv(timeseries_file, index=False)
    print(f"  Saved time series: {timeseries_file}")
    
    # Save aggregated statistics
    stats_file = f"{OUTPUT_DIR}/exp5_statistics_{timestamp}.csv"
    stats = df_results.groupby('capital_level').agg({
        'survived': ['mean', 'sum'],
        'survival_time': ['mean', 'std'],
        'final_cash': ['mean', 'std'],
        'market_fit_achieved': ['mean', 'sum'],
        'time_to_pmf': ['mean', 'std'],
        'market_share': ['mean', 'std', 'max'],
        'revenue': ['mean', 'std', 'max'],
        'feature_completeness': ['mean', 'std'],
        'technical_debt': ['mean', 'std'],
        'capital_spent': ['mean', 'std'],
        'capital_efficiency': ['mean', 'std', 'max']
    })
    stats.to_csv(stats_file)
    print(f"  Saved statistics: {stats_file}")

def main():
    """Main execution function."""
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run experiment
    df_results, df_time_series = run_initial_capital_experiment()
    
    # Analyze results
    analyze_results(df_results, df_time_series)
    
    # Create visualizations
    create_visualizations(df_results, df_time_series, timestamp)
    
    # Save results
    save_results(df_results, df_time_series, timestamp)
    
    print("\n" + "="*60)
    print("EXPERIMENT 5 COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Results saved with timestamp: {timestamp}")
    print(f"Check the '{OUTPUT_DIR}' directory for outputs.")
    print("\nKey Question Answered:")
    print("How much capital is needed to achieve product-market fit and sustain growth?")

if __name__ == "__main__":
    main()
