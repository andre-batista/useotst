"""
Experiment 6: Burn Rate vs Growth Analysis

Tests different burn rate strategies and their impact on growth and sustainability.
Compares aggressive growth (high burn) vs conservative growth (low burn).
Analyzes the trade-off between growth speed and financial sustainability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import run_simulation, get_default_params, CompanyModel
import os
from datetime import datetime

# Configuration
NUM_RUNS_PER_CONFIG = 20  # Number of simulations per burn rate strategy
STEPS = 200
OUTPUT_DIR = "results"

# Burn rate strategies to test
BURN_RATE_STRATEGIES = [
    {
        'name': 'Ultra Conservative',
        'initial_burn': 5.0,
        'description': 'Minimal spending - bootstrapped approach',
        'team_size': {'eng': 5, 'sales': 3, 'mkt': 2}
    },
    {
        'name': 'Conservative',
        'initial_burn': 8.0,
        'description': 'Low burn rate - sustainable growth',
        'team_size': {'eng': 8, 'sales': 4, 'mkt': 2}
    },
    {
        'name': 'Moderate',
        'initial_burn': 10.0,
        'description': 'Standard burn rate - balanced approach',
        'team_size': {'eng': 10, 'sales': 5, 'mkt': 3}
    },
    {
        'name': 'Aggressive',
        'initial_burn': 15.0,
        'description': 'High burn rate - fast growth focus',
        'team_size': {'eng': 15, 'sales': 8, 'mkt': 5}
    },
    {
        'name': 'Very Aggressive',
        'initial_burn': 20.0,
        'description': 'Very high burn - blitzscaling',
        'team_size': {'eng': 20, 'sales': 10, 'mkt': 7}
    },
    {
        'name': 'Extreme Growth',
        'initial_burn': 30.0,
        'description': 'Extreme burn - all-in growth',
        'team_size': {'eng': 25, 'sales': 15, 'mkt': 10}
    }
]

def ensure_output_dir():
    """Creates output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def run_burn_rate_experiment():
    """Runs burn rate experiment."""
    
    print("="*60)
    print("EXPERIMENT 6: BURN RATE VS GROWTH ANALYSIS")
    print("="*60)
    print(f"Runs per strategy: {NUM_RUNS_PER_CONFIG}")
    print(f"Simulation steps: {STEPS}\n")
    
    all_results = []
    all_time_series = []
    
    for strategy in BURN_RATE_STRATEGIES:
        print(f"\n{'='*60}")
        print(f"Testing: {strategy['name']}")
        print(f"Initial burn rate: ${strategy['initial_burn']}M/month")
        print(f"Team: {strategy['team_size']['eng']} eng, "
              f"{strategy['team_size']['sales']} sales, {strategy['team_size']['mkt']} mkt")
        print(f"Description: {strategy['description']}")
        print(f"{'='*60}")
        
        strategy_results = []
        
        for run in range(NUM_RUNS_PER_CONFIG):
            print(f"  Run {run+1}/{NUM_RUNS_PER_CONFIG}...", end=" ")
            
            # Get default parameters
            params = get_default_params()
            
            # Run simulation with strategy-specific team size
            result = run_simulation(
                num_engineers=strategy['team_size']['eng'],
                num_sales=strategy['team_size']['sales'],
                num_marketing=strategy['team_size']['mkt'],
                num_hr=1,
                num_mgmt=1,
                steps=STEPS,
                params=params,
                verbose=False
            )
            
            # Modify initial burn rate in the model
            # Note: The actual burn rate is calculated as base_burn + (team_size * cost_per_person)
            # We approximate this in our analysis
            
            data = result['data']
            final_state = result['final_state']
            
            # Calculate metrics
            # Growth rate (market share increase per step)
            if len(data) > 10:
                early_share = data['Market Share'].iloc[:10].mean()
                late_share = data['Market Share'].iloc[-10:].mean()
                growth_rate = (late_share - early_share) / len(data) if len(data) > 0 else 0
            else:
                growth_rate = 0
            
            # Runway efficiency (market share gained per month of runway used)
            # For survivors: measure return on initial capital investment
            # For failures: measure how much market share achieved before running out of cash
            initial_runway = 100.0
            
            if not result['game_over']:
                # Survived: calculate efficiency as market share per unit of initial capital
                # This measures ROI: how much market capture per dollar invested
                runway_efficiency = final_state['market_share'] / initial_runway if initial_runway > 0 else 0
            else:
                # Failed: calculate how much runway was consumed
                runway_used = initial_runway - final_state['cash_runway']
                runway_efficiency = final_state['market_share'] / runway_used if runway_used > 0 else 0
            
            # Time to profitability
            time_to_profit = STEPS
            for step in range(len(data)):
                if data.iloc[step]['Revenue'] > strategy['initial_burn']:
                    time_to_profit = step
                    break
            
            # Peak market share
            peak_share = data['Market Share'].max() if len(data) > 0 else 0
            
            # Calculate burn multiple (revenue / burn rate)
            avg_revenue = data['Revenue'].mean() if len(data) > 0 else 0
            burn_multiple = avg_revenue / strategy['initial_burn'] if strategy['initial_burn'] > 0 else 0
            
            # Calculate survival time correctly
            # If survived, time = STEPS; if failed, time = game_over_step
            if result['game_over']:
                survival_time = result.get('game_over_step', len(data))
            else:
                survival_time = STEPS
            
            strategy_results.append({
                'strategy_name': strategy['name'],
                'initial_burn_rate': strategy['initial_burn'],
                'team_size_total': sum(strategy['team_size'].values()),
                'run': run + 1,
                'survived': not result['game_over'],
                'survival_time': survival_time,
                'market_fit': final_state['market_fit'],
                'market_share': final_state['market_share'],
                'peak_market_share': peak_share,
                'growth_rate': growth_rate,
                'feature_completeness': final_state['feature_completeness'],
                'technical_debt': final_state['technical_debt'],
                'bug_count': final_state['bug_count'],
                'code_quality': final_state['code_quality'],
                'brand_awareness': final_state['brand_awareness'],
                'revenue': final_state['revenue'],
                'cash_runway': final_state['cash_runway'],
                'runway_used': runway_used,
                'runway_efficiency': runway_efficiency,
                'funding_rounds': final_state['funding_rounds'],
                'pivots_used': final_state['pivots_used'],
                'time_to_profitability': time_to_profit,
                'burn_multiple': burn_multiple,
            })
            
            # Store time series data
            time_series_data = result['data'].copy()
            time_series_data['strategy_name'] = strategy['name']
            time_series_data['initial_burn_rate'] = strategy['initial_burn']
            time_series_data['run'] = run + 1
            all_time_series.append(time_series_data)
            
            print(f"✓ (Survived: {not result['game_over']}, "
                  f"Share: {final_state['market_share']:.1f}%, "
                  f"Runway: {final_state['cash_runway']:.0f}m)")
        
        all_results.extend(strategy_results)
        
        # Print summary for this strategy
        df_strategy = pd.DataFrame(strategy_results)
        print(f"\n  Summary for {strategy['name']}:")
        print(f"    Survival rate: {df_strategy['survived'].mean():.1%}")
        print(f"    Avg market share: {df_strategy['market_share'].mean():.1f}%")
        print(f"    Avg growth rate: {df_strategy['growth_rate'].mean():.4f}%/step")
        print(f"    Avg runway efficiency: {df_strategy['runway_efficiency'].mean():.4f}")
        print(f"    Avg funding rounds: {df_strategy['funding_rounds'].mean():.1f}")
    
    return pd.DataFrame(all_results), pd.concat(all_time_series, ignore_index=True)

def analyze_results(df_results, df_time_series):
    """Analyzes and visualizes experiment results."""
    
    print("\n" + "="*60)
    print("ANALYSIS: BURN RATE VS GROWTH")
    print("="*60)
    
    # Group by strategy
    grouped = df_results.groupby('strategy_name')
    
    # 1. Survival and Growth Analysis
    print("\n1. SURVIVAL AND GROWTH")
    print("-" * 60)
    survival_stats = grouped.agg({
        'survived': ['mean', 'sum'],
        'survival_time': ['mean', 'std'],
        'growth_rate': ['mean', 'std'],
        'peak_market_share': ['mean', 'max']
    })
    print(survival_stats)
    
    # 2. Market Performance
    print("\n2. MARKET PERFORMANCE")
    print("-" * 60)
    market_stats = grouped.agg({
        'market_share': ['mean', 'std', 'max'],
        'market_fit': ['mean', 'std'],
        'brand_awareness': ['mean', 'std'],
        'revenue': ['mean', 'std', 'max']
    })
    print(market_stats)
    
    # 3. Financial Efficiency
    print("\n3. FINANCIAL EFFICIENCY")
    print("-" * 60)
    efficiency_stats = grouped.agg({
        'runway_used': ['mean', 'std'],
        'runway_efficiency': ['mean', 'std'],
        'cash_runway': ['mean', 'std'],
        'burn_multiple': ['mean', 'std'],
        'time_to_profitability': ['mean', 'std']
    })
    print(efficiency_stats)
    
    # 4. Funding Requirements
    print("\n4. FUNDING REQUIREMENTS")
    print("-" * 60)
    funding_stats = grouped.agg({
        'funding_rounds': ['mean', 'std', 'max'],
        'pivots_used': ['mean', 'sum']
    })
    print(funding_stats)
    
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
    
    # Best survival rate
    best_survival = df_results.groupby('strategy_name')['survived'].mean().idxmax()
    best_survival_rate = df_results.groupby('strategy_name')['survived'].mean().max()
    print(f"Best for survival: {best_survival} ({best_survival_rate:.1%})")
    
    # Best growth rate
    best_growth = df_results.groupby('strategy_name')['growth_rate'].mean().idxmax()
    best_growth_rate = df_results.groupby('strategy_name')['growth_rate'].mean().max()
    print(f"Best growth rate: {best_growth} ({best_growth_rate:.4f}%/step)")
    
    # Best runway efficiency
    best_efficiency = df_results.groupby('strategy_name')['runway_efficiency'].mean().idxmax()
    best_efficiency_val = df_results.groupby('strategy_name')['runway_efficiency'].mean().max()
    print(f"Most runway efficient: {best_efficiency} ({best_efficiency_val:.4f})")
    
    # Best market share
    best_share = df_results.groupby('strategy_name')['market_share'].mean().idxmax()
    best_share_val = df_results.groupby('strategy_name')['market_share'].mean().max()
    print(f"Best market share: {best_share} ({best_share_val:.1f}%)")
    
    # Fewest funding rounds needed
    least_funding = df_results.groupby('strategy_name')['funding_rounds'].mean().idxmin()
    least_funding_val = df_results.groupby('strategy_name')['funding_rounds'].mean().min()
    print(f"Least funding needed: {least_funding} ({least_funding_val:.1f} rounds)")

def create_visualizations(df_results, df_time_series, timestamp):
    """Creates comprehensive visualizations."""
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    ensure_output_dir()
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    strategies = df_results['strategy_name'].unique()
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(strategies)))
    strategy_colors = dict(zip(strategies, colors))
    
    # Figure 1: Core Trade-offs
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle('Experiment 6: Burn Rate vs Growth - Core Trade-offs', fontsize=16, fontweight='bold')
    
    # Survival Rate vs Burn Rate
    burn_rates = df_results.groupby('strategy_name')['initial_burn_rate'].first()
    survival_rates = df_results.groupby('strategy_name')['survived'].mean()
    axes[0, 0].plot(burn_rates, survival_rates, marker='o', linestyle='', markersize=8, color='darkblue')
    axes[0, 0].set_xlabel('Initial Burn Rate ($/month)')
    axes[0, 0].set_ylabel('Survival Rate')
    axes[0, 0].set_title('Survival Rate vs Burn Rate')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_ylim([0, 1.25])
    
    # Growth Rate vs Burn Rate
    growth_rates = df_results.groupby('strategy_name')['growth_rate'].mean()
    axes[0, 1].plot(burn_rates, growth_rates, marker='s', linestyle='', markersize=8, color='darkgreen')
    axes[0, 1].set_xlabel('Initial Burn Rate ($/month)')
    axes[0, 1].set_ylabel('Growth Rate (%/step)')
    axes[0, 1].set_title('Growth Rate vs Burn Rate')
    axes[0, 1].grid(alpha=0.3)
    
    # Market Share vs Burn Rate
    market_shares = df_results.groupby('strategy_name')['market_share'].mean()
    market_shares_std = df_results.groupby('strategy_name')['market_share'].std()
    axes[0, 2].errorbar(burn_rates, market_shares, yerr=market_shares_std,
                       marker='D', linestyle='', markersize=8, capsize=5, color='darkred')
    axes[0, 2].set_xlabel('Initial Burn Rate ($/month)')
    axes[0, 2].set_ylabel('Market Share (%)')
    axes[0, 2].set_title('Market Share vs Burn Rate')
    axes[0, 2].grid(alpha=0.3)
    
    # Runway Efficiency
    efficiency = df_results.groupby('strategy_name')['runway_efficiency'].mean()
    axes[1, 0].bar(range(len(efficiency)), efficiency.values,
                   color=[strategy_colors[s] for s in efficiency.index])
    axes[1, 0].set_xticks(range(len(efficiency)))
    axes[1, 0].set_xticklabels(efficiency.index, rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_ylabel('Market Share / Runway Used')
    axes[1, 0].set_title('Runway Efficiency')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Funding Rounds Required
    funding = df_results.groupby('strategy_name')['funding_rounds'].mean()
    axes[1, 1].bar(range(len(funding)), funding.values,
                   color=[strategy_colors[s] for s in funding.index])
    axes[1, 1].set_xticks(range(len(funding)))
    axes[1, 1].set_xticklabels(funding.index, rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_ylabel('Average Funding Rounds')
    axes[1, 1].set_title('Funding Requirements')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Burn Multiple
    burn_mult = df_results.groupby('strategy_name')['burn_multiple'].mean()
    axes[1, 2].bar(range(len(burn_mult)), burn_mult.values,
                   color=[strategy_colors[s] for s in burn_mult.index])
    axes[1, 2].set_xticks(range(len(burn_mult)))
    axes[1, 2].set_xticklabels(burn_mult.index, rotation=45, ha='right', fontsize=8)
    axes[1, 2].set_ylabel('Revenue / Burn Rate')
    axes[1, 2].set_title('Burn Multiple')
    axes[1, 2].grid(axis='y', alpha=0.3)
    axes[1, 2].axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')
    axes[1, 2].legend()
    
    plt.tight_layout()
    filename1 = f"{OUTPUT_DIR}/exp6_tradeoffs_{timestamp}.png"
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename1}")
    plt.close()
    
    # Figure 2: Time Series Evolution
    fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig2.suptitle('Experiment 6: Evolution Over Time by Burn Rate', fontsize=16, fontweight='bold')
    
    time_metrics = [
        ('Market Share', 'Market Share (%)'),
        ('Revenue', 'Revenue'),
        ('Cash Runway', 'Cash Runway (months)'),
        ('Market Fit', 'Market Fit')
    ]
    
    for idx, (metric, ylabel) in enumerate(time_metrics):
        ax = axes[idx // 2, idx % 2]
        
        for strategy in strategies:
            strategy_data = df_time_series[df_time_series['strategy_name'] == strategy]
            grouped = strategy_data.groupby(strategy_data.index % STEPS)[metric]
            mean_vals = grouped.mean()
            std_vals = grouped.std()
            
            ax.plot(mean_vals.index, mean_vals.values,
                   label=strategy, color=strategy_colors[strategy], linewidth=2)
            ax.fill_between(mean_vals.index,
                           mean_vals.values - std_vals.values,
                           mean_vals.values + std_vals.values,
                           alpha=0.2, color=strategy_colors[strategy])
        
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filename2 = f"{OUTPUT_DIR}/exp6_evolution_{timestamp}.png"
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename2}")
    plt.close()
    
    # Figure 3: Scatter Analysis
    fig3, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig3.suptitle('Experiment 6: Burn Rate Relationships', fontsize=16, fontweight='bold')
    
    scatter_pairs = [
        ('initial_burn_rate', 'market_share', 'Burn Rate ($/month)', 'Market Share (%)'),
        ('initial_burn_rate', 'survival_time', 'Burn Rate ($/month)', 'Survival Time (steps)'),
        ('runway_used', 'market_share', 'Runway Used (months)', 'Market Share (%)'),
        ('growth_rate', 'runway_efficiency', 'Growth Rate (%/step)', 'Runway Efficiency')
    ]
    
    for idx, (x_metric, y_metric, x_label, y_label) in enumerate(scatter_pairs):
        ax = axes[idx // 2, idx % 2]
        
        for strategy in strategies:
            strategy_data = df_results[df_results['strategy_name'] == strategy]
            ax.scatter(strategy_data[x_metric], strategy_data[y_metric],
                      label=strategy, color=strategy_colors[strategy], alpha=0.6, s=50)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'{y_label} vs {x_label}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filename3 = f"{OUTPUT_DIR}/exp6_scatter_{timestamp}.png"
    plt.savefig(filename3, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename3}")
    plt.close()
    
    # Figure 4: Efficiency Analysis
    fig4, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig4.suptitle('Experiment 6: Growth Efficiency Analysis', fontsize=16, fontweight='bold')
    
    # Growth vs Survival trade-off
    for strategy in strategies:
        strategy_data = df_results[df_results['strategy_name'] == strategy]
        avg_growth = strategy_data['growth_rate'].mean()
        avg_survival = strategy_data['survived'].mean()
        axes[0].scatter(avg_growth, avg_survival, 
                       label=strategy, color=strategy_colors[strategy], 
                       s=200, alpha=0.7, edgecolors='black', linewidth=2)
    
    axes[0].set_xlabel('Average Growth Rate (%/step)')
    axes[0].set_ylabel('Survival Rate')
    axes[0].set_title('Growth vs Survival Trade-off')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(left=0)
    axes[0].set_ylim([0, 1.25])
    
    # Efficiency frontier
    for strategy in strategies:
        strategy_data = df_results[df_results['strategy_name'] == strategy]
        avg_efficiency = strategy_data['runway_efficiency'].mean()
        avg_share = strategy_data['market_share'].mean()
        axes[1].scatter(avg_efficiency, avg_share,
                       label=strategy, color=strategy_colors[strategy],
                       s=200, alpha=0.7, edgecolors='black', linewidth=2)
    
    axes[1].set_xlabel('Runway Efficiency (Share/Runway)')
    axes[1].set_ylabel('Average Market Share (%)')
    axes[1].set_title('Efficiency Frontier')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    filename4 = f"{OUTPUT_DIR}/exp6_efficiency_{timestamp}.png"
    plt.savefig(filename4, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename4}")
    plt.close()
    
    # Figure 5: Box Plots
    fig5, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig5.suptitle('Experiment 6: Metric Distributions by Burn Rate', fontsize=16, fontweight='bold')
    
    box_metrics = [
        ('market_share', 'Market Share (%)'),
        ('growth_rate', 'Growth Rate (%/step)'),
        ('runway_efficiency', 'Runway Efficiency'),
        ('survival_time', 'Survival Time (steps)'),
        ('funding_rounds', 'Funding Rounds'),
        ('revenue', 'Revenue')
    ]
    
    for idx, (metric, title) in enumerate(box_metrics):
        ax = axes[idx // 3, idx % 3]
        
        data_to_plot = [df_results[df_results['strategy_name'] == s][metric].values 
                       for s in strategies]
        
        bp = ax.boxplot(data_to_plot, labels=strategies, patch_artist=True)
        
        for patch, strategy in zip(bp['boxes'], strategies):
            patch.set_facecolor(strategy_colors[strategy])
            patch.set_alpha(0.6)
        
        ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(f'{title} Distribution')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename5 = f"{OUTPUT_DIR}/exp6_distributions_{timestamp}.png"
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
    summary_file = f"{OUTPUT_DIR}/exp6_summary_{timestamp}.csv"
    df_results.to_csv(summary_file, index=False)
    print(f"  Saved summary: {summary_file}")
    
    # Save time series data
    timeseries_file = f"{OUTPUT_DIR}/exp6_timeseries_{timestamp}.csv"
    df_time_series.to_csv(timeseries_file, index=False)
    print(f"  Saved time series: {timeseries_file}")
    
    # Save aggregated statistics
    stats_file = f"{OUTPUT_DIR}/exp6_statistics_{timestamp}.csv"
    stats = df_results.groupby('strategy_name').agg({
        'survived': ['mean', 'sum'],
        'survival_time': ['mean', 'std'],
        'market_share': ['mean', 'std', 'max'],
        'growth_rate': ['mean', 'std'],
        'runway_efficiency': ['mean', 'std'],
        'revenue': ['mean', 'std', 'max'],
        'cash_runway': ['mean', 'std'],
        'funding_rounds': ['mean', 'std', 'max'],
        'burn_multiple': ['mean', 'std'],
        'feature_completeness': ['mean', 'std'],
        'technical_debt': ['mean', 'std']
    })
    stats.to_csv(stats_file)
    print(f"  Saved statistics: {stats_file}")

def main():
    """Main execution function."""
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run experiment
    df_results, df_time_series = run_burn_rate_experiment()
    
    # Analyze results
    analyze_results(df_results, df_time_series)
    
    # Create visualizations
    create_visualizations(df_results, df_time_series, timestamp)
    
    # Save results
    save_results(df_results, df_time_series, timestamp)
    
    print("\n" + "="*60)
    print("EXPERIMENT 6 COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Results saved with timestamp: {timestamp}")
    print(f"Check the '{OUTPUT_DIR}' directory for outputs.")
    print("\nKey Question Answered:")
    print("What is the optimal balance between growth speed and financial sustainability?")

if __name__ == "__main__":
    main()
