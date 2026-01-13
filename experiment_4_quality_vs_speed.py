"""
Experiment 4: Quality vs Speed Analysis

Compares different development strategies and their long-term impact.
Tests "Move Fast and Break Things" vs "Quality First" approaches.
Analyzes trade-offs between feature velocity, technical debt, and sustainability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import run_simulation, get_default_params
import os
from datetime import datetime

# Configuration
NUM_RUNS_PER_CONFIG = 20  # Number of simulations per strategy
STEPS = 200
OUTPUT_DIR = "results"

# Development strategies to test
STRATEGIES = [
    {
        'name': 'Move Fast Break Things',
        'description': 'Prioritize speed over quality - rapid feature development',
        'params': {
            'dev_to_features': 0.020,        # Develop very fast
            'dev_to_debt': 0.10,             # Generate substantial technical debt (2x default)
            'dev_to_bugs': 0.06,             # Generate many bugs (2x default)
            'quality_reduces_bugs': 0.015,   # Weak QA (0.75x default)
            'quality_reduces_debt': 0.010,   # Weak refactoring (0.67x default)
            'quality_improves_code': 0.003,  # Slow code quality improvement (0.6x default)
            'refactor_reduces_debt': 0.025,  # Weak refactoring impact (0.625x default)
            'refactor_improves_quality': 0.012,  # Weak quality improvement (0.6x default)
        }
    },
    {
        'name': 'Balanced Pragmatic',
        'description': 'Balance between speed and quality - pragmatic approach',
        'params': {
            'dev_to_features': 0.012,        # Moderate development speed
            'dev_to_debt': 0.05,             # Moderate technical debt (default)
            'dev_to_bugs': 0.03,             # Moderate bugs (default)
            'quality_reduces_bugs': 0.02,    # Standard QA (default)
            'quality_reduces_debt': 0.015,   # Standard refactoring (default)
            'quality_improves_code': 0.005,  # Standard quality improvement (default)
            'refactor_reduces_debt': 0.04,   # Standard refactoring impact (default)
            'refactor_improves_quality': 0.02,  # Standard quality improvement (default)
        }
    },
    {
        'name': 'Quality First',
        'description': 'Prioritize quality over speed - sustainable development',
        'params': {
            'dev_to_features': 0.008,        # Develop slower but cleaner
            'dev_to_debt': 0.020,            # Low technical debt (0.4x default)
            'dev_to_bugs': 0.012,            # Few bugs (0.4x default)
            'quality_reduces_bugs': 0.035,   # Strong QA (1.75x default)
            'quality_reduces_debt': 0.025,   # Strong refactoring (1.67x default)
            'quality_improves_code': 0.008,  # Fast code quality improvement (1.6x default)
            'refactor_reduces_debt': 0.060,  # Strong refactoring impact (1.5x default)
            'refactor_improves_quality': 0.030,  # Strong quality improvement (1.5x default)
        }
    },
    {
        'name': 'Extreme Quality',
        'description': 'Extreme focus on quality - very slow but pristine code',
        'params': {
            'dev_to_features': 0.005,        # Very slow development
            'dev_to_debt': 0.010,            # Minimal technical debt (0.2x default)
            'dev_to_bugs': 0.006,            # Very few bugs (0.2x default)
            'quality_reduces_bugs': 0.050,   # Very strong QA (2.5x default)
            'quality_reduces_debt': 0.040,   # Very strong refactoring (2.67x default)
            'quality_improves_code': 0.012,  # Very fast quality improvement (2.4x default)
            'refactor_reduces_debt': 0.080,  # Very strong refactoring impact (2x default)
            'refactor_improves_quality': 0.040,  # Very strong quality improvement (2x default)
        }
    },
    {
        'name': 'Technical Debt Spiral',
        'description': 'Unsustainable speed - generates excessive debt',
        'params': {
            'dev_to_features': 0.025,        # Extremely fast
            'dev_to_debt': 0.15,             # Excessive technical debt (3x default)
            'dev_to_bugs': 0.10,             # Many bugs (3.3x default)
            'quality_reduces_bugs': 0.010,   # Very weak QA (0.5x default)
            'quality_reduces_debt': 0.008,   # Very weak refactoring (0.53x default)
            'quality_improves_code': 0.002,  # Very slow quality improvement (0.4x default)
            'refactor_reduces_debt': 0.020,  # Very weak refactoring impact (0.5x default)
            'refactor_improves_quality': 0.010,  # Very weak quality improvement (0.5x default)
        }
    }
]

def ensure_output_dir():
    """Creates output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def run_quality_vs_speed_experiment():
    """Runs quality vs speed experiment."""
    
    print("="*60)
    print("EXPERIMENT 4: QUALITY VS SPEED ANALYSIS")
    print("="*60)
    print(f"Runs per strategy: {NUM_RUNS_PER_CONFIG}")
    print(f"Simulation steps: {STEPS}\n")
    
    all_results = []
    all_time_series = []
    
    for strategy in STRATEGIES:
        print(f"\n{'='*60}")
        print(f"Testing: {strategy['name']}")
        print(f"Description: {strategy['description']}")
        print(f"Key parameters:")
        for param, value in strategy['params'].items():
            print(f"  {param}: {value}")
        print(f"{'='*60}")
        
        strategy_results = []
        
        # Get default parameters and update with strategy-specific ones
        for run in range(NUM_RUNS_PER_CONFIG):
            print(f"  Run {run+1}/{NUM_RUNS_PER_CONFIG}...", end=" ")
            
            # Start with default parameters
            params = get_default_params()
            # Override with strategy-specific parameters
            params.update(strategy['params'])
            
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
            
            # Calculate derived metrics
            final_state = result['final_state']
            data = result['data']
            
            # Calculate sustainability index
            sustainability = (
                (100 - final_state['technical_debt']) * 0.3 +
                (100 - final_state['bug_count']) * 0.3 +
                final_state['code_quality'] * 0.4
            ) / 100
            
            # Calculate velocity (features per step)
            velocity = final_state['feature_completeness'] / STEPS if not result['game_over'] else 0
            
            # Calculate debt accumulation rate
            if len(data) > 10:
                debt_accumulation = (data['Technical Debt'].iloc[-1] - data['Technical Debt'].iloc[10]) / (len(data) - 10)
            else:
                debt_accumulation = 0
            
            strategy_results.append({
                'strategy_name': strategy['name'],
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
                'sustainability_index': sustainability,
                'velocity': velocity,
                'debt_accumulation_rate': debt_accumulation,
            })
            
            # Store time series data
            time_series_data = result['data'].copy()
            time_series_data['strategy_name'] = strategy['name']
            time_series_data['run'] = run + 1
            all_time_series.append(time_series_data)
            
            print(f"OK (Survived: {not result['game_over']}, "
                  f"Features: {final_state['feature_completeness']:.1f}, "
                  f"Debt: {final_state['technical_debt']:.1f})")
        
        all_results.extend(strategy_results)
        
        # Print summary for this strategy
        df_strategy = pd.DataFrame(strategy_results)
        print(f"\n  Summary for {strategy['name']}:")
        print(f"    Survival rate: {df_strategy['survived'].mean():.1%}")
        print(f"    Avg feature completeness: {df_strategy['feature_completeness'].mean():.1f}")
        print(f"    Avg technical debt: {df_strategy['technical_debt'].mean():.1f}")
        print(f"    Avg bug count: {df_strategy['bug_count'].mean():.1f}")
        print(f"    Avg code quality: {df_strategy['code_quality'].mean():.1f}")
        print(f"    Avg sustainability index: {df_strategy['sustainability_index'].mean():.2f}")
        print(f"    Avg velocity: {df_strategy['velocity'].mean():.4f} features/step")
    
    return pd.DataFrame(all_results), pd.concat(all_time_series, ignore_index=True)

def analyze_results(df_results, df_time_series):
    """Analyzes and visualizes experiment results."""
    
    print("\n" + "="*60)
    print("ANALYSIS: QUALITY VS SPEED COMPARISON")
    print("="*60)
    
    # Group by strategy
    grouped = df_results.groupby('strategy_name')
    
    # 1. Survival and Performance Analysis
    print("\n1. SURVIVAL AND PERFORMANCE")
    print("-" * 60)
    survival_stats = grouped.agg({
        'survived': ['mean', 'sum'],
        'survival_time': ['mean', 'std'],
        'market_share': ['mean', 'std', 'max']
    })
    print(survival_stats)
    
    # 2. Feature Development Analysis
    print("\n2. FEATURE DEVELOPMENT")
    print("-" * 60)
    feature_stats = grouped.agg({
        'feature_completeness': ['mean', 'std', 'max'],
        'velocity': ['mean', 'std'],
        'market_fit': ['mean', 'std']
    })
    print(feature_stats)
    
    # 3. Quality Metrics Analysis
    print("\n3. QUALITY METRICS")
    print("-" * 60)
    quality_stats = grouped.agg({
        'technical_debt': ['mean', 'std', 'min', 'max'],
        'bug_count': ['mean', 'std', 'min', 'max'],
        'code_quality': ['mean', 'std', 'min', 'max'],
        'sustainability_index': ['mean', 'std']
    })
    print(quality_stats)
    
    # 4. Technical Debt Analysis
    print("\n4. TECHNICAL DEBT ACCUMULATION")
    print("-" * 60)
    debt_stats = grouped.agg({
        'debt_accumulation_rate': ['mean', 'std']
    })
    print(debt_stats)
    
    # 5. Financial Performance
    print("\n5. FINANCIAL PERFORMANCE")
    print("-" * 60)
    financial_stats = grouped.agg({
        'revenue': ['mean', 'std', 'max'],
        'cash_runway': ['mean', 'std'],
        'funding_rounds': ['mean', 'max']
    })
    print(financial_stats)
    
    # 6. Key Insights
    print("\n6. KEY INSIGHTS")
    print("-" * 60)
    
    # Best strategy for survival
    best_survival = df_results.groupby('strategy_name')['survived'].mean().idxmax()
    best_survival_rate = df_results.groupby('strategy_name')['survived'].mean().max()
    print(f"Best for survival: {best_survival} ({best_survival_rate:.1%})")
    
    # Best strategy for features
    best_features = df_results.groupby('strategy_name')['feature_completeness'].mean().idxmax()
    best_features_val = df_results.groupby('strategy_name')['feature_completeness'].mean().max()
    print(f"Best for features: {best_features} ({best_features_val:.1f})")
    
    # Best strategy for quality
    best_quality = df_results.groupby('strategy_name')['code_quality'].mean().idxmax()
    best_quality_val = df_results.groupby('strategy_name')['code_quality'].mean().max()
    print(f"Best for code quality: {best_quality} ({best_quality_val:.1f})")
    
    # Lowest technical debt
    lowest_debt = df_results.groupby('strategy_name')['technical_debt'].mean().idxmin()
    lowest_debt_val = df_results.groupby('strategy_name')['technical_debt'].mean().min()
    print(f"Lowest technical debt: {lowest_debt} ({lowest_debt_val:.1f})")
    
    # Best sustainability
    best_sustain = df_results.groupby('strategy_name')['sustainability_index'].mean().idxmax()
    best_sustain_val = df_results.groupby('strategy_name')['sustainability_index'].mean().max()
    print(f"Best sustainability: {best_sustain} ({best_sustain_val:.2f})")
    
    # Best market share
    best_market = df_results.groupby('strategy_name')['market_share'].mean().idxmax()
    best_market_val = df_results.groupby('strategy_name')['market_share'].mean().max()
    print(f"Best market share: {best_market} ({best_market_val:.1f}%)")

def create_visualizations(df_results, df_time_series, timestamp):
    """Creates comprehensive visualizations."""
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    ensure_output_dir()
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    strategies = df_results['strategy_name'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))
    strategy_colors = dict(zip(strategies, colors))
    
    # Figure 1: Core Trade-offs Overview
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle('Experiment 4: Quality vs Speed - Core Trade-offs', fontsize=16, fontweight='bold')
    
    # Feature Completeness
    feature_avg = df_results.groupby('strategy_name')['feature_completeness'].mean()
    feature_std = df_results.groupby('strategy_name')['feature_completeness'].std()
    axes[0, 0].bar(range(len(feature_avg)), feature_avg.values,
                   yerr=feature_std.values,
                   color=[strategy_colors[s] for s in feature_avg.index],
                   capsize=5)
    axes[0, 0].set_xticks(range(len(feature_avg)))
    axes[0, 0].set_xticklabels(feature_avg.index, rotation=45, ha='right', fontsize=8)
    axes[0, 0].set_ylabel('Feature Completeness')
    axes[0, 0].set_title('Feature Development')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Technical Debt
    debt_avg = df_results.groupby('strategy_name')['technical_debt'].mean()
    debt_std = df_results.groupby('strategy_name')['technical_debt'].std()
    axes[0, 1].bar(range(len(debt_avg)), debt_avg.values,
                   yerr=debt_std.values,
                   color=[strategy_colors[s] for s in debt_avg.index],
                   capsize=5)
    axes[0, 1].set_xticks(range(len(debt_avg)))
    axes[0, 1].set_xticklabels(debt_avg.index, rotation=45, ha='right', fontsize=8)
    axes[0, 1].set_ylabel('Technical Debt')
    axes[0, 1].set_title('Technical Debt Accumulation')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Bug Count
    bug_avg = df_results.groupby('strategy_name')['bug_count'].mean()
    bug_std = df_results.groupby('strategy_name')['bug_count'].std()
    axes[0, 2].bar(range(len(bug_avg)), bug_avg.values,
                   yerr=bug_std.values,
                   color=[strategy_colors[s] for s in bug_avg.index],
                   capsize=5)
    axes[0, 2].set_xticks(range(len(bug_avg)))
    axes[0, 2].set_xticklabels(bug_avg.index, rotation=45, ha='right', fontsize=8)
    axes[0, 2].set_ylabel('Bug Count')
    axes[0, 2].set_title('Bug Accumulation')
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # Code Quality
    quality_avg = df_results.groupby('strategy_name')['code_quality'].mean()
    quality_std = df_results.groupby('strategy_name')['code_quality'].std()
    axes[1, 0].bar(range(len(quality_avg)), quality_avg.values,
                   yerr=quality_std.values,
                   color=[strategy_colors[s] for s in quality_avg.index],
                   capsize=5)
    axes[1, 0].set_xticks(range(len(quality_avg)))
    axes[1, 0].set_xticklabels(quality_avg.index, rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_ylabel('Code Quality')
    axes[1, 0].set_title('Code Quality Score')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Sustainability Index
    sustain_avg = df_results.groupby('strategy_name')['sustainability_index'].mean()
    sustain_std = df_results.groupby('strategy_name')['sustainability_index'].std()
    axes[1, 1].bar(range(len(sustain_avg)), sustain_avg.values,
                   yerr=sustain_std.values,
                   color=[strategy_colors[s] for s in sustain_avg.index],
                   capsize=5)
    axes[1, 1].set_xticks(range(len(sustain_avg)))
    axes[1, 1].set_xticklabels(sustain_avg.index, rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_ylabel('Sustainability Index')
    axes[1, 1].set_title('Overall Sustainability')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Survival Rate
    survival_rate = df_results.groupby('strategy_name')['survived'].mean()
    axes[1, 2].bar(range(len(survival_rate)), survival_rate.values,
                   color=[strategy_colors[s] for s in survival_rate.index])
    axes[1, 2].set_xticks(range(len(survival_rate)))
    axes[1, 2].set_xticklabels(survival_rate.index, rotation=45, ha='right', fontsize=8)
    axes[1, 2].set_ylabel('Survival Rate')
    axes[1, 2].set_title('Company Survival Rate')
    axes[1, 2].set_ylim([0, 1])
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename1 = f"{OUTPUT_DIR}/exp4_core_tradeoffs_{timestamp}.png"
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename1}")
    plt.close()
    
    # Figure 2: Time Series Evolution - Quality Metrics
    fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig2.suptitle('Experiment 4: Quality Metrics Evolution Over Time', fontsize=16, fontweight='bold')
    
    quality_metrics = [
        ('Feature Completeness', 'Features'),
        ('Technical Debt', 'Technical Debt'),
        ('Bug Count', 'Bugs'),
        ('Code Quality', 'Quality Score')
    ]
    
    for idx, (metric, ylabel) in enumerate(quality_metrics):
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
    filename2 = f"{OUTPUT_DIR}/exp4_quality_evolution_{timestamp}.png"
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename2}")
    plt.close()
    
    # Figure 3: Time Series - Business Metrics
    fig3, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig3.suptitle('Experiment 4: Business Metrics Evolution Over Time', fontsize=16, fontweight='bold')
    
    business_metrics = [
        ('Market Fit', 'Market Fit'),
        ('Market Share', 'Market Share (%)'),
        ('Revenue', 'Revenue'),
        ('Cash Runway', 'Cash Runway (months)')
    ]
    
    for idx, (metric, ylabel) in enumerate(business_metrics):
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
    filename3 = f"{OUTPUT_DIR}/exp4_business_evolution_{timestamp}.png"
    plt.savefig(filename3, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename3}")
    plt.close()
    
    # Figure 4: Scatter Plots - Key Relationships
    fig4, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig4.suptitle('Experiment 4: Strategy Trade-offs Analysis', fontsize=16, fontweight='bold')
    
    scatter_pairs = [
        ('feature_completeness', 'technical_debt', 'Features', 'Technical Debt'),
        ('velocity', 'sustainability_index', 'Velocity (features/step)', 'Sustainability'),
        ('technical_debt', 'market_share', 'Technical Debt', 'Market Share (%)'),
        ('bug_count', 'revenue', 'Bug Count', 'Revenue')
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
    filename4 = f"{OUTPUT_DIR}/exp4_tradeoffs_{timestamp}.png"
    plt.savefig(filename4, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename4}")
    plt.close()
    
    # Figure 5: Box Plots for Distribution Analysis
    fig5, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig5.suptitle('Experiment 4: Distribution of Key Metrics', fontsize=16, fontweight='bold')
    
    box_metrics = [
        ('feature_completeness', 'Feature Completeness'),
        ('technical_debt', 'Technical Debt'),
        ('bug_count', 'Bug Count'),
        ('market_share', 'Market Share (%)'),
        ('revenue', 'Revenue'),
        ('sustainability_index', 'Sustainability Index')
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
    filename5 = f"{OUTPUT_DIR}/exp4_distributions_{timestamp}.png"
    plt.savefig(filename5, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename5}")
    plt.close()
    
    # Figure 6: Velocity vs Quality Trade-off
    fig6, ax = plt.subplots(figsize=(12, 8))
    fig6.suptitle('Experiment 4: The Velocity-Quality Trade-off', fontsize=16, fontweight='bold')
    
    for strategy in strategies:
        strategy_data = df_results[df_results['strategy_name'] == strategy]
        ax.scatter(strategy_data['velocity'], strategy_data['code_quality'],
                  label=strategy, color=strategy_colors[strategy], 
                  alpha=0.6, s=100, edgecolors='black', linewidth=1)
    
    # Add strategy means as larger points
    for strategy in strategies:
        strategy_data = df_results[df_results['strategy_name'] == strategy]
        mean_velocity = strategy_data['velocity'].mean()
        mean_quality = strategy_data['code_quality'].mean()
        ax.scatter(mean_velocity, mean_quality, 
                  color=strategy_colors[strategy], s=300, marker='*',
                  edgecolors='black', linewidth=2, zorder=10)
    
    ax.set_xlabel('Development Velocity (features per step)', fontsize=12)
    ax.set_ylabel('Code Quality Score', fontsize=12)
    ax.set_title('Trade-off between Speed and Quality\n(Stars indicate strategy means)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filename6 = f"{OUTPUT_DIR}/exp4_velocity_quality_{timestamp}.png"
    plt.savefig(filename6, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename6}")
    plt.close()

def save_results(df_results, df_time_series, timestamp):
    """Saves results to CSV files."""
    
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    ensure_output_dir()
    
    # Save summary results
    summary_file = f"{OUTPUT_DIR}/exp4_summary_{timestamp}.csv"
    df_results.to_csv(summary_file, index=False)
    print(f"  Saved summary: {summary_file}")
    
    # Save time series data
    timeseries_file = f"{OUTPUT_DIR}/exp4_timeseries_{timestamp}.csv"
    df_time_series.to_csv(timeseries_file, index=False)
    print(f"  Saved time series: {timeseries_file}")
    
    # Save aggregated statistics
    stats_file = f"{OUTPUT_DIR}/exp4_statistics_{timestamp}.csv"
    stats = df_results.groupby('strategy_name').agg({
        'survived': ['mean', 'sum'],
        'survival_time': ['mean', 'std'],
        'feature_completeness': ['mean', 'std', 'max'],
        'technical_debt': ['mean', 'std', 'min', 'max'],
        'bug_count': ['mean', 'std', 'min', 'max'],
        'code_quality': ['mean', 'std', 'min', 'max'],
        'market_fit': ['mean', 'std'],
        'market_share': ['mean', 'std', 'max'],
        'revenue': ['mean', 'std', 'max'],
        'cash_runway': ['mean', 'std'],
        'sustainability_index': ['mean', 'std'],
        'velocity': ['mean', 'std'],
        'debt_accumulation_rate': ['mean', 'std']
    })
    stats.to_csv(stats_file)
    print(f"  Saved statistics: {stats_file}")

def main():
    """Main execution function."""
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run experiment
    df_results, df_time_series = run_quality_vs_speed_experiment()
    
    # Analyze results
    analyze_results(df_results, df_time_series)
    
    # Create visualizations
    create_visualizations(df_results, df_time_series, timestamp)
    
    # Save results
    save_results(df_results, df_time_series, timestamp)
    
    print("\n" + "="*60)
    print("EXPERIMENT 4 COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Results saved with timestamp: {timestamp}")
    print(f"Check the '{OUTPUT_DIR}' directory for outputs.")
    print("\nKey Question Answered:")
    print("Which strategy provides the best balance between speed and sustainability?")

if __name__ == "__main__":
    main()
