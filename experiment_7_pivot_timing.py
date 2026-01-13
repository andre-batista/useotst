"""
Experiment 7: Pivot Timing Analysis
===================================

Tests different pivot timing strategies to determine optimal timing for strategic pivots.

Configurations tested:
1. Never Pivot - Persevere with initial strategy
2. Early Pivot - Pivot at step 30
3. Mid Pivot - Pivot at step 60
4. Late Pivot - Pivot at step 100
5. Adaptive Pivot - Pivot when metrics indicate failure

Key Metrics:
- Market fit progression
- Market share achieved
- Survival rate
- Time to PMF
- Recovery effectiveness

Output:
- CSV files with detailed results
- Visualization plots comparing strategies
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import from newscript.py
from model import run_simulation, get_default_params, CompanyModel

# Experiment Configuration
NUM_RUNS_PER_CONFIG = 20
STEPS = 200
OUTPUT_DIR = "results"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pivot timing configurations
EXPERIMENT_CONFIGS = [
    {
        'name': 'Never Pivot',
        'description': 'Persevere with initial strategy, no pivots',
        'pivot_timing': None,
        'initial_market_fit': 25,  # Start with poor fit
        'allow_adaptive_pivot': False
    },
    {
        'name': 'Early Pivot',
        'description': 'Pivot at step 30 (1.5 years)',
        'pivot_timing': 30,
        'initial_market_fit': 25,
        'allow_adaptive_pivot': False
    },
    {
        'name': 'Mid Pivot',
        'description': 'Pivot at step 60 (3 years)',
        'pivot_timing': 60,
        'initial_market_fit': 25,
        'allow_adaptive_pivot': False
    },
    {
        'name': 'Late Pivot',
        'description': 'Pivot at step 100 (5 years)',
        'pivot_timing': 100,
        'initial_market_fit': 25,
        'allow_adaptive_pivot': False
    },
    {
        'name': 'Adaptive Pivot',
        'description': 'Pivot when market fit < 30 after step 40',
        'pivot_timing': 'adaptive',
        'initial_market_fit': 25,
        'allow_adaptive_pivot': True
    },
]


def run_experiment_with_pivot(config, run_number, verbose=False):
    """
    Run a single simulation with specified pivot timing.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with pivot timing parameters
    run_number : int
        Current run number
    verbose : bool
        Print progress information
    
    Returns:
    --------
    dict
        Results dictionary with metrics
    """
    params = get_default_params()
    
    # Create model with poor initial market fit
    company = CompanyModel(
        num_engineers=10,
        num_sales=5,
        num_marketing=3,
        num_hr=1,
        num_mgmt=1,
        params=params
    )
    
    # Set initial conditions for struggling startup
    company.product.market_fit = config['initial_market_fit']
    company.product.feature_completeness = 15.0
    company.cash_runway = 80.0  # Limited runway adds pressure
    
    # Track metrics over time
    time_series = []
    pivot_occurred = False
    pivot_step = None
    adaptive_pivot_triggered = False
    
    # Disable model's automatic pivot mechanism initially
    original_pivots = company.pivots_remaining
    if config['pivot_timing'] is not None and not config['allow_adaptive_pivot']:
        company.pivots_remaining = 0  # Disable automatic pivots
    
    game_over = False
    game_over_step = None
    
    for step in range(STEPS):
        # Check for forced pivot at specified step
        if config['pivot_timing'] is not None and isinstance(config['pivot_timing'], int):
            if step == config['pivot_timing'] and not pivot_occurred:
                # Restore pivot ability and execute
                company.pivots_remaining = 1
                company.execute_pivot()
                pivot_occurred = True
                pivot_step = step
                company.pivots_remaining = 0  # Prevent further pivots
        
        # Adaptive pivot logic
        elif config['pivot_timing'] == 'adaptive':
            if (step > 40 and 
                company.product.market_fit < 30 and 
                company.revenue < 5 and 
                not pivot_occurred and
                company.pivots_remaining > 0):
                company.execute_pivot()
                pivot_occurred = True
                pivot_step = step
                adaptive_pivot_triggered = True
                company.pivots_remaining = 0  # Prevent further pivots
        
        # Execute step
        company.step()
        
        # Collect data
        time_series.append({
            'step': step,
            'market_fit': company.product.market_fit,
            'market_share': company.product.market_share,
            'feature_completeness': company.product.feature_completeness,
            'technical_debt': company.product.technical_debt,
            'bug_count': company.product.bug_count,
            'cash_runway': company.cash_runway,
            'revenue': company.revenue,
            'brand_awareness': company.product.brand_awareness,
            'team_morale': company.performance_metrics.get('team_morale', 0),
            'pivot_occurred': pivot_occurred
        })
        
        # Check for game over
        if company.cash_runway < 0:
            game_over = True
            game_over_step = step
            if verbose:
                print(f"  Run {run_number}: Game Over at step {step}")
            break
    
    # Calculate final metrics
    df = pd.DataFrame(time_series)
    
    # Calculate time to PMF (market_fit > 70)
    time_to_pmf = None
    pmf_achieved = False
    if df['market_fit'].max() > 70:
        pmf_steps = df[df['market_fit'] > 70]
        if len(pmf_steps) > 0:
            time_to_pmf = pmf_steps.iloc[0]['step']
            pmf_achieved = True
    
    # Calculate recovery metrics if pivot occurred
    pre_pivot_market_fit = None
    post_pivot_market_fit = None
    recovery_rate = None
    
    if pivot_occurred and pivot_step is not None:
        pre_pivot_data = df[df['step'] < pivot_step]
        post_pivot_data = df[df['step'] >= pivot_step]
        
        if len(pre_pivot_data) > 0:
            pre_pivot_market_fit = pre_pivot_data['market_fit'].iloc[-1]
        
        if len(post_pivot_data) > 20:  # At least 20 steps after pivot
            post_pivot_market_fit = post_pivot_data['market_fit'].iloc[20]
            if pre_pivot_market_fit is not None:
                recovery_rate = (post_pivot_market_fit - pre_pivot_market_fit) / 20
    
    # Calculate peak metrics
    peak_market_share = df['market_share'].max()
    peak_market_fit = df['market_fit'].max()
    avg_revenue = df['revenue'].mean()
    
    # Calculate sustainability score
    final_cash = df['cash_runway'].iloc[-1] if not game_over else 0
    sustainability_score = (
        (1 if not game_over else 0) * 40 +
        (peak_market_fit / 100) * 30 +
        (peak_market_share / 100) * 20 +
        (min(final_cash / 50, 1)) * 10
    )
    
    results = {
        'config_name': config['name'],
        'run': run_number,
        'pivot_timing': config['pivot_timing'],
        'pivot_occurred': pivot_occurred,
        'pivot_step': pivot_step,
        'adaptive_pivot': adaptive_pivot_triggered,
        'survived': not game_over,
        'game_over_step': game_over_step,
        'final_market_fit': df['market_fit'].iloc[-1],
        'final_market_share': df['market_share'].iloc[-1],
        'peak_market_fit': peak_market_fit,
        'peak_market_share': peak_market_share,
        'pmf_achieved': pmf_achieved,
        'time_to_pmf': time_to_pmf,
        'pre_pivot_market_fit': pre_pivot_market_fit,
        'post_pivot_market_fit': post_pivot_market_fit,
        'recovery_rate': recovery_rate,
        'final_technical_debt': df['technical_debt'].iloc[-1],
        'final_bug_count': df['bug_count'].iloc[-1],
        'final_cash_runway': final_cash,
        'avg_revenue': avg_revenue,
        'sustainability_score': sustainability_score,
        'time_series': df
    }
    
    return results


def run_experiment():
    """
    Run the complete experiment across all configurations.
    
    Returns:
    --------
    list
        List of result dictionaries
    """
    print("=" * 70)
    print("EXPERIMENT 7: PIVOT TIMING ANALYSIS")
    print("=" * 70)
    print(f"\nConfigurations: {len(EXPERIMENT_CONFIGS)}")
    print(f"Runs per configuration: {NUM_RUNS_PER_CONFIG}")
    print(f"Steps per run: {STEPS}")
    print(f"Total simulations: {len(EXPERIMENT_CONFIGS) * NUM_RUNS_PER_CONFIG}\n")
    
    all_results = []
    
    for config_idx, config in enumerate(EXPERIMENT_CONFIGS):
        print(f"\n[{config_idx + 1}/{len(EXPERIMENT_CONFIGS)}] Running: {config['name']}")
        print(f"  Description: {config['description']}")
        
        for run in range(NUM_RUNS_PER_CONFIG):
            if (run + 1) % 5 == 0:
                print(f"  Progress: {run + 1}/{NUM_RUNS_PER_CONFIG} runs completed")
            
            result = run_experiment_with_pivot(config, run + 1)
            all_results.append(result)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)
    
    return all_results


def analyze_results(results):
    """
    Analyze experiment results and compute aggregate statistics.
    
    Parameters:
    -----------
    results : list
        List of result dictionaries
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with aggregate statistics by configuration
    """
    print("\n" + "=" * 70)
    print("ANALYSIS: PIVOT TIMING COMPARISON")
    print("=" * 70)
    
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'time_series'} for r in results])
    
    # Group by configuration
    grouped = df.groupby('config_name')
    
    analysis = grouped.agg({
        'survived': ['mean', 'sum'],
        'pmf_achieved': ['mean', 'sum'],
        'time_to_pmf': ['mean', 'median', 'std'],
        'final_market_fit': ['mean', 'median', 'std', 'max'],
        'final_market_share': ['mean', 'median', 'std', 'max'],
        'peak_market_fit': ['mean', 'median', 'max'],
        'peak_market_share': ['mean', 'median', 'max'],
        'recovery_rate': ['mean', 'median'],
        'sustainability_score': ['mean', 'std'],
        'avg_revenue': ['mean', 'median'],
        'game_over_step': ['mean', 'median']
    }).round(2)
    
    # Calculate success rate (survived + achieved significant market share)
    success_metrics = df.groupby('config_name').apply(
        lambda x: pd.Series({
            'survival_rate': (x['survived'].sum() / len(x) * 100),
            'pmf_rate': (x['pmf_achieved'].sum() / len(x) * 100),
            'avg_final_market_share': x['final_market_share'].mean(),
            'success_rate': ((x['survived'] & (x['final_market_share'] > 5)).sum() / len(x) * 100)
        })
    ).round(2)
    
    print("\n📊 SURVIVAL AND SUCCESS RATES:")
    print(success_metrics)
    
    print("\n📈 MARKET METRICS:")
    market_cols = [col for col in analysis.columns if 'market' in str(col).lower()]
    print(analysis[market_cols])
    
    print("\n⏱️ TIME TO PRODUCT-MARKET FIT:")
    pmf_cols = [col for col in analysis.columns if 'pmf' in str(col).lower() or 'time_to' in str(col).lower()]
    print(analysis[pmf_cols])
    
    print("\n🔄 PIVOT ANALYSIS:")
    pivot_analysis = df.groupby('config_name').agg({
        'pivot_occurred': 'sum',
        'pivot_step': ['mean', 'median'],
        'recovery_rate': ['mean', 'median']
    }).round(2)
    print(pivot_analysis)
    
    # Find best strategy
    best_config = success_metrics['success_rate'].idxmax()
    best_rate = success_metrics.loc[best_config, 'success_rate']
    
    print(f"\n🏆 BEST STRATEGY: {best_config}")
    print(f"   Success Rate: {best_rate:.1f}%")
    print(f"   Survival Rate: {success_metrics.loc[best_config, 'survival_rate']:.1f}%")
    print(f"   PMF Achievement Rate: {success_metrics.loc[best_config, 'pmf_rate']:.1f}%")
    print(f"   Avg Market Share: {success_metrics.loc[best_config, 'avg_final_market_share']:.2f}%")
    
    return analysis, success_metrics


def create_visualizations(results):
    """
    Create comprehensive visualizations of the experiment results.
    
    Parameters:
    -----------
    results : list
        List of result dictionaries
    """
    print("\n📊 Generating visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Prepare data
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'time_series'} for r in results])
    
    # Figure 1: Success Metrics Comparison
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
    fig1.suptitle('Experiment 7: Pivot Timing - Success Metrics Comparison', 
                  fontsize=16, fontweight='bold')
    
    # Survival rate
    survival_data = df.groupby('config_name')['survived'].mean() * 100
    survival_data.plot(kind='bar', ax=axes1[0, 0], color='steelblue')
    axes1[0, 0].set_title('Survival Rate by Strategy')
    axes1[0, 0].set_ylabel('Survival Rate (%)')
    axes1[0, 0].set_xlabel('')
    axes1[0, 0].tick_params(axis='x', rotation=45)
    axes1[0, 0].grid(True, alpha=0.3)
    axes1[0, 0].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    axes1[0, 0].legend()
    
    # PMF Achievement Rate
    pmf_data = df.groupby('config_name')['pmf_achieved'].mean() * 100
    pmf_data.plot(kind='bar', ax=axes1[0, 1], color='green')
    axes1[0, 1].set_title('Product-Market Fit Achievement Rate')
    axes1[0, 1].set_ylabel('PMF Achievement Rate (%)')
    axes1[0, 1].set_xlabel('')
    axes1[0, 1].tick_params(axis='x', rotation=45)
    axes1[0, 1].grid(True, alpha=0.3)
    
    # Final Market Share (box plot)
    df.boxplot(column='final_market_share', by='config_name', ax=axes1[1, 0])
    axes1[1, 0].set_title('Final Market Share Distribution')
    axes1[1, 0].set_ylabel('Market Share (%)')
    axes1[1, 0].set_xlabel('Strategy')
    axes1[1, 0].tick_params(axis='x', rotation=45)
    plt.sca(axes1[1, 0])
    plt.xticks(rotation=45, ha='right')
    
    # Sustainability Score
    df.boxplot(column='sustainability_score', by='config_name', ax=axes1[1, 1])
    axes1[1, 1].set_title('Sustainability Score Distribution')
    axes1[1, 1].set_ylabel('Sustainability Score (0-100)')
    axes1[1, 1].set_xlabel('Strategy')
    axes1[1, 1].tick_params(axis='x', rotation=45)
    plt.sca(axes1[1, 1])
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig1_path = os.path.join(OUTPUT_DIR, f"exp7_success_metrics_{timestamp}.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig1_path}")
    plt.close()
    
    # Figure 2: Time Series - Market Fit Evolution
    fig2, ax2 = plt.subplots(figsize=(15, 8))
    fig2.suptitle('Experiment 7: Market Fit Evolution Over Time', 
                  fontsize=16, fontweight='bold')
    
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for idx, config_name in enumerate(df['config_name'].unique()):
        config_results = [r for r in results if r['config_name'] == config_name]
        
        # Calculate mean trajectory
        all_steps = []
        max_len = 0
        for r in config_results:
            ts = r['time_series']
            all_steps.append(ts['market_fit'].values)
            max_len = max(max_len, len(ts['market_fit']))
        
        # Pad shorter sequences with NaN to handle companies that failed early
        padded_steps = []
        for s in all_steps:
            if len(s) < max_len:
                padded = np.pad(s, (0, max_len - len(s)), constant_values=np.nan)
            else:
                padded = s
            padded_steps.append(padded)
        
        # Calculate mean ignoring NaN values
        mean_market_fit = np.nanmean(padded_steps, axis=0)
        std_market_fit = np.nanstd(padded_steps, axis=0)
        steps = range(len(mean_market_fit))
        
        ax2.plot(steps, mean_market_fit, label=config_name, 
                color=colors[idx], linewidth=2)
        ax2.fill_between(steps, 
                         mean_market_fit - std_market_fit,
                         mean_market_fit + std_market_fit,
                         alpha=0.2, color=colors[idx])
        
        # Mark average pivot point if applicable
        pivot_results = [r for r in config_results if r['pivot_occurred']]
        if pivot_results:
            avg_pivot_step = np.mean([r['pivot_step'] for r in pivot_results])
            ax2.axvline(x=avg_pivot_step, color=colors[idx], 
                       linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Simulation Step (months)')
    ax2.set_ylabel('Market Fit')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=70, color='green', linestyle='--', alpha=0.3, label='PMF Threshold')
    
    fig2_path = os.path.join(OUTPUT_DIR, f"exp7_market_fit_evolution_{timestamp}.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig2_path}")
    plt.close()
    
    # Figure 3: Time to PMF Analysis
    fig3, axes3 = plt.subplots(1, 2, figsize=(15, 6))
    fig3.suptitle('Experiment 7: Time to Product-Market Fit Analysis', 
                  fontsize=16, fontweight='bold')
    
    # Time to PMF distribution
    pmf_df = df[df['pmf_achieved'] == True].copy()
    if len(pmf_df) > 0:
        pmf_df.boxplot(column='time_to_pmf', by='config_name', ax=axes3[0])
        axes3[0].set_title('Time to PMF Distribution (PMF Achieved Only)')
        axes3[0].set_ylabel('Steps to PMF')
        axes3[0].set_xlabel('Strategy')
        axes3[0].tick_params(axis='x', rotation=45)
        plt.sca(axes3[0])
        plt.xticks(rotation=45, ha='right')
    else:
        axes3[0].text(0.5, 0.5, 'No PMF achieved in any configuration', 
                     ha='center', va='center')
        axes3[0].set_title('Time to PMF Distribution')
    
    # Peak Market Fit vs Pivot Timing
    for idx, config_name in enumerate(df['config_name'].unique()):
        config_df = df[df['config_name'] == config_name]
        pivot_steps = []
        peak_fits = []
        
        for _, row in config_df.iterrows():
            if row['pivot_occurred'] and row['pivot_step'] is not None:
                pivot_steps.append(row['pivot_step'])
                peak_fits.append(row['peak_market_fit'])
        
        if pivot_steps:
            axes3[1].scatter(pivot_steps, peak_fits, label=config_name, 
                           s=100, alpha=0.6, color=colors[idx])
    
    axes3[1].set_xlabel('Pivot Step')
    axes3[1].set_ylabel('Peak Market Fit Achieved')
    axes3[1].set_title('Pivot Timing vs Peak Market Fit')
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig3_path = os.path.join(OUTPUT_DIR, f"exp7_pmf_analysis_{timestamp}.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig3_path}")
    plt.close()
    
    # Figure 4: Recovery Analysis
    fig4, axes4 = plt.subplots(1, 2, figsize=(15, 6))
    fig4.suptitle('Experiment 7: Pivot Recovery Analysis', 
                  fontsize=16, fontweight='bold')
    
    # Recovery rate by strategy
    recovery_df = df[df['recovery_rate'].notna()].copy()
    if len(recovery_df) > 0:
        recovery_data = recovery_df.groupby('config_name')['recovery_rate'].mean()
        recovery_data.plot(kind='bar', ax=axes4[0], color='teal')
        axes4[0].set_title('Average Recovery Rate After Pivot')
        axes4[0].set_ylabel('Recovery Rate (Market Fit gain per step)')
        axes4[0].set_xlabel('')
        axes4[0].tick_params(axis='x', rotation=45)
        axes4[0].grid(True, alpha=0.3)
        axes4[0].axhline(y=0, color='r', linestyle='-', alpha=0.5)
    else:
        axes4[0].text(0.5, 0.5, 'No recovery data available', 
                     ha='center', va='center')
        axes4[0].set_title('Average Recovery Rate After Pivot')
    
    # Pre vs Post Pivot Market Fit
    if len(recovery_df) > 0:
        config_names = recovery_df['config_name'].unique()
        x = np.arange(len(config_names))
        width = 0.35
        
        pre_pivot = [recovery_df[recovery_df['config_name'] == name]['pre_pivot_market_fit'].mean() 
                    for name in config_names]
        post_pivot = [recovery_df[recovery_df['config_name'] == name]['post_pivot_market_fit'].mean() 
                     for name in config_names]
        
        axes4[1].bar(x - width/2, pre_pivot, width, label='Pre-Pivot', color='orange', alpha=0.8)
        axes4[1].bar(x + width/2, post_pivot, width, label='Post-Pivot (20 steps)', color='green', alpha=0.8)
        axes4[1].set_xlabel('Strategy')
        axes4[1].set_ylabel('Market Fit')
        axes4[1].set_title('Market Fit Before and After Pivot')
        axes4[1].set_xticks(x)
        axes4[1].set_xticklabels(config_names, rotation=45, ha='right')
        axes4[1].legend()
        axes4[1].grid(True, alpha=0.3)
    else:
        axes4[1].text(0.5, 0.5, 'No pivot comparison data available', 
                     ha='center', va='center')
        axes4[1].set_title('Market Fit Before and After Pivot')
    
    plt.tight_layout()
    fig4_path = os.path.join(OUTPUT_DIR, f"exp7_recovery_analysis_{timestamp}.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig4_path}")
    plt.close()
    
    # Figure 5: Financial Impact
    fig5, axes5 = plt.subplots(2, 2, figsize=(15, 12))
    fig5.suptitle('Experiment 7: Financial and Operational Impact', 
                  fontsize=16, fontweight='bold')
    
    # Average Revenue
    revenue_data = df.groupby('config_name')['avg_revenue'].mean()
    revenue_data.plot(kind='bar', ax=axes5[0, 0], color='gold')
    axes5[0, 0].set_title('Average Revenue')
    axes5[0, 0].set_ylabel('Revenue')
    axes5[0, 0].set_xlabel('')
    axes5[0, 0].tick_params(axis='x', rotation=45)
    axes5[0, 0].grid(True, alpha=0.3)
    
    # Final Cash Runway (survivors only)
    survivors_df = df[df['survived'] == True].copy()
    if len(survivors_df) > 0:
        survivors_df.boxplot(column='final_cash_runway', by='config_name', ax=axes5[0, 1])
        axes5[0, 1].set_title('Final Cash Runway (Survivors Only)')
        axes5[0, 1].set_ylabel('Cash Runway (months)')
        axes5[0, 1].set_xlabel('Strategy')
        axes5[0, 1].tick_params(axis='x', rotation=45)
        plt.sca(axes5[0, 1])
        plt.xticks(rotation=45, ha='right')
    else:
        axes5[0, 1].text(0.5, 0.5, 'No survivors', ha='center', va='center')
        axes5[0, 1].set_title('Final Cash Runway (Survivors Only)')
    
    # Game Over Step (failures only)
    failures_df = df[df['survived'] == False].copy()
    if len(failures_df) > 0:
        failures_df.boxplot(column='game_over_step', by='config_name', ax=axes5[1, 0])
        axes5[1, 0].set_title('Time of Failure (Failed Companies Only)')
        axes5[1, 0].set_ylabel('Game Over Step')
        axes5[1, 0].set_xlabel('Strategy')
        axes5[1, 0].tick_params(axis='x', rotation=45)
        plt.sca(axes5[1, 0])
        plt.xticks(rotation=45, ha='right')
    else:
        axes5[1, 0].text(0.5, 0.5, 'No failures', ha='center', va='center')
        axes5[1, 0].set_title('Time of Failure (Failed Companies Only)')
    
    # Technical Debt at end
    df.boxplot(column='final_technical_debt', by='config_name', ax=axes5[1, 1])
    axes5[1, 1].set_title('Final Technical Debt')
    axes5[1, 1].set_ylabel('Technical Debt')
    axes5[1, 1].set_xlabel('Strategy')
    axes5[1, 1].tick_params(axis='x', rotation=45)
    plt.sca(axes5[1, 1])
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    fig5_path = os.path.join(OUTPUT_DIR, f"exp7_financial_impact_{timestamp}.png")
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig5_path}")
    plt.close()
    
    print("  ✓ All visualizations created successfully")


def save_results(results, analysis, success_metrics):
    """
    Save results to CSV files.
    
    Parameters:
    -----------
    results : list
        List of result dictionaries
    analysis : pd.DataFrame
        Aggregate analysis results
    success_metrics : pd.DataFrame
        Success metrics by configuration
    """
    print("\n💾 Saving results to CSV files...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results (without time series)
    df_results = pd.DataFrame([{k: v for k, v in r.items() if k != 'time_series'} 
                               for r in results])
    results_path = os.path.join(OUTPUT_DIR, f"exp7_detailed_results_{timestamp}.csv")
    df_results.to_csv(results_path, index=False)
    print(f"  ✓ Saved: {results_path}")
    
    # Save aggregate analysis
    analysis_path = os.path.join(OUTPUT_DIR, f"exp7_aggregate_analysis_{timestamp}.csv")
    analysis.to_csv(analysis_path)
    print(f"  ✓ Saved: {analysis_path}")
    
    # Save success metrics
    success_path = os.path.join(OUTPUT_DIR, f"exp7_success_metrics_{timestamp}.csv")
    success_metrics.to_csv(success_path)
    print(f"  ✓ Saved: {success_path}")
    
    print("  ✓ All results saved successfully")


def main():
    """
    Main execution function for Experiment 7.
    """
    print("\n" + "🔬" * 35)
    print("EXPERIMENT 7: PIVOT TIMING ANALYSIS")
    print("🔬" * 35 + "\n")
    
    # Run experiment
    results = run_experiment()
    
    # Analyze results
    analysis, success_metrics = analyze_results(results)
    
    # Create visualizations
    create_visualizations(results)
    
    # Save results
    save_results(results, analysis, success_metrics)
    
    print("\n" + "=" * 70)
    print("✅ EXPERIMENT 7 COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nResults saved in: {OUTPUT_DIR}/")
    print("\nKey Findings:")
    print("- Check visualizations for timing strategy comparisons")
    print("- Review CSV files for detailed metrics")
    print("- Compare early vs late vs adaptive pivot strategies")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
