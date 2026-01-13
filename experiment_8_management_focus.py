"""
Experiment 8: Management Focus Impact Analysis
===============================================

Tests different management focus strategies to determine optimal leadership approach.

Configurations tested:
1. Always Engineering Focus - Prioritize technical development
2. Always Sales Focus - Prioritize revenue generation
3. Always Marketing Focus - Prioritize brand and market awareness
4. Always Quality Focus - Prioritize bug fixes and technical debt
5. Adaptive/Balanced (Default) - Dynamic focus based on situation

Key Metrics:
- Product quality vs market performance
- Technical debt accumulation
- Revenue generation
- Market share growth
- Survival rate

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
from model import run_simulation, get_default_params, CompanyModel, ManagementAgent

# Experiment Configuration
NUM_RUNS_PER_CONFIG = 20
STEPS = 200
OUTPUT_DIR = "results"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Management focus configurations
EXPERIMENT_CONFIGS = [
    {
        'name': 'Always Engineering',
        'description': 'Management always focuses on engineering priorities',
        'focus': 'engineering',
        'strategy_adjustments': {
            'dev_vs_quality': 0.8,  # More development
            'features_vs_debt': 0.9  # More features
        }
    },
    {
        'name': 'Always Sales',
        'description': 'Management always focuses on sales and revenue',
        'focus': 'sales',
        'strategy_adjustments': {
            'growth_vs_stability': 0.8  # More growth-oriented
        }
    },
    {
        'name': 'Always Marketing',
        'description': 'Management always focuses on marketing and brand',
        'focus': 'marketing',
        'strategy_adjustments': {
            'growth_vs_stability': 0.7
        }
    },
    {
        'name': 'Always Quality',
        'description': 'Management always focuses on quality and technical debt',
        'focus': 'quality',
        'strategy_adjustments': {
            'dev_vs_quality': 0.3,  # Much more quality
            'features_vs_debt': 0.4  # Much more refactoring
        }
    },
    {
        'name': 'Adaptive/Balanced',
        'description': 'Dynamic focus based on situation (default behavior)',
        'focus': 'balanced',
        'strategy_adjustments': {}
    },
]


class CustomManagementAgent(ManagementAgent):
    """Custom management agent with forced focus strategy."""
    
    def __init__(self, model, forced_focus=None):
        super().__init__(model)
        self.forced_focus = forced_focus
    
    def assess_situation(self, product, avg_capacities):
        """Override to force specific focus if configured."""
        if self.forced_focus and self.forced_focus != 'balanced':
            self.current_focus = self.forced_focus
            return self.forced_focus
        
        # Otherwise use default adaptive logic
        return super().assess_situation(product, avg_capacities)


def run_experiment_with_focus(config, run_number, verbose=False):
    """
    Run a single simulation with specified management focus.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with management focus parameters
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
    
    # Create model
    company = CompanyModel(
        num_engineers=10,
        num_sales=5,
        num_marketing=3,
        num_hr=1,
        num_mgmt=1,
        params=params
    )
    
    # Replace management agent with custom one
    if len(company.teams['mgmt']) > 0:
        old_mgmt = company.teams['mgmt'][0]
        custom_mgmt = CustomManagementAgent(company, forced_focus=config['focus'])
        # Copy attributes from old management agent
        custom_mgmt.management_capacity = old_mgmt.management_capacity
        custom_mgmt.strategic_vision = old_mgmt.strategic_vision
        custom_mgmt.crisis_management = old_mgmt.crisis_management
        company.teams['mgmt'][0] = custom_mgmt
    
    # Apply strategy adjustments
    for key, value in config['strategy_adjustments'].items():
        if key in company.strategy:
            company.strategy[key] = value
    
    # Adjust sprint budgets based on focus
    if config['focus'] == 'engineering':
        company.sprint_budget['engineering'] = 120.0
        company.sprint_budget['sales'] = 60.0
        company.sprint_budget['marketing'] = 40.0
    elif config['focus'] == 'sales':
        company.sprint_budget['engineering'] = 80.0
        company.sprint_budget['sales'] = 100.0
        company.sprint_budget['marketing'] = 50.0
    elif config['focus'] == 'marketing':
        company.sprint_budget['engineering'] = 80.0
        company.sprint_budget['sales'] = 60.0
        company.sprint_budget['marketing'] = 90.0
    elif config['focus'] == 'quality':
        company.sprint_budget['engineering'] = 100.0  # Same budget but different allocation
        company.sprint_budget['sales'] = 80.0
        company.sprint_budget['marketing'] = 60.0
    
    # Track metrics over time
    time_series = []
    game_over = False
    game_over_step = None
    
    for step in range(STEPS):
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
            'code_quality': company.product.code_quality,
            'cash_runway': company.cash_runway,
            'revenue': company.revenue,
            'brand_awareness': company.product.brand_awareness,
            'team_morale': company.performance_metrics.get('team_morale', 0),
            'customer_satisfaction': company.performance_metrics.get('customer_satisfaction', 0),
            'organizational_alignment': company.product.organizational_alignment,
            'num_engineers': len(company.teams['engineers']),
            'num_sales': len(company.teams['sales']),
            'num_marketing': len(company.teams['marketing'])
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
    
    # Calculate peak metrics
    peak_market_share = df['market_share'].max()
    peak_market_fit = df['market_fit'].max()
    peak_revenue = df['revenue'].max()
    
    # Calculate average metrics
    avg_technical_debt = df['technical_debt'].mean()
    avg_bug_count = df['bug_count'].mean()
    avg_code_quality = df['code_quality'].mean()
    avg_team_morale = df['team_morale'].mean()
    avg_customer_satisfaction = df['customer_satisfaction'].mean()
    
    # Calculate quality-speed tradeoff metrics
    final_features = df['feature_completeness'].iloc[-1]
    final_quality = df['code_quality'].iloc[-1]
    quality_speed_balance = (final_quality / 100) * (final_features / (final_features + 50))
    
    # Calculate growth rate
    if len(df) > 20:
        early_market_share = df.iloc[20]['market_share']
        late_market_share = df.iloc[-1]['market_share']
        growth_rate = (late_market_share - early_market_share) / (len(df) - 20)
    else:
        growth_rate = 0
    
    # Calculate sustainability score
    final_cash = df['cash_runway'].iloc[-1] if not game_over else 0
    sustainability_score = (
        (1 if not game_over else 0) * 30 +
        (peak_market_fit / 100) * 25 +
        (peak_market_share / 100) * 20 +
        (avg_code_quality / 100) * 15 +
        (min(final_cash / 50, 1)) * 10
    )
    
    # Calculate technical health score
    technical_health = (
        (avg_code_quality / 100) * 40 +
        (1 - min(avg_technical_debt / 100, 1)) * 35 +
        (1 - min(avg_bug_count / 30, 1)) * 25
    ) * 100
    
    # Calculate business performance score
    business_performance = (
        (peak_market_share / 100) * 40 +
        (peak_market_fit / 100) * 30 +
        (min(peak_revenue / 50, 1)) * 30
    ) * 100
    
    results = {
        'config_name': config['name'],
        'run': run_number,
        'focus_strategy': config['focus'],
        'survived': not game_over,
        'game_over_step': game_over_step,
        'final_market_fit': df['market_fit'].iloc[-1],
        'final_market_share': df['market_share'].iloc[-1],
        'peak_market_fit': peak_market_fit,
        'peak_market_share': peak_market_share,
        'peak_revenue': peak_revenue,
        'pmf_achieved': pmf_achieved,
        'time_to_pmf': time_to_pmf,
        'final_technical_debt': df['technical_debt'].iloc[-1],
        'final_bug_count': df['bug_count'].iloc[-1],
        'final_code_quality': df['code_quality'].iloc[-1],
        'final_feature_completeness': df['feature_completeness'].iloc[-1],
        'avg_technical_debt': avg_technical_debt,
        'avg_bug_count': avg_bug_count,
        'avg_code_quality': avg_code_quality,
        'avg_team_morale': avg_team_morale,
        'avg_customer_satisfaction': avg_customer_satisfaction,
        'final_cash_runway': final_cash,
        'avg_revenue': df['revenue'].mean(),
        'growth_rate': growth_rate,
        'quality_speed_balance': quality_speed_balance,
        'sustainability_score': sustainability_score,
        'technical_health': technical_health,
        'business_performance': business_performance,
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
    print("EXPERIMENT 8: MANAGEMENT FOCUS IMPACT ANALYSIS")
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
            
            result = run_experiment_with_focus(config, run + 1)
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
    print("ANALYSIS: MANAGEMENT FOCUS COMPARISON")
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
        'peak_market_share': ['mean', 'median', 'max'],
        'peak_revenue': ['mean', 'median', 'max'],
        'avg_technical_debt': ['mean', 'std'],
        'avg_bug_count': ['mean', 'std'],
        'avg_code_quality': ['mean', 'std'],
        'technical_health': ['mean', 'std'],
        'business_performance': ['mean', 'std'],
        'sustainability_score': ['mean', 'std'],
        'growth_rate': ['mean', 'median']
    }).round(2)
    
    # Calculate success metrics
    success_metrics = df.groupby('config_name').apply(
        lambda x: pd.Series({
            'survival_rate': (x['survived'].sum() / len(x) * 100),
            'pmf_rate': (x['pmf_achieved'].sum() / len(x) * 100),
            'avg_final_market_share': x['final_market_share'].mean(),
            'avg_technical_health': x['technical_health'].mean(),
            'avg_business_performance': x['business_performance'].mean(),
            'success_rate': ((x['survived'] & (x['final_market_share'] > 5)).sum() / len(x) * 100)
        })
    ).round(2)
    
    print("\n📊 SURVIVAL AND SUCCESS RATES:")
    print(success_metrics)
    
    print("\n📈 BUSINESS PERFORMANCE:")
    business_cols = [col for col in analysis.columns if any(x in str(col).lower() for x in ['market', 'revenue', 'business'])]
    print(analysis[business_cols])
    
    print("\n🔧 TECHNICAL HEALTH:")
    tech_cols = [col for col in analysis.columns if any(x in str(col).lower() for x in ['debt', 'bug', 'quality', 'technical'])]
    print(analysis[tech_cols])
    
    print("\n⚖️ TRADEOFF ANALYSIS:")
    tradeoff_data = df.groupby('config_name').agg({
        'avg_code_quality': 'mean',
        'final_feature_completeness': 'mean',
        'growth_rate': 'mean',
        'avg_technical_debt': 'mean'
    }).round(2)
    print(tradeoff_data)
    
    # Find best strategies for different objectives
    print("\n🏆 BEST STRATEGIES BY OBJECTIVE:")
    
    best_survival = success_metrics['survival_rate'].idxmax()
    print(f"   Survival: {best_survival} ({success_metrics.loc[best_survival, 'survival_rate']:.1f}%)")
    
    best_market_share = success_metrics['avg_final_market_share'].idxmax()
    print(f"   Market Share: {best_market_share} ({success_metrics.loc[best_market_share, 'avg_final_market_share']:.2f}%)")
    
    best_technical = success_metrics['avg_technical_health'].idxmax()
    print(f"   Technical Health: {best_technical} ({success_metrics.loc[best_technical, 'avg_technical_health']:.1f})")
    
    best_business = success_metrics['avg_business_performance'].idxmax()
    print(f"   Business Performance: {best_business} ({success_metrics.loc[best_business, 'avg_business_performance']:.1f})")
    
    best_overall = success_metrics['success_rate'].idxmax()
    print(f"   Overall Success: {best_overall} ({success_metrics.loc[best_overall, 'success_rate']:.1f}%)")
    
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
    
    # Figure 1: Overall Performance Comparison
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
    fig1.suptitle('Experiment 8: Management Focus - Overall Performance', 
                  fontsize=16, fontweight='bold')
    
    # Survival rate
    survival_data = df.groupby('config_name')['survived'].mean() * 100
    survival_data.plot(kind='bar', ax=axes1[0, 0], color='steelblue')
    axes1[0, 0].set_title('Survival Rate by Management Focus')
    axes1[0, 0].set_ylabel('Survival Rate (%)')
    axes1[0, 0].set_xlabel('')
    axes1[0, 0].tick_params(axis='x', rotation=45)
    axes1[0, 0].grid(True, alpha=0.3)
    axes1[0, 0].axhline(y=50, color='r', linestyle='--', alpha=0.5)
    
    # Business Performance Score
    business_data = df.groupby('config_name')['business_performance'].mean()
    business_data.plot(kind='bar', ax=axes1[0, 1], color='green')
    axes1[0, 1].set_title('Business Performance Score')
    axes1[0, 1].set_ylabel('Score (0-100)')
    axes1[0, 1].set_xlabel('')
    axes1[0, 1].tick_params(axis='x', rotation=45)
    axes1[0, 1].grid(True, alpha=0.3)
    
    # Technical Health Score
    tech_data = df.groupby('config_name')['technical_health'].mean()
    tech_data.plot(kind='bar', ax=axes1[1, 0], color='orange')
    axes1[1, 0].set_title('Technical Health Score')
    axes1[1, 0].set_ylabel('Score (0-100)')
    axes1[1, 0].set_xlabel('')
    axes1[1, 0].tick_params(axis='x', rotation=45)
    axes1[1, 0].grid(True, alpha=0.3)
    
    # Sustainability Score
    sust_data = df.groupby('config_name')['sustainability_score'].mean()
    sust_data.plot(kind='bar', ax=axes1[1, 1], color='purple')
    axes1[1, 1].set_title('Sustainability Score')
    axes1[1, 1].set_ylabel('Score (0-100)')
    axes1[1, 1].set_xlabel('')
    axes1[1, 1].tick_params(axis='x', rotation=45)
    axes1[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig1_path = os.path.join(OUTPUT_DIR, f"exp8_overall_performance_{timestamp}.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig1_path}")
    plt.close()
    
    # Figure 2: Technical vs Business Tradeoff
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
    fig2.suptitle('Experiment 8: Technical Quality vs Business Performance Tradeoff', 
                  fontsize=16, fontweight='bold')
    
    # Scatter: Technical Health vs Business Performance
    for idx, config_name in enumerate(df['config_name'].unique()):
        config_df = df[df['config_name'] == config_name]
        axes2[0].scatter(config_df['technical_health'], 
                        config_df['business_performance'],
                        label=config_name, s=100, alpha=0.6)
    
    axes2[0].set_xlabel('Technical Health Score')
    axes2[0].set_ylabel('Business Performance Score')
    axes2[0].set_title('Technical Health vs Business Performance')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    # Box plot: Code Quality vs Market Share
    comparison_df = df[['config_name', 'avg_code_quality', 'final_market_share']].copy()
    config_names = comparison_df['config_name'].unique()
    
    x_pos = np.arange(len(config_names))
    quality_means = [comparison_df[comparison_df['config_name']==name]['avg_code_quality'].mean() 
                     for name in config_names]
    share_means = [comparison_df[comparison_df['config_name']==name]['final_market_share'].mean() 
                   for name in config_names]
    
    width = 0.35
    axes2[1].bar(x_pos - width/2, quality_means, width, label='Avg Code Quality', alpha=0.8)
    axes2[1].bar(x_pos + width/2, share_means, width, label='Final Market Share', alpha=0.8)
    axes2[1].set_xlabel('Management Focus')
    axes2[1].set_ylabel('Value')
    axes2[1].set_title('Code Quality vs Market Share')
    axes2[1].set_xticks(x_pos)
    axes2[1].set_xticklabels(config_names, rotation=45, ha='right')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR, f"exp8_tradeoff_analysis_{timestamp}.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig2_path}")
    plt.close()
    
    # Figure 3: Time Series Evolution
    fig3, axes3 = plt.subplots(2, 2, figsize=(15, 12))
    fig3.suptitle('Experiment 8: Metrics Evolution Over Time', 
                  fontsize=16, fontweight='bold')
    
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    # Market Fit Evolution
    for idx, config_name in enumerate(df['config_name'].unique()):
        config_results = [r for r in results if r['config_name'] == config_name]
        all_steps = [r['time_series']['market_fit'].values for r in config_results]
        min_len = min(len(s) for s in all_steps)
        all_steps = [s[:min_len] for s in all_steps]
        mean_values = np.mean(all_steps, axis=0)
        std_values = np.std(all_steps, axis=0)
        steps = range(len(mean_values))
        
        axes3[0, 0].plot(steps, mean_values, label=config_name, 
                        color=colors[idx], linewidth=2)
        axes3[0, 0].fill_between(steps, mean_values - std_values, 
                                 mean_values + std_values,
                                 alpha=0.2, color=colors[idx])
    
    axes3[0, 0].set_xlabel('Step')
    axes3[0, 0].set_ylabel('Market Fit')
    axes3[0, 0].set_title('Market Fit Evolution')
    axes3[0, 0].legend(loc='best', fontsize=8)
    axes3[0, 0].grid(True, alpha=0.3)
    
    # Technical Debt Evolution
    for idx, config_name in enumerate(df['config_name'].unique()):
        config_results = [r for r in results if r['config_name'] == config_name]
        all_steps = [r['time_series']['technical_debt'].values for r in config_results]
        min_len = min(len(s) for s in all_steps)
        all_steps = [s[:min_len] for s in all_steps]
        mean_values = np.mean(all_steps, axis=0)
        steps = range(len(mean_values))
        
        axes3[0, 1].plot(steps, mean_values, label=config_name, 
                        color=colors[idx], linewidth=2)
    
    axes3[0, 1].set_xlabel('Step')
    axes3[0, 1].set_ylabel('Technical Debt')
    axes3[0, 1].set_title('Technical Debt Evolution')
    axes3[0, 1].legend(loc='best', fontsize=8)
    axes3[0, 1].grid(True, alpha=0.3)
    
    # Market Share Evolution
    for idx, config_name in enumerate(df['config_name'].unique()):
        config_results = [r for r in results if r['config_name'] == config_name]
        all_steps = [r['time_series']['market_share'].values for r in config_results]
        min_len = min(len(s) for s in all_steps)
        all_steps = [s[:min_len] for s in all_steps]
        mean_values = np.mean(all_steps, axis=0)
        steps = range(len(mean_values))
        
        axes3[1, 0].plot(steps, mean_values, label=config_name, 
                        color=colors[idx], linewidth=2)
    
    axes3[1, 0].set_xlabel('Step')
    axes3[1, 0].set_ylabel('Market Share (%)')
    axes3[1, 0].set_title('Market Share Evolution')
    axes3[1, 0].legend(loc='best', fontsize=8)
    axes3[1, 0].grid(True, alpha=0.3)
    
    # Revenue Evolution
    for idx, config_name in enumerate(df['config_name'].unique()):
        config_results = [r for r in results if r['config_name'] == config_name]
        all_steps = [r['time_series']['revenue'].values for r in config_results]
        min_len = min(len(s) for s in all_steps)
        all_steps = [s[:min_len] for s in all_steps]
        mean_values = np.mean(all_steps, axis=0)
        steps = range(len(mean_values))
        
        axes3[1, 1].plot(steps, mean_values, label=config_name, 
                        color=colors[idx], linewidth=2)
    
    axes3[1, 1].set_xlabel('Step')
    axes3[1, 1].set_ylabel('Revenue')
    axes3[1, 1].set_title('Revenue Evolution')
    axes3[1, 1].legend(loc='best', fontsize=8)
    axes3[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig3_path = os.path.join(OUTPUT_DIR, f"exp8_time_series_{timestamp}.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig3_path}")
    plt.close()
    
    # Figure 4: Detailed Technical Metrics
    fig4, axes4 = plt.subplots(2, 2, figsize=(15, 12))
    fig4.suptitle('Experiment 8: Technical Metrics Comparison', 
                  fontsize=16, fontweight='bold')
    
    # Average Technical Debt
    df.boxplot(column='avg_technical_debt', by='config_name', ax=axes4[0, 0])
    axes4[0, 0].set_title('Average Technical Debt')
    axes4[0, 0].set_ylabel('Technical Debt')
    axes4[0, 0].set_xlabel('')
    axes4[0, 0].tick_params(axis='x', rotation=45)
    plt.sca(axes4[0, 0])
    plt.xticks(rotation=45, ha='right')
    
    # Average Bug Count
    df.boxplot(column='avg_bug_count', by='config_name', ax=axes4[0, 1])
    axes4[0, 1].set_title('Average Bug Count')
    axes4[0, 1].set_ylabel('Bug Count')
    axes4[0, 1].set_xlabel('')
    axes4[0, 1].tick_params(axis='x', rotation=45)
    plt.sca(axes4[0, 1])
    plt.xticks(rotation=45, ha='right')
    
    # Average Code Quality
    df.boxplot(column='avg_code_quality', by='config_name', ax=axes4[1, 0])
    axes4[1, 0].set_title('Average Code Quality')
    axes4[1, 0].set_ylabel('Code Quality (0-100)')
    axes4[1, 0].set_xlabel('')
    axes4[1, 0].tick_params(axis='x', rotation=45)
    plt.sca(axes4[1, 0])
    plt.xticks(rotation=45, ha='right')
    
    # Final Feature Completeness
    df.boxplot(column='final_feature_completeness', by='config_name', ax=axes4[1, 1])
    axes4[1, 1].set_title('Final Feature Completeness')
    axes4[1, 1].set_ylabel('Feature Completeness')
    axes4[1, 1].set_xlabel('')
    axes4[1, 1].tick_params(axis='x', rotation=45)
    plt.sca(axes4[1, 1])
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    fig4_path = os.path.join(OUTPUT_DIR, f"exp8_technical_metrics_{timestamp}.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig4_path}")
    plt.close()
    
    # Figure 5: Market Performance
    fig5, axes5 = plt.subplots(2, 2, figsize=(15, 12))
    fig5.suptitle('Experiment 8: Market Performance Comparison', 
                  fontsize=16, fontweight='bold')
    
    # Final Market Share
    df.boxplot(column='final_market_share', by='config_name', ax=axes5[0, 0])
    axes5[0, 0].set_title('Final Market Share Distribution')
    axes5[0, 0].set_ylabel('Market Share (%)')
    axes5[0, 0].set_xlabel('')
    axes5[0, 0].tick_params(axis='x', rotation=45)
    plt.sca(axes5[0, 0])
    plt.xticks(rotation=45, ha='right')
    
    # Peak Market Share
    peak_data = df.groupby('config_name')['peak_market_share'].mean()
    peak_data.plot(kind='bar', ax=axes5[0, 1], color='teal')
    axes5[0, 1].set_title('Average Peak Market Share')
    axes5[0, 1].set_ylabel('Market Share (%)')
    axes5[0, 1].set_xlabel('')
    axes5[0, 1].tick_params(axis='x', rotation=45)
    axes5[0, 1].grid(True, alpha=0.3)
    
    # Growth Rate
    growth_data = df.groupby('config_name')['growth_rate'].mean()
    growth_data.plot(kind='bar', ax=axes5[1, 0], color='gold')
    axes5[1, 0].set_title('Average Growth Rate')
    axes5[1, 0].set_ylabel('Growth Rate (% per step)')
    axes5[1, 0].set_xlabel('')
    axes5[1, 0].tick_params(axis='x', rotation=45)
    axes5[1, 0].grid(True, alpha=0.3)
    axes5[1, 0].axhline(y=0, color='r', linestyle='-', alpha=0.5)
    
    # Average Revenue
    revenue_data = df.groupby('config_name')['avg_revenue'].mean()
    revenue_data.plot(kind='bar', ax=axes5[1, 1], color='darkgreen')
    axes5[1, 1].set_title('Average Revenue')
    axes5[1, 1].set_ylabel('Revenue')
    axes5[1, 1].set_xlabel('')
    axes5[1, 1].tick_params(axis='x', rotation=45)
    axes5[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig5_path = os.path.join(OUTPUT_DIR, f"exp8_market_performance_{timestamp}.png")
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig5_path}")
    plt.close()
    
    # Figure 6: Team Dynamics
    fig6, axes6 = plt.subplots(1, 2, figsize=(15, 6))
    fig6.suptitle('Experiment 8: Team Dynamics and Morale', 
                  fontsize=16, fontweight='bold')
    
    # Average Team Morale
    morale_data = df.groupby('config_name')['avg_team_morale'].mean()
    morale_data.plot(kind='bar', ax=axes6[0], color='skyblue')
    axes6[0].set_title('Average Team Morale')
    axes6[0].set_ylabel('Team Morale (0-100)')
    axes6[0].set_xlabel('')
    axes6[0].tick_params(axis='x', rotation=45)
    axes6[0].grid(True, alpha=0.3)
    
    # Customer Satisfaction
    satisfaction_data = df.groupby('config_name')['avg_customer_satisfaction'].mean()
    satisfaction_data.plot(kind='bar', ax=axes6[1], color='lightcoral')
    axes6[1].set_title('Average Customer Satisfaction')
    axes6[1].set_ylabel('Customer Satisfaction (0-100)')
    axes6[1].set_xlabel('')
    axes6[1].tick_params(axis='x', rotation=45)
    axes6[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig6_path = os.path.join(OUTPUT_DIR, f"exp8_team_dynamics_{timestamp}.png")
    plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {fig6_path}")
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
    results_path = os.path.join(OUTPUT_DIR, f"exp8_detailed_results_{timestamp}.csv")
    df_results.to_csv(results_path, index=False)
    print(f"  ✓ Saved: {results_path}")
    
    # Save aggregate analysis
    analysis_path = os.path.join(OUTPUT_DIR, f"exp8_aggregate_analysis_{timestamp}.csv")
    analysis.to_csv(analysis_path)
    print(f"  ✓ Saved: {analysis_path}")
    
    # Save success metrics
    success_path = os.path.join(OUTPUT_DIR, f"exp8_success_metrics_{timestamp}.csv")
    success_metrics.to_csv(success_path)
    print(f"  ✓ Saved: {success_path}")
    
    print("  ✓ All results saved successfully")


def main():
    """
    Main execution function for Experiment 8.
    """
    print("\n" + "🔬" * 35)
    print("EXPERIMENT 8: MANAGEMENT FOCUS IMPACT ANALYSIS")
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
    print("✅ EXPERIMENT 8 COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nResults saved in: {OUTPUT_DIR}/")
    print("\nKey Findings:")
    print("- Compare technical vs business tradeoffs")
    print("- Identify optimal management focus strategy")
    print("- Review time series evolution by focus type")
    print("- Analyze quality-speed balance across strategies")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
