"""
Experiment 2: Team Size Optimization

Tests different team size configurations to find the optimal balance.
Focuses on varying engineer count while keeping other teams constant.
Analyzes impact on survival, market share, revenue, and sustainability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import run_simulation, get_default_params
import os
from datetime import datetime

# Configuration
NUM_RUNS_PER_CONFIG = 15  # Number of simulations per team configuration
STEPS = 200
OUTPUT_DIR = "results"

# Team configurations to test
TEAM_CONFIGS = [
    # Varying engineer count (primary focus)
    {'name': 'Small Eng Team', 'eng': 5, 'sales': 5, 'mkt': 3, 'hr': 1, 'mgmt': 1},
    {'name': 'Medium Eng Team', 'eng': 10, 'sales': 5, 'mkt': 3, 'hr': 1, 'mgmt': 1},
    {'name': 'Large Eng Team', 'eng': 15, 'sales': 5, 'mkt': 3, 'hr': 1, 'mgmt': 1},
    {'name': 'XLarge Eng Team', 'eng': 20, 'sales': 5, 'mkt': 3, 'hr': 1, 'mgmt': 1},
    
    # Varying sales team size
    {'name': 'Small Sales Team', 'eng': 10, 'sales': 3, 'mkt': 3, 'hr': 1, 'mgmt': 1},
    {'name': 'Large Sales Team', 'eng': 10, 'sales': 8, 'mkt': 3, 'hr': 1, 'mgmt': 1},
    
    # Varying marketing team size
    {'name': 'Small Mkt Team', 'eng': 10, 'sales': 5, 'mkt': 1, 'hr': 1, 'mgmt': 1},
    {'name': 'Large Mkt Team', 'eng': 10, 'sales': 5, 'mkt': 6, 'hr': 1, 'mgmt': 1},
    
    # Testing different total team sizes with balanced ratios
    {'name': 'Startup (2:1:1)', 'eng': 6, 'sales': 3, 'mkt': 3, 'hr': 1, 'mgmt': 1},
    {'name': 'Scale-up (2:1:1)', 'eng': 20, 'sales': 10, 'mkt': 8, 'hr': 2, 'mgmt': 2},
]

def ensure_output_dir():
    """Creates output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def calculate_total_headcount(config):
    """Calculates total headcount for a team configuration."""
    return config['eng'] + config['sales'] + config['mkt'] + config['hr'] + config['mgmt']

def calculate_burn_rate_estimate(config):
    """Estimates monthly burn rate based on team size (rough estimate: $10k/person)."""
    return calculate_total_headcount(config) * 2.0  # Matches model's initial burn_rate calculation

def run_team_size_analysis():
    """Runs team size optimization analysis."""
    
    print("="*60)
    print("EXPERIMENT 2: TEAM SIZE OPTIMIZATION")
    print("="*60)
    print(f"Runs per configuration: {NUM_RUNS_PER_CONFIG}")
    print(f"Simulation steps: {STEPS}\n")
    
    all_results = []
    total_configs = len(TEAM_CONFIGS)
    
    for idx, config in enumerate(TEAM_CONFIGS):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{total_configs}] Testing: {config['name']}")
        print(f"  Team: {config['eng']} Eng, {config['sales']} Sales, {config['mkt']} Mkt, {config['hr']} HR, {config['mgmt']} Mgmt")
        print(f"  Total headcount: {calculate_total_headcount(config)}")
        print(f"  Estimated burn rate: ${calculate_burn_rate_estimate(config):.0f}k/month")
        print(f"{'='*60}")
        
        # Run multiple simulations with this configuration
        for run_id in range(NUM_RUNS_PER_CONFIG):
            params = get_default_params()
            
            try:
                result = run_simulation(
                    num_engineers=config['eng'],
                    num_sales=config['sales'],
                    num_marketing=config['mkt'],
                    num_hr=config['hr'],
                    num_mgmt=config['mgmt'],
                    steps=STEPS,
                    params=params,
                    verbose=False
                )
                
                # Extract metrics
                final_state = result['final_state']
                data = result['data']
                
                # Calculate additional metrics
                survived = not result['game_over']
                survival_duration = result['game_over_step'] if result['game_over'] else STEPS
                
                # Time to reach milestones
                market_fit_50_step = None
                market_share_5_step = None
                
                for step in range(len(data)):
                    if market_fit_50_step is None and data.iloc[step]['Market Fit'] >= 50:
                        market_fit_50_step = step
                    if market_share_5_step is None and data.iloc[step]['Market Share'] >= 5:
                        market_share_5_step = step
                
                # Average metrics over time
                avg_market_fit = data['Market Fit'].mean()
                avg_revenue = data['Revenue'].mean()
                avg_team_morale = data['Team Morale'].mean()
                max_market_share = data['Market Share'].max()
                
                # Calculate efficiency metrics
                total_headcount = calculate_total_headcount(config)
                revenue_per_employee = final_state['revenue'] / total_headcount if total_headcount > 0 else 0
                market_share_per_employee = final_state['market_share'] / total_headcount if total_headcount > 0 else 0
                
                # Store results
                all_results.append({
                    'config_name': config['name'],
                    'num_engineers': config['eng'],
                    'num_sales': config['sales'],
                    'num_marketing': config['mkt'],
                    'num_hr': config['hr'],
                    'num_mgmt': config['mgmt'],
                    'total_headcount': total_headcount,
                    'run_id': run_id,
                    'survived': survived,
                    'survival_duration': survival_duration,
                    'final_market_fit': final_state['market_fit'],
                    'final_market_share': final_state['market_share'],
                    'final_revenue': final_state['revenue'],
                    'final_cash_runway': final_state['cash_runway'],
                    'final_tech_debt': final_state['technical_debt'],
                    'final_bugs': final_state['bug_count'],
                    'avg_market_fit': avg_market_fit,
                    'avg_revenue': avg_revenue,
                    'avg_team_morale': avg_team_morale,
                    'max_market_share': max_market_share,
                    'pivots_used': final_state['pivots_used'],
                    'funding_rounds': final_state['funding_rounds'],
                    'market_fit_50_step': market_fit_50_step if market_fit_50_step is not None else STEPS,
                    'market_share_5_step': market_share_5_step if market_share_5_step is not None else STEPS,
                    'revenue_per_employee': revenue_per_employee,
                    'market_share_per_employee': market_share_per_employee,
                })
                
                if (run_id + 1) % 5 == 0:
                    print(f"  Completed {run_id + 1}/{NUM_RUNS_PER_CONFIG} runs")
                    
            except Exception as e:
                print(f"  Error in run {run_id}: {str(e)}")
                continue
    
    return pd.DataFrame(all_results)

def analyze_and_visualize(df):
    """Analyzes results and creates visualizations."""
    
    ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw data
    csv_path = os.path.join(OUTPUT_DIR, f"experiment_2_raw_data_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Raw data saved to: {csv_path}")
    
    # Calculate summary statistics
    summary = df.groupby('config_name').agg({
        'survived': ['mean', 'count'],
        'survival_duration': 'mean',
        'final_market_fit': ['mean', 'std'],
        'final_market_share': ['mean', 'std'],
        'final_revenue': ['mean', 'std'],
        'final_cash_runway': ['mean', 'std'],
        'final_tech_debt': 'mean',
        'final_bugs': 'mean',
        'avg_team_morale': 'mean',
        'max_market_share': 'mean',
        'pivots_used': 'mean',
        'funding_rounds': 'mean',
        'market_fit_50_step': 'mean',
        'market_share_5_step': 'mean',
        'revenue_per_employee': 'mean',
        'market_share_per_employee': 'mean',
        'total_headcount': 'first',
        'num_engineers': 'first',
    }).round(2)
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, f"experiment_2_summary_{timestamp}.csv")
    summary.to_csv(summary_path)
    print(f"✅ Summary statistics saved to: {summary_path}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    # Figure 1: Engineer Count Impact
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('Experiment 2: Impact of Engineer Count', fontsize=16, fontweight='bold')
    
    # Filter only engineer-varying configs
    eng_configs = df[df['config_name'].str.contains('Eng Team')].copy()
    eng_summary = eng_configs.groupby('num_engineers').agg({
        'survived': 'mean',
        'final_market_fit': 'mean',
        'final_market_share': 'mean',
        'final_revenue': 'mean',
        'avg_team_morale': 'mean',
        'revenue_per_employee': 'mean',
    }).reset_index()
    
    # Plot 1: Survival & Market Metrics
    ax = axes1[0, 0]
    ax2 = ax.twinx()
    line1 = ax.plot(eng_summary['num_engineers'], eng_summary['final_market_fit'], 
                   'o-', color='blue', linewidth=2, markersize=10, label='Market Fit')
    line2 = ax.plot(eng_summary['num_engineers'], eng_summary['final_market_share'], 
                   's-', color='green', linewidth=2, markersize=10, label='Market Share')
    line3 = ax2.plot(eng_summary['num_engineers'], eng_summary['survived'] * 100, 
                    '^-', color='red', linewidth=2, markersize=10, label='Survival Rate (%)')
    ax.set_xlabel('Number of Engineers', fontsize=12, fontweight='bold')
    ax.set_ylabel('Market Fit / Market Share', fontsize=11)
    ax2.set_ylabel('Survival Rate (%)', fontsize=11, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.set_title('Survival and Market Performance', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best')
    
    # Plot 2: Revenue
    ax = axes1[0, 1]
    ax.plot(eng_summary['num_engineers'], eng_summary['final_revenue'], 
           'o-', color='purple', linewidth=2, markersize=10)
    ax.set_xlabel('Number of Engineers', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Revenue', fontsize=11)
    ax.set_title('Revenue by Team Size', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Efficiency (Revenue per Employee)
    ax = axes1[1, 0]
    ax.plot(eng_summary['num_engineers'], eng_summary['revenue_per_employee'], 
           'd-', color='orange', linewidth=2, markersize=10)
    ax.set_xlabel('Number of Engineers', fontsize=12, fontweight='bold')
    ax.set_ylabel('Revenue per Employee', fontsize=11)
    ax.set_title('Team Efficiency', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Team Morale
    ax = axes1[1, 1]
    ax.plot(eng_summary['num_engineers'], eng_summary['avg_team_morale'], 
           'h-', color='teal', linewidth=2, markersize=10)
    ax.set_xlabel('Number of Engineers', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Team Morale', fontsize=11)
    ax.set_title('Team Morale by Size', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot1_path = os.path.join(OUTPUT_DIR, f"experiment_2_engineer_impact_{timestamp}.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"✅ Engineer impact visualization saved to: {plot1_path}")
    
    # Figure 2: All Team Configurations Comparison
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 12))
    fig2.suptitle('Experiment 2: All Team Configurations Comparison', fontsize=16, fontweight='bold')
    
    config_summary = df.groupby('config_name').agg({
        'survived': 'mean',
        'final_market_share': 'mean',
        'final_revenue': 'mean',
        'revenue_per_employee': 'mean',
        'total_headcount': 'first',
    }).reset_index()
    
    # Sort by survival rate
    config_summary = config_summary.sort_values('survived', ascending=False)
    
    # Plot 1: Survival Rates
    ax = axes2[0, 0]
    bars = ax.barh(config_summary['config_name'], config_summary['survived'] * 100, color='steelblue')
    ax.set_xlabel('Survival Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Survival Rate by Configuration', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
               ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Plot 2: Market Share
    ax = axes2[0, 1]
    ax.barh(config_summary['config_name'], config_summary['final_market_share'], color='green', alpha=0.7)
    ax.set_xlabel('Average Final Market Share (%)', fontsize=12, fontweight='bold')
    ax.set_title('Market Share by Configuration', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Revenue
    ax = axes2[1, 0]
    ax.barh(config_summary['config_name'], config_summary['final_revenue'], color='purple', alpha=0.7)
    ax.set_xlabel('Average Final Revenue', fontsize=12, fontweight='bold')
    ax.set_title('Revenue by Configuration', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Efficiency (scatter: headcount vs revenue per employee)
    ax = axes2[1, 1]
    scatter = ax.scatter(config_summary['total_headcount'], 
                        config_summary['revenue_per_employee'],
                        s=200, c=config_summary['survived']*100, 
                        cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=1.5)
    ax.set_xlabel('Total Headcount', fontsize=12, fontweight='bold')
    ax.set_ylabel('Revenue per Employee', fontsize=12, fontweight='bold')
    ax.set_title('Efficiency: Headcount vs Revenue per Employee', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Survival Rate (%)', fontsize=10)
    
    plt.tight_layout()
    plot2_path = os.path.join(OUTPUT_DIR, f"experiment_2_all_configs_{timestamp}.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"✅ All configurations visualization saved to: {plot2_path}")
    
    return summary

def print_key_findings(df):
    """Prints key findings from the analysis."""
    
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    # Overall best configuration
    config_summary = df.groupby('config_name').agg({
        'survived': 'mean',
        'final_market_share': 'mean',
        'final_revenue': 'mean',
        'revenue_per_employee': 'mean',
        'total_headcount': 'first',
        'num_engineers': 'first',
        'num_sales': 'first',
        'num_marketing': 'first',
    })
    
    # Best for survival
    best_survival = config_summary['survived'].idxmax()
    print(f"\n🏆 Best for Survival: {best_survival}")
    print(f"   Survival rate: {config_summary.loc[best_survival, 'survived']*100:.1f}%")
    print(f"   Team size: {config_summary.loc[best_survival, 'total_headcount']:.0f} people")
    
    # Best for market share
    best_market = config_summary['final_market_share'].idxmax()
    print(f"\n📈 Best for Market Share: {best_market}")
    print(f"   Avg market share: {config_summary.loc[best_market, 'final_market_share']:.1f}%")
    print(f"   Survival rate: {config_summary.loc[best_market, 'survived']*100:.1f}%")
    
    # Best for revenue
    best_revenue = config_summary['final_revenue'].idxmax()
    print(f"\n💰 Best for Revenue: {best_revenue}")
    print(f"   Avg revenue: {config_summary.loc[best_revenue, 'final_revenue']:.1f}")
    print(f"   Survival rate: {config_summary.loc[best_revenue, 'survived']*100:.1f}%")
    
    # Most efficient (revenue per employee)
    best_efficiency = config_summary['revenue_per_employee'].idxmax()
    print(f"\n⚡ Most Efficient: {best_efficiency}")
    print(f"   Revenue per employee: {config_summary.loc[best_efficiency, 'revenue_per_employee']:.2f}")
    print(f"   Team size: {config_summary.loc[best_efficiency, 'total_headcount']:.0f} people")
    
    # Engineer count analysis
    print(f"\n👨‍💻 Engineer Count Analysis:")
    eng_configs = df[df['config_name'].str.contains('Eng Team')]
    if not eng_configs.empty:
        eng_summary = eng_configs.groupby('num_engineers').agg({
            'survived': 'mean',
            'final_market_share': 'mean',
            'revenue_per_employee': 'mean'
        })
        
        print(f"   5 engineers: {eng_summary.loc[5, 'survived']*100:.1f}% survival, "
              f"{eng_summary.loc[5, 'final_market_share']:.1f}% market share")
        print(f"   10 engineers: {eng_summary.loc[10, 'survived']*100:.1f}% survival, "
              f"{eng_summary.loc[10, 'final_market_share']:.1f}% market share")
        print(f"   15 engineers: {eng_summary.loc[15, 'survived']*100:.1f}% survival, "
              f"{eng_summary.loc[15, 'final_market_share']:.1f}% market share")
        print(f"   20 engineers: {eng_summary.loc[20, 'survived']*100:.1f}% survival, "
              f"{eng_summary.loc[20, 'final_market_share']:.1f}% market share")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if config_summary.loc[best_survival, 'survived'] == 1.0:
        print(f"   ✓ For maximum survival: Use {best_survival}")
    else:
        print(f"   ⚠ No configuration achieved 100% survival")
    
    print(f"   ✓ For market dominance: Use {best_market}")
    print(f"   ✓ For efficiency: Use {best_efficiency}")

def main():
    """Main execution function."""
    
    # Run analysis
    df = run_team_size_analysis()
    
    # Analyze and visualize
    summary = analyze_and_visualize(df)
    
    # Print key findings
    print_key_findings(df)
    
    print("\n" + "="*60)
    print("✅ EXPERIMENT 2 COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    main()
