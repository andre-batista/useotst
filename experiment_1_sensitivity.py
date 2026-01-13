"""
Experiment 1: Parameter Sensitivity Analysis

Tests how single parameter variations affect simulation outcomes.
Analyzes the impact of key parameters on market_fit, market_share, and company survival.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import run_simulation, get_default_params
import os
from datetime import datetime

# Configuration
NUM_RUNS_PER_CONFIG = 10  # Number of simulations per parameter value
STEPS = 200
OUTPUT_DIR = "results"

# Parameters to test with their ranges
PARAMETERS_TO_TEST = {
    'base_turnover_rate': [0.5, 1.0, 2.0, 4.0, 8.0],
    'dev_to_features': [0.005, 0.010, 0.015, 0.020, 0.025],
    'hr_talent_boost': [0.1, 0.3, 0.5, 0.7, 1.0],
    'strategic_focus_strength': [0.5, 1.0, 2.0, 3.0, 4.0],
}

def ensure_output_dir():
    """Creates output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def run_sensitivity_analysis():
    """Runs parameter sensitivity analysis."""
    
    print("="*60)
    print("EXPERIMENT 1: PARAMETER SENSITIVITY ANALYSIS")
    print("="*60)
    print(f"Runs per configuration: {NUM_RUNS_PER_CONFIG}")
    print(f"Simulation steps: {STEPS}\n")
    
    all_results = []
    total_configs = sum(len(values) for values in PARAMETERS_TO_TEST.values())
    current_config = 0
    
    for param_name, param_values in PARAMETERS_TO_TEST.items():
        print(f"\n{'='*60}")
        print(f"Testing parameter: {param_name}")
        print(f"{'='*60}")
        
        for param_value in param_values:
            current_config += 1
            print(f"\n[{current_config}/{total_configs}] Testing {param_name} = {param_value}")
            
            # Run multiple simulations with this parameter value
            for run_id in range(NUM_RUNS_PER_CONFIG):
                # Get default parameters
                params = get_default_params()
                
                # Override the parameter being tested
                params[param_name] = param_value
                
                # Run simulation
                try:
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
                    
                    # Extract key metrics
                    final_state = result['final_state']
                    data = result['data']
                    
                    # Calculate additional metrics
                    survived = not result['game_over']
                    avg_market_fit = data['Market Fit'].mean()
                    final_market_fit = final_state['market_fit']
                    final_market_share = final_state['market_share']
                    final_revenue = final_state['revenue']
                    final_tech_debt = final_state['technical_debt']
                    final_bugs = final_state['bug_count']
                    pivots_used = final_state['pivots_used']
                    funding_rounds = final_state['funding_rounds']
                    
                    # Store results
                    all_results.append({
                        'parameter': param_name,
                        'parameter_value': param_value,
                        'run_id': run_id,
                        'survived': survived,
                        'game_over_step': result['game_over_step'] if result['game_over'] else STEPS,
                        'final_market_fit': final_market_fit,
                        'avg_market_fit': avg_market_fit,
                        'final_market_share': final_market_share,
                        'final_revenue': final_revenue,
                        'final_tech_debt': final_tech_debt,
                        'final_bugs': final_bugs,
                        'pivots_used': pivots_used,
                        'funding_rounds': funding_rounds,
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
    csv_path = os.path.join(OUTPUT_DIR, f"experiment_1_raw_data_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Raw data saved to: {csv_path}")
    
    # Calculate summary statistics
    summary = df.groupby(['parameter', 'parameter_value']).agg({
        'survived': ['mean', 'count'],
        'game_over_step': 'mean',
        'final_market_fit': ['mean', 'std'],
        'avg_market_fit': ['mean', 'std'],
        'final_market_share': ['mean', 'std'],
        'final_revenue': ['mean', 'std'],
        'final_tech_debt': ['mean', 'std'],
        'final_bugs': ['mean', 'std'],
        'pivots_used': 'mean',
        'funding_rounds': 'mean'
    }).round(2)
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, f"experiment_1_summary_{timestamp}.csv")
    summary.to_csv(summary_path)
    print(f"✅ Summary statistics saved to: {summary_path}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Experiment 1: Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    for idx, param_name in enumerate(PARAMETERS_TO_TEST.keys()):
        ax = axes[idx // 2, idx % 2]
        
        param_data = df[df['parameter'] == param_name]
        
        # Calculate means for each parameter value
        grouped = param_data.groupby('parameter_value').agg({
            'final_market_fit': 'mean',
            'final_market_share': 'mean',
            'survived': 'mean'
        }).reset_index()
        
        # Create twin axes
        ax2 = ax.twinx()
        
        # Plot market fit and market share
        line1 = ax.plot(grouped['parameter_value'], grouped['final_market_fit'], 
                       'o-', color='blue', linewidth=2, markersize=8, label='Market Fit')
        line2 = ax.plot(grouped['parameter_value'], grouped['final_market_share'], 
                       's-', color='green', linewidth=2, markersize=8, label='Market Share')
        
        # Plot survival rate on secondary axis
        line3 = ax2.plot(grouped['parameter_value'], grouped['survived'] * 100, 
                        '^-', color='red', linewidth=2, markersize=8, label='Survival Rate (%)')
        
        # Styling
        ax.set_xlabel(param_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Market Fit / Market Share', fontsize=10)
        ax2.set_ylabel('Survival Rate (%)', fontsize=10, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax.set_title(f'Impact of {param_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"experiment_1_sensitivity_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {plot_path}")
    
    # Create detailed plot for survival analysis
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
    fig2.suptitle('Experiment 1: Survival and Quality Metrics', fontsize=16, fontweight='bold')
    
    for idx, param_name in enumerate(PARAMETERS_TO_TEST.keys()):
        ax = axes2[idx // 2, idx % 2]
        
        param_data = df[df['parameter'] == param_name]
        
        grouped = param_data.groupby('parameter_value').agg({
            'final_tech_debt': 'mean',
            'final_bugs': 'mean',
            'survived': 'mean'
        }).reset_index()
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(grouped['parameter_value'], grouped['final_tech_debt'], 
                       'o-', color='orange', linewidth=2, markersize=8, label='Tech Debt')
        line2 = ax.plot(grouped['parameter_value'], grouped['final_bugs'], 
                       's-', color='purple', linewidth=2, markersize=8, label='Bugs')
        line3 = ax2.plot(grouped['parameter_value'], grouped['survived'] * 100, 
                        '^-', color='red', linewidth=2, markersize=8, label='Survival Rate (%)')
        
        ax.set_xlabel(param_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Tech Debt / Bugs', fontsize=10)
        ax2.set_ylabel('Survival Rate (%)', fontsize=10, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax.set_title(f'Quality Impact: {param_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best', fontsize=9)
    
    plt.tight_layout()
    plot2_path = os.path.join(OUTPUT_DIR, f"experiment_1_quality_{timestamp}.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"✅ Quality visualization saved to: {plot2_path}")
    
    return summary

def print_key_findings(df):
    """Prints key findings from the analysis."""
    
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    for param_name in PARAMETERS_TO_TEST.keys():
        param_data = df[df['parameter'] == param_name]
        
        print(f"\n📌 {param_name}:")
        
        # Find best and worst parameter values
        summary = param_data.groupby('parameter_value').agg({
            'survived': 'mean',
            'final_market_fit': 'mean',
            'final_market_share': 'mean'
        })
        
        best_value = summary['survived'].idxmax()
        worst_value = summary['survived'].idxmin()
        
        best_stats = summary.loc[best_value]
        worst_stats = summary.loc[worst_value]
        
        print(f"  ✅ Best value: {best_value}")
        print(f"     - Survival rate: {best_stats['survived']*100:.1f}%")
        print(f"     - Avg Market Fit: {best_stats['final_market_fit']:.1f}")
        print(f"     - Avg Market Share: {best_stats['final_market_share']:.1f}%")
        
        print(f"  ❌ Worst value: {worst_value}")
        print(f"     - Survival rate: {worst_stats['survived']*100:.1f}%")
        print(f"     - Avg Market Fit: {worst_stats['final_market_fit']:.1f}")
        print(f"     - Avg Market Share: {worst_stats['final_market_share']:.1f}%")

def main():
    """Main execution function."""
    
    # Run analysis
    df = run_sensitivity_analysis()
    
    # Analyze and visualize
    summary = analyze_and_visualize(df)
    
    # Print key findings
    print_key_findings(df)
    
    print("\n" + "="*60)
    print("✅ EXPERIMENT 1 COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    main()
