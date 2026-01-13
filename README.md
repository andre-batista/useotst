# Understanding Software Engineering Organizations Through Systems Thinking

An agent-based simulation model for studying software engineering organizations dynamics, team interactions, and strategic decision-making. This research project uses Mesa framework to simulate how different organizational factors affect organization success and survival.

## Overview

This model simulates a software company with multiple teams (Engineering, Sales, Marketing, HR, and Management) interacting to develop a product and capture market share. The simulation tracks key metrics like market fit, revenue, technical debt, and organizational dynamics to understand what factors contribute to startup success.

### Key Features

- **Multi-Agent Teams**: Simulates five distinct teams with unique capabilities and interactions
- **Product Evolution**: Tracks feature completeness, code quality, technical debt, and market fit
- **Financial Modeling**: Includes cash runway, burn rate, revenue generation, and funding dynamics
- **Strategic Management**: Management agents make strategic decisions based on company state
- **Market Dynamics**: Simulates market share growth, brand awareness, and customer acquisition
- **Crisis Management**: Includes pivot mechanisms and emergency response strategies

## Model Components

### Agent Types

1. **EngineerAgent**: Develops features and maintains code quality
   - Development capacity
   - Explanation capacity (documentation/knowledge sharing)

2. **SalesAgent**: Converts leads into revenue
   - Selling capacity
   - Understanding capacity (product knowledge)

3. **MarketingAgent**: Generates leads and improves market fit
   - Improvement ideas capacity
   - Publicize capacity (brand awareness)

4. **HumanResourcesAgent**: Coordinates teams and manages talent
   - Coordination capacity
   - Talent development
   - Conflict resolution
   - Hiring quality

5. **ManagementAgent**: Strategic planning and crisis management
   - Management capacity
   - Strategic vision
   - Crisis management
   - Focus areas (engineering, sales, marketing, quality, balanced)

### Product Attributes

- Feature completeness
- Code quality
- Technical debt
- Bug count
- Market fit
- Market share
- Brand awareness
- Lead quality
- Organizational alignment
- Organizational conflict

## Experiments

The repository includes eight experimental studies:

1. **[experiment_1_sensitivity.py](experiment_1_sensitivity.py)**: Parameter sensitivity analysis
   - Tests impact of turnover rate, development speed, HR effectiveness, and strategic focus

2. **[experiment_2_team_size.py](experiment_2_team_size.py)**: Team size optimization
   - Analyzes optimal team configurations and headcount balances

3. **[experiment_3_unbalanced_teams.py](experiment_3_unbalanced_teams.py)**: Unbalanced team compositions
   - Studies effects of skewed team ratios

4. **[experiment_4_quality_vs_speed.py](experiment_4_quality_vs_speed.py)**: Quality vs speed trade-offs
   - Explores different development strategies

5. **[experiment_5_initial_capital.py](experiment_5_initial_capital.py)**: Initial capital impact
   - Tests how starting funding affects survival and growth

6. **[experiment_6_burn_rate.py](experiment_6_burn_rate.py)**: Burn rate management
   - Analyzes financial efficiency strategies

7. **[experiment_7_pivot_timing.py](experiment_7_pivot_timing.py)**: Pivot timing strategies
   - Studies when and how pivots affect outcomes

8. **[experiment_8_management_focus.py](experiment_8_management_focus.py)**: Management focus strategies
   - Compares different strategic focus approaches

## Installation

### Requirements

- Python 3.7+
- Dependencies listed in [requirements.txt](requirements.txt)

### Setup

```bash
# Clone the repository
git clone https://github.com/andre-batista/useotst.git
cd useotst

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `numpy`: Numerical computations
- `mesa`: Agent-based modeling framework
- `pandas`: Data analysis and manipulation
- `matplotlib`: Visualization

## Usage

### Running the Model

```python
from model import run_simulation, get_default_params

# Run with default parameters
result = run_simulation(
    num_engineers=10,
    num_sales=5,
    num_marketing=3,
    num_hr=1,
    num_mgmt=1,
    steps=200,
    verbose=True
)

# Access results
print(f"Survived: {not result['game_over']}")
print(f"Final market share: {result['final_state']['market_share']:.2f}%")
```

### Running Experiments

Each experiment can be run independently:

```bash
# Run parameter sensitivity analysis
python experiment_1_sensitivity.py

# Run team size optimization
python experiment_2_team_size.py

# Run other experiments...
python experiment_3_unbalanced_teams.py
```

Results are saved to the `results/` directory as CSV files with timestamped filenames.

### Customizing Parameters

```python
from model import get_default_params

params = get_default_params()

# Modify parameters
params['base_turnover_rate'] = 2.0
params['dev_to_features'] = 0.015
params['hr_talent_boost'] = 0.5

# Run with custom parameters
result = run_simulation(params=params)
```

## Model Parameters

Key adjustable parameters include:

- **Turnover**: `base_turnover_rate` - Employee turnover percentage
- **Development**: `dev_to_features` - Feature development rate
- **Quality**: `quality_to_code_quality` - Code quality improvement rate
- **Debt**: `debt_generation_rate` - Technical debt accumulation
- **Marketing**: `mkt_to_market_fit` - Market fit improvement rate
- **Sales**: `sales_to_revenue` - Revenue conversion efficiency
- **HR**: `hr_talent_boost` - Talent development effectiveness
- **Management**: `strategic_focus_strength` - Strategic decision impact

## Output and Results

### Data Collection

The model collects time-series data for:
- Product metrics (features, quality, debt, bugs, market fit)
- Market metrics (market share, brand awareness, revenue)
- Organizational metrics (alignment, conflict, team morale)
- Financial metrics (cash runway, burn rate, funding rounds)
- Team capacities and headcount

### Result Files

Each experiment generates:
- **Summary statistics**: Aggregate metrics across runs
- **Time-series data**: Step-by-step evolution of key variables
- **Detailed results**: Per-run outcomes and configurations
- **Aggregate analysis**: Statistical summaries and comparisons

## Research Applications

This model is suitable for studying:
- Startup team composition and sizing
- Technical debt management strategies
- Product-market fit dynamics
- Cash runway and funding strategies
- Strategic pivot timing
- Organizational coordination challenges
- Growth vs quality trade-offs

## Manuscript

A research manuscript is available in the [manuscript/](manuscript/) directory, written in LaTeX using the Wiley article class.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Time Unit

By default, each simulation step represents **one month**. This can be configured in [model.py](model.py) by changing the `TIME_UNIT` constant.

## Contributing

This is a research project. For questions or collaboration inquiries, please open an issue.

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{alves2026understanding,
  title = {Understanding Software Engineering Organizations Through Systems Thinking},
  author = {Juliana Alves, Henrique Alves, Andr\'e Batista},
  year = {2026},
  url = {https://github.com/andre-batista/useotst}
}
```

## Acknowledgments

Built using the [Mesa](https://mesa.readthedocs.io/) agent-based modeling framework.
