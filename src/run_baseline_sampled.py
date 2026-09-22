"""
Run corrected baseline experiments with deterministic sampled burst scenarios.

The script writes results after each strategy, so partial runs remain usable if a
large LP is interrupted.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.append(SRC_DIR)

from instance_generator import (  # noqa: E402
    DisruptionScenario,
    generate_burst_scenarios,
    generate_independent_scenarios,
    generate_small_instance,
)
from lp_solver import Solution, solve_robust_model  # noqa: E402


def deterministic_even_sample(scenarios: List[DisruptionScenario],
                              sample_size: int) -> List[DisruptionScenario]:
    """Keep nominal scenario and sample the remaining enumeration evenly."""
    if sample_size >= len(scenarios):
        return scenarios
    if sample_size <= 1:
        return [scenarios[0]]

    non_nominal = scenarios[1:]
    indices = np.linspace(0, len(non_nominal) - 1, sample_size - 1, dtype=int)
    return [scenarios[0]] + [non_nominal[i] for i in indices]


def solution_to_row(name: str, sol: Solution) -> Dict:
    return {
        'Strategy': name,
        'Total Cost ($M)': sol.total_cost / 1e6,
        'Investment ($M)': sol.investment_cost / 1e6,
        'Worst Operating ($M)': sol.worst_case_cost / 1e6,
        'Depot Capacity (tons)': ' / '.join([f'{c:.0f}' for c in sol.C_depot]),
        'Mode Capacity (t/mo)': ' / '.join([f'{q:.0f}' for q in sol.q_mode]),
        'Worst Scenario': f'{len(sol.worst_scenario_events)} events',
        'Num Scenarios': sol.num_scenarios,
        'Solve Time (s)': round(sol.solve_time, 2),
        'LP Status': sol.lp_status,
    }


def solution_to_json(sol: Solution) -> Dict:
    return {
        'total_cost': float(sol.total_cost),
        'investment_cost': float(sol.investment_cost),
        'worst_case_cost': float(sol.worst_case_cost),
        'C_depot': sol.C_depot.tolist(),
        'q_mode': sol.q_mode.tolist(),
        'worst_scenario_id': int(sol.worst_scenario_id),
        'worst_scenario_events': sol.worst_scenario_events,
        'solve_time': float(sol.solve_time),
        'num_scenarios': int(sol.num_scenarios),
        'lp_status': sol.lp_status,
    }


def save_progress(output_dir: str, results: Dict[str, Solution], metadata: Dict):
    os.makedirs(output_dir, exist_ok=True)
    rows = [solution_to_row(name, sol) for name, sol in results.items()]
    pd.DataFrame(rows).to_csv(f'{output_dir}/summary.csv', index=False)

    detailed = {
        'metadata': metadata,
        'results': {name: solution_to_json(sol) for name, sol in results.items()},
    }
    with open(f'{output_dir}/detailed_results.json', 'w') as f:
        json.dump(detailed, f, indent=2)


def plot_results(output_dir: str, results: Dict[str, Solution]):
    if not results:
        return
    figures_dir = f'{output_dir}/figures'
    os.makedirs(figures_dir, exist_ok=True)

    strategies = list(results.keys())
    costs = [results[s].total_cost / 1e6 for s in strategies]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(strategies, costs, color='steelblue', edgecolor='black')
    ax.set_ylabel('Total Cost ($M)')
    ax.set_title('Baseline Cost Comparison')
    ax.tick_params(axis='x', rotation=35)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/baseline_cost_comparison.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=500)
    parser.add_argument(
        '--output-root',
        default=os.path.join(PROJECT_ROOT, 'results', '02_baselines'),
    )
    args = parser.parse_args()

    params = generate_small_instance()
    scale_name = (
        f'small_T{params.T}_M{params.M}_J{params.J}_K{params.K}_'
        f'Gamma{params.Gamma}_N{args.sample_size}_sampled'
    )
    output_dir = f'{args.output_root}/{scale_name}'
    os.makedirs(f'{output_dir}/logs', exist_ok=True)

    print('=' * 80)
    print(f'CORRECTED BASELINE EXPERIMENTS ({scale_name})')
    print('=' * 80)

    all_burst_scenarios = generate_burst_scenarios(params, max_scenarios=None)
    burst_scenarios = deterministic_even_sample(all_burst_scenarios, args.sample_size)
    ind_scenarios = generate_independent_scenarios(params, Gamma_total=params.Gamma * params.M)

    metadata = {
        'scale_name': scale_name,
        'sample_size': args.sample_size,
        'sample_method': 'deterministic_even_sample_with_nominal',
        'total_burst_scenarios_available': len(all_burst_scenarios),
        'burst_scenarios_used': len(burst_scenarios),
        'independent_scenarios_used': len(ind_scenarios),
    }

    results: Dict[str, Solution] = {}

    def run_and_save(name: str, scenario_list: List[DisruptionScenario], **kwargs):
        print(f'\n[{len(results) + 1}] {name}')
        start = time.time()
        sol = solve_robust_model(params, scenario_list, **kwargs)
        print(f'{name} wall time: {time.time() - start:.2f}s')
        results[name] = sol
        save_progress(output_dir, results, metadata)
        plot_results(output_dir, results)
        print(f'Saved progress to {output_dir}')

    run_and_save('nominal', [DisruptionScenario([])])
    run_and_save('independent', ind_scenarios)
    run_and_save('burst', burst_scenarios)

    fixed_q = results['burst'].q_mode.copy()
    print(f'\nInventory-only fixed q_mode from burst: {fixed_q}')
    run_and_save('inventory_only', burst_scenarios, fixed_q=fixed_q)

    fixed_C = results['burst'].C_depot.copy()
    print(f'\nRedundancy-only fixed C_depot from burst: {fixed_C}')
    run_and_save('redundancy_only', burst_scenarios, fixed_C=fixed_C)

    run_and_save('joint', burst_scenarios)

    diff = abs(results['joint'].total_cost - results['burst'].total_cost)
    print(f'\nJoint vs burst absolute difference: {diff:.6f}')
    print('DONE')


if __name__ == '__main__':
    main()
