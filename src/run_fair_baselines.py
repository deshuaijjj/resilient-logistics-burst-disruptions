"""
Run fair baseline experiments for burst-resilient logistics design.

These baselines avoid using the burst optimum itself as the fixed value for a
single-lever strategy. Results are saved after every strategy so interrupted
runs remain auditable.
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


def parse_shares(value: str, expected_len: int) -> np.ndarray:
    shares = np.array([float(x.strip()) for x in value.split(',')])
    if len(shares) != expected_len:
        raise ValueError(f'Expected {expected_len} q shares, got {len(shares)}')
    if np.any(shares < 0) or np.sum(shares) <= 0:
        raise ValueError('q shares must be nonnegative and have positive sum')
    return shares / np.sum(shares)


def first_stage_cost(params, C: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(params.f_depot * C) + np.sum(params.v_mode * q * params.T))


def make_balanced_q(params, q_margin: float, q_shares: np.ndarray) -> np.ndarray:
    average_monthly_demand = float(np.sum(params.D_demand) / params.T)
    return average_monthly_demand * q_margin * q_shares


def make_budget_matched_C(params, fixed_q: np.ndarray, target_budget: float,
                          reference_C: np.ndarray) -> np.ndarray:
    q_cost = float(np.sum(params.v_mode * fixed_q * params.T))
    c_budget = max(0.0, target_budget - q_cost)
    if np.sum(reference_C) <= 0:
        proportions = np.ones(params.J) / params.J
    else:
        proportions = reference_C / np.sum(reference_C)
    cost_per_scale = float(np.sum(params.f_depot * proportions))
    if cost_per_scale <= 0:
        return np.zeros(params.J)
    return proportions * (c_budget / cost_per_scale)


def make_budget_matched_q(params, fixed_C: np.ndarray, target_budget: float,
                          q_shares: np.ndarray) -> np.ndarray:
    c_cost = float(np.sum(params.f_depot * fixed_C))
    q_budget = max(0.0, target_budget - c_cost)
    cost_per_scale = float(np.sum(params.v_mode * q_shares * params.T))
    if cost_per_scale <= 0:
        return np.zeros(params.M)
    return q_shares * (q_budget / cost_per_scale)


def solution_to_row(name: str, sol: Solution, definition: str) -> Dict:
    return {
        'Strategy': name,
        'Definition': definition,
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


def save_progress(output_dir: str, results: Dict[str, Solution],
                  definitions: Dict[str, str], metadata: Dict):
    os.makedirs(output_dir, exist_ok=True)
    rows = [solution_to_row(name, sol, definitions[name]) for name, sol in results.items()]
    pd.DataFrame(rows).to_csv(f'{output_dir}/summary.csv', index=False)

    detailed = {
        'metadata': metadata,
        'definitions': definitions,
        'results': {name: solution_to_json(sol) for name, sol in results.items()},
    }
    with open(f'{output_dir}/detailed_results.json', 'w') as f:
        json.dump(detailed, f, indent=2)


def plot_progress(output_dir: str, results: Dict[str, Solution]):
    if not results:
        return
    figures_dir = f'{output_dir}/figures'
    os.makedirs(figures_dir, exist_ok=True)

    strategies = list(results.keys())
    x = np.arange(len(strategies))

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.5))
    total_costs = [results[s].total_cost / 1e6 for s in strategies]
    ax.bar(x, total_costs, color='steelblue', edgecolor='black')
    ax.set_ylabel('Total cost ($M)')
    ax.set_title('Fair Baseline Cost Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in strategies], fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/fair_baseline_costs.png', dpi=160)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.5))
    C_totals = [np.sum(results[s].C_depot) for s in strategies]
    q_totals = [np.sum(results[s].q_mode) for s in strategies]
    width = 0.38
    ax.bar(x - width / 2, C_totals, width, label='Total C')
    ax.bar(x + width / 2, q_totals, width, label='Total q')
    ax.set_ylabel('Capacity')
    ax.set_title('Fair Baseline Capacity Decisions')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in strategies], fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/fair_baseline_capacities.png', dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=500)
    parser.add_argument('--q-margin', type=float, default=2.0)
    parser.add_argument('--q-shares', default='0.5,0.3,0.2')
    parser.add_argument(
        '--output-root',
        default=os.path.join(PROJECT_ROOT, 'results', '06_fair_baselines'),
    )
    args = parser.parse_args()

    params = generate_small_instance()
    q_shares = parse_shares(args.q_shares, params.M)

    scale_name = (
        f'small_T{params.T}_M{params.M}_J{params.J}_K{params.K}_'
        f'Gamma{params.Gamma}_N{args.sample_size}_sampled'
    )
    output_dir = f'{args.output_root}/{scale_name}'
    os.makedirs(f'{output_dir}/logs', exist_ok=True)

    print('=' * 80)
    print(f'FAIR BASELINE EXPERIMENTS ({scale_name})')
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
        'q_margin': args.q_margin,
        'q_shares': q_shares.tolist(),
    }

    results: Dict[str, Solution] = {}
    definitions: Dict[str, str] = {}

    def run_and_save(name: str, definition: str,
                     scenario_list: List[DisruptionScenario], **kwargs):
        print(f'\n[{len(results) + 1}] {name}')
        print(definition)
        start = time.time()
        sol = solve_robust_model(params, scenario_list, **kwargs)
        print(f'{name} wall time: {time.time() - start:.2f}s')
        results[name] = sol
        definitions[name] = definition
        save_progress(output_dir, results, definitions, metadata)
        plot_progress(output_dir, results)
        print(f'Saved progress to {output_dir}')

    run_and_save(
        'nominal',
        'No disruptions; optimize C and q.',
        [DisruptionScenario([])],
    )
    run_and_save(
        'independent',
        'Independent single-period disruptions; optimize C and q.',
        ind_scenarios,
    )
    run_and_save(
        'burst_joint',
        'Burst disruptions; optimize C and q.',
        burst_scenarios,
    )

    nominal = results['nominal']
    burst_joint = results['burst_joint']
    balanced_q = make_balanced_q(params, args.q_margin, q_shares)

    run_and_save(
        'inventory_only_nominal_q',
        'Burst disruptions; fix q at nominal solution and optimize C.',
        burst_scenarios,
        fixed_q=nominal.q_mode.copy(),
    )
    run_and_save(
        'inventory_only_balanced_q',
        (
            'Burst disruptions; fix q at exogenous balanced multi-mode portfolio '
            f'(margin={args.q_margin}, shares={q_shares.tolist()}) and optimize C.'
        ),
        burst_scenarios,
        fixed_q=balanced_q,
    )
    run_and_save(
        'redundancy_only_nominal_C',
        'Burst disruptions; fix C at nominal solution and optimize q.',
        burst_scenarios,
        fixed_C=nominal.C_depot.copy(),
    )

    budget_C = make_budget_matched_C(
        params,
        fixed_q=nominal.q_mode.copy(),
        target_budget=burst_joint.investment_cost,
        reference_C=nominal.C_depot.copy(),
    )
    budget_q = make_budget_matched_q(
        params,
        fixed_C=nominal.C_depot.copy(),
        target_budget=burst_joint.investment_cost,
        q_shares=q_shares,
    )

    metadata['budget_matched_inventory_fixed_C'] = budget_C.tolist()
    metadata['budget_matched_inventory_fixed_q'] = nominal.q_mode.tolist()
    metadata['budget_matched_redundancy_fixed_C'] = nominal.C_depot.tolist()
    metadata['budget_matched_redundancy_fixed_q'] = budget_q.tolist()
    metadata['burst_joint_investment_cost'] = float(burst_joint.investment_cost)
    metadata['budget_matched_inventory_investment_cost'] = first_stage_cost(
        params, budget_C, nominal.q_mode.copy()
    )
    metadata['budget_matched_redundancy_investment_cost'] = first_stage_cost(
        params, nominal.C_depot.copy(), budget_q
    )

    run_and_save(
        'budget_matched_inventory',
        (
            'Burst disruptions; fixed q=nominal and fixed C scaled in nominal '
            'proportions to match burst_joint investment cost.'
        ),
        burst_scenarios,
        fixed_C=budget_C,
        fixed_q=nominal.q_mode.copy(),
    )
    run_and_save(
        'budget_matched_redundancy',
        (
            'Burst disruptions; fixed C=nominal and fixed q allocated by exogenous '
            'shares to match burst_joint investment cost.'
        ),
        burst_scenarios,
        fixed_C=nominal.C_depot.copy(),
        fixed_q=budget_q,
    )

    print('\nDONE')


if __name__ == '__main__':
    main()
