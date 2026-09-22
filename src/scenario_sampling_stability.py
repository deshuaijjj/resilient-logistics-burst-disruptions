"""
Scenario Sampling Stability Test

Test how cost changes with different sample sizes: N=100, 500, 1000
Only test burst robust strategy to save time.
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.append(SRC_DIR)

from instance_generator import generate_small_instance, generate_burst_scenarios
from lp_solver import solve_robust_model
import pandas as pd
import numpy as np
import time


def deterministic_even_sample(scenarios, sample_size):
    """Keep nominal scenario and sample the remaining enumeration evenly."""
    if sample_size >= len(scenarios):
        return scenarios
    if sample_size <= 1:
        return [scenarios[0]]
    non_nominal = scenarios[1:]
    indices = np.linspace(0, len(non_nominal) - 1, sample_size - 1, dtype=int)
    return [scenarios[0]] + [non_nominal[i] for i in indices]

def main():
    print("="*80)
    print("SCENARIO SAMPLING STABILITY TEST")
    print("="*80)

    # Generate instance
    params = generate_small_instance()
    print(f"\nInstance: T={params.T}, M={params.M}, J={params.J}, K={params.K}, Gamma={params.Gamma}")

    # Generate all scenarios once
    print("\nGenerating all burst scenarios...")
    all_scenarios = generate_burst_scenarios(params, max_scenarios=None)
    print(f"Total scenarios available: {len(all_scenarios)}")

    # Test different sample sizes
    sample_sizes = [100, 500, 1000]
    results = []

    for N in sample_sizes:
        print(f"\n{'='*60}")
        print(f"Testing N={N} scenarios...")
        print(f"{'='*60}")

        sampled_scenarios = deterministic_even_sample(all_scenarios, N)

        print(f"Solving burst robust with {N} scenarios...")
        start_time = time.time()
        solution = solve_robust_model(params, sampled_scenarios)
        solve_time = time.time() - start_time

        if solution.lp_status != 'optimal':
            print(f"WARNING: LP status = {solution.lp_status}")

        result = {
            'sample_size': N,
            'total_cost_M': solution.total_cost / 1e6,
            'investment_cost_M': solution.investment_cost / 1e6,
            'worst_operating_cost_M': solution.worst_case_cost / 1e6,
            'C_depot_0': solution.C_depot[0] if len(solution.C_depot) > 0 else 0,
            'C_depot_1': solution.C_depot[1] if len(solution.C_depot) > 1 else 0,
            'q_mode_0': solution.q_mode[0] if len(solution.q_mode) > 0 else 0,
            'q_mode_1': solution.q_mode[1] if len(solution.q_mode) > 1 else 0,
            'q_mode_2': solution.q_mode[2] if len(solution.q_mode) > 2 else 0,
            'worst_scenario_id': solution.worst_scenario_id,
            'num_worst_events': len(sampled_scenarios[solution.worst_scenario_id].events),
            'solve_time_s': solve_time,
            'lp_status': solution.lp_status
        }

        results.append(result)

        output_dir = (
            f'{PROJECT_ROOT}/results/04_sampling_stability/'
            f'small_T{params.T}_M{params.M}_J{params.J}_K{params.K}_Gamma{params.Gamma}'
        )
        os.makedirs(output_dir, exist_ok=True)
        sample_label = '_'.join([f'N{n}' for n in sample_sizes])
        output_path = f'{output_dir}/scenario_sample_stability_{sample_label}.csv'
        pd.DataFrame(results).to_csv(output_path, index=False)

        print(f"\nResults for N={N}:")
        print(f"  Total cost: ${result['total_cost_M']:.2f}M")
        print(f"  Investment: ${result['investment_cost_M']:.2f}M")
        print(f"  Worst operating: ${result['worst_operating_cost_M']:.2f}M")
        print(f"  C_depot: [{result['C_depot_0']:.0f}, {result['C_depot_1']:.0f}]")
        print(f"  q_mode: [{result['q_mode_0']:.0f}, {result['q_mode_1']:.0f}, {result['q_mode_2']:.0f}]")
        print(f"  Worst scenario: {solution.worst_scenario_id} ({result['num_worst_events']} events)")
        print(f"  Solve time: {solve_time:.2f}s")

    # Save results
    df = pd.DataFrame(results)
    output_dir = (
        f'{PROJECT_ROOT}/results/04_sampling_stability/'
        f'small_T{params.T}_M{params.M}_J{params.J}_K{params.K}_Gamma{params.Gamma}'
    )
    os.makedirs(output_dir, exist_ok=True)
    sample_label = '_'.join([f'N{n}' for n in sample_sizes])
    output_path = f'{output_dir}/scenario_sample_stability_{sample_label}.csv'
    df.to_csv(output_path, index=False)

    print(f"\n{'='*80}")
    print("STABILITY ANALYSIS")
    print(f"{'='*80}")

    costs = [r['total_cost_M'] for r in results]
    print(f"\nCost progression:")
    for i, r in enumerate(results):
        if i == 0:
            print(f"  N={r['sample_size']:4d}: ${r['total_cost_M']:8.2f}M (baseline)")
        else:
            diff = r['total_cost_M'] - results[0]['total_cost_M']
            pct = 100 * diff / results[0]['total_cost_M']
            print(f"  N={r['sample_size']:4d}: ${r['total_cost_M']:8.2f}M (+${diff:.2f}M, +{pct:.2f}%)")

    print(f"\nCost range: ${min(costs):.2f}M - ${max(costs):.2f}M")
    print(f"Cost std dev: ${np.std(costs):.2f}M")

    print(f"\nCapacity stability:")
    C0_vals = [r['C_depot_0'] for r in results]
    C1_vals = [r['C_depot_1'] for r in results]
    q0_vals = [r['q_mode_0'] for r in results]
    q1_vals = [r['q_mode_1'] for r in results]
    q2_vals = [r['q_mode_2'] for r in results]

    print(f"  C_depot[0]: {set(C0_vals)}")
    print(f"  C_depot[1]: {set(C1_vals)}")
    print(f"  q_mode[0]: {[f'{q:.0f}' for q in q0_vals]}")
    print(f"  q_mode[1]: {[f'{q:.0f}' for q in q1_vals]}")
    print(f"  q_mode[2]: {[f'{q:.0f}' for q in q2_vals]}")

    # Recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")

    cost_variation = (max(costs) - min(costs)) / min(costs) * 100

    if cost_variation < 1.0:
        print(f"Cost variation < 1% → N=500 is sufficient for baseline experiments")
        recommended_N = 500
    elif cost_variation < 3.0:
        print(f"Cost variation < 3% → N=1000 recommended for robustness")
        recommended_N = 1000
    else:
        print(f"Cost variation = {cost_variation:.2f}% → Consider full enumeration or larger N")
        recommended_N = 1000

    print(f"\nRecommended sample size: N={recommended_N}")
    print(f"Saved to: {output_path}")

if __name__ == '__main__':
    main()
