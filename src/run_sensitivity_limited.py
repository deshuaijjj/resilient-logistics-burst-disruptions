"""
Sensitivity Analysis with Limited Scenarios (N=300-500 per test)

Tests:
1. Gamma sensitivity: {0, 1, 2, 3}
2. R_max sensitivity: {0.5, 0.75, 1.0, 1.25, 1.5} × baseline
3. s_initial sensitivity: {0, 0.5, 1.0, 1.5} × baseline
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.append(SRC_DIR)

from instance_generator import (
    InstanceParameters,
    generate_small_instance,
    generate_burst_scenarios
)
from lp_solver import solve_robust_model
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

def sensitivity_gamma(base_params, gamma_values, max_scenarios=400, output_path=None):
    """Test cost monotonicity with Gamma"""
    print("\n" + "="*80)
    print("SENSITIVITY: GAMMA (Disruption Budget)")
    print("="*80)

    results = []
    for gamma in gamma_values:
        print(f"\n--- Gamma = {gamma} ---")

        params = InstanceParameters(
            M=base_params.M, J=base_params.J, D=base_params.D,
            K=base_params.K, T=base_params.T,
            f_depot=base_params.f_depot, v_mode=base_params.v_mode,
            c_mode=base_params.c_mode, h_depot=base_params.h_depot,
            p_unmet=base_params.p_unmet,
            tau_mode_depot=base_params.tau_mode_depot,
            tau_depot_dest=base_params.tau_depot_dest,
            R_max=base_params.R_max, D_demand=base_params.D_demand,
            s_initial=base_params.s_initial,
            L_min=base_params.L_min, L_max=base_params.L_max,
            Gamma=gamma
        )

        scenarios = generate_burst_scenarios(params, max_scenarios=None)
        scenarios = deterministic_even_sample(scenarios, max_scenarios)
        print(f"  Using {len(scenarios)} scenarios")

        start = time.time()
        sol = solve_robust_model(params, scenarios)
        solve_time = time.time() - start

        results.append({
            'Gamma': gamma,
            'Total_Cost_M': sol.total_cost / 1e6,
            'Investment_M': sol.investment_cost / 1e6,
            'Operating_M': sol.worst_case_cost / 1e6,
            'C_depot_total': np.sum(sol.C_depot),
            'q_mode_total': np.sum(sol.q_mode),
            'Num_Scenarios': len(scenarios),
            'Solve_Time_s': solve_time
        })

        if output_path:
            pd.DataFrame(results).to_csv(output_path, index=False)

        print(f"  Cost: ${sol.total_cost/1e6:.2f}M, Time: {solve_time:.1f}s")

    df = pd.DataFrame(results)
    return df

def sensitivity_r_max(base_params, scale_factors, max_scenarios=400, output_path=None):
    """Test reception bottleneck effect"""
    print("\n" + "="*80)
    print("SENSITIVITY: R_MAX (Reception Capacity)")
    print("="*80)

    results = []
    base_R_max = base_params.R_max.copy()

    for scale in scale_factors:
        print(f"\n--- R_max scale = {scale} ---")

        scaled_R_max = base_R_max * scale
        params = InstanceParameters(
            M=base_params.M, J=base_params.J, D=base_params.D,
            K=base_params.K, T=base_params.T,
            f_depot=base_params.f_depot, v_mode=base_params.v_mode,
            c_mode=base_params.c_mode, h_depot=base_params.h_depot,
            p_unmet=base_params.p_unmet,
            tau_mode_depot=base_params.tau_mode_depot,
            tau_depot_dest=base_params.tau_depot_dest,
            R_max=scaled_R_max, D_demand=base_params.D_demand,
            s_initial=base_params.s_initial,
            L_min=base_params.L_min, L_max=base_params.L_max,
            Gamma=base_params.Gamma
        )

        scenarios = generate_burst_scenarios(params, max_scenarios=None)
        scenarios = deterministic_even_sample(scenarios, max_scenarios)
        print(f"  Using {len(scenarios)} scenarios, R_max={scaled_R_max[0,0]:.0f}")

        start = time.time()
        sol = solve_robust_model(params, scenarios)
        solve_time = time.time() - start

        results.append({
            'R_max_scale': scale,
            'R_max_value': scaled_R_max[0, 0],
            'Total_Cost_M': sol.total_cost / 1e6,
            'C_depot_0': sol.C_depot[0],
            'C_depot_1': sol.C_depot[1] if len(sol.C_depot) > 1 else 0,
            'C_depot_total': np.sum(sol.C_depot),
            'q_mode_total': np.sum(sol.q_mode),
            'Solve_Time_s': solve_time
        })

        if output_path:
            pd.DataFrame(results).to_csv(output_path, index=False)

        print(f"  C_depot: {sol.C_depot}, Cost: ${sol.total_cost/1e6:.2f}M")

    df = pd.DataFrame(results)
    return df

def sensitivity_s_initial(base_params, scale_factors, max_scenarios=400, output_path=None):
    """Test initial inventory locking effect"""
    print("\n" + "="*80)
    print("SENSITIVITY: S_INITIAL (Initial Inventory)")
    print("="*80)

    results = []
    base_s_initial = base_params.s_initial.copy()

    for scale in scale_factors:
        print(f"\n--- s_initial scale = {scale} ---")

        scaled_s_initial = base_s_initial * scale
        params = InstanceParameters(
            M=base_params.M, J=base_params.J, D=base_params.D,
            K=base_params.K, T=base_params.T,
            f_depot=base_params.f_depot, v_mode=base_params.v_mode,
            c_mode=base_params.c_mode, h_depot=base_params.h_depot,
            p_unmet=base_params.p_unmet,
            tau_mode_depot=base_params.tau_mode_depot,
            tau_depot_dest=base_params.tau_depot_dest,
            R_max=base_params.R_max, D_demand=base_params.D_demand,
            s_initial=scaled_s_initial,
            L_min=base_params.L_min, L_max=base_params.L_max,
            Gamma=base_params.Gamma
        )

        scenarios = generate_burst_scenarios(params, max_scenarios=None)
        scenarios = deterministic_even_sample(scenarios, max_scenarios)
        print(f"  Using {len(scenarios)} scenarios, s_initial={scaled_s_initial[0,0]:.0f}")

        start = time.time()
        sol = solve_robust_model(params, scenarios)
        solve_time = time.time() - start

        results.append({
            's_initial_scale': scale,
            's_initial_value': scaled_s_initial[0, 0],
            'Total_Cost_M': sol.total_cost / 1e6,
            'C_depot_0': sol.C_depot[0],
            'C_depot_1': sol.C_depot[1] if len(sol.C_depot) > 1 else 0,
            'C_depot_total': np.sum(sol.C_depot),
            'q_mode_total': np.sum(sol.q_mode),
            'Solve_Time_s': solve_time
        })

        if output_path:
            pd.DataFrame(results).to_csv(output_path, index=False)

        print(f"  C_depot: {sol.C_depot}, Cost: ${sol.total_cost/1e6:.2f}M")

    df = pd.DataFrame(results)
    return df

def main():
    print("="*80)
    print("SENSITIVITY ANALYSIS (LIMITED SCENARIOS)")
    print("="*80)

    base_params = generate_small_instance()
    max_scenarios = 400
    scale_name = (
        f"small_T{base_params.T}_M{base_params.M}_J{base_params.J}_K{base_params.K}_"
        f"Gamma{base_params.Gamma}_N{max_scenarios}_sampled"
    )
    output_dir = f'{PROJECT_ROOT}/results/05_sensitivity/{scale_name}'
    os.makedirs(output_dir, exist_ok=True)

    # Test 1: Gamma
    gamma_path = f"{output_dir}/sensitivity_gamma.csv"
    df_gamma = sensitivity_gamma(
        base_params, [0, 1, 2, 3],
        max_scenarios=max_scenarios,
        output_path=gamma_path,
    )
    df_gamma.to_csv(gamma_path, index=False)
    print(f"\n✓ Saved: {gamma_path}")

    # Plot Gamma
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(df_gamma['Gamma'], df_gamma['Total_Cost_M'], 'o-', linewidth=2, markersize=10, color='steelblue')
    ax.set_xlabel('Gamma (Disruption Budget)', fontsize=12)
    ax.set_ylabel('Total Cost ($M)', fontsize=12)
    ax.set_title('Cost Monotonicity with Gamma', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sensitivity_gamma.png", dpi=150)
    print(f"✓ Saved: {output_dir}/sensitivity_gamma.png")
    plt.close()

    # Test 2: R_max
    r_max_path = f"{output_dir}/sensitivity_r_max.csv"
    df_r_max = sensitivity_r_max(
        base_params, [0.5, 0.75, 1.0, 1.25, 1.5],
        max_scenarios=max_scenarios,
        output_path=r_max_path,
    )
    df_r_max.to_csv(r_max_path, index=False)
    print(f"\n✓ Saved: {r_max_path}")

    # Plot R_max
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(df_r_max['R_max_scale'], df_r_max['C_depot_total'], 'o-', linewidth=2, markersize=10, color='coral')
    ax.set_xlabel('R_max Scale Factor', fontsize=12)
    ax.set_ylabel('Total C_depot (tons)', fontsize=12)
    ax.set_title('Reception Bottleneck Effect', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sensitivity_r_max.png", dpi=150)
    print(f"✓ Saved: {output_dir}/sensitivity_r_max.png")
    plt.close()

    # Test 3: s_initial
    s_initial_path = f"{output_dir}/sensitivity_s_initial.csv"
    df_s_initial = sensitivity_s_initial(
        base_params, [0, 0.5, 1.0, 1.5],
        max_scenarios=max_scenarios,
        output_path=s_initial_path,
    )
    df_s_initial.to_csv(s_initial_path, index=False)
    print(f"\n✓ Saved: {s_initial_path}")

    # Plot s_initial
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(df_s_initial['s_initial_scale'], df_s_initial['C_depot_total'], 'o-', linewidth=2, markersize=10, color='green')
    ax.set_xlabel('s_initial Scale Factor', fontsize=12)
    ax.set_ylabel('Total C_depot (tons)', fontsize=12)
    ax.set_title('Initial Inventory Locking Effect', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sensitivity_s_initial.png", dpi=150)
    print(f"✓ Saved: {output_dir}/sensitivity_s_initial.png")
    plt.close()

    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")

if __name__ == '__main__':
    main()
