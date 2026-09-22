"""
Deep-Space Logistics: Sensitivity Analysis Experiments

This module implements systematic sensitivity analysis to validate paper propositions:
- Proposition: Cost monotonicity with Gamma
- Proposition: Substitution effect between inventory and redundancy
- Proposition: Reception bottleneck effect
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from typing import List, Dict, Tuple

from instance_generator import (
    InstanceParameters,
    DisruptionScenario,
    generate_small_instance,
    generate_burst_scenarios
)
from lp_solver import solve_robust_model, Solution


def sensitivity_gamma(base_params: InstanceParameters,
                      gamma_values: List[int],
                      output_dir: str) -> pd.DataFrame:
    """
    Sensitivity analysis: vary Gamma (disruption budget)

    Check whether resilience cost increases monotonically with Gamma.

    Args:
        base_params: Base instance parameters
        gamma_values: List of Gamma values to test (e.g., [0, 1, 2, 3])
        output_dir: Output directory for results

    Returns:
        DataFrame with results for each Gamma
    """
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: GAMMA (Disruption Budget)")
    print("="*80)

    results = []

    for gamma in gamma_values:
        print(f"\n--- Testing Gamma = {gamma} ---")

        # Create modified instance
        params = InstanceParameters(
            M=base_params.M, J=base_params.J, D=base_params.D,
            K=base_params.K, T=base_params.T,
            f_depot=base_params.f_depot,
            v_mode=base_params.v_mode,
            c_mode=base_params.c_mode,
            h_depot=base_params.h_depot,
            p_unmet=base_params.p_unmet,
            tau_mode_depot=base_params.tau_mode_depot,
            tau_depot_dest=base_params.tau_depot_dest,
            R_max=base_params.R_max,
            D_demand=base_params.D_demand,
            s_initial=base_params.s_initial,
            L_min=base_params.L_min,
            L_max=base_params.L_max,
            Gamma=gamma
        )

        # Generate scenarios for this Gamma
        scenarios = generate_burst_scenarios(params, max_scenarios=500)
        print(f"  Generated {len(scenarios)} scenarios")

        # Solve
        solution = solve_robust_model(params, scenarios)

        results.append({
            'Gamma': gamma,
            'Total_Cost': solution.total_cost / 1e6,
            'Investment_Cost': solution.investment_cost / 1e6,
            'Operating_Cost': solution.worst_case_cost / 1e6,
            'C_depot_total': np.sum(solution.C_depot),
            'q_mode_total': np.sum(solution.q_mode),
            'Num_Scenarios': len(scenarios),
            'Solve_Time': solution.solve_time
        })

    df = pd.DataFrame(results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/sensitivity_gamma.csv", index=False)
    print(f"\nSaved: {output_dir}/sensitivity_gamma.csv")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df['Gamma'], df['Total_Cost'], 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('Gamma (Disruption Budget)', fontsize=12)
    ax1.set_ylabel('Total Cost ($M)', fontsize=12)
    ax1.set_title('Cost vs. Disruption Budget', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)

    ax2.plot(df['Gamma'], df['C_depot_total'], 'o-', linewidth=2, markersize=8,
             label='Total Depot Capacity', color='coral')
    ax2.plot(df['Gamma'], df['q_mode_total'], 's-', linewidth=2, markersize=8,
             label='Total Mode Capacity', color='seagreen')
    ax2.set_xlabel('Gamma (Disruption Budget)', fontsize=12)
    ax2.set_ylabel('Capacity', fontsize=12)
    ax2.set_title('Capacity Investment vs. Gamma', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sensitivity_gamma.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/sensitivity_gamma.png")
    plt.close()

    return df


def sensitivity_r_max(base_params: InstanceParameters,
                      r_scaling_factors: List[float],
                      burst_scenarios: List[DisruptionScenario],
                      output_dir: str) -> pd.DataFrame:
    """
    Sensitivity analysis: vary R_max (reception capacity)

    Check how reception bottlenecks affect the marginal value of transport redundancy.

    Args:
        base_params: Base instance parameters
        r_scaling_factors: Scaling factors for R_max (e.g., [0.5, 0.75, 1.0, 1.25, 1.5])
        burst_scenarios: Burst disruption scenarios
        output_dir: Output directory

    Returns:
        DataFrame with results
    """
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: R_MAX (Reception Capacity)")
    print("="*80)

    results = []

    for scale in r_scaling_factors:
        print(f"\n--- Testing R_max scale = {scale} ---")

        # Create modified instance
        params = InstanceParameters(
            M=base_params.M, J=base_params.J, D=base_params.D,
            K=base_params.K, T=base_params.T,
            f_depot=base_params.f_depot,
            v_mode=base_params.v_mode,
            c_mode=base_params.c_mode,
            h_depot=base_params.h_depot,
            p_unmet=base_params.p_unmet,
            tau_mode_depot=base_params.tau_mode_depot,
            tau_depot_dest=base_params.tau_depot_dest,
            R_max=base_params.R_max * scale,
            D_demand=base_params.D_demand,
            s_initial=base_params.s_initial,
            L_min=base_params.L_min,
            L_max=base_params.L_max,
            Gamma=base_params.Gamma
        )

        # Solve
        solution = solve_robust_model(params, burst_scenarios)

        results.append({
            'R_max_Scale': scale,
            'Total_Cost': solution.total_cost / 1e6,
            'Investment_Cost': solution.investment_cost / 1e6,
            'Operating_Cost': solution.worst_case_cost / 1e6,
            'C_depot_total': np.sum(solution.C_depot),
            'q_mode_total': np.sum(solution.q_mode),
            'Solve_Time': solution.solve_time
        })

    df = pd.DataFrame(results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/sensitivity_r_max.csv", index=False)
    print(f"\nSaved: {output_dir}/sensitivity_r_max.csv")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df['R_max_Scale'], df['Total_Cost'], 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('R_max Scaling Factor', fontsize=12)
    ax1.set_ylabel('Total Cost ($M)', fontsize=12)
    ax1.set_title('Cost vs. Reception Capacity', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)

    ax2.plot(df['R_max_Scale'], df['C_depot_total'], 'o-', linewidth=2, markersize=8,
             label='Depot Capacity', color='coral')
    ax2.plot(df['R_max_Scale'], df['q_mode_total'], 's-', linewidth=2, markersize=8,
             label='Mode Capacity', color='seagreen')
    ax2.set_xlabel('R_max Scaling Factor', fontsize=12)
    ax2.set_ylabel('Capacity', fontsize=12)
    ax2.set_title('Capacity vs. Reception Capacity', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sensitivity_r_max.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/sensitivity_r_max.png")
    plt.close()

    return df


def sensitivity_s_initial(base_params: InstanceParameters,
                          s_scaling_factors: List[float],
                          burst_scenarios: List[DisruptionScenario],
                          output_dir: str) -> pd.DataFrame:
    """
    Sensitivity analysis: vary s_initial (initial inventory)

    Check whether initial inventory locks in capacity decisions.

    Args:
        base_params: Base instance parameters
        s_scaling_factors: Scaling factors for s_initial (e.g., [0, 0.5, 1.0, 1.5])
        burst_scenarios: Burst disruption scenarios
        output_dir: Output directory

    Returns:
        DataFrame with results
    """
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: S_INITIAL (Initial Inventory)")
    print("="*80)

    results = []

    for scale in s_scaling_factors:
        print(f"\n--- Testing s_initial scale = {scale} ---")

        # Create modified instance
        params = InstanceParameters(
            M=base_params.M, J=base_params.J, D=base_params.D,
            K=base_params.K, T=base_params.T,
            f_depot=base_params.f_depot,
            v_mode=base_params.v_mode,
            c_mode=base_params.c_mode,
            h_depot=base_params.h_depot,
            p_unmet=base_params.p_unmet,
            tau_mode_depot=base_params.tau_mode_depot,
            tau_depot_dest=base_params.tau_depot_dest,
            R_max=base_params.R_max,
            D_demand=base_params.D_demand,
            s_initial=base_params.s_initial * scale,
            L_min=base_params.L_min,
            L_max=base_params.L_max,
            Gamma=base_params.Gamma
        )

        # Solve
        solution = solve_robust_model(params, burst_scenarios)

        results.append({
            'S_initial_Scale': scale,
            'Total_Cost': solution.total_cost / 1e6,
            'Investment_Cost': solution.investment_cost / 1e6,
            'Operating_Cost': solution.worst_case_cost / 1e6,
            'C_depot_0': solution.C_depot[0],
            'C_depot_1': solution.C_depot[1],
            'C_depot_total': np.sum(solution.C_depot),
            'q_mode_total': np.sum(solution.q_mode),
            'Solve_Time': solution.solve_time
        })

    df = pd.DataFrame(results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/sensitivity_s_initial.csv", index=False)
    print(f"\nSaved: {output_dir}/sensitivity_s_initial.csv")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df['S_initial_Scale'], df['Total_Cost'], 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('S_initial Scaling Factor', fontsize=12)
    ax1.set_ylabel('Total Cost ($M)', fontsize=12)
    ax1.set_title('Cost vs. Initial Inventory', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)

    ax2.plot(df['S_initial_Scale'], df['C_depot_0'], 'o-', linewidth=2, markersize=8,
             label='Depot 0 Capacity', color='coral')
    ax2.plot(df['S_initial_Scale'], df['C_depot_1'], 's-', linewidth=2, markersize=8,
             label='Depot 1 Capacity', color='seagreen')
    ax2.plot(df['S_initial_Scale'], df['C_depot_total'], '^-', linewidth=2, markersize=8,
             label='Total Depot Capacity', color='purple')
    ax2.set_xlabel('S_initial Scaling Factor', fontsize=12)
    ax2.set_ylabel('Depot Capacity (tons)', fontsize=12)
    ax2.set_title('Depot Capacity vs. Initial Inventory', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sensitivity_s_initial.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/sensitivity_s_initial.png")
    plt.close()

    return df


def main():
    """Run all sensitivity analyses"""
    print("\n" + "="*80)
    print("DEEP-SPACE LOGISTICS: SENSITIVITY ANALYSIS")
    print("="*80)

    # Load base instance
    base_params = generate_small_instance()

    # Generate burst scenarios once
    print("\nGenerating burst scenarios...")
    burst_scenarios = generate_burst_scenarios(base_params, max_scenarios=500)
    print(f"Using {len(burst_scenarios)} burst scenarios")

    output_dir = "../results/sensitivity"

    # 1. Gamma sensitivity
    print("\n" + "="*80)
    gamma_values = [0, 1, 2, 3]
    df_gamma = sensitivity_gamma(base_params, gamma_values, output_dir)

    # 2. R_max sensitivity
    r_scaling_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    df_r_max = sensitivity_r_max(base_params, r_scaling_factors, burst_scenarios, output_dir)

    # 3. s_initial sensitivity
    s_scaling_factors = [0.0, 0.5, 1.0, 1.5]
    df_s_initial = sensitivity_s_initial(base_params, s_scaling_factors, burst_scenarios, output_dir)

    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("  - sensitivity_gamma.csv/.png")
    print("  - sensitivity_r_max.csv/.png")
    print("  - sensitivity_s_initial.csv/.png")


if __name__ == "__main__":
    main()
