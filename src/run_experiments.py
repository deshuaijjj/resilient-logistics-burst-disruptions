"""
Deep-Space Logistics: Baseline Strategies and Experiments

This module implements all baseline strategies and runs comparative experiments.

BASELINE DEFINITIONS (strict definitions used in the manuscript):

1. Nominal:
   - Scenario set: no disruption only
   - Decision variables: jointly optimize C_depot and q_mode
   - Purpose: provide an optimistic baseline

2. Burst Robust:
   - Scenario set: burst disruption scenarios with up to Gamma events
   - Decision variables: jointly optimize C_depot and q_mode
   - Purpose: joint design under burst uncertainty

3. Independent Robust:
   - Scenario set: independent disruption scenarios under budgeted uncertainty
   - Decision variables: jointly optimize C_depot and q_mode
   - Purpose: contrast independent failures with time-correlated bursts

4. Inventory-Only:
   - Scenario set: same as burst robust
   - Decision variables: fix q_mode at a reasonable value and optimize C_depot
   - Purpose: quantify the cost of relying only on inventory capacity

5. Redundancy-Only:
   - Scenario set: same as burst robust
   - Decision variables: fix C_depot at a reasonable value and optimize q_mode
   - Purpose: quantify the cost of relying only on transport redundancy

6. Joint Design:
   - Same definition as Burst Robust, rerun as an independent check
   - Purpose: verify consistency of the joint optimization result
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
import json
import os

from instance_generator import (
    InstanceParameters,
    DisruptionScenario,
    generate_small_instance,
    generate_burst_scenarios,
    generate_independent_scenarios
)
from lp_solver import solve_robust_model, Solution


def run_baseline_experiments(params: InstanceParameters,
                             burst_scenarios: List[DisruptionScenario],
                             ind_scenarios: List[DisruptionScenario]) -> Dict[str, Solution]:
    """
    Run all baseline experiments with corrected definitions

    Args:
        params: Problem instance
        burst_scenarios: Burst disruption scenarios (complete or sampled)
        ind_scenarios: Independent disruption scenarios

    Returns:
        Dictionary mapping strategy name to Solution
    """
    results = {}

    print("\n" + "="*80)
    print("BASELINE EXPERIMENTS (CORRECTED DEFINITIONS)")
    print("="*80)

    # ==================== Baseline 1: Nominal ====================
    # Definition: no disruptions; optimize both C and q.
    print("\n[1/6] Nominal Strategy...")
    print("  Definition: No disruptions, optimize both C and q")
    nominal_scenarios = [DisruptionScenario([])]  # Only no-disruption scenario
    results['nominal'] = solve_robust_model(params, nominal_scenarios)

    # ==================== Baseline 2: Independent Robust ====================
    # Definition: standard budgeted uncertainty; optimize both C and q.
    print("\n[2/6] Independent Robust...")
    print(f"  Definition: Budgeted uncertainty ({len(ind_scenarios)} scenarios), optimize both C and q")
    results['independent'] = solve_robust_model(params, ind_scenarios)

    # ==================== Baseline 3: Burst Robust ====================
    # Definition: burst disruption model; optimize both C and q.
    print("\n[3/6] Burst Robust (our model)...")
    print(f"  Definition: Burst disruptions ({len(burst_scenarios)} scenarios), optimize both C and q")
    results['burst'] = solve_robust_model(params, burst_scenarios)

    # ==================== Baseline 4: Inventory-Only ====================
    # Definition: fix q at a balanced multi-mode allocation and optimize only C.
    # Correction: use a reasonable multi-mode allocation rather than nominal single-mode q.
    print("\n[4/6] Inventory-Only...")
    print("  Definition: Fixed q at balanced multi-mode allocation, optimize only C")

    # Fixed value: use burst q as a representative diversified transport portfolio.
    # Alternatively, a uniform total_capacity / M allocation could be used.
    if 'burst' in results:
        fixed_q = results['burst'].q_mode.copy()
        print(f"  Fixed q_mode = {fixed_q} (from burst solution)")
    else:
        # Fallback: balanced allocation.
        total_monthly_demand = np.sum(params.D_demand) / params.T
        fixed_q = np.ones(params.M) * (total_monthly_demand / params.M * 1.2)  # 120% of average
        print(f"  Fixed q_mode = {fixed_q} (balanced allocation)")

    results['inventory_only'] = solve_robust_model(
        params, burst_scenarios, fixed_q=fixed_q
    )

    # ==================== Baseline 5: Redundancy-Only ====================
    # Definition: fix C at a reasonable inventory capacity and optimize only q.
    # Correction: use burst C or another defensible inventory-capacity value.
    print("\n[5/6] Redundancy-Only...")
    print("  Definition: Fixed C at reasonable inventory capacity, optimize only q")

    # Fixed value: use burst C as a representative inventory-capacity level.
    if 'burst' in results:
        fixed_C = results['burst'].C_depot.copy()
        print(f"  Fixed C_depot = {fixed_C} (from burst solution)")
    else:
        # Fallback: use nominal C.
        fixed_C = results['nominal'].C_depot.copy()
        print(f"  Fixed C_depot = {fixed_C} (from nominal solution)")

    results['redundancy_only'] = solve_robust_model(
        params, burst_scenarios, fixed_C=fixed_C
    )

    # ==================== Baseline 6: Joint Design ====================
    # Definition: same as burst, but rerun independently for verification.
    # This is not an alias of results['burst'].
    print("\n[6/6] Joint Design (verification)...")
    print("  Definition: Same as burst, but re-run to verify joint optimization")
    results['joint'] = solve_robust_model(params, burst_scenarios)

    # Verify that joint and burst results match.
    if abs(results['joint'].total_cost - results['burst'].total_cost) > 1e-3:
        print("  WARNING: Joint and Burst results differ! This should not happen.")
    else:
        print("  Verification: Joint = Burst (as expected)")

    return results


def generate_summary_table(results: Dict[str, Solution], params: InstanceParameters) -> pd.DataFrame:
    """Generate summary comparison table"""
    data = []

    for name, sol in results.items():
        row = {
            'Strategy': name,
            'Total Cost ($M)': sol.total_cost / 1e6,
            'Investment ($M)': sol.investment_cost / 1e6,
            'Worst Operating ($M)': sol.worst_case_cost / 1e6,
            'Depot Capacity (tons)': ' / '.join([f"{c:.0f}" for c in sol.C_depot]),
            'Mode Capacity (t/mo)': ' / '.join([f"{q:.0f}" for q in sol.q_mode]),
            'Worst Scenario': f"{len(sol.worst_scenario_events)} events",
            'Num Scenarios': sol.num_scenarios,
            'Solve Time (s)': f"{sol.solve_time:.2f}",
            'LP Status': sol.lp_status
        }
        data.append(row)

    df = pd.DataFrame(data)
    return df


def plot_results(results: Dict[str, Solution], output_dir: str):
    """Generate visualization plots"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Cost comparison bar chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    strategies = ['nominal', 'independent', 'burst', 'inventory_only', 'redundancy_only', 'joint']
    strategies = [s for s in strategies if s in results]

    inv_costs = [results[s].investment_cost / 1e6 for s in strategies]
    op_costs = [results[s].worst_case_cost / 1e6 for s in strategies]

    x = np.arange(len(strategies))
    width = 0.35

    ax.bar(x - width/2, inv_costs, width, label='Investment Cost', color='steelblue')
    ax.bar(x + width/2, op_costs, width, label='Worst Operating Cost', color='coral')

    ax.set_ylabel('Cost (Million $)', fontsize=12)
    ax.set_title('Cost Comparison Across Strategies', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in strategies],
                        rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/cost_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/cost_comparison.png")
    plt.close()

    # 2. Design decisions comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Depot capacity
    depot_data = np.array([results[s].C_depot for s in strategies])
    n_depots = depot_data.shape[1]

    x = np.arange(len(strategies))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for j in range(n_depots):
        ax1.bar(x + j*width/n_depots - width/2, depot_data[:, j], width/n_depots,
                label=f'Depot {j+1}', alpha=0.8, color=colors[j % len(colors)])

    ax1.set_ylabel('Capacity (tons)', fontsize=12)
    ax1.set_title('Depot Capacity Investment', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace('_', ' ').title() for s in strategies],
                         rotation=45, ha='right', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Mode capacity
    mode_data = np.array([results[s].q_mode for s in strategies])
    n_modes = mode_data.shape[1]

    for m in range(n_modes):
        ax2.bar(x + m*width/n_modes - width/2, mode_data[:, m], width/n_modes,
                label=f'Mode {m+1}', alpha=0.8, color=colors[m % len(colors)])

    ax2.set_ylabel('Capacity (tons/month)', fontsize=12)
    ax2.set_title('Transportation Capacity Contract', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace('_', ' ').title() for s in strategies],
                         rotation=45, ha='right', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/design_decisions.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/design_decisions.png")
    plt.close()


def save_results(results: Dict[str, Solution],
                 summary_df: pd.DataFrame,
                 output_dir: str):
    """Save results to CSV and JSON"""
    os.makedirs(output_dir, exist_ok=True)

    # Save summary table
    summary_df.to_csv(f"{output_dir}/summary.csv", index=False)
    print(f"Saved: {output_dir}/summary.csv")

    # Save detailed results as JSON
    detailed = {}
    for name, sol in results.items():
        detailed[name] = {
            'total_cost': float(sol.total_cost),
            'investment_cost': float(sol.investment_cost),
            'worst_case_cost': float(sol.worst_case_cost),
            'C_depot': sol.C_depot.tolist(),
            'q_mode': sol.q_mode.tolist(),
            'worst_scenario_id': sol.worst_scenario_id,
            'worst_scenario_events': sol.worst_scenario_events,
            'solve_time': sol.solve_time,
            'num_scenarios': sol.num_scenarios,
            'lp_status': sol.lp_status
        }

    with open(f"{output_dir}/detailed_results.json", 'w') as f:
        json.dump(detailed, f, indent=2)
    print(f"Saved: {output_dir}/detailed_results.json")


def main():
    """Main experiment runner"""
    print("\n" + "="*80)
    print("DEEP-SPACE LOGISTICS: CORRECTED BASELINE EXPERIMENTS")
    print("="*80)

    # Setup
    params = generate_small_instance()

    print("\n[SETUP] Generating scenarios...")

    # Generate burst scenarios with proper handling
    print("\n  Generating burst disruption scenarios...")
    burst_scenarios_all = generate_burst_scenarios(params, max_scenarios=None)
    print(f"  Total burst scenarios available: {len(burst_scenarios_all)}")

    # For small instance, use all scenarios if <500, otherwise sample deterministically
    if len(burst_scenarios_all) <= 500:
        burst_scenarios = burst_scenarios_all
        print(f"  Using all {len(burst_scenarios)} burst scenarios")
    else:
        # Deterministic sampling: take every k-th scenario
        k = len(burst_scenarios_all) // 500
        burst_scenarios = burst_scenarios_all[::k][:500]
        print(f"  Sampled {len(burst_scenarios)} burst scenarios (every {k}-th)")

    # Generate independent scenarios
    print("\n  Generating independent disruption scenarios...")
    ind_scenarios = generate_independent_scenarios(params, Gamma_total=params.Gamma * params.M)
    print(f"  Independent scenarios: {len(ind_scenarios)}")

    # Run experiments
    try:
        results = run_baseline_experiments(params, burst_scenarios, ind_scenarios)
    except Exception as e:
        print(f"\n ERROR during experiments: {e}")
        import traceback
        traceback.print_exc()
        return

    # Generate outputs
    print("\n" + "="*80)
    print("GENERATING OUTPUTS")
    print("="*80)

    summary_df = generate_summary_table(results, params)
    print("\n[SUMMARY TABLE]")
    print(summary_df.to_string(index=False))

    output_dir = (
        f"../results/02_baselines/"
        f"small_T{params.T}_M{params.M}_J{params.J}_K{params.K}_Gamma{params.Gamma}_N{len(burst_scenarios)}_sampled"
    )
    save_results(results, summary_df, output_dir)

    print("\n[GENERATING PLOTS]")
    plot_results(results, output_dir)

    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    if 'nominal' in results and 'burst' in results:
        nominal_cost = results['nominal'].total_cost
        burst_cost = results['burst'].total_cost
        increase = (burst_cost - nominal_cost) / nominal_cost * 100

        print(f"\n1. Cost of Resilience:")
        print(f"   Nominal: ${nominal_cost/1e6:.2f}M")
        print(f"   Burst Robust: ${burst_cost/1e6:.2f}M")
        print(f"   Increase: {increase:.1f}%")

    if 'independent' in results and 'burst' in results:
        ind_inv = results['independent'].investment_cost
        burst_inv = results['burst'].investment_cost
        inv_diff = (burst_inv - ind_inv) / ind_inv * 100

        print(f"\n2. Burst vs. Independent Model:")
        print(f"   Independent investment: ${ind_inv/1e6:.2f}M")
        print(f"   Burst investment: ${burst_inv/1e6:.2f}M")
        print(f"   Difference: {inv_diff:+.1f}%")

    if 'inventory_only' in results and 'redundancy_only' in results and 'joint' in results:
        inv_only_cost = results['inventory_only'].total_cost
        red_only_cost = results['redundancy_only'].total_cost
        joint_cost = results['joint'].total_cost

        best_single = min(inv_only_cost, red_only_cost)
        improvement = (best_single - joint_cost) / best_single * 100

        print(f"\n3. Joint Design Value:")
        print(f"   Inventory-only: ${inv_only_cost/1e6:.2f}M")
        print(f"   Redundancy-only: ${red_only_cost/1e6:.2f}M")
        print(f"   Joint design: ${joint_cost/1e6:.2f}M")
        print(f"   Improvement over best single: {improvement:.1f}%")

    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - summary.csv")
    print(f"  - detailed_results.json")
    print(f"  - cost_comparison.png")
    print(f"  - design_decisions.png")


if __name__ == "__main__":
    main()
