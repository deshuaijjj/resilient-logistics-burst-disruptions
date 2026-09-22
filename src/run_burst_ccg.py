"""
Simplified column-and-constraint generation over the finite burst library.

The restricted master is the existing sparse extensive-form LP on an active
scenario subset. The separation step freezes the master design and evaluates it
against the full burst library, adding the worst scenario when necessary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.append(SRC_DIR)

from instance_generator import generate_burst_scenarios, generate_small_instance  # noqa: E402
from lp_solver import solve_robust_model  # noqa: E402
from recourse_certifier import (  # noqa: E402
    certification_to_summary_row,
    certification_to_jsonable,
    certify_fixed_design,
    deterministic_even_sample_indices,
)


def scenario_events_json(scenarios, scenario_id: int) -> str:
    return json.dumps([[int(v) for v in event] for event in scenarios[scenario_id].events])


def run_scenario_generation_ccg(
    initial_sample_size: int,
    max_iterations: int,
    gap_tol_abs: float,
    gap_tol_rel: float,
    scenario_limit: int,
    output_root: str,
    progress_interval: int,
):
    params = generate_small_instance()
    all_scenarios = generate_burst_scenarios(params, max_scenarios=None)
    separation_ids = None
    if scenario_limit and scenario_limit > 0:
        separation_ids = list(range(min(scenario_limit, len(all_scenarios))))
        separation_count = len(separation_ids)
    else:
        separation_count = len(all_scenarios)

    active_ids = deterministic_even_sample_indices(
        len(all_scenarios),
        min(initial_sample_size, len(all_scenarios)),
    )
    active_ids = sorted(set(active_ids))
    active_set = set(active_ids)

    os.makedirs(output_root, exist_ok=True)
    iterations = []
    final_cert = None
    final_solution = None
    converged = False
    total_start = time.time()

    for iteration in range(1, max_iterations + 1):
        print('\n' + '=' * 80)
        print(f'CCG ITERATION {iteration}: {len(active_ids):,} active scenarios')
        print('=' * 80)
        iter_start = time.time()
        active_scenarios = [all_scenarios[i] for i in active_ids]

        master_start = time.time()
        master_sol = solve_robust_model(params, active_scenarios)
        master_wall = time.time() - master_start
        if master_sol.lp_status != 'optimal':
            raise RuntimeError(f'Master failed at iteration {iteration}: {master_sol.lp_status}')

        separation_start = time.time()
        cert, _ = certify_fixed_design(
            params=params,
            scenarios=all_scenarios,
            C_depot=master_sol.C_depot,
            q_mode=master_sol.q_mode,
            design_name=f'ccg_iter_{iteration}',
            sampled_total_cost=master_sol.total_cost,
            sampled_worst_operating_cost=master_sol.worst_case_cost,
            scenario_ids=separation_ids,
            progress_interval=progress_interval,
        )
        separation_wall = time.time() - separation_start
        lower_bound = master_sol.total_cost
        upper_bound = cert.full_total_cost
        gap = upper_bound - lower_bound
        denom = max(abs(upper_bound), 1.0)
        gap_pct = 100.0 * gap / denom
        worst_id = cert.worst.scenario_id
        new_scenario = worst_id not in active_set
        tol = max(gap_tol_abs, gap_tol_rel * max(abs(upper_bound), 1.0))
        converged = (not new_scenario) and gap <= tol

        row = {
            'iteration': iteration,
            'master_scenarios': len(active_ids),
            'added_scenario_id': None if not new_scenario else worst_id,
            'new_scenario': new_scenario,
            'lower_bound_M': lower_bound / 1e6,
            'upper_bound_M': upper_bound / 1e6,
            'gap_M': gap / 1e6,
            'gap_pct': gap_pct,
            'master_objective_M': master_sol.total_cost / 1e6,
            'master_worst_operating_M': master_sol.worst_case_cost / 1e6,
            'adversarial_operating_M': cert.full_worst_operating_cost / 1e6,
            'adversarial_total_cost_M': cert.full_total_cost / 1e6,
            'C_depot_0': master_sol.C_depot[0],
            'C_depot_1': master_sol.C_depot[1] if len(master_sol.C_depot) > 1 else 0.0,
            'q_mode_0': master_sol.q_mode[0],
            'q_mode_1': master_sol.q_mode[1] if len(master_sol.q_mode) > 1 else 0.0,
            'q_mode_2': master_sol.q_mode[2] if len(master_sol.q_mode) > 2 else 0.0,
            'worst_scenario_id': worst_id,
            'worst_events_json': scenario_events_json(all_scenarios, worst_id),
            'worst_unmet_tons': cert.worst.unmet_tons,
            'master_solve_time_s': master_sol.solve_time,
            'master_wall_time_s': master_wall,
            'separation_time_s': separation_wall,
            'iteration_wall_time_s': time.time() - iter_start,
            'lp_status': master_sol.lp_status,
            'converged': converged,
        }
        iterations.append(row)
        pd.DataFrame(iterations).to_csv(
            os.path.join(output_root, 'ccg_iterations.csv'),
            index=False,
        )

        final_solution = master_sol
        final_cert = cert

        print(
            f'Iteration {iteration}: LB=${lower_bound / 1e6:,.2f}M, '
            f'UB=${upper_bound / 1e6:,.2f}M, gap={gap_pct:.6f}%, '
            f'worst={worst_id}, new={new_scenario}'
        )

        if converged:
            print('CCG converged.')
            break
        if new_scenario:
            active_set.add(worst_id)
            active_ids = sorted(active_set)
        else:
            print('Worst scenario is already active, but gap tolerance is not met.')
            break

    assert final_solution is not None
    assert final_cert is not None
    total_wall = time.time() - total_start
    summary_row = certification_to_summary_row(
        final_cert,
        final_solution.C_depot,
        final_solution.q_mode,
        source_path='ccg',
        design_sample_size=len(active_ids),
    )
    summary_row.update({
        'method': 'finite_library_ccg',
        'iterations': len(iterations),
        'scenarios_used': len(active_ids),
        'master_objective_M': final_solution.total_cost / 1e6,
        'master_worst_operating_M': final_solution.worst_case_cost / 1e6,
        'total_wall_time_s': total_wall,
        'converged': converged,
        'separation_scenarios': separation_count,
    })

    pd.DataFrame([summary_row]).to_csv(os.path.join(output_root, 'ccg_summary.csv'), index=False)
    with open(os.path.join(output_root, 'ccg_active_scenarios.json'), 'w') as f:
        json.dump({
            'active_ids': active_ids,
            'active_events': {
                str(i): [[int(v) for v in event] for event in all_scenarios[i].events]
                for i in active_ids
            },
        }, f, indent=2)
    with open(os.path.join(output_root, 'ccg_final_solution.json'), 'w') as f:
        json.dump({
            'solution': {
                'C_depot': final_solution.C_depot.tolist(),
                'q_mode': final_solution.q_mode.tolist(),
                'investment_cost': final_solution.investment_cost,
                'worst_case_cost': final_solution.worst_case_cost,
                'total_cost': final_solution.total_cost,
                'worst_scenario_id_active': final_solution.worst_scenario_id,
                'worst_scenario_events': final_solution.worst_scenario_events,
                'solve_time': final_solution.solve_time,
                'num_scenarios': final_solution.num_scenarios,
                'lp_status': final_solution.lp_status,
            },
            'certification': certification_to_jsonable(final_cert),
            'summary': summary_row,
        }, f, indent=2)

    return final_solution, final_cert, pd.DataFrame(iterations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial-sample-size', type=int, default=1000)
    parser.add_argument('--max-iterations', type=int, default=10)
    parser.add_argument('--gap-tol-abs', type=float, default=1.0)
    parser.add_argument('--gap-tol-rel', type=float, default=1e-8)
    parser.add_argument(
        '--scenario-limit',
        type=int,
        default=0,
        help='Limit separation to first N full-library scenarios for smoke tests.',
    )
    parser.add_argument(
        '--output-root',
        default=os.path.join(PROJECT_ROOT, 'results', '09_burst_ccg'),
    )
    parser.add_argument('--progress-interval', type=int, default=500)
    args = parser.parse_args()

    run_scenario_generation_ccg(
        initial_sample_size=args.initial_sample_size,
        max_iterations=args.max_iterations,
        gap_tol_abs=args.gap_tol_abs,
        gap_tol_rel=args.gap_tol_rel,
        scenario_limit=args.scenario_limit,
        output_root=args.output_root,
        progress_interval=args.progress_interval,
    )


if __name__ == '__main__':
    main()
