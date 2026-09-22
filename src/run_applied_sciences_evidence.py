"""
Additional evidence for the Applied Sciences revision.

This script adds three lightweight checks requested in the revision plan:
random-seed sampling robustness, holdout evaluation of a sampled design, and
small stylized instance variants. It writes both raw CSV outputs and compact
LaTeX tables for manuscript use.
"""

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.append(SRC_DIR)

from instance_generator import (  # noqa: E402
    DisruptionScenario,
    InstanceParameters,
    generate_burst_scenarios,
    generate_small_instance,
)
from lp_solver import Solution, solve_robust_model  # noqa: E402


RESULT_ROOT = os.path.join(PROJECT_ROOT, 'results', '07_applied_sciences_evidence')
TABLE_DIR = os.path.join(PROJECT_ROOT, 'results', 'tables_for_paper')


def deterministic_even_sample(
    scenarios: List[DisruptionScenario],
    sample_size: int,
) -> List[DisruptionScenario]:
    """Keep nominal scenario and sample the remaining enumeration evenly."""
    if sample_size >= len(scenarios):
        return scenarios
    if sample_size <= 1:
        return [scenarios[0]]
    non_nominal = scenarios[1:]
    indices = np.linspace(0, len(non_nominal) - 1, sample_size - 1, dtype=int)
    return [scenarios[0]] + [non_nominal[i] for i in indices]


def random_sample(
    scenarios: List[DisruptionScenario],
    sample_size: int,
    seed: int,
) -> List[DisruptionScenario]:
    """Keep nominal scenario and randomly sample non-nominal scenarios."""
    if sample_size >= len(scenarios):
        return scenarios
    if sample_size <= 1:
        return [scenarios[0]]
    rng = np.random.default_rng(seed)
    non_nominal = scenarios[1:]
    sample_count = min(sample_size - 1, len(non_nominal))
    indices = rng.choice(len(non_nominal), size=sample_count, replace=False)
    indices.sort()
    return [scenarios[0]] + [non_nominal[int(i)] for i in indices]


def solution_to_row(label: str, sol: Solution, extra: Dict | None = None) -> Dict:
    row = {
        'label': label,
        'total_cost_M': sol.total_cost / 1e6,
        'investment_cost_M': sol.investment_cost / 1e6,
        'worst_operating_cost_M': sol.worst_case_cost / 1e6,
        'C_depot_0': sol.C_depot[0] if len(sol.C_depot) > 0 else 0.0,
        'C_depot_1': sol.C_depot[1] if len(sol.C_depot) > 1 else 0.0,
        'C_depot_total': float(np.sum(sol.C_depot)),
        'q_mode_0': sol.q_mode[0] if len(sol.q_mode) > 0 else 0.0,
        'q_mode_1': sol.q_mode[1] if len(sol.q_mode) > 1 else 0.0,
        'q_mode_2': sol.q_mode[2] if len(sol.q_mode) > 2 else 0.0,
        'q_mode_total': float(np.sum(sol.q_mode)),
        'worst_scenario_id': sol.worst_scenario_id,
        'num_worst_events': len(sol.worst_scenario_events),
        'solve_time_s': sol.solve_time,
        'num_scenarios': sol.num_scenarios,
        'lp_status': sol.lp_status,
    }
    if extra:
        row.update(extra)
    return row


def save_latex_table(
    df: pd.DataFrame,
    name: str,
    latex_df: pd.DataFrame | None = None,
    escape: bool = False,
):
    os.makedirs(TABLE_DIR, exist_ok=True)
    csv_path = os.path.join(TABLE_DIR, f'{name}.csv')
    tex_path = os.path.join(TABLE_DIR, f'{name}.tex')
    df.to_csv(csv_path, index=False)
    out = latex_df if latex_df is not None else df
    out.to_latex(tex_path, index=False, float_format='%.2f', escape=escape)
    print(f'Saved {csv_path}')
    print(f'Saved {tex_path}')


def fmt(x: float, decimals: int = 2) -> str:
    value = float(x)
    if abs(value) < 0.5 * 10 ** (-decimals):
        value = 0.0
    return f'{value:,.{decimals}f}'


def random_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ('Total cost (\\$M)', 'total_cost_M'),
        ('Investment (\\$M)', 'investment_cost_M'),
        ('Worst operating (\\$M)', 'worst_operating_cost_M'),
        ('Depot 0 (tons)', 'C_depot_0'),
        ('Depot 1 (tons)', 'C_depot_1'),
        ('Mode 0 (t/mo)', 'q_mode_0'),
        ('Mode 1 (t/mo)', 'q_mode_1'),
        ('Mode 2 (t/mo)', 'q_mode_2'),
    ]
    rows = []
    for label, col in metrics:
        values = df[col].astype(float)
        rows.append({
            'Metric': label,
            'Mean': values.mean(),
            'Min': values.min(),
            'Max': values.max(),
            'Std. dev.': values.std(ddof=0),
        })
    return pd.DataFrame(rows)


def paper_random_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.copy()
    for col in ['Mean', 'Min', 'Max', 'Std. dev.']:
        table[col] = table[col].map(fmt)
    return table


def paper_holdout_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    table = table.rename(columns={
        'design_sample_size': 'Design sample',
        'holdout_size': 'Evaluation set',
        'optimization_total_cost_M': 'Sampled design cost (\\$M)',
        'holdout_total_cost_M': 'Holdout cost (\\$M)',
        'holdout_worst_operating_cost_M': 'Holdout worst operating (\\$M)',
        'cost_gap_M': 'Gap (\\$M)',
        'C_depot': 'Depot capacity',
        'q_mode': 'Mode capacity',
    })
    for col in [
        'Sampled design cost (\\$M)',
        'Holdout cost (\\$M)',
        'Holdout worst operating (\\$M)',
        'Gap (\\$M)',
    ]:
        table[col] = table[col].map(fmt)
    for col in ['Design sample', 'Evaluation set']:
        table[col] = table[col].map(lambda x: f'{int(x):,}')
    return table


def paper_variant_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    labels = {
        'base': 'Base',
        'lower_reception': 'Lower reception',
        'higher_emergency_cost': 'Higher emergency cost',
        'early_spikes': 'Earlier demand spikes',
    }
    table['variant'] = table['variant'].map(lambda x: labels.get(x, str(x).replace('_', ' ')))
    table = table.rename(columns={
        'variant': 'Variant',
        'description': 'Description',
        'total_cost_M': 'Total cost (\\$M)',
        'investment_cost_M': 'Investment (\\$M)',
        'worst_operating_cost_M': 'Worst operating (\\$M)',
        'C_depot_total': 'Depot capacity',
        'q_mode_total': 'Mode capacity',
    })
    keep = [
        'Variant', 'Description', 'Total cost (\\$M)', 'Investment (\\$M)',
        'Worst operating (\\$M)', 'Depot capacity', 'Mode capacity',
    ]
    table = table[keep]
    for col in [
        'Total cost (\\$M)', 'Investment (\\$M)', 'Worst operating (\\$M)',
        'Depot capacity', 'Mode capacity',
    ]:
        table[col] = table[col].map(fmt)
    return table


def paper_extended_design_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    table['Depot capacity'] = table.apply(
        lambda r: f"{float(r['C_depot_0']):.0f} / {float(r['C_depot_1']):.0f}",
        axis=1,
    )
    table['Mode capacity'] = table.apply(
        lambda r: (
            f"{float(r['q_mode_0']):.0f} / "
            f"{float(r['q_mode_1']):.0f} / "
            f"{float(r['q_mode_2']):.0f}"
        ),
        axis=1,
    )
    table = table.rename(columns={
        'sample_size': 'Sample size',
        'total_cost_M': 'Total cost (\\$M)',
        'investment_cost_M': 'Investment (\\$M)',
        'worst_operating_cost_M': 'Worst operating (\\$M)',
        'num_worst_events': 'Worst events',
    })
    keep = [
        'Sample size', 'Total cost (\\$M)', 'Investment (\\$M)',
        'Worst operating (\\$M)', 'Depot capacity', 'Mode capacity',
        'Worst events',
    ]
    table = table[keep]
    for col in ['Total cost (\\$M)', 'Investment (\\$M)', 'Worst operating (\\$M)']:
        table[col] = table[col].map(fmt)
    table['Sample size'] = table['Sample size'].map(lambda x: f'{int(x):,}')
    table['Worst events'] = table['Worst events'].map(lambda x: f'{int(x):,}')
    return table


def load_existing_design(sample_size: int) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Load an existing deterministic burst design if available."""
    path = os.path.join(
        PROJECT_ROOT,
        'results',
        '02_baselines',
        f'small_T24_M3_J2_K2_Gamma2_N{sample_size}_sampled',
        'detailed_results.json',
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    results = data.get('results', data)
    result = results.get('burst') or results.get('burst_joint')
    if result is None:
        return None
    return (
        np.array(result['C_depot'], dtype=float),
        np.array(result['q_mode'], dtype=float),
        float(result['total_cost']) / 1e6,
    )


def existing_design_row(sample_size: int) -> Dict | None:
    """Read an existing deterministic design result as a table row."""
    csv_path = os.path.join(RESULT_ROOT, f'deterministic_N{sample_size}_design.csv')
    if os.path.exists(csv_path):
        row = pd.read_csv(csv_path).iloc[0].to_dict()
        return row

    path = os.path.join(
        PROJECT_ROOT,
        'results',
        '02_baselines',
        f'small_T24_M3_J2_K2_Gamma2_N{sample_size}_sampled',
        'detailed_results.json',
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    results = data.get('results', data)
    result = results.get('burst') or results.get('burst_joint')
    if result is None:
        return None
    C = np.array(result['C_depot'], dtype=float)
    q = np.array(result['q_mode'], dtype=float)
    return {
        'sample_size': sample_size,
        'total_cost_M': float(result['total_cost']) / 1e6,
        'investment_cost_M': float(result['investment_cost']) / 1e6,
        'worst_operating_cost_M': float(result['worst_case_cost']) / 1e6,
        'C_depot_0': C[0] if len(C) > 0 else 0.0,
        'C_depot_1': C[1] if len(C) > 1 else 0.0,
        'q_mode_0': q[0] if len(q) > 0 else 0.0,
        'q_mode_1': q[1] if len(q) > 1 else 0.0,
        'q_mode_2': q[2] if len(q) > 2 else 0.0,
        'worst_scenario_id': int(result.get('worst_scenario_id', -1)),
        'num_worst_events': len(result.get('worst_scenario_events', [])),
        'solve_time_s': float(result.get('solve_time', 0.0)),
        'lp_status': result.get('lp_status', ''),
    }


def run_random_seed_robustness(
    params: InstanceParameters,
    all_scenarios: List[DisruptionScenario],
    sample_size: int,
    seeds: List[int],
    output_dir: str,
) -> pd.DataFrame:
    print('\n' + '=' * 80)
    print('RANDOM-SEED SAMPLING ROBUSTNESS')
    print('=' * 80)
    rows = []
    out_path = os.path.join(output_dir, 'random_seed_robustness_raw.csv')
    for seed in seeds:
        print(f'\nSeed {seed}, N={sample_size}')
        scenarios = random_sample(all_scenarios, sample_size, seed)
        start = time.time()
        sol = solve_robust_model(params, scenarios)
        wall_time = time.time() - start
        row = solution_to_row(
            f'seed_{seed}',
            sol,
            {'seed': seed, 'sample_size': sample_size, 'wall_time_s': wall_time},
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    summary = random_summary_table(df)
    summary.to_csv(os.path.join(output_dir, 'random_seed_robustness_summary.csv'), index=False)
    save_latex_table(
        summary,
        'random_seed_robustness',
        latex_df=paper_random_summary_table(summary),
        escape=False,
    )
    return df


def run_holdout_evaluation(
    params: InstanceParameters,
    all_scenarios: List[DisruptionScenario],
    design_sample_size: int,
    holdout_size: int | None,
    output_dir: str,
) -> pd.DataFrame:
    print('\n' + '=' * 80)
    print('HOLDOUT EVALUATION')
    print('=' * 80)
    loaded = load_existing_design(design_sample_size)
    if loaded is None:
        print(f'No existing N={design_sample_size} burst design found; solving it.')
        design_scenarios = deterministic_even_sample(all_scenarios, design_sample_size)
        design_sol = solve_robust_model(params, design_scenarios)
        C_depot = design_sol.C_depot.copy()
        q_mode = design_sol.q_mode.copy()
        optimization_total_cost_M = design_sol.total_cost / 1e6
    else:
        C_depot, q_mode, optimization_total_cost_M = loaded
        print(f'Loaded existing N={design_sample_size} design.')

    if holdout_size is None or holdout_size >= len(all_scenarios):
        holdout_scenarios = all_scenarios
        holdout_label = len(all_scenarios)
        holdout_method = 'full_enumeration'
    else:
        holdout_scenarios = deterministic_even_sample(all_scenarios, holdout_size)
        holdout_label = len(holdout_scenarios)
        holdout_method = 'deterministic_even_holdout'

    print(f'Evaluating fixed design on {holdout_label} scenarios.')
    holdout_sol = solve_robust_model(
        params,
        holdout_scenarios,
        fixed_C=C_depot,
        fixed_q=q_mode,
    )
    holdout_total_cost_M = holdout_sol.total_cost / 1e6
    row = {
        'design_sample_size': design_sample_size,
        'holdout_size': holdout_label,
        'holdout_method': holdout_method,
        'optimization_total_cost_M': optimization_total_cost_M,
        'holdout_total_cost_M': holdout_total_cost_M,
        'holdout_worst_operating_cost_M': holdout_sol.worst_case_cost / 1e6,
        'cost_gap_M': holdout_total_cost_M - optimization_total_cost_M,
        'C_depot': ' / '.join(f'{x:.0f}' for x in C_depot),
        'q_mode': ' / '.join(f'{x:.0f}' for x in q_mode),
        'holdout_worst_scenario_id': holdout_sol.worst_scenario_id,
        'holdout_worst_events': len(holdout_sol.worst_scenario_events),
        'solve_time_s': holdout_sol.solve_time,
        'lp_status': holdout_sol.lp_status,
    }
    df = pd.DataFrame([row])
    df.to_csv(os.path.join(output_dir, 'holdout_validation.csv'), index=False)
    save_latex_table(
        df[[
            'design_sample_size', 'holdout_size', 'optimization_total_cost_M',
            'holdout_total_cost_M', 'holdout_worst_operating_cost_M',
            'cost_gap_M', 'C_depot', 'q_mode',
        ]],
        'holdout_validation',
        latex_df=paper_holdout_table(df[[
            'design_sample_size', 'holdout_size', 'optimization_total_cost_M',
            'holdout_total_cost_M', 'holdout_worst_operating_cost_M',
            'cost_gap_M', 'C_depot', 'q_mode',
        ]]),
        escape=False,
    )
    return df


def make_variant(base: InstanceParameters, variant: str) -> tuple[InstanceParameters, str]:
    params = copy.deepcopy(base)
    if variant == 'base':
        return params, 'Base stylized case'
    if variant == 'lower_reception':
        params.R_max = params.R_max * 0.75
        return params, 'Destination reception capacity scaled to 0.75x'
    if variant == 'higher_emergency_cost':
        params.v_mode = params.v_mode.copy()
        params.c_mode = params.c_mode.copy()
        params.v_mode[2] *= 1.5
        params.c_mode[2] *= 1.25
        return params, 'Emergency/high-cost mode made more expensive'
    if variant == 'early_spikes':
        demand = params.D_demand.copy()
        demand[0, 0, [3, 9, 15, 21]] = 200
        demand[0, 0, [2, 8, 14, 20]] = 400
        demand[0, 1, [6, 12, 18]] = 100
        demand[0, 1, [5, 11, 17]] = 250
        params.D_demand = demand
        return params, 'Demand spike timing shifted one month earlier'
    raise ValueError(f'Unknown variant: {variant}')


def run_instance_variants(
    base_params: InstanceParameters,
    sample_size: int,
    output_dir: str,
) -> pd.DataFrame:
    print('\n' + '=' * 80)
    print('STYLIZED INSTANCE VARIANTS')
    print('=' * 80)
    variants = ['base', 'lower_reception', 'higher_emergency_cost', 'early_spikes']
    rows = []
    out_path = os.path.join(output_dir, 'instance_variants.csv')
    for variant in variants:
        params, description = make_variant(base_params, variant)
        print(f'\nVariant: {variant} ({description})')
        scenarios = generate_burst_scenarios(params, max_scenarios=None)
        sampled = deterministic_even_sample(scenarios, sample_size)
        sol = solve_robust_model(params, sampled)
        row = solution_to_row(
            variant,
            sol,
            {'variant': variant, 'description': description, 'sample_size': len(sampled)},
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    save_latex_table(
        df,
        'instance_variants',
        latex_df=paper_variant_table(df),
        escape=False,
    )
    return df


def run_extended_design_comparison(
    params: InstanceParameters,
    all_scenarios: List[DisruptionScenario],
    sample_size: int,
    output_dir: str,
) -> pd.DataFrame:
    print('\n' + '=' * 80)
    print('EXTENDED DETERMINISTIC SAMPLE DESIGN')
    print('=' * 80)
    rows = []
    base_row = existing_design_row(1000)
    if base_row is not None:
        rows.append(base_row)

    row = existing_design_row(sample_size)
    if row is None:
        print(f'No existing N={sample_size} deterministic design found; solving it.')
        scenarios = deterministic_even_sample(all_scenarios, sample_size)
        start = time.time()
        sol = solve_robust_model(params, scenarios)
        wall_time = time.time() - start
        row = solution_to_row(
            f'deterministic_N{sample_size}',
            sol,
            {'sample_size': sample_size, 'wall_time_s': wall_time},
        )
        row.pop('label', None)
        pd.DataFrame([row]).to_csv(
            os.path.join(output_dir, f'deterministic_N{sample_size}_design.csv'),
            index=False,
        )
    rows.append(row)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=['sample_size'], keep='last').sort_values('sample_size')
    df.to_csv(os.path.join(output_dir, 'extended_sampling_design.csv'), index=False)
    save_latex_table(
        df,
        'extended_sampling_design',
        latex_df=paper_extended_design_table(df),
        escape=False,
    )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-sample-size', type=int, default=500)
    parser.add_argument('--random-seeds', default='0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--holdout-design-size', type=int, default=1000)
    parser.add_argument(
        '--holdout-size',
        type=int,
        default=2000,
        help='Use 0 or a value >= full enumeration size for full enumeration.',
    )
    parser.add_argument('--variant-sample-size', type=int, default=400)
    parser.add_argument('--extended-design-size', type=int, default=2000)
    parser.add_argument(
        '--skip-random',
        action='store_true',
        help='Skip random-seed robustness solves.',
    )
    parser.add_argument(
        '--skip-holdout',
        action='store_true',
        help='Skip holdout evaluation.',
    )
    parser.add_argument(
        '--skip-variants',
        action='store_true',
        help='Skip stylized instance variants.',
    )
    parser.add_argument(
        '--skip-extended-design',
        action='store_true',
        help='Skip deterministic larger-sample design comparison.',
    )
    args = parser.parse_args()

    os.makedirs(RESULT_ROOT, exist_ok=True)
    params = generate_small_instance()
    print('Generating full Gamma=2 burst scenario enumeration...')
    all_scenarios = generate_burst_scenarios(params, max_scenarios=None)
    print(f'Available scenarios: {len(all_scenarios)}')

    with open(os.path.join(RESULT_ROOT, 'metadata.json'), 'w') as f:
        json.dump({
            'random_sample_size': args.random_sample_size,
            'random_seeds': args.random_seeds,
            'holdout_design_size': args.holdout_design_size,
            'holdout_size': args.holdout_size,
            'variant_sample_size': args.variant_sample_size,
            'total_burst_scenarios_available': len(all_scenarios),
        }, f, indent=2)

    if not args.skip_random:
        seeds = [int(x.strip()) for x in args.random_seeds.split(',') if x.strip()]
        run_random_seed_robustness(
            params,
            all_scenarios,
            args.random_sample_size,
            seeds,
            RESULT_ROOT,
        )

    if not args.skip_holdout:
        holdout_size = None if args.holdout_size <= 0 else args.holdout_size
        run_holdout_evaluation(
            params,
            all_scenarios,
            args.holdout_design_size,
            holdout_size,
            RESULT_ROOT,
        )

    if not args.skip_variants:
        run_instance_variants(params, args.variant_sample_size, RESULT_ROOT)

    if not args.skip_extended_design:
        run_extended_design_comparison(
            params,
            all_scenarios,
            args.extended_design_size,
            RESULT_ROOT,
        )

    print('\nDONE')
    print(f'Raw outputs: {RESULT_ROOT}')
    print(f'Paper tables: {TABLE_DIR}')


if __name__ == '__main__':
    main()
