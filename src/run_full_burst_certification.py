"""
Evaluate fixed first-stage designs on a burst scenario library.

This script is intended for the revised paper workflow: designs may be produced
from sampled extensive-form solves, but each candidate design is then frozen and
tested against the same burst scenario library.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.append(SRC_DIR)

from instance_generator import generate_burst_scenarios, generate_small_instance  # noqa: E402
from recourse_certifier import (  # noqa: E402
    first_stage_cost,
    certification_to_jsonable,
    certification_to_summary_row,
    certify_fixed_design,
)


DEFAULT_RESULT_FILES = [
    os.path.join(
        PROJECT_ROOT,
        'results',
        '02_baselines',
        'small_T24_M3_J2_K2_Gamma2_N1000_sampled',
        'detailed_results.json',
    ),
    os.path.join(
        PROJECT_ROOT,
        'results',
        '06_fair_baselines',
        'small_T24_M3_J2_K2_Gamma2_N1000_sampled',
        'detailed_results.json',
    ),
]


def parse_strategy_filter(value: str) -> set[str] | None:
    if not value or value.strip().lower() in {'all', '*'}:
        return None
    return {part.strip() for part in value.split(',') if part.strip()}


def load_designs(paths: Iterable[str], strategy_filter: set[str] | None = None) -> List[Dict]:
    """Load first-stage decisions from one or more detailed result JSON files."""
    designs = []
    seen = set()
    for path in paths:
        if not os.path.exists(path):
            print(f'Warning: missing result file {path}')
            continue
        with open(path) as f:
            data = json.load(f)
        results = data.get('results', data)
        for name, result in results.items():
            if strategy_filter is not None and name not in strategy_filter:
                continue
            key = name
            if key in seen:
                print(f'Skipping duplicate design label {name} from {path}')
                continue
            seen.add(key)
            designs.append({
                'design_label': name,
                'source_path': path,
                'sample_num_scenarios': int(result.get('num_scenarios', 0)),
                'C_depot': np.array(result['C_depot'], dtype=float),
                'q_mode': np.array(result['q_mode'], dtype=float),
                'sampled_total_cost': float(result.get('total_cost', np.nan)),
                'sampled_worst_operating_cost': float(result.get('worst_case_cost', np.nan)),
            })
    return designs


def finite_or_none(value: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result-json',
        action='append',
        default=None,
        help='Detailed result JSON containing C_depot/q_mode decisions. May repeat.',
    )
    parser.add_argument(
        '--strategies',
        default='nominal,independent,burst,burst_joint,inventory_only_nominal_q,'
        'inventory_only_balanced_q,redundancy_only_nominal_C,budget_matched_inventory,'
        'budget_matched_redundancy',
        help='Comma-separated strategy names to certify, or all.',
    )
    parser.add_argument(
        '--output-root',
        default=os.path.join(PROJECT_ROOT, 'results', '08_full_burst_certification'),
    )
    parser.add_argument(
        '--scenario-limit',
        type=int,
        default=0,
        help='Use first N scenarios for smoke tests. 0 means full enumeration.',
    )
    parser.add_argument(
        '--save-scenario-costs',
        action='store_true',
        help='Write per-scenario recourse costs for each design.',
    )
    parser.add_argument('--progress-interval', type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    params = generate_small_instance()
    all_scenarios = generate_burst_scenarios(params, max_scenarios=None)
    scenario_ids = None
    if args.scenario_limit and args.scenario_limit > 0:
        scenario_ids = range(min(args.scenario_limit, len(all_scenarios)))
        scenario_label = len(list(scenario_ids))
        scenario_ids = range(scenario_label)
    else:
        scenario_label = len(all_scenarios)

    paths = args.result_json if args.result_json else DEFAULT_RESULT_FILES
    strategy_filter = parse_strategy_filter(args.strategies)
    designs = load_designs(paths, strategy_filter=strategy_filter)
    if not designs:
        raise SystemExit('No designs found to certify.')

    print(f'Certifying {len(designs)} designs on {scenario_label:,} scenarios.')
    summary_rows = []
    detailed = {
        'metadata': {
            'scenario_limit': args.scenario_limit,
            'certification_scenarios': scenario_label,
            'total_burst_scenarios_available': len(all_scenarios),
            'result_json': paths,
            'strategies': args.strategies,
        },
        'certifications': {},
    }
    worst_rows = []
    for design in designs:
        label = design['design_label']
        print('\n' + '=' * 80)
        print(f'CERTIFYING: {label}')
        print('=' * 80)
        start = time.time()
        scenario_cost_path = None
        if args.save_scenario_costs:
            scenario_cost_path = os.path.join(args.output_root, f'scenario_costs_{label}.csv')

        try:
            cert, _ = certify_fixed_design(
                params=params,
                scenarios=all_scenarios,
                C_depot=design['C_depot'],
                q_mode=design['q_mode'],
                design_name=label,
                sampled_total_cost=finite_or_none(design['sampled_total_cost']),
                sampled_worst_operating_cost=finite_or_none(design['sampled_worst_operating_cost']),
                scenario_ids=scenario_ids,
                progress_interval=args.progress_interval,
                save_scenario_costs_path=scenario_cost_path,
            )
        except ValueError as exc:
            wall_time = time.time() - start
            C = design['C_depot']
            q = design['q_mode']
            row = {
                'design_label': label,
                'design_source': design['source_path'],
                'design_sample_size': design['sample_num_scenarios'],
                'certification_scenarios': scenario_label,
                'investment_cost_M': first_stage_cost(params, C, q) / 1e6,
                'optimization_total_cost_M': (
                    None if finite_or_none(design['sampled_total_cost']) is None
                    else finite_or_none(design['sampled_total_cost']) / 1e6
                ),
                'optimization_worst_operating_cost_M': (
                    None if finite_or_none(design['sampled_worst_operating_cost']) is None
                    else finite_or_none(design['sampled_worst_operating_cost']) / 1e6
                ),
                'certified_total_cost_M': None,
                'certified_worst_operating_cost_M': None,
                'cost_gap_M': None,
                'relative_gap_pct': None,
                'operating_gap_M': None,
                'C_depot_0': C[0] if len(C) > 0 else 0.0,
                'C_depot_1': C[1] if len(C) > 1 else 0.0,
                'q_mode_0': q[0] if len(q) > 0 else 0.0,
                'q_mode_1': q[1] if len(q) > 1 else 0.0,
                'q_mode_2': q[2] if len(q) > 2 else 0.0,
                'worst_scenario_id': None,
                'num_worst_events': None,
                'worst_events_json': None,
                'worst_unmet_tons': None,
                'eval_time_s': wall_time,
                'lp_status': f'invalid_design: {exc}',
                'wall_time_s': wall_time,
            }
            summary_rows.append(row)
            detailed['certifications'][label] = {'status': row['lp_status']}
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(args.output_root, 'full_certification_summary.csv'),
                index=False,
            )
            with open(os.path.join(args.output_root, 'full_certification_detailed.json'), 'w') as f:
                json.dump(detailed, f, indent=2)
            print(f'Skipping invalid design {label}: {exc}')
            continue
        wall_time = time.time() - start
        row = certification_to_summary_row(
            cert,
            design['C_depot'],
            design['q_mode'],
            source_path=design['source_path'],
            design_sample_size=design['sample_num_scenarios'],
        )
        row['wall_time_s'] = wall_time
        summary_rows.append(row)
        worst_rows.append({
            'design_label': label,
            'scenario_id': cert.worst.scenario_id,
            'num_events': len(cert.worst.events),
            'events': json.dumps(cert.worst.events),
            'operating_cost_M': cert.worst.operating_cost / 1e6,
            'total_cost_M': cert.full_total_cost / 1e6,
            'unmet_tons': cert.worst.unmet_tons,
            'status': cert.worst.status,
        })
        detailed['certifications'][label] = certification_to_jsonable(cert)
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(args.output_root, 'full_certification_summary.csv'),
            index=False,
        )
        pd.DataFrame(worst_rows).to_csv(
            os.path.join(args.output_root, 'full_certification_worst_scenarios.csv'),
            index=False,
        )
        with open(os.path.join(args.output_root, 'full_certification_detailed.json'), 'w') as f:
            json.dump(detailed, f, indent=2)

    print('\nDONE')
    print(f'Outputs: {args.output_root}')


if __name__ == '__main__':
    main()
