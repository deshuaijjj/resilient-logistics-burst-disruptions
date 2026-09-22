"""Construct and solve the exact dominance-pruned burst uncertainty set.

For this model, disruption matrix ``Delta`` only tightens nonnegative mode
capacity constraints through ``q_m * (1 - Delta_mt)``.  For a fixed first-stage
design, if ``Delta_a <= Delta_b`` componentwise, every recourse solution
feasible under ``Delta_b`` is feasible under ``Delta_a``.  Therefore

    Q(C, q; Delta_a) <= Q(C, q; Delta_b).

Any disruption matrix dominated by another matrix can consequently be removed
without changing the finite-library worst-case value for any design.  The
robust master may be built using only componentwise-maximal matrices.

The script writes only to a new output directory.  Existing 08/09 experiments
are loaded as read-only references and are never rerun or overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import scipy

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.append(SRC_DIR)

from instance_generator import (  # noqa: E402
    DisruptionScenario,
    generate_burst_scenarios,
    generate_small_instance,
)
from lp_solver import solve_robust_model  # noqa: E402
from recourse_certifier import certify_fixed_design  # noqa: E402


@dataclass(frozen=True)
class CanonicalMatrix:
    bits: int
    severity: int
    representative_id: int
    event_set_ids: tuple[int, ...]


def scenario_bits(scenario: DisruptionScenario, num_modes: int, num_periods: int) -> int:
    """Encode a binary disruption matrix as one Python integer."""
    flat = scenario.to_delta_matrix(num_modes, num_periods).reshape(-1)
    bits = 0
    for position in np.flatnonzero(flat):
        bits |= 1 << int(position)
    return bits


def canonicalize_scenarios(
    scenarios: Sequence[DisruptionScenario],
    num_modes: int,
    num_periods: int,
) -> List[CanonicalMatrix]:
    """Collapse duplicate event-set encodings into unique disruption matrices."""
    grouped: Dict[int, List[int]] = {}
    for scenario_id, scenario in enumerate(scenarios):
        bits = scenario_bits(scenario, num_modes, num_periods)
        grouped.setdefault(bits, []).append(scenario_id)
    return [
        CanonicalMatrix(
            bits=bits,
            severity=bin(bits).count('1'),  # == bits.bit_count(); portable to Python < 3.10
            representative_id=ids[0],
            event_set_ids=tuple(ids),
        )
        for bits, ids in grouped.items()
    ]


def is_subset_bits(candidate: int, possible_superset: int) -> bool:
    """Return true when every disrupted cell in candidate is in the superset."""
    return candidate & ~possible_superset == 0


def maximal_matrices(records: Sequence[CanonicalMatrix]) -> List[CanonicalMatrix]:
    """Return all componentwise-maximal unique disruption matrices.

    Records are processed by decreasing cardinality.  A distinct equal-size
    binary set cannot contain another, so every possible strict superset has
    already been considered when a record is tested.
    """
    ordered = sorted(
        records,
        key=lambda record: (-record.severity, record.representative_id),
    )
    maximal: List[CanonicalMatrix] = []
    for record in ordered:
        if any(
            record.bits != incumbent.bits
            and is_subset_bits(record.bits, incumbent.bits)
            for incumbent in maximal
        ):
            continue
        maximal.append(record)
    return maximal


def hash_canonical_records(records: Sequence[CanonicalMatrix]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.bits):
        byte_count = max(1, (record.bits.bit_length() + 7) // 8)
        digest.update(record.bits.to_bytes(byte_count, 'little'))
        digest.update(b'\0')
    return digest.hexdigest()


def severity_counts(records: Sequence[CanonicalMatrix]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        key = str(record.severity)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def load_reference(path: str) -> Dict | None:
    if not path or not os.path.exists(path):
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    row = frame.iloc[0]
    return {
        'source_path': os.path.abspath(path),
        'certified_total_cost_M': float(row['certified_total_cost_M']),
        'worst_unmet_tons': float(row['worst_unmet_tons']),
        'iterations': int(row['iterations']),
        'scenarios_used': int(row['scenarios_used']),
        'total_wall_time_s': float(row['total_wall_time_s']),
        'C_depot': [float(row['C_depot_0']), float(row['C_depot_1'])],
        'q_mode': [
            float(row['q_mode_0']),
            float(row['q_mode_1']),
            float(row['q_mode_2']),
        ],
    }


def write_json(path: str, data: Dict | List) -> None:
    with open(path, 'w') as stream:
        json.dump(data, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output-root',
        default=os.path.join(PROJECT_ROOT, 'results', '11_dominance_pruned_oracle'),
    )
    parser.add_argument(
        '--reference-ccg',
        default=os.path.join(PROJECT_ROOT, 'results', '09_burst_ccg', 'ccg_summary.csv'),
    )
    parser.add_argument(
        '--solve-exact-master',
        action='store_true',
        help='Solve one new extensive-form master on maximal matrices only.',
    )
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    params = generate_small_instance()
    scenarios = generate_burst_scenarios(params, max_scenarios=None)

    prune_start = time.time()
    canonical = canonicalize_scenarios(scenarios, params.M, params.T)
    maximal = maximal_matrices(canonical)
    prune_time = time.time() - prune_start
    maximal_ids = sorted(record.representative_id for record in maximal)
    reference = load_reference(args.reference_ccg)

    summary = {
        'proof_scope': {
            'ordering': 'Delta_a <= Delta_b componentwise',
            'capacity_rhs': 'q_m * (1 - Delta_mt)',
            'assumptions': [
                'q_m is nonnegative',
                'Delta appears only by tightening mode-capacity constraints',
                'all other recourse constraints and costs are Delta-independent',
                'recourse minimizes operating cost',
            ],
            'consequence': (
                'feasible(Delta_b) subset feasible(Delta_a), hence '
                'Q(Delta_a) <= Q(Delta_b); dominated matrices are exact-prunable'
            ),
        },
        'event_set_count': len(scenarios),
        'unique_matrix_count': len(canonical),
        'duplicate_event_set_encodings': len(scenarios) - len(canonical),
        'maximal_matrix_count': len(maximal),
        'dominated_unique_matrix_count': len(canonical) - len(maximal),
        'unique_matrix_pruning_pct': 100.0 * (len(canonical) - len(maximal)) / len(canonical),
        'event_set_pruning_pct': 100.0 * (len(scenarios) - len(maximal)) / len(scenarios),
        'maximal_severity_counts': severity_counts(maximal),
        'pruning_time_s': prune_time,
        'canonical_sha256': hash_canonical_records(canonical),
        'maximal_sha256': hash_canonical_records(maximal),
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'existing_ccg_reference': reference,
    }
    write_json(os.path.join(args.output_root, 'dominance_summary.json'), summary)
    write_json(
        os.path.join(args.output_root, 'maximal_scenarios.json'),
        [
            {
                'representative_id': record.representative_id,
                'severity': record.severity,
                'event_set_ids': list(record.event_set_ids),
                'events': [
                    [int(value) for value in event]
                    for event in scenarios[record.representative_id].events
                ],
            }
            for record in sorted(maximal, key=lambda item: item.representative_id)
        ],
    )
    print(json.dumps(summary, indent=2))

    if not args.solve_exact_master:
        return

    total_start = time.time()
    selected_scenarios = [scenarios[scenario_id] for scenario_id in maximal_ids]
    solution = solve_robust_model(params, selected_scenarios)
    if solution.lp_status != 'optimal':
        raise RuntimeError(f'Dominance-pruned master failed: {solution.lp_status}')

    certification, _ = certify_fixed_design(
        params=params,
        scenarios=scenarios,
        C_depot=solution.C_depot,
        q_mode=solution.q_mode,
        design_name='dominance_pruned_exact',
        sampled_total_cost=solution.total_cost,
        sampled_worst_operating_cost=solution.worst_case_cost,
        scenario_ids=maximal_ids,
        progress_interval=250,
    )
    total_wall = time.time() - total_start
    result = {
        'method': 'dominance_pruned_exact_extensive_form',
        'exactness_scope': 'enumerated finite burst-event library',
        'master_scenarios': len(maximal_ids),
        'master_total_cost_M': solution.total_cost / 1e6,
        'certified_total_cost_M': certification.full_total_cost / 1e6,
        'certification_gap_M': certification.certification_gap / 1e6,
        'certification_gap_pct': certification.certification_gap_pct,
        'worst_unmet_tons': certification.worst.unmet_tons,
        'worst_scenario_id': certification.worst.scenario_id,
        'C_depot': solution.C_depot.tolist(),
        'q_mode': solution.q_mode.tolist(),
        'master_solve_time_s': solution.solve_time,
        'maximal_set_certification_time_s': certification.eval_time_s,
        'pruning_time_s': prune_time,
        'total_wall_time_s': total_wall,
        'lp_status': solution.lp_status,
        'certification_status': certification.status,
        'solution': {
            'investment_cost': solution.investment_cost,
            'worst_case_cost': solution.worst_case_cost,
            'total_cost': solution.total_cost,
            'worst_scenario_id_in_pruned_master': solution.worst_scenario_id,
            'worst_scenario_events': solution.worst_scenario_events,
        },
        'certification': asdict(certification),
    }
    if reference is not None:
        result['comparison_to_existing_ccg'] = {
            'objective_gap_M': (
                result['certified_total_cost_M'] - reference['certified_total_cost_M']
            ),
            'C_max_abs_difference': float(np.max(np.abs(
                solution.C_depot - np.asarray(reference['C_depot'])
            ))),
            'q_max_abs_difference': float(np.max(np.abs(
                solution.q_mode - np.asarray(reference['q_mode'])
            ))),
            'wall_time_ratio_existing_ccg_over_pruned': (
                reference['total_wall_time_s'] / total_wall
            ),
        }
    write_json(os.path.join(args.output_root, 'dominance_pruned_result.json'), result)
    pd.DataFrame([{
        key: value
        for key, value in result.items()
        if not isinstance(value, (dict, list))
    }]).to_csv(os.path.join(args.output_root, 'summary.csv'), index=False)
    print('\nDOMINANCE-PRUNED RESULT')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
