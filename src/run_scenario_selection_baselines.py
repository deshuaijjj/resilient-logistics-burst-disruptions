"""Run external finite-scenario selection baselines on the burst library.

This script deliberately treats the existing results in ``results/08_*`` and
``results/09_*`` as read-only references.  It creates new designs by selecting
the same number of scenarios with simple literature-grounded policies, solves
each selected-scenario master once, and evaluates only those new designs on the
unchanged full burst-event library.

The implemented selectors are:

* ``random``: uniform sampling without replacement;
* ``maxsum_top``: rank by the number of disrupted mode-period cells, with a
  seeded random tie-break;
* ``maxsum_weighted``: sample without replacement with probability
  proportional to squared disruption severity.  This mirrors the Maxsum
  weighting protocol of Goerigk and Kurtz (2025), adapted from objective-cost
  scenarios to binary disruption matrices;
* ``kmeans``: MiniBatch K-means on flattened disruption matrices, followed by
  selection of the nearest observed scenario to each nonempty centroid.

No selector is an exact worst-case oracle.  Full-library certification remains
the common evaluation oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence

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
from lp_solver import Solution, solve_robust_model  # noqa: E402
from recourse_certifier import (  # noqa: E402
    CertificationResult,
    certify_fixed_design,
)


DEFAULT_METHODS = ('random', 'maxsum_top', 'maxsum_weighted')
VALID_METHODS = DEFAULT_METHODS + ('kmeans',)


def parse_csv_strings(value: str) -> List[str]:
    return [part.strip() for part in value.split(',') if part.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(',') if part.strip()]


def disruption_matrix(
    scenario: DisruptionScenario,
    num_modes: int,
    num_periods: int,
) -> np.ndarray:
    return scenario.to_delta_matrix(num_modes, num_periods).reshape(-1)


def build_disruption_matrix(
    scenarios: Sequence[DisruptionScenario],
    num_modes: int,
    num_periods: int,
) -> np.ndarray:
    """Return the flattened binary disruption matrix for every scenario."""
    return np.stack([
        disruption_matrix(scenario, num_modes, num_periods)
        for scenario in scenarios
    ]).astype(np.uint8, copy=False)


def validate_sample_size(num_scenarios: int, sample_size: int) -> None:
    if sample_size < 1:
        raise ValueError('sample_size must be at least one')
    if sample_size > num_scenarios:
        raise ValueError(
            f'sample_size={sample_size} exceeds library size={num_scenarios}'
        )


def _with_nominal(selected_non_nominal: Iterable[int]) -> List[int]:
    """Return sorted full-library IDs with the nominal scenario first."""
    selected = sorted({int(i) for i in selected_non_nominal if int(i) != 0})
    return [0] + selected


def select_random_ids(
    num_scenarios: int,
    sample_size: int,
    seed: int,
) -> List[int]:
    validate_sample_size(num_scenarios, sample_size)
    if sample_size == 1:
        return [0]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        np.arange(1, num_scenarios, dtype=int),
        size=sample_size - 1,
        replace=False,
    )
    return _with_nominal(chosen)


def select_maxsum_top_ids(
    matrix: np.ndarray,
    sample_size: int,
    seed: int,
) -> List[int]:
    """Select highest-severity scenarios with a seeded random tie-break."""
    validate_sample_size(matrix.shape[0], sample_size)
    if sample_size == 1:
        return [0]
    candidate_ids = np.arange(1, matrix.shape[0], dtype=int)
    severity = matrix[1:].sum(axis=1)
    rng = np.random.default_rng(seed)
    tie_break = rng.random(len(candidate_ids))
    order = np.lexsort((tie_break, -severity))
    chosen = candidate_ids[order[:sample_size - 1]]
    return _with_nominal(chosen)


def select_maxsum_weighted_ids(
    matrix: np.ndarray,
    sample_size: int,
    seed: int,
) -> List[int]:
    """Sample scenarios with probability proportional to severity squared."""
    validate_sample_size(matrix.shape[0], sample_size)
    if sample_size == 1:
        return [0]
    candidate_ids = np.arange(1, matrix.shape[0], dtype=int)
    severity = matrix[1:].sum(axis=1).astype(float)
    weights = np.square(severity)
    if not np.any(weights > 0):
        return select_random_ids(matrix.shape[0], sample_size, seed)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        candidate_ids,
        size=sample_size - 1,
        replace=False,
        p=weights / weights.sum(),
    )
    return _with_nominal(chosen)


def select_kmeans_ids(
    matrix: np.ndarray,
    sample_size: int,
    seed: int,
) -> List[int]:
    """Select observed scenarios nearest to MiniBatch K-means centroids."""
    validate_sample_size(matrix.shape[0], sample_size)
    if sample_size == 1:
        return [0]
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError('kmeans requires scikit-learn') from exc

    points = matrix[1:].astype(np.float32, copy=False)
    num_clusters = sample_size - 1
    model = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=seed,
        batch_size=min(4096, len(points)),
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.01,
    )
    labels = model.fit_predict(points)
    centers = model.cluster_centers_

    best_distance = np.full(num_clusters, np.inf)
    best_local_id = np.full(num_clusters, -1, dtype=int)
    for local_id, cluster_id in enumerate(labels):
        distance = float(np.sum(np.square(points[local_id] - centers[cluster_id])))
        if distance < best_distance[cluster_id]:
            best_distance[cluster_id] = distance
            best_local_id[cluster_id] = local_id

    chosen = {int(local_id + 1) for local_id in best_local_id if local_id >= 0}
    if len(chosen) < num_clusters:
        # Empty clusters are filled deterministically from severe unselected
        # scenarios.  The fill policy is recorded through the selected IDs.
        severity = matrix[1:].sum(axis=1)
        rng = np.random.default_rng(seed)
        tie_break = rng.random(len(points))
        order = np.lexsort((tie_break, -severity))
        for local_id in order:
            chosen.add(int(local_id + 1))
            if len(chosen) == num_clusters:
                break

    return _with_nominal(chosen)


def select_scenario_ids(
    method: str,
    matrix: np.ndarray,
    sample_size: int,
    seed: int,
) -> List[int]:
    if method == 'random':
        return select_random_ids(matrix.shape[0], sample_size, seed)
    if method == 'maxsum_top':
        return select_maxsum_top_ids(matrix, sample_size, seed)
    if method == 'maxsum_weighted':
        return select_maxsum_weighted_ids(matrix, sample_size, seed)
    if method == 'kmeans':
        return select_kmeans_ids(matrix, sample_size, seed)
    raise ValueError(f'Unknown method {method!r}; choose from {VALID_METHODS}')


def hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def selection_summary(matrix: np.ndarray, selected_ids: Sequence[int]) -> Dict:
    selected = matrix[np.asarray(selected_ids, dtype=int)]
    severity = selected.sum(axis=1)
    unique_count = len({row.tobytes() for row in selected})
    return {
        'selected_scenarios': len(selected_ids),
        'selected_unique_matrices': unique_count,
        'selected_duplicate_matrices': len(selected_ids) - unique_count,
        'severity_min': int(np.min(severity)),
        'severity_mean': float(np.mean(severity)),
        'severity_median': float(np.median(severity)),
        'severity_max': int(np.max(severity)),
    }


def solution_json(sol: Solution) -> Dict:
    return {
        'total_cost': float(sol.total_cost),
        'investment_cost': float(sol.investment_cost),
        'worst_case_cost': float(sol.worst_case_cost),
        'C_depot': sol.C_depot.tolist(),
        'q_mode': sol.q_mode.tolist(),
        'worst_scenario_id_in_selected_set': int(sol.worst_scenario_id),
        'worst_scenario_events': sol.worst_scenario_events,
        'solve_time_s': float(sol.solve_time),
        'num_scenarios': int(sol.num_scenarios),
        'lp_status': sol.lp_status,
    }


def certification_json(cert: CertificationResult) -> Dict:
    data = asdict(cert)
    return data


def result_row(
    label: str,
    method: str,
    seed: int,
    selection_time_s: float,
    selected_ids: Sequence[int],
    matrix: np.ndarray,
    sol: Solution,
    cert: CertificationResult,
    reference_total_cost_m: float | None,
) -> Dict:
    train_worst_full_id = int(selected_ids[int(sol.worst_scenario_id)])
    row = {
        'label': label,
        'method': method,
        'seed': seed,
        **selection_summary(matrix, selected_ids),
        'selection_time_s': selection_time_s,
        'solve_time_s': sol.solve_time,
        'certification_time_s': cert.eval_time_s,
        'training_total_cost_M': sol.total_cost / 1e6,
        'training_worst_operating_cost_M': sol.worst_case_cost / 1e6,
        'certified_total_cost_M': cert.full_total_cost / 1e6,
        'certified_worst_operating_cost_M': cert.full_worst_operating_cost / 1e6,
        'certification_gap_M': cert.certification_gap / 1e6,
        'certification_gap_pct': cert.certification_gap_pct,
        'worst_unmet_tons': cert.worst.unmet_tons,
        'training_worst_full_scenario_id': train_worst_full_id,
        'certified_worst_scenario_id': cert.worst.scenario_id,
        'contains_reference_worst_1220': 1220 in selected_ids,
        'contains_reference_worst_1295': 1295 in selected_ids,
        'C_depot_0': sol.C_depot[0],
        'C_depot_1': sol.C_depot[1] if len(sol.C_depot) > 1 else 0.0,
        'q_mode_0': sol.q_mode[0],
        'q_mode_1': sol.q_mode[1] if len(sol.q_mode) > 1 else 0.0,
        'q_mode_2': sol.q_mode[2] if len(sol.q_mode) > 2 else 0.0,
        'lp_status': sol.lp_status,
        'certification_status': cert.status,
    }
    if reference_total_cost_m is not None:
        row['certified_gap_to_existing_ccg_M'] = (
            row['certified_total_cost_M'] - reference_total_cost_m
        )
        row['certified_gap_to_existing_ccg_pct'] = (
            100.0 * row['certified_gap_to_existing_ccg_M'] / reference_total_cost_m
        )
    return row


def load_existing_ccg_reference(path: str) -> Dict | None:
    if not path or not os.path.exists(path):
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    row = frame.iloc[0].to_dict()
    return {
        'source_path': os.path.abspath(path),
        'certified_total_cost_M': float(row['certified_total_cost_M']),
        'worst_unmet_tons': float(row['worst_unmet_tons']),
        'iterations': int(row['iterations']),
        'scenarios_used': int(row['scenarios_used']),
        'total_wall_time_s': float(row['total_wall_time_s']),
    }


def write_progress(
    output_root: str,
    rows: Sequence[Dict],
    metadata: Dict,
    run_details: Dict,
) -> None:
    os.makedirs(output_root, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(output_root, 'summary.csv'), index=False)
    with open(os.path.join(output_root, 'metadata.json'), 'w') as stream:
        json.dump(metadata, stream, indent=2)
    with open(os.path.join(output_root, 'detailed_results.json'), 'w') as stream:
        json.dump(run_details, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=1002)
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS))
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument(
        '--output-root',
        default=os.path.join(PROJECT_ROOT, 'results', '10_scenario_selection_baselines'),
    )
    parser.add_argument(
        '--reference-ccg',
        default=os.path.join(PROJECT_ROOT, 'results', '09_burst_ccg', 'ccg_summary.csv'),
    )
    parser.add_argument(
        '--scenario-limit',
        type=int,
        default=0,
        help='Limit full-library certification for smoke tests; 0 uses all scenarios.',
    )
    parser.add_argument(
        '--selection-only',
        action='store_true',
        help='Generate and audit selected IDs without solving optimization models.',
    )
    parser.add_argument(
        '--rerun-existing',
        action='store_true',
        help='Rerun labels already present in the new summary file.',
    )
    args = parser.parse_args()

    methods = parse_csv_strings(args.methods)
    unknown = sorted(set(methods) - set(VALID_METHODS))
    if unknown:
        raise SystemExit(f'Unknown methods: {unknown}; choose from {VALID_METHODS}')
    seeds = parse_csv_ints(args.seeds)
    if not seeds:
        raise SystemExit('At least one seed is required.')

    params = generate_small_instance()
    scenarios = generate_burst_scenarios(params, max_scenarios=None)
    validate_sample_size(len(scenarios), args.sample_size)
    print('Building disruption matrix...')
    matrix = build_disruption_matrix(scenarios, params.M, params.T)
    library_unique = len({row.tobytes() for row in matrix})

    reference = load_existing_ccg_reference(args.reference_ccg)
    reference_cost_m = None if reference is None else reference['certified_total_cost_M']
    metadata = {
        'purpose': 'external scenario-selection baselines; existing 08/09 are read-only',
        'sample_size': args.sample_size,
        'methods': methods,
        'seeds': seeds,
        'full_event_sets': len(scenarios),
        'full_unique_disruption_matrices': library_unique,
        'matrix_sha256': hash_array(matrix),
        'scenario_limit': args.scenario_limit,
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'existing_ccg_reference': reference,
        'literature_boundary': {
            'random': 'uniform scenario initialization baseline',
            'maxsum_top': 'deterministic ranking adaptation of Maxsum severity',
            'maxsum_weighted': (
                'squared-weight sampling adaptation of Goerigk-Kurtz Maxsum; '
                'binary disruption severity replaces objective-cost sum'
            ),
            'kmeans': 'generic clustering baseline; no Goerigk-Khosravi guarantee transfers',
        },
    }

    os.makedirs(args.output_root, exist_ok=True)
    summary_path = os.path.join(args.output_root, 'summary.csv')
    if os.path.exists(summary_path):
        existing_rows = pd.read_csv(summary_path).to_dict(orient='records')
    else:
        existing_rows = []
    rows = list(existing_rows)
    completed_labels = {str(row['label']) for row in rows}

    detail_path = os.path.join(args.output_root, 'detailed_results.json')
    if os.path.exists(detail_path):
        with open(detail_path) as stream:
            run_details = json.load(stream)
    else:
        run_details = {'runs': {}}

    selection_dir = os.path.join(args.output_root, 'selected_ids')
    os.makedirs(selection_dir, exist_ok=True)

    for method in methods:
        for seed in seeds:
            label = f'{method}_K{args.sample_size}_seed{seed}'
            if label in completed_labels and not args.rerun_existing:
                print(f'Skipping completed new baseline {label}')
                continue

            print('\n' + '=' * 80)
            print(f'NEW EXTERNAL BASELINE: {label}')
            print('=' * 80)
            selection_start = time.time()
            selected_ids = select_scenario_ids(
                method=method,
                matrix=matrix,
                sample_size=args.sample_size,
                seed=seed,
            )
            selection_time = time.time() - selection_start
            if len(selected_ids) != args.sample_size:
                raise RuntimeError(
                    f'{label} selected {len(selected_ids)} scenarios, '
                    f'expected {args.sample_size}'
                )
            selection_data = {
                'label': label,
                'method': method,
                'seed': seed,
                'selected_ids': selected_ids,
                **selection_summary(matrix, selected_ids),
            }
            with open(os.path.join(selection_dir, f'{label}.json'), 'w') as stream:
                json.dump(selection_data, stream, indent=2)
            print(json.dumps({k: v for k, v in selection_data.items() if k != 'selected_ids'}, indent=2))

            if args.selection_only:
                continue

            selected_scenarios = [scenarios[i] for i in selected_ids]
            sol = solve_robust_model(params, selected_scenarios)
            if sol.lp_status != 'optimal':
                raise RuntimeError(f'{label} master status: {sol.lp_status}')

            scenario_ids = None
            if args.scenario_limit > 0:
                scenario_ids = range(min(args.scenario_limit, len(scenarios)))
            cert, _ = certify_fixed_design(
                params=params,
                scenarios=scenarios,
                C_depot=sol.C_depot,
                q_mode=sol.q_mode,
                design_name=label,
                sampled_total_cost=sol.total_cost,
                sampled_worst_operating_cost=sol.worst_case_cost,
                scenario_ids=scenario_ids,
                progress_interval=500,
            )
            row = result_row(
                label=label,
                method=method,
                seed=seed,
                selection_time_s=selection_time,
                selected_ids=selected_ids,
                matrix=matrix,
                sol=sol,
                cert=cert,
                reference_total_cost_m=reference_cost_m,
            )
            rows = [existing for existing in rows if str(existing['label']) != label]
            rows.append(row)
            completed_labels.add(label)
            run_details['runs'][label] = {
                'selection': selection_data,
                'solution': solution_json(sol),
                'certification': certification_json(cert),
            }
            write_progress(args.output_root, rows, metadata, run_details)
            print(pd.DataFrame([row]).to_string(index=False))

    # Selection-only runs still retain full provenance and selected IDs.
    if args.selection_only:
        with open(os.path.join(args.output_root, 'metadata.json'), 'w') as stream:
            json.dump(metadata, stream, indent=2)


if __name__ == '__main__':
    main()
