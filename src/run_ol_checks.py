"""Two extra checks for the Optimization Letters manuscript (2026-11-10).

Both write only into new result directories and reuse the code paths of the
recorded stages.

  onedesign  results/25_one_sampled_design/
             The design of the first random sample (random_seed_0 of
             results/17_sampling_seeds) evaluated in every one of the 29,916
             scenarios: how many leave demand unmet after month 1, how many
             cost more than the sample reported, and its cost with no outage.
  samesize   results/24_same_size_samples/
             100 random samples of 1,648 scenarios (the size of the kept set),
             same protocol as results/17_sampling_seeds.

  python3 src/run_ol_checks.py --stage onedesign
  python3 src/run_ol_checks.py --stage onedesign --source 24_same_size_samples --out 26_one_sampled_design_same_size
  python3 src/run_ol_checks.py --stage samesize [--seeds 100]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from revision_common import RecourseEvaluator, build_library, environment_record, write_json  # noqa: E402
from run_proano_revision import FROZEN, stage_seeds  # noqa: E402

TOL_T, TOL_D = 1e-6, 1.0


def stage_onedesign(lib, out_dir: str, label: str = 'random_seed_0', source: str = '17_sampling_seeds') -> dict:
    seeds = pd.read_csv(os.path.join(PROJECT_ROOT, 'results', source, 'seeds.csv'))
    row = seeds[seeds['label'] == label].iloc[0]
    C = [row['C_depot_0'], row['C_depot_1']]
    q = [row['q_mode_0'], row['q_mode_1'], row['q_mode_2']]
    ev = RecourseEvaluator(lib.params)
    D = np.asarray(lib.params.D_demand)
    startup_t = float(D[:, :, 0].sum())
    rows = []
    for sid in range(len(lib.scenarios)):
        r = ev.evaluate(C, q, lib.delta(sid))
        d = lib.delta(sid)
        rows.append({'scenario_id': sid, 'status': r.status, 'operating_cost': r.objective,
                     'unmet_tons': r.unmet_tons, 'lines01_overlap_months': int((d[0] * d[1]).sum()),
                     'line0_months': int(d[0].sum()), 'line1_months': int(d[1].sum()),
                     'line2_months': int(d[2].sum())})
    f = pd.DataFrame(rows)
    f.to_csv(os.path.join(out_dir, f'{label}_scenario_costs.csv'), index=False)
    short = f[f['unmet_tons'] > startup_t + TOL_T]
    above = f[f['operating_cost'] > row['in_sample_worst_recourse'] + TOL_D]
    r_opt = ev.evaluate(FROZEN['exact_C'], FROZEN['exact_q'], lib.delta(0))
    opt_invest = FROZEN['exact_total_cost'] - max(
        ev.evaluate(FROZEN['exact_C'], FROZEN['exact_q'], lib.delta(s)).objective for s in (1220, 1295))
    out = {
        'label': label, 'source': source, 'C': C, 'q': q, 'scenarios': int(len(f)),
        'all_optimal': bool((f['status'] == 'optimal').all()),
        'startup_unmet_tons': startup_t,
        'worst_operating_cost': float(f['operating_cost'].max()),
        'matches_recorded_worst': bool(abs(f['operating_cost'].max() - row['complete_library_worst_recourse']) <= TOL_D),
        'scenarios_with_unmet_after_month_1': int(len(short)),
        'share_pct': 100.0 * len(short) / len(f),
        'min_lines01_overlap_among_them': int(short['lines01_overlap_months'].min()) if len(short) else None,
        'scenarios_costing_more_than_reported': int(len(above)),
        'share_costing_more_pct': 100.0 * len(above) / len(f),
        'no_outage_total_cost_sampled': float(row['investment_cost'] + f.loc[0, 'operating_cost']),
        'no_outage_total_cost_optimal': float(opt_invest + r_opt.objective),
        'investment_sampled': float(row['investment_cost']), 'investment_optimal': float(opt_invest),
        'environment': environment_record(),
    }
    out['no_outage_difference'] = out['no_outage_total_cost_optimal'] - out['no_outage_total_cost_sampled']
    write_json(os.path.join(out_dir, 'one_design_summary.json'), out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['onedesign', 'samesize'])
    ap.add_argument('--seeds', type=int, default=100)
    ap.add_argument('--sample-size', type=int, default=1648)
    ap.add_argument('--source', default='17_sampling_seeds', help='onedesign: results directory holding seeds.csv')
    ap.add_argument('--out', default='25_one_sampled_design', help='onedesign: results directory to write')
    args = ap.parse_args()
    lib = build_library()
    started = time.time()
    if args.stage == 'onedesign':
        out_dir = os.path.join(PROJECT_ROOT, 'results', args.out)
        os.makedirs(out_dir, exist_ok=True)
        out = stage_onedesign(lib, out_dir, source=args.source)
        print(json.dumps({k: v for k, v in out.items() if k != 'environment'}, indent=1))
    else:
        out_dir = os.path.join(PROJECT_ROOT, 'results', '24_same_size_samples')
        os.makedirs(out_dir, exist_ok=True)
        payload = stage_seeds(lib, out_dir, args.seeds, args.sample_size)
        print(json.dumps(payload.get('summary', {}), indent=1, default=str)[:3000])
    print(f'done in {time.time() - started:.0f}s')


if __name__ == '__main__':
    main()
