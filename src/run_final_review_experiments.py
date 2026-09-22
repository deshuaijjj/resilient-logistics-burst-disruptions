"""Final-review experiments (2026-09-20) for the IEEE Aerospace 2027 manuscript.

Every stage writes only into its own new results directory; results/08-18 are
never modified.  Nothing here is a timed experiment: wall times are recorded for
information, but the machine may be shared and no timing from these stages is
reported in the manuscript.  Work counts (masters, recourse LPs) are deterministic.

Stages
  scenlevel  results/19_scenario_level/        post-processing of the per-encoding costs in
                                               results/13: where each fixed design falls short
  support    results/20_support_certificate/   two-encoding support of the optimum, leave-one-out
                                               around encoding 1295, closed form of the maximal set
  ccgstart   results/21_ccg_start_sets/        CCG from the nominal encoding alone, with
                                               maximal-set and with all-encoding separation
  libsens    results/22_library_sensitivity/   duration caps and event budget: rebuild the library,
                                               filter, solve the reduced extensive form
  penalty    results/23_penalty_scale/         both designs under rescaled unmet-demand penalties

Usage
  python3 src/run_final_review_experiments.py --all
  python3 src/run_final_review_experiments.py --stage scenlevel --stage support
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import statistics
import sys
import time
import traceback
from typing import Dict, List, Sequence

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from revision_common import (  # noqa: E402
    FROZEN, PROJECT_ROOT, Library, RecourseEvaluator, build_library, ccg_loop,
    deterministic_even_sample_indices, environment_record, sha256_file, solve_extensive_form,
    write_json,
)
from instance_generator import generate_small_instance  # noqa: E402

STAGE_DIRS = {
    'scenlevel': '19_scenario_level',
    'support': '20_support_certificate',
    'ccgstart': '21_ccg_start_sets',
    'libsens': '22_library_sensitivity',
    'penalty': '23_penalty_scale',
}
ALL_STAGES = ['scenlevel', 'support', 'ccgstart', 'libsens', 'penalty']
TOL_DOLLARS = 1.0
TOL_TONS = 1e-6


def log(msg: str) -> None:
    print(msg, flush=True)


def finish_stage(out_dir: str, stage: str, payload: dict, started: float) -> None:
    payload['stage'] = stage
    payload['wall_time_s'] = time.time() - started
    payload['timing_protocol'] = 'untimed: machine may be shared; no wall time from this stage is reported'
    payload['environment'] = environment_record()
    write_json(os.path.join(out_dir, f'{stage}_result.json'), payload)
    manifest = {}
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path) and name != 'manifest.json':
            manifest[name] = sha256_file(path)
    write_json(os.path.join(out_dir, 'manifest.json'), {'files': manifest, 'environment': payload['environment']})
    log(f'[{stage}] done in {payload["wall_time_s"]:.1f}s -> {out_dir}')


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def evaluate_on(evaluator: RecourseEvaluator, lib: Library, C, q, ids: Sequence[int]) -> Dict:
    """Fixed-design recourse on the given encodings; returns worst cost and unmet statistics."""
    worst, worst_id, worst_unmet, max_unmet, max_unmet_id, failed = -1.0, None, None, -1.0, None, 0
    costs = {}
    for sid in ids:
        r = evaluator.evaluate(C, q, lib.delta(sid))
        if r.status != 'optimal':
            failed += 1
            continue
        costs[int(sid)] = r.objective
        if r.objective > worst:
            worst, worst_id, worst_unmet = r.objective, int(sid), r.unmet_tons
        if r.unmet_tons > max_unmet:
            max_unmet, max_unmet_id = r.unmet_tons, int(sid)
    ties = sorted(s for s, v in costs.items() if abs(v - worst) <= TOL_DOLLARS)
    return {'evaluated': len(ids), 'failed_solves': failed, 'worst_recourse': worst,
            'worst_scenario_id': worst_id, 'worst_events': [list(e) for e in lib.scenarios[worst_id].events],
            'worst_unmet_tons': worst_unmet, 'cost_maximizers_within_tolerance': ties,
            'max_unmet_tons_in_cost_min_solutions': max_unmet, 'max_unmet_scenario_id': max_unmet_id}


def closed_form_maximal_bits(lib: Library) -> set:
    """Bitsets of compatible sets of maximum-duration events, as many as the budget allows."""
    from run_dominance_pruned_oracle import scenario_bits
    from instance_generator import DisruptionScenario
    p = lib.params
    top = [e for e in lib.single_events if e[2] == int(p.L_max[e[0]])]

    def compatible(a, b):
        if a[0] != b[0]:
            return True
        return a[1] + a[2] <= b[1] or b[1] + b[2] <= a[1]

    out = set()
    if p.Gamma == 1:
        for e in top:
            out.add(scenario_bits(DisruptionScenario([e]), p.M, p.T))
    elif p.Gamma == 2:
        for i, a in enumerate(top):
            for b in top[i + 1:]:
                if compatible(a, b):
                    out.add(scenario_bits(DisruptionScenario([a, b]), p.M, p.T))
    else:
        raise ValueError('closed form only stated for an event budget of one or two')
    return out


def total_demand_by_month(params) -> np.ndarray:
    return np.asarray(params.D_demand).sum(axis=(0, 1))


# --------------------------------------------------------------------------- #
# scenlevel
# --------------------------------------------------------------------------- #

def stage_scenlevel(results_root: str, out_dir: str) -> dict:
    src = os.path.join(results_root, '13_fixed_design_verification')
    summary = json.load(open(os.path.join(src, 'verification_summary.json')))
    params = generate_small_instance()
    startup_unmet = float(np.asarray(params.D_demand)[:, :, 0].sum())
    startup_penalty = float((np.asarray(params.D_demand)[:, :, 0] * np.asarray(params.p_unmet)).sum())

    def read(name):
        rows = []
        with open(os.path.join(src, f'{name}_scenario_costs.csv'), newline='') as stream:
            for r in csv.DictReader(stream):
                rows.append({'id': int(r['scenario_id']), 'events': json.loads(r['events']),
                             'cost': float(r['operating_cost']), 'unmet': float(r['unmet_tons']),
                             'status': r['status']})
        return rows

    data = {name: read(name) for name in ('exact', 'sampled', 'nominal_only')}
    invest = {name: float(summary['designs'][name]['investment_cost']) for name in data}
    forced_invest = float((np.asarray(params.f_depot) * np.asarray(params.s_initial).sum(axis=1)).sum())
    out = {'source': 'results/13_fixed_design_verification/*_scenario_costs.csv',
           'startup_unmet_tons': startup_unmet, 'startup_penalty_dollars': startup_penalty,
           'forced_depot_investment_dollars': forced_invest,
           'controllable_investment_dollars': {n: invest[n] - forced_invest for n in invest},
           'designs': {}, 'source_sha256': {f'{n}_scenario_costs.csv': sha256_file(os.path.join(src, f'{n}_scenario_costs.csv')) for n in data}}

    for name, rows in data.items():
        above = [r for r in rows if r['unmet'] > startup_unmet + TOL_TONS]
        worst = max(r['cost'] for r in rows)
        out['designs'][name] = {
            'encodings': len(rows), 'all_optimal': all(r['status'] == 'optimal' for r in rows),
            'investment_cost': invest[name],
            'encodings_with_shortfall_beyond_startup': len(above),
            'share_with_shortfall_beyond_startup_pct': 100.0 * len(above) / len(rows),
            'max_unmet_tons': max(r['unmet'] for r in rows), 'min_unmet_tons': min(r['unmet'] for r in rows),
            'worst_recourse': worst,
            'cost_maximizers_within_one_dollar': sorted(r['id'] for r in rows if abs(r['cost'] - worst) <= TOL_DOLLARS),
            'nominal_scenario_recourse': rows[0]['cost'],
            'nominal_scenario_total_net_of_startup': invest[name] + rows[0]['cost'] - startup_penalty,
            'complete_library_total_net_of_startup': invest[name] + worst - startup_penalty,
        }

    # ---- the sampled design's failures ------------------------------------------------
    sam = data['sampled']
    in_sample_worst = FROZEN['sampled_in_sample_total_cost'] - invest['sampled']
    fail = [r for r in sam if r['unmet'] > startup_unmet + TOL_TONS]
    detail = []
    for r in fail:
        ev = sorted(r['events'])
        months = {m: set() for m in range(params.M)}
        for m, a, L in ev:
            months[m] |= set(range(a + 1, a + L + 1))           # one-based months
        both = months[0] & months[1]
        detail.append({'id': r['id'], 'events': ev, 'extra_unmet_tons': r['unmet'] - startup_unmet,
                       'cost': r['cost'], 'modes': sorted({e[0] for e in ev}),
                       'durations': [e[2] for e in ev], 'months_both_primary_modes_out': len(both),
                       'first_month_out': min(a + 1 for _, a, _ in ev),
                       'synchronized_six_month_pair': len(ev) == 2 and ev[0][1] == ev[1][1] and ev[0][2] == ev[1][2] == 6})
    sync = [d for d in detail if d['synchronized_six_month_pair']]
    out['sampled_failures'] = {
        'count': len(fail),
        'share_pct': 100.0 * len(fail) / len(sam),
        'ids': [d['id'] for d in detail],
        'all_are_pairs_on_modes_0_and_1': all(d['modes'] == [0, 1] and len(d['events']) == 2 for d in detail),
        'min_event_duration_months': min(min(d['durations']) for d in detail),
        'min_months_both_primary_modes_out': min(d['months_both_primary_modes_out'] for d in detail),
        'extra_unmet_tons_min': min(d['extra_unmet_tons'] for d in detail),
        'extra_unmet_tons_max': max(d['extra_unmet_tons'] for d in detail),
        'extra_unmet_tons_histogram': {f'{k:.2f}': v for k, v in sorted(
            {x: sum(1 for d in detail if abs(d['extra_unmet_tons'] - x) < 1e-6) for x in {round(d['extra_unmet_tons'], 6) for d in detail}}.items())},
        'synchronized_six_month_pairs': len(sync),
        'synchronized_start_months_one_based': sorted(d['first_month_out'] for d in sync),
        'synchronized_not_in_month_one': sum(1 for d in sync if d['first_month_out'] > 1),
        'synchronized_not_in_month_one_extra_unmet_tons': sorted({round(d['extra_unmet_tons'], 6) for d in sync if d['first_month_out'] > 1}),
        'latest_failing_first_month_out': max(d['first_month_out'] for d in detail),
        'encodings_above_in_sample_worst_recourse': sum(1 for r in sam if r['cost'] > in_sample_worst + TOL_DOLLARS),
        'in_sample_worst_recourse': in_sample_worst,
        'encodings_above_exact_optimum_worst_recourse': sum(1 for r in sam if r['cost'] > FROZEN['exact_worst_recourse'] + TOL_DOLLARS),
        'detail': detail,
    }

    # ---- is it an initialization effect?  drop every encoding with an outage in month 1 ----
    def restricted(rows):
        keep = [r for r in rows if all(a >= 1 for _, a, _ in r['events'])]
        top = max(keep, key=lambda r: r['cost'])
        return keep, top
    for name in ('sampled', 'exact'):
        keep, top = restricted(data[name])
        out['designs'][name]['no_month_one_outage'] = {
            'encodings': len(keep), 'worst_recourse': top['cost'], 'worst_scenario_id': top['id'],
            'worst_events': top['events'], 'worst_unmet_tons': top['unmet'],
            'total_cost': invest[name] + top['cost'],
            'increase_over_sampled_in_sample_pct': 100.0 * (invest[name] + top['cost'] - FROZEN['sampled_in_sample_total_cost']) / FROZEN['sampled_in_sample_total_cost'],
        }

    # ---- price of protection: exact minus sampled, encoding by encoding ------------------
    diff = [invest['exact'] + e['cost'] - invest['sampled'] - s['cost'] for e, s in zip(data['exact'], data['sampled'])]
    assert all(e['id'] == s['id'] for e, s in zip(data['exact'], data['sampled']))
    dearer = [d for d in diff if d > 0]
    cheaper = [d for d in diff if d < -TOL_DOLLARS]
    nominal_net = {n: out['designs'][n]['nominal_scenario_total_net_of_startup'] for n in ('exact', 'sampled')}
    out['exact_minus_sampled'] = {
        'nominal_scenario_difference_dollars': diff[0],
        'nominal_scenario_difference_pct_of_net_total': 100.0 * diff[0] / nominal_net['sampled'],
        'nominal_scenario_difference_pct_of_total': 100.0 * diff[0] / (invest['sampled'] + data['sampled'][0]['cost']),
        'encodings_where_exact_costs_more': len(dearer), 'max_extra_cost_dollars': max(dearer),
        'median_extra_cost_dollars': statistics.median(dearer),
        'encodings_where_exact_costs_less': len(cheaper), 'max_saving_dollars': -min(cheaper),
        'cheaper_ids_equal_failure_ids': sorted(i for i, d in enumerate(diff) if d < -TOL_DOLLARS) == sorted(out['sampled_failures']['ids']),
    }
    # bound on extra unmet tonnage implied by the objective alone
    p_min = float(np.asarray(params.p_unmet).min())
    out['exact_extra_unmet_bound_tons'] = (FROZEN['exact_worst_recourse'] - startup_penalty) / p_min
    out['checks'] = {
        'exact_never_beyond_startup': out['designs']['exact']['encodings_with_shortfall_beyond_startup'] == 0,
        'sampled_unique_maximizer_is_1295': out['designs']['sampled']['cost_maximizers_within_one_dollar'] == [1295],
    }
    write_json(os.path.join(out_dir, 'scenario_level_summary.json'), out)
    log(f"  exact beyond startup: {out['designs']['exact']['encodings_with_shortfall_beyond_startup']}, "
        f"sampled: {out['sampled_failures']['count']}, nominal-only: {out['designs']['nominal_only']['encodings_with_shortfall_beyond_startup']}")
    return {'summary_file': 'scenario_level_summary.json', 'checks': out['checks']}


# --------------------------------------------------------------------------- #
# support
# --------------------------------------------------------------------------- #

def stage_support(lib: Library, results_root: str, out_dir: str) -> dict:
    ev = RecourseEvaluator(lib.params)
    out: Dict = {}

    # closed form of the maximal set
    cf = closed_form_maximal_bits(lib)
    filt = {rec.bits for rec in lib.maximal}
    top = [e for e in lib.single_events if e[2] == int(lib.params.L_max[e[0]])]
    sev = {}
    for b in cf:
        sev[bin(b).count('1')] = sev.get(bin(b).count('1'), 0) + 1
    out['closed_form_maximal_set'] = {
        'maximum_duration_events': len(top),
        'maximum_duration_events_by_mode': [sum(1 for e in top if e[0] == m) for m in range(lib.params.M)],
        'closed_form_matrices': len(cf), 'filter_matrices': len(filt), 'sets_equal': cf == filt,
        'closed_form_cells_histogram': {str(k): v for k, v in sorted(sev.items())},
    }
    log(f"  closed form == filter output: {cf == filt} ({len(cf)} vs {len(filt)})")

    # subsets that support the optimum
    subsets = {'ids_1220_1295': [1220, 1295], 'id_1295_only': [1295], 'id_1220_only': [1220],
               'nominal_plus_1220_1295': [0, 1220, 1295]}
    out['subset_extensive_forms'] = {}
    for name, ids in subsets.items():
        res = solve_extensive_form(ev, lib, ids, log=lambda m: None)
        res['scenario_ids'] = ids
        res['objective_minus_exact_optimum'] = res['total_cost'] - FROZEN['exact_total_cost']
        res['q_max_abs_difference_from_exact'] = float(np.max(np.abs(np.asarray(res['q']) - np.asarray(FROZEN['exact_q']))))
        out['subset_extensive_forms'][name] = res
        log(f"  EF over {ids}: {res['total_cost']:.2f}  q={[round(v, 3) for v in res['q']]}")
    two = out['subset_extensive_forms']['ids_1220_1295']
    ver = json.load(open(os.path.join(results_root, '13_fixed_design_verification', 'verification_summary.json')))
    ub = float(ver['designs']['exact']['complete_library_total_cost'])
    out['certificate'] = {
        'lower_bound_two_encoding_relaxation': two['total_cost'],
        'upper_bound_complete_library_value_of_exact_design': ub,
        'upper_bound_source': 'results/13_fixed_design_verification/verification_summary.json (29,916 encodings)',
        'gap_dollars': ub - two['total_cost'],
        'variables_in_relaxation': two['dims']['variables'],
        'closed': abs(ub - two['total_cost']) <= TOL_DOLLARS,
    }

    # leave-one-out: every maximal matrix except the one induced by encoding 1295
    b1295 = lib.bits[1295]
    loo_ids = [rec.representative_id for rec in lib.maximal if rec.bits != b1295]
    res = solve_extensive_form(ev, lib, loo_ids, log=lambda m: None)
    full = evaluate_on(ev, lib, res['C'], res['q'], lib.maximal_ids)
    n, k = len(lib.maximal), 999
    out['leave_one_out_1295'] = {
        'scenarios_in_design_set': len(loo_ids), 'design': {'C': res['C'], 'q': res['q']},
        'in_set_total_cost': res['total_cost'], 'investment_cost': res['investment_cost'],
        'complete_library_evaluation_on_maximal_set': full,
        'complete_library_total_cost': res['investment_cost'] + full['worst_recourse'],
        'gap_vs_exact_optimum': res['investment_cost'] + full['worst_recourse'] - FROZEN['exact_total_cost'],
        'probability_uniform_sample_of_999_maximal_matrices_omits_it_pct': 100.0 * (1 - k / n),
    }
    log(f"  leave-one-out: q={[round(v, 3) for v in res['q']]} complete-library {out['leave_one_out_1295']['complete_library_total_cost']:.2f}")
    write_json(os.path.join(out_dir, 'support_certificate.json'), out)
    return {'summary_file': 'support_certificate.json',
            'checks': {'closed_form_equals_filter': cf == filt, 'certificate_closed': out['certificate']['closed']}}


# --------------------------------------------------------------------------- #
# ccgstart
# --------------------------------------------------------------------------- #

def stage_ccgstart(lib: Library, out_dir: str, skip_all_encodings: bool) -> dict:
    out: Dict = {}
    runs = [('nominal_start_maximal_separation', lib.maximal_ids)]
    if not skip_all_encodings:
        runs.append(('nominal_start_all_encoding_separation', None))
    for name, sep in runs:
        log(f'  CCG {name}')
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):            # lp_solver prints a full HiGHS log per master
            res = ccg_loop(lib, initial_ids=[0], separation_ids=sep, max_iterations=60, log=lambda m: None)
        res['initial_ids'] = [0]
        res['final_master_variables'] = 6 + 528 * len(res['active_ids_final'])
        res['max_master_variables'] = max(6 + 528 * it['master_scenarios'] for it in res['iterations'])
        res['added_ids_in_order'] = [it['added_scenario_id'] for it in res['iterations'] if it['added_scenario_id'] is not None]
        res['objective_minus_exact_optimum'] = res['total_cost'] - FROZEN['exact_total_cost']
        res['q_max_abs_difference_from_exact'] = float(np.max(np.abs(np.asarray(res['q']) - np.asarray(FROZEN['exact_q']))))
        out[name] = res
        log(f"    iterations={res['num_iterations']} LPs={res['recourse_lps']} active={res['active_ids_final']} "
            f"obj-ref={res['objective_minus_exact_optimum']:.3e}")
    write_json(os.path.join(out_dir, 'ccg_start_sets.json'), out)
    return {'summary_file': 'ccg_start_sets.json',
            'checks': {k: bool(v['converged'] and abs(v['objective_minus_exact_optimum']) <= TOL_DOLLARS) for k, v in out.items()}}


# --------------------------------------------------------------------------- #
# libsens
# --------------------------------------------------------------------------- #

def stage_libsens(out_dir: str) -> dict:
    base = generate_small_instance()
    D = total_demand_by_month(base)
    S0 = float(np.asarray(base.s_initial).sum())
    variants = [
        ('baseline', {'L_max': [6, 6, 2], 'Gamma': 2}),
        ('mode2_cap_1', {'L_max': [6, 6, 1], 'Gamma': 2}),
        ('mode2_cap_4', {'L_max': [6, 6, 4], 'Gamma': 2}),
        ('mode2_cap_6', {'L_max': [6, 6, 6], 'Gamma': 2}),
        ('primary_cap_5', {'L_max': [5, 5, 2], 'Gamma': 2}),
        ('primary_cap_4', {'L_max': [4, 4, 2], 'Gamma': 2}),
        ('primary_cap_2', {'L_max': [2, 2, 2], 'Gamma': 2}),
        ('single_event', {'L_max': [6, 6, 2], 'Gamma': 1}),
    ]
    rows = []
    for name, change in variants:
        params = copy.deepcopy(base)
        params.L_max = np.array(change['L_max'])
        params.L_min = np.minimum(np.asarray(base.L_min), params.L_max)
        params.Gamma = int(change['Gamma'])
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            lib = build_library(params)
        cf = closed_form_maximal_bits(lib)
        filt = {rec.bits for rec in lib.maximal}
        ev = RecourseEvaluator(params)
        res = solve_extensive_form(ev, lib, lib.maximal_ids, log=lambda m: None)
        full = evaluate_on(ev, lib, res['C'], res['q'], lib.maximal_ids)
        Lp = int(min(params.L_max[0], params.L_max[1]))
        # both primary modes out in months 1..Lp: restored-mode cargo first arrives in month Lp+3
        window = float(D[1:Lp + 2].sum())
        predicted = max(0.0, (window - S0) / Lp) if params.Gamma >= 2 else 0.0
        row = {
            'variant': name, 'L_min': [int(v) for v in params.L_min], 'L_max': [int(v) for v in params.L_max],
            'Gamma': params.Gamma, 'encodings': len(lib.scenarios), 'distinct_matrices': len(lib.canonical),
            'maximal_matrices': len(lib.maximal), 'closed_form_equals_filter': cf == filt,
            'total_cost': res['total_cost'], 'investment_cost': res['investment_cost'], 'C': res['C'], 'q': res['q'],
            'eta': res['eta'], 'maximal_set_evaluation': full,
            'eta_matches_evaluation': abs(res['eta'] - full['worst_recourse']) <= TOL_DOLLARS,
            'demand_window_months_2_to_Lplus2_tons': window, 'initial_stock_tons': S0,
            'q2_predicted_by_supply_window': predicted,
            'q2_minus_prediction': res['q'][2] - predicted,
            'ef_solve_time_s': res['solve_time_s'],
        }
        rows.append(row)
        log(f"  {name:14s} L_max={change['L_max']} Gamma={change['Gamma']} |Omega|={row['encodings']} |Umax|={row['maximal_matrices']} "
            f"cf={row['closed_form_equals_filter']} total={row['total_cost']:.2f} q={[round(v, 3) for v in res['q']]} "
            f"q2_pred={predicted:.3f} max_unmet={full['max_unmet_tons_in_cost_min_solutions']:.3f}")
    write_json(os.path.join(out_dir, 'library_sensitivity.json'), {'variants': rows})
    base_row = rows[0]
    return {'summary_file': 'library_sensitivity.json',
            'checks': {'baseline_reproduces_exact_optimum': abs(base_row['total_cost'] - FROZEN['exact_total_cost']) <= TOL_DOLLARS,
                       'closed_form_holds_in_every_variant': all(r['closed_form_equals_filter'] for r in rows),
                       'eta_matches_in_every_variant': all(r['eta_matches_evaluation'] for r in rows)}}


# --------------------------------------------------------------------------- #
# penalty
# --------------------------------------------------------------------------- #

def stage_penalty(out_dir: str) -> dict:
    base = generate_small_instance()
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        lib0 = build_library(base)
    sample_ids = deterministic_even_sample_indices(len(lib0.scenarios), 1000)
    rows = []
    for scale in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        params = copy.deepcopy(base)
        params.p_unmet = np.asarray(base.p_unmet, dtype=float) * scale
        lib = Library(params, lib0.scenarios, lib0.canonical, lib0.maximal, lib0.bits, lib0.single_events, lib0.prune_time_s)
        ev = RecourseEvaluator(params)
        startup_penalty = float((np.asarray(params.D_demand)[:, :, 0] * params.p_unmet).sum())
        entry = {'penalty_scale': scale, 'p_unmet': params.p_unmet.tolist(), 'startup_penalty_dollars': startup_penalty}
        for label, ids in (('exact', lib.maximal_ids), ('sampled', sample_ids)):
            res = solve_extensive_form(ev, lib, ids, log=lambda m: None)
            full = evaluate_on(ev, lib, res['C'], res['q'], lib.maximal_ids)
            entry[label] = {'design_set_size': len(ids), 'design_set_total_cost': res['total_cost'],
                            'investment_cost': res['investment_cost'], 'C': res['C'], 'q': res['q'],
                            'complete_library_total_cost': res['investment_cost'] + full['worst_recourse'],
                            'maximal_set_evaluation': full}
        e, s = entry['exact'], entry['sampled']
        entry['sampled_increase_pct'] = 100.0 * (s['complete_library_total_cost'] - s['design_set_total_cost']) / s['design_set_total_cost']
        entry['exact_below_sampled_pct'] = 100.0 * (s['complete_library_total_cost'] - e['complete_library_total_cost']) / s['complete_library_total_cost']
        entry['q_exact_max_abs_difference_from_baseline'] = float(np.max(np.abs(np.asarray(e['q']) - np.asarray(FROZEN['exact_q']))))
        entry['q_sampled_max_abs_difference_from_baseline'] = float(np.max(np.abs(np.asarray(s['q']) - np.asarray(FROZEN['sampled_q']))))
        rows.append(entry)
        log(f"  scale={scale:g}: exact q={[round(v, 3) for v in e['q']]} unmet={e['maximal_set_evaluation']['worst_unmet_tons']:.1f} | "
            f"sampled q={[round(v, 3) for v in s['q']]} unmet={s['maximal_set_evaluation']['worst_unmet_tons']:.1f} "
            f"worst={s['maximal_set_evaluation']['worst_scenario_id']} | +{entry['sampled_increase_pct']:.2f}% / -{entry['exact_below_sampled_pct']:.2f}%")
    write_json(os.path.join(out_dir, 'penalty_scale.json'), {'scales': rows})
    unit = [r for r in rows if r['penalty_scale'] == 1.0][0]
    return {'summary_file': 'penalty_scale.json',
            'checks': {'unit_scale_reproduces_exact': abs(unit['exact']['complete_library_total_cost'] - FROZEN['exact_total_cost']) <= TOL_DOLLARS,
                       'unit_scale_reproduces_sampled': abs(unit['sampled']['complete_library_total_cost'] - FROZEN['sampled_full_total_cost']) <= TOL_DOLLARS}}


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', action='append', choices=ALL_STAGES)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--results-root', default=os.path.join(PROJECT_ROOT, 'results'))
    ap.add_argument('--out-root', default=None, help='write stage directories here instead of --results-root (smoke tests)')
    ap.add_argument('--skip-all-encoding-ccg', action='store_true', help='ccgstart: only the maximal-set separation run')
    args = ap.parse_args()
    stages = ALL_STAGES if args.all or not args.stage else args.stage
    lib = None
    status = {}
    for stage in stages:
        out_dir = os.path.join(args.out_root or args.results_root, STAGE_DIRS[stage])
        os.makedirs(out_dir, exist_ok=True)
        started = time.time()
        log(f'[{stage}] start')
        try:
            if stage in ('support', 'ccgstart') and lib is None:
                import io, contextlib
                with contextlib.redirect_stdout(io.StringIO()):
                    lib = build_library()
                assert len(lib.scenarios) == FROZEN['event_set_count'] and len(lib.maximal) == FROZEN['maximal_matrix_count']
            if stage == 'scenlevel':
                payload = stage_scenlevel(args.results_root, out_dir)
            elif stage == 'support':
                payload = stage_support(lib, args.results_root, out_dir)
            elif stage == 'ccgstart':
                payload = stage_ccgstart(lib, out_dir, args.skip_all_encoding_ccg)
            elif stage == 'libsens':
                payload = stage_libsens(out_dir)
            else:
                payload = stage_penalty(out_dir)
            finish_stage(out_dir, stage, payload, started)
            status[stage] = payload.get('checks')
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            status[stage] = f'FAILED: {exc}'
    log('SUMMARY: ' + json.dumps(status))


if __name__ == '__main__':
    main()
