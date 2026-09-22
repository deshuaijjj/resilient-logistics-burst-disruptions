"""September 2026 revision experiments for the IEEE Aerospace 2027 manuscript.

Stages (each writes only into its own new results directory; results/08-11 are
never touched):

  gate     results/12_revision_gate/            library regression + reduced-EF rerun with timers
  rerun    results/12_revision_gate/ccg_rerun/  recorded CCG configuration rerun in the same session
  verify   results/13_fixed_design_verification/ complete-library evaluation of three fixed designs,
                                                 per-scenario costs, monotonicity + canonicalization audit
  dual     results/14_dual_oracle/              strong-duality checks, capacity shadow prices,
                                                 single-level worst-case MILP oracle
  benders  results/15_benders_cross_check/      Kelley/Benders cutting planes, maximal-set separation,
                                                 final complete-library check, optimal-face bounds
  ccgmax   results/16_ccg_maximal_separation/   CCG with separation over the maximal set only
  seeds    results/17_sampling_seeds/           even-index regression + random 1,000-encoding samples
  fullef   results/18_full_ef_attempt/          monolithic LP over all distinct matrices (optional)

Usage:
  python3 src/run_proano_revision.py --all
  python3 src/run_proano_revision.py --stage gate --stage seeds --seeds 20
  python3 src/run_proano_revision.py --stage fullef --fullef-time-limit 10800
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from revision_common import (  # noqa: E402
    FROZEN, MILLION, PROJECT_ROOT, Library, RecourseEvaluator, build_library, ccg_loop,
    deterministic_even_sample_indices, environment_record, first_stage_cost, frozen_designs,
    kelley_cutting_plane, maximal_within, random_sample_ids, sha256_file, solve_extensive_form,
    worst_case_milp, write_json, certify_fixed_design,
)
from lp_solver import solve_robust_model  # noqa: E402

STAGE_DIRS = {
    'gate': '12_revision_gate',
    'rerun': '12_revision_gate',
    'verify': '13_fixed_design_verification',
    'dual': '14_dual_oracle',
    'benders': '15_benders_cross_check',
    'ccgmax': '16_ccg_maximal_separation',
    'seeds': '17_sampling_seeds',
    'fullef': '18_full_ef_attempt',
}
ALL_STAGES = ['gate', 'rerun', 'verify', 'dual', 'benders', 'ccgmax', 'seeds']
TOL_DOLLARS = 1.0          # absolute agreement tolerance for objectives (1 US dollar)
TOL_CAPACITY = 1e-6        # t or t/month
TOL_MILP = 10.0            # dollars; the MILP oracle runs in $M with mip_rel_gap 1e-9 (about 3.5 dollars at 3.5e9)


def log(msg: str) -> None:
    print(msg, flush=True)


def finish_stage(out_dir: str, stage: str, payload: dict, started: float) -> None:
    payload['stage'] = stage
    payload['wall_time_s'] = time.time() - started
    payload['environment'] = environment_record()
    write_json(os.path.join(out_dir, f'{stage}_result.json'), payload)
    manifest = {}
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path) and name != 'manifest.json':
            manifest[name] = sha256_file(path)
    write_json(os.path.join(out_dir, 'manifest.json'), {'files': manifest, 'environment': payload['environment']})
    log(f'[{stage}] done in {payload["wall_time_s"]:.1f}s -> {out_dir}')


def agreement(C, q, total, ref_C=FROZEN['exact_C'], ref_q=FROZEN['exact_q'], ref_total=FROZEN['exact_total_cost']):
    return {
        'objective_difference': float(total - ref_total),
        'C_max_abs_difference': float(np.max(np.abs(np.asarray(C) - np.asarray(ref_C)))),
        'q_max_abs_difference': float(np.max(np.abs(np.asarray(q) - np.asarray(ref_q)))),
        'objective_within_tolerance': bool(abs(total - ref_total) <= TOL_DOLLARS),
    }


# --------------------------------------------------------------------------- #
# gate + rerun
# --------------------------------------------------------------------------- #

def stage_gate(lib: Library, out_dir: str) -> dict:
    started = time.time()
    summary = lib.summary()
    checks = {
        'event_set_count_ok': summary['event_set_count'] == FROZEN['event_set_count'],
        'unique_matrix_count_ok': summary['unique_matrix_count'] == FROZEN['unique_matrix_count'],
        'maximal_matrix_count_ok': summary['maximal_matrix_count'] == FROZEN['maximal_matrix_count'],
        'canonical_sha256_ok': summary['canonical_sha256'] == FROZEN['canonical_sha256'],
        'maximal_sha256_ok': summary['maximal_sha256'] == FROZEN['maximal_sha256'],
    }
    # Reduced extensive form rerun with the recorded timer boundaries.
    maximal_ids = lib.maximal_ids
    t0 = time.time()
    master = solve_robust_model(lib.params, [lib.scenarios[i] for i in maximal_ids])
    if master.lp_status != 'optimal':
        raise RuntimeError(f'reduced EF failed: {master.lp_status}')
    cert, _ = certify_fixed_design(
        params=lib.params, scenarios=lib.scenarios, C_depot=master.C_depot, q_mode=master.q_mode,
        design_name='reduced_ef_rerun', sampled_total_cost=master.total_cost,
        sampled_worst_operating_cost=master.worst_case_cost, scenario_ids=maximal_ids,
        progress_interval=0)
    reduced_wall = time.time() - t0
    reduced = {
        'master_scenarios': len(maximal_ids),
        'master_total_cost': master.total_cost, 'certified_total_cost': cert.full_total_cost,
        'certification_gap': cert.full_total_cost - master.total_cost,
        'C': master.C_depot.tolist(), 'q': master.q_mode.tolist(),
        'worst_scenario_id': int(cert.worst.scenario_id), 'worst_unmet_tons': cert.worst.unmet_tons,
        'pruning_time_s': lib.prune_time_s,
        'master_assembly_plus_lp_time_s': master.solve_time,
        'maximal_set_certification_time_s': cert.eval_time_s,
        'wall_time_excluding_pruning_s': reduced_wall,
        'agreement_with_frozen': agreement(master.C_depot, master.q_mode, cert.full_total_cost),
    }
    checks['reduced_ef_objective_ok'] = reduced['agreement_with_frozen']['objective_within_tolerance']
    checks['reduced_ef_design_ok'] = (reduced['agreement_with_frozen']['C_max_abs_difference'] <= TOL_CAPACITY
                                      and reduced['agreement_with_frozen']['q_max_abs_difference'] <= TOL_CAPACITY)
    payload = {'library': summary, 'checks': checks, 'all_checks_pass': all(checks.values()),
               'reduced_ef_rerun': reduced, 'frozen_reference': FROZEN}
    write_json(os.path.join(out_dir, 'reduced_rerun.json'), reduced)
    finish_stage(out_dir, 'gate', payload, started)
    return payload


def stage_rerun(lib: Library, out_dir: str) -> dict:
    """Recorded CCG configuration rerun in this session (current code defaults)."""
    from run_burst_ccg import run_scenario_generation_ccg
    started = time.time()
    ccg_dir = os.path.join(out_dir, 'ccg_rerun')
    config = {'initial_sample_size': 1000, 'max_iterations': 10, 'gap_tol_abs': 1.0,
              'gap_tol_rel': 1e-8, 'scenario_limit': 0, 'progress_interval': 100000}
    t0 = time.time()
    solution, cert, iterations = run_scenario_generation_ccg(output_root=ccg_dir, **config)
    ccg_wall = time.time() - t0
    it = iterations.to_dict(orient='records')
    ccg = {
        'config_used': config, 'iterations': it, 'num_iterations': len(it),
        'total_cost': solution.total_cost, 'C': solution.C_depot.tolist(), 'q': solution.q_mode.tolist(),
        'final_upper_bound': cert.full_total_cost, 'worst_scenario_id': int(cert.worst.scenario_id),
        'worst_unmet_tons': cert.worst.unmet_tons,
        'separation_time_total_s': float(sum(r['separation_time_s'] for r in it)),
        'master_solve_time_total_s': float(sum(r['master_solve_time_s'] for r in it)),
        'wall_time_s': ccg_wall,
        'recourse_lps': int(len(lib.scenarios) * len(it)),
        'agreement_with_frozen': agreement(solution.C_depot, solution.q_mode, cert.full_total_cost),
    }
    reduced_path = os.path.join(out_dir, 'reduced_rerun.json')
    reduced = json.load(open(reduced_path)) if os.path.exists(reduced_path) else None
    timing = {
        'note': ('Protocol-matched observations from one session on one machine; single observations, '
                 'not statistics.  Do not divide these by the recorded June/August values.'),
        'reduced_ef': None if reduced is None else {
            'pruning_time_s': reduced['pruning_time_s'],
            'master_assembly_plus_lp_time_s': reduced['master_assembly_plus_lp_time_s'],
            'maximal_set_certification_time_s': reduced['maximal_set_certification_time_s'],
            'wall_time_excluding_pruning_s': reduced['wall_time_excluding_pruning_s'],
            'recourse_lps': len(lib.maximal_ids), 'master_solves': 1,
        },
        'ccg_full_library_separation': {
            'master_solve_time_total_s': ccg['master_solve_time_total_s'],
            'separation_time_total_s': ccg['separation_time_total_s'],
            'wall_time_s': ccg['wall_time_s'], 'iterations': ccg['num_iterations'],
            'recourse_lps': ccg['recourse_lps'], 'master_solves': ccg['num_iterations'],
        },
    }
    write_json(os.path.join(out_dir, 'same_session_timing.json'), timing)
    payload = {'ccg_rerun': ccg, 'same_session_timing': timing}
    finish_stage(out_dir, 'rerun', payload, started)
    return payload


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #

def nominal_only_design(lib: Library) -> dict:
    master = solve_robust_model(lib.params, [lib.scenarios[0]])
    if master.lp_status != 'optimal':
        raise RuntimeError(f'nominal-only EF failed: {master.lp_status}')
    return {'C': master.C_depot.tolist(), 'q': master.q_mode.tolist(),
            'nominal_total_cost': master.total_cost, 'source': 'solve_robust_model on the nominal encoding only'}


def monotonicity_audit(lib: Library, costs: dict, tol: float = TOL_DOLLARS) -> dict:
    """costs: scenario_id -> recourse cost (dollars) for every encoding."""
    # canonicalization lemma: equal matrices -> equal costs
    dup_spread = 0.0
    for rec in lib.canonical:
        if len(rec.event_set_ids) > 1:
            vals = [costs[i] for i in rec.event_set_ids]
            dup_spread = max(dup_spread, max(vals) - min(vals))
    maximal = [(r.bits, costs[r.representative_id]) for r in lib.maximal]
    maximal_bits = {b for b, _ in maximal}
    violations, checked_pairs, dominated_checked = [], 0, 0
    worst_margin = -np.inf   # max over dominated of (cost_dom - min cost of containing maximal)
    for rec in lib.canonical:
        if rec.bits in maximal_bits:
            continue
        c_dom = costs[rec.representative_id]
        dominated_checked += 1
        containing = [c_max for b_max, c_max in maximal if rec.bits & ~b_max == 0]
        checked_pairs += len(containing)
        if not containing:
            violations.append({'representative_id': rec.representative_id, 'reason': 'no containing maximal matrix'})
            continue
        margin = c_dom - min(containing)
        worst_margin = max(worst_margin, margin)
        if margin > tol:
            violations.append({'representative_id': rec.representative_id, 'cost_dominated': c_dom,
                               'min_cost_containing_maximal': min(containing), 'excess': margin})
    return {
        'dominated_matrices_checked': dominated_checked, 'comparable_pairs_checked': checked_pairs,
        'violations': len(violations), 'violation_examples': violations[:10],
        'max_excess_over_containing_maximal': float(worst_margin),
        'max_cost_spread_within_duplicate_groups': float(dup_spread), 'tolerance': tol,
    }


def stage_verify(lib: Library, out_dir: str) -> dict:
    started = time.time()
    designs = frozen_designs()
    designs['nominal_only'] = nominal_only_design(lib)
    results = {}
    for name in ('nominal_only', 'sampled', 'exact'):
        d = designs[name]
        csv_path = os.path.join(out_dir, f'{name}_scenario_costs.csv')
        log(f'[verify] complete-library evaluation of {name} design q={np.round(d["q"], 4).tolist()}')
        cert, frame = certify_fixed_design(
            params=lib.params, scenarios=lib.scenarios, C_depot=d['C'], q_mode=d['q'], design_name=name,
            progress_interval=5000, save_scenario_costs_path=csv_path)
        costs = {int(r.scenario_id): float(r.operating_cost) for r in frame.itertuples()}
        statuses = set(frame['status'])
        # maximal-set maximum (should coincide with the complete-library maximum)
        max_over_maximal = max(costs[i] for i in lib.maximal_ids)
        argmax_all = sorted(i for i, v in costs.items() if v >= cert.full_worst_operating_cost - TOL_DOLLARS)
        log(f'[verify] {name}: worst recourse {cert.full_worst_operating_cost/MILLION:.6f}M, auditing monotonicity ...')
        audit = monotonicity_audit(lib, costs)
        results[name] = {
            'design': d, 'investment_cost': cert.investment_cost,
            'complete_library_worst_recourse': cert.full_worst_operating_cost,
            'complete_library_total_cost': cert.full_total_cost,
            'worst_scenario_id': int(cert.worst.scenario_id), 'worst_events': cert.worst.events,
            'worst_unmet_tons': cert.worst.unmet_tons, 'worst_unmet_by_commodity': cert.worst.unmet_by_commodity,
            'cost_maximizers_within_tolerance': argmax_all[:50], 'num_cost_maximizers_within_tolerance': len(argmax_all),
            'max_over_maximal_set': max_over_maximal,
            'maximal_equals_complete': bool(abs(max_over_maximal - cert.full_worst_operating_cost) <= TOL_DOLLARS),
            'evaluated_scenarios': cert.evaluated_scenarios, 'lp_statuses': sorted(statuses),
            'eval_time_s': cert.eval_time_s, 'scenario_costs_csv': os.path.basename(csv_path),
            'monotonicity_audit': audit,
        }
    checks = {
        'sampled_worst_recourse_ok': abs(results['sampled']['complete_library_worst_recourse'] - FROZEN['sampled_full_worst_recourse']) <= TOL_DOLLARS,
        'exact_worst_recourse_ok': abs(results['exact']['complete_library_worst_recourse'] - FROZEN['exact_worst_recourse']) <= TOL_DOLLARS,
        'nominal_design_matches_e08_row': bool(np.max(np.abs(np.asarray(designs['nominal_only']['q']) - np.asarray(FROZEN['nominal_row_q']))) <= 1e-6),
        'all_lp_optimal': all(r['lp_statuses'] == ['optimal'] for r in results.values()),
        'monotonicity_zero_violations': all(r['monotonicity_audit']['violations'] == 0 for r in results.values()),
        'maximal_set_equals_complete_for_all': all(r['maximal_equals_complete'] for r in results.values()),
    }
    payload = {'designs': results, 'checks': checks, 'all_checks_pass': all(checks.values())}
    write_json(os.path.join(out_dir, 'verification_summary.json'), payload)
    finish_stage(out_dir, 'verify', payload, started)
    return payload


# --------------------------------------------------------------------------- #
# dual
# --------------------------------------------------------------------------- #

def stage_dual(lib: Library, out_dir: str, oracle_time_limit: float) -> dict:
    started = time.time()
    ev = RecourseEvaluator(lib.params)
    designs = frozen_designs()
    designs['nominal_only'] = nominal_only_design(lib)
    rng = np.random.default_rng(0)
    probe_ids = [0] + list(FROZEN['reference_worst_ids']) + [int(i) for i in rng.choice(lib.maximal_ids, 3, replace=False)]
    duality, shadow, oracle = {}, {}, {}
    P_M = float(np.max(lib.params.p_unmet)) / MILLION
    for name in ('nominal_only', 'sampled', 'exact'):
        d = designs[name]
        duality[name] = []
        for sid in probe_ids:
            Delta = lib.delta(sid)
            r = ev.evaluate(d['C'], d['q'], Delta, want_duals=True)
            dual_obj = ev.dual_objective(d['C'], d['q'], Delta, r.y_ub, r.y_eq)
            const, gC, gq = ev.cut(Delta, r.y_ub, r.y_eq)
            cut_value = const + float(gC @ np.asarray(d['C'])) + float(gq @ np.asarray(d['q']))
            duality[name].append({
                'scenario_id': int(sid), 'events': [list(map(int, e)) for e in lib.scenarios[sid].events],
                'primal': r.objective, 'dual': dual_obj, 'cut_value_at_design': cut_value,
                'abs_gap': abs(r.objective - dual_obj), 'rel_gap': abs(r.objective - dual_obj) / max(1.0, abs(r.objective)),
                'max_ineq_marginal': float(np.max(r.y_ub)), 'min_ineq_marginal': float(np.min(r.y_ub)),
            })
        shadow[name] = {}
        for sid in FROZEN['reference_worst_ids']:
            Delta = lib.delta(sid)
            r = ev.evaluate(d['C'], d['q'], Delta, want_duals=True)
            cm = ev.capacity_marginals(Delta, r.y_ub)
            shadow[name][str(sid)] = {
                'subgradient_wrt_q_dollars_per_t_month': cm['subgradient_wrt_q'],
                'min_lambda_dollars': cm['min_lambda'],
                'first_stage_price_dollars_per_t_month': (lib.params.v_mode * lib.params.T).tolist(),
                'recourse_cost': r.objective, 'unmet_tons': r.unmet_tons,
            }
        # enumeration reference over the maximal set (exact by the proposition)
        ref, _ = certify_fixed_design(params=lib.params, scenarios=lib.scenarios, C_depot=d['C'], q_mode=d['q'],
                                      design_name=name, scenario_ids=lib.maximal_ids, progress_interval=0)
        oracle[name] = {'maximal_set_reference_worst_recourse': ref.full_worst_operating_cost,
                        'maximal_set_reference_worst_id': int(ref.worst.scenario_id), 'runs': []}
        for Lam in (P_M, 100.0 * P_M):
            log(f'[dual] MILP oracle for {name}, big-M {Lam:g} $M ...')
            res = worst_case_milp(ev, lib, d['C'], d['q'], big_m_M=Lam, time_limit_s=oracle_time_limit)
            if 'worst_recourse' in res:
                res['difference_vs_enumeration'] = res['worst_recourse'] - ref.full_worst_operating_cost
                res['agrees_with_enumeration'] = bool(abs(res['difference_vs_enumeration']) <= TOL_MILP)
            oracle[name]['runs'].append(res)
    checks = {
        'strong_duality_all_within_1e-6_rel': all(x['rel_gap'] <= 1e-6 for v in duality.values() for x in v),
        'ineq_marginals_nonpositive': all(x['max_ineq_marginal'] <= 1e-9 for v in duality.values() for x in v),
        'oracle_agrees_all': all(r.get('agrees_with_enumeration', False) for v in oracle.values() for r in v['runs']),
        'oracle_big_m_insensitive': all(
            len(v['runs']) == 2 and 'worst_recourse' in v['runs'][0] and 'worst_recourse' in v['runs'][1]
            and abs(v['runs'][0]['worst_recourse'] - v['runs'][1]['worst_recourse']) <= TOL_MILP
            for v in oracle.values()),
        'capacity_multipliers_within_lemma_bound': all(
            -x['min_ineq_marginal'] <= float(np.max(lib.params.p_unmet)) + 1e-6 for v in duality.values() for x in v),
    }
    payload = {'designs': {k: {'C': v['C'], 'q': v['q'], 'source': v['source']} for k, v in designs.items()},
               'strong_duality': duality, 'shadow_prices': shadow, 'oracle': oracle,
               'lemma_bound_P_dollars': float(np.max(lib.params.p_unmet)),
               'checks': checks, 'all_checks_pass': all(checks.values())}
    write_json(os.path.join(out_dir, 'dual_checks.json'), {'strong_duality': duality, 'shadow_prices': shadow})
    write_json(os.path.join(out_dir, 'oracle_results.json'), oracle)
    finish_stage(out_dir, 'dual', payload, started)
    return payload


# --------------------------------------------------------------------------- #
# benders, ccgmax
# --------------------------------------------------------------------------- #

def final_full_check(lib: Library, C, q, name: str) -> dict:
    cert, _ = certify_fixed_design(params=lib.params, scenarios=lib.scenarios, C_depot=C, q_mode=q,
                                   design_name=name, progress_interval=0)
    return {'complete_library_worst_recourse': cert.full_worst_operating_cost,
            'complete_library_total_cost': cert.full_total_cost, 'worst_scenario_id': int(cert.worst.scenario_id),
            'worst_unmet_tons': cert.worst.unmet_tons, 'evaluated_scenarios': cert.evaluated_scenarios,
            'eval_time_s': cert.eval_time_s}


def stage_benders(lib: Library, out_dir: str, cuts_per_iter: int) -> dict:
    started = time.time()
    ev = RecourseEvaluator(lib.params)
    result = kelley_cutting_plane(ev, lib, lib.maximal_ids, tol_abs=TOL_DOLLARS, max_iters=300,
                                  cuts_per_iter=cuts_per_iter, log=log)
    result['separation_set'] = 'componentwise-maximal matrices'
    result['final_complete_library_check'] = final_full_check(lib, result['C'], result['q'], 'kelley_final')
    result['agreement_with_frozen'] = agreement(result['C'], result['q'], result['upper_bound'])
    checks = {
        'objective_ok': result['agreement_with_frozen']['objective_within_tolerance'],
        'design_ok': result['agreement_with_frozen']['C_max_abs_difference'] <= 1e-3
                     and result['agreement_with_frozen']['q_max_abs_difference'] <= 1e-3,
        'complete_library_confirms': abs(result['final_complete_library_check']['complete_library_total_cost']
                                         - result['upper_bound']) <= TOL_DOLLARS,
        'optimal_face_is_a_point': result['optimal_face_max_width'] <= 0.1,   # t or t/month, with a 1-dollar objective slack
    }
    payload = {'benders': result, 'checks': checks, 'all_checks_pass': all(checks.values())}
    write_json(os.path.join(out_dir, 'benders_result.json'), result)
    finish_stage(out_dir, 'benders', payload, started)
    return payload


def stage_ccgmax(lib: Library, out_dir: str) -> dict:
    started = time.time()
    initial = deterministic_even_sample_indices(len(lib.scenarios), 1000)
    result = ccg_loop(lib, initial, lib.maximal_ids, log=log)
    result['separation_set'] = 'componentwise-maximal matrices'
    result['final_complete_library_check'] = final_full_check(lib, result['C'], result['q'], 'ccgmax_final')
    result['agreement_with_frozen'] = agreement(result['C'], result['q'], result['final_upper_bound'])
    checks = {
        'converged': result['converged'],
        'objective_ok': result['agreement_with_frozen']['objective_within_tolerance'],
        'design_ok': result['agreement_with_frozen']['C_max_abs_difference'] <= 1e-6
                     and result['agreement_with_frozen']['q_max_abs_difference'] <= 1e-6,
        'complete_library_confirms': abs(result['final_complete_library_check']['complete_library_total_cost']
                                         - result['final_upper_bound']) <= TOL_DOLLARS,
    }
    payload = {'ccg_maximal_separation': result, 'checks': checks, 'all_checks_pass': all(checks.values())}
    write_json(os.path.join(out_dir, 'ccg_maximal_result.json'), result)
    finish_stage(out_dir, 'ccgmax', payload, started)
    return payload


# --------------------------------------------------------------------------- #
# seeds
# --------------------------------------------------------------------------- #

def stage_seeds(lib: Library, out_dir: str, num_seeds: int, sample_size: int) -> dict:
    started = time.time()
    N = len(lib.scenarios)
    jobs = [('even_index', deterministic_even_sample_indices(N, sample_size))]
    jobs += [(f'random_seed_{s}', random_sample_ids(N, sample_size, s)) for s in range(num_seeds)]
    rows = []
    for label, ids in jobs:
        log(f'[seeds] {label}: {len(ids)} encodings')
        t0 = time.time()
        master = solve_robust_model(lib.params, [lib.scenarios[i] for i in ids])
        if master.lp_status != 'optimal':
            raise RuntimeError(f'{label}: sampled EF failed: {master.lp_status}')
        cert, _ = certify_fixed_design(params=lib.params, scenarios=lib.scenarios, C_depot=master.C_depot,
                                       q_mode=master.q_mode, design_name=label, sampled_total_cost=master.total_cost,
                                       sampled_worst_operating_cost=master.worst_case_cost,
                                       scenario_ids=lib.maximal_ids, progress_interval=0)
        _, n_distinct = maximal_within(lib, ids)
        rows.append({
            'label': label, 'sample_size': len(ids), 'distinct_matrices_in_sample': n_distinct,
            'contains_1295': 1295 in ids, 'contains_1220': 1220 in ids,
            'in_sample_total_cost': master.total_cost, 'in_sample_worst_recourse': master.worst_case_cost,
            'complete_library_total_cost': cert.full_total_cost,
            'complete_library_worst_recourse': cert.full_worst_operating_cost,
            'gap_vs_in_sample': cert.full_total_cost - master.total_cost,
            'gap_vs_exact_optimum': cert.full_total_cost - FROZEN['exact_total_cost'],
            'in_sample_rel_dev_from_exact_pct': 100.0 * (master.total_cost - FROZEN['exact_total_cost']) / FROZEN['exact_total_cost'],
            'C_depot_0': master.C_depot[0], 'C_depot_1': master.C_depot[1],
            'q_mode_0': master.q_mode[0], 'q_mode_1': master.q_mode[1], 'q_mode_2': master.q_mode[2],
            'investment_cost': master.investment_cost,
            'worst_scenario_id': int(cert.worst.scenario_id), 'worst_events_json': json.dumps(cert.worst.events),
            'worst_unmet_tons': cert.worst.unmet_tons,
            'master_assembly_plus_lp_time_s': master.solve_time, 'maximal_set_eval_time_s': cert.eval_time_s,
            'wall_time_s': time.time() - t0,
        })
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'seeds.csv'), index=False)
    frame = pd.DataFrame(rows)
    even = frame[frame['label'] == 'even_index'].iloc[0]
    rnd = frame[frame['label'] != 'even_index']
    gaps = rnd['gap_vs_exact_optimum'].to_numpy()
    summary = {
        'sample_size': sample_size, 'random_samples': int(len(rnd)),
        'even_index_regression': {
            'in_sample_total_cost': float(even['in_sample_total_cost']),
            'complete_library_total_cost': float(even['complete_library_total_cost']),
            'in_sample_ok': abs(even['in_sample_total_cost'] - FROZEN['sampled_in_sample_total_cost']) <= TOL_DOLLARS,
            'complete_library_ok': abs(even['complete_library_total_cost'] - FROZEN['sampled_full_total_cost']) <= TOL_DOLLARS,
        },
        'random_samples_with_gap_over_1e9': int((gaps > 1e9).sum()),
        'random_samples_with_gap_over_1e6': int((gaps > 1e6).sum()),
        'random_samples_containing_1295': int(rnd['contains_1295'].sum()),
        'random_samples_containing_both_reference_ids': int((rnd['contains_1295'] & rnd['contains_1220']).sum()),
        'complete_library_total_cost_min': float(rnd['complete_library_total_cost'].min()),
        'complete_library_total_cost_max': float(rnd['complete_library_total_cost'].max()),
        'in_sample_rel_dev_from_exact_pct_max': float(rnd['in_sample_rel_dev_from_exact_pct'].abs().max()),
        'worst_unmet_distribution': {str(k): int(v) for k, v in rnd['worst_unmet_tons'].round(3).value_counts().items()},
        'q_mode_2_distribution': {str(k): int(v) for k, v in rnd['q_mode_2'].round(3).value_counts().items()},
        'inclusion_probability_single_encoding': (sample_size - 1) / (N - 1),
        'inclusion_probability_both_reference_ids': (sample_size - 1) / (N - 1) * (sample_size - 2) / (N - 2),
    }
    payload = {'summary': summary, 'rows': rows}
    write_json(os.path.join(out_dir, 'seeds_summary.json'), summary)
    finish_stage(out_dir, 'seeds', payload, started)
    return payload


# --------------------------------------------------------------------------- #
# fullef
# --------------------------------------------------------------------------- #

def stage_fullef(lib: Library, out_dir: str, time_limit: float, limit: int, method: str) -> dict:
    started = time.time()
    ev = RecourseEvaluator(lib.params)
    ids = lib.canonical_ids if limit <= 0 else lib.canonical_ids[:limit]
    log(f'[fullef] monolithic EF over {len(ids)} distinct matrices, method={method}, time limit {time_limit}s')
    result = solve_extensive_form(ev, lib, ids, method=method, time_limit_s=time_limit, log=log)
    result['scenario_set'] = 'all distinct matrices' if limit <= 0 else f'first {limit} canonical representatives'
    if result.get('success'):
        result['agreement_with_frozen'] = agreement(result['C'], result['q'], result['total_cost'])
        result['final_complete_library_check'] = final_full_check(lib, result['C'], result['q'], 'fullef_final')
    payload = {'full_ef': result}
    write_json(os.path.join(out_dir, 'full_ef_result.json'), result)
    finish_stage(out_dir, 'fullef', payload, started)
    return payload


# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', action='append', choices=list(STAGE_DIRS), default=None)
    parser.add_argument('--all', action='store_true', help=f'run {ALL_STAGES}')
    parser.add_argument('--results-root', default=os.path.join(PROJECT_ROOT, 'results'))
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--sample-size', type=int, default=1000)
    parser.add_argument('--cuts-per-iteration', type=int, default=1)
    parser.add_argument('--oracle-time-limit', type=float, default=1800.0)
    parser.add_argument('--fullef-time-limit', type=float, default=10800.0)
    parser.add_argument('--fullef-limit', type=int, default=0, help='smoke test: use only the first N canonical matrices')
    parser.add_argument('--fullef-method', default='highs-ipm')
    args = parser.parse_args()
    stages = list(ALL_STAGES) if args.all else (args.stage or [])
    if not stages:
        parser.error('give --all or at least one --stage')

    log(f'Stages: {stages}')
    lib = build_library()
    log(f'Library: {lib.summary()}')
    outcomes = {}
    for stage in stages:
        out_dir = os.path.join(args.results_root, STAGE_DIRS[stage])
        os.makedirs(out_dir, exist_ok=True)
        log(f'\n===== stage {stage} -> {out_dir}')
        try:
            if stage == 'gate':
                outcomes[stage] = stage_gate(lib, out_dir)['all_checks_pass']
            elif stage == 'rerun':
                stage_rerun(lib, out_dir); outcomes[stage] = True
            elif stage == 'verify':
                outcomes[stage] = stage_verify(lib, out_dir)['all_checks_pass']
            elif stage == 'dual':
                outcomes[stage] = stage_dual(lib, out_dir, args.oracle_time_limit)['all_checks_pass']
            elif stage == 'benders':
                outcomes[stage] = stage_benders(lib, out_dir, args.cuts_per_iteration)['all_checks_pass']
            elif stage == 'ccgmax':
                outcomes[stage] = stage_ccgmax(lib, out_dir)['all_checks_pass']
            elif stage == 'seeds':
                outcomes[stage] = stage_seeds(lib, out_dir, args.seeds, args.sample_size)['summary']['even_index_regression']['complete_library_ok']
            elif stage == 'fullef':
                outcomes[stage] = bool(stage_fullef(lib, out_dir, args.fullef_time_limit, args.fullef_limit,
                                                    args.fullef_method)['full_ef'].get('success'))
        except Exception as exc:  # keep going; record the failure
            outcomes[stage] = False
            write_json(os.path.join(out_dir, f'{stage}_error.json'),
                       {'stage': stage, 'error': repr(exc), 'traceback': traceback.format_exc(),
                        'environment': environment_record()})
            log(f'[{stage}] FAILED: {exc!r}')
    log('\nSUMMARY: ' + json.dumps({k: bool(v) for k, v in outcomes.items()}))


if __name__ == '__main__':
    main()
