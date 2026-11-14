"""Experiments reported in the Optimization Letters manuscript.

Scenario definition and the three scenario sets: src/ol_scenarios.py (an
outage may start in any period; what lies after the horizon is cut off).
Example: the lunar example with demand from month 2 on
(run_ol_revision.base_instance) and the random examples around it
(run_ol_revision.random_instance).

  counts     results/34_scenario_counts/
             For k = 1..4 outages per scenario: number of scenarios, number in
             the set of Theorem 1 and in the set of Theorem 2, number of single
             outages each needs, time to write the two sets down, and, where
             the whole set can be listed, time to list it and to find the set
             of Theorem 1 by comparing pairs.
  treesize   results/34_scenario_counts/tree_sizes.json
             Number of nodes of the scenario tree of the multi-stage problem
             for every scenario (k = 1..3) and for the two sets (k = 1..4).
  milpsep    results/35_milp_separation/
             Column-and-constraint generation in which the worst scenario of a
             round is found by a mixed-integer program with one 0-1 variable
             per single outage: all single outages, or only the longest ones
             that end within the horizon (Theorem 1).
  worstfloor results/35_milp_separation/worst_scenario_unlimited_capacity.json
             The worst scenario of each optimal design operated with unlimited
             capacity: the demand that is unmet under every design.
  verify     results/36_all_scenarios_check/
             The optimal design operated in every scenario, one by one.
  family     results/37_random_examples/
             The two variants of milpsep on random examples.
  famselect  results/37_random_examples/seed_selection.json
             Which seeds make up the random examples: a seed is used if all of
             its demand can be served when no line fails and capacity is
             unlimited. family and learnsfam use the same seeds.
  learns     results/38_learning_operator/
             The lunar example for an operator who learns of outages as they
             happen (linear program on the scenario tree), next to the
             operator who knows the scenario in advance.
  learnsccg  results/38_learning_operator/lunar_k<k>_ccg*.json
             The same comparison when the two-stage problem is too large for
             one linear program (three outages): its optimal design comes from
             column-and-constraint generation, and the multi-stage design is
             operated two-stage scenario by scenario.
  learnsfam  results/39_learning_operator_random/
             The same comparison on the random examples.
  samefirst  results/42_longest_same_first_period/
             Baseline that needs no proof: every outage replaced by the longest
             outage with the same first period (at most k of them). Two-stage
             search with these 0-1 variables; multi-stage tree over this set.
             --ks: k for the two-stage runs; --only: k for the trees.
  famsamefirst results/42_longest_same_first_period/family.json
             The same baseline on the random examples (two-stage search and
             multi-stage tree, k = 2).
  trees      results/43_trees_one_session/lunar.json
             The tree linear program over at most k longest outages beginning in
             any period, over the set of Theorem 1 and over that of Proposition
             1, in one session, with simplex iterations (--ks, --repeats for k=2).
  famtrees   results/43_trees_one_session/family.json
             The same three trees on the random examples, one run each.
  variantsregret results/40_late_demand_variants/regret.json
             The design found with the scenarios of Proposition 1 in a variant
             where they understate the cost, operated over those of Theorem 1.
  variants   results/40_late_demand_variants/
             Variants of the lunar example in which all demand falls late in
             the horizon and holding is costlier: optimal cost and design of
             the multi-stage problem over the scenarios of Theorem 2 and over
             those of Theorem 1, and of the two-stage problem.

  python3 src/run_ol_final.py --stage counts
  python3 src/run_ol_final.py --stage treesize
  python3 src/run_ol_final.py --stage milpsep --ks 2,3,4 --repeats 3
  python3 src/run_ol_final.py --stage verify --ks 2,3 --jobs 8
  python3 src/run_ol_final.py --stage family --instances 30
  python3 src/run_ol_final.py --stage famselect --instances 30
  python3 src/run_ol_final.py --stage learns --k 2 --repeats 3
  python3 src/run_ol_final.py --stage learnsccg --k 3 --method highs-ds --time-limit 28800
  python3 src/run_ol_final.py --stage learnsfam --instances 30
  python3 src/run_ol_final.py --stage variants
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Sequence

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

import ol_scenarios as sc  # noqa: E402
from revision_common import RecourseEvaluator, environment_record, write_json  # noqa: E402
from run_ol_revision import (base_instance, ccg_milp, pairwise_filter, random_instance,  # noqa: E402
                             select_random_instances)

TOL_DOLLARS = 1.0
STAGES = {'counts': '34_scenario_counts', 'treesize': '34_scenario_counts',
          'milpsep': '35_milp_separation', 'worstfloor': '35_milp_separation',
          'verify': '36_all_scenarios_check',
          'family': '37_random_examples', 'famselect': '37_random_examples', 'learns': '38_learning_operator', 'learnsccg': '38_learning_operator',
          'learnsfam': '39_learning_operator_random', 'variants': '40_late_demand_variants',
          'samefirst': '42_longest_same_first_period', 'famsamefirst': '42_longest_same_first_period',
          'trees': '43_trees_one_session', 'famtrees': '43_trees_one_session', 'variantsregret': '40_late_demand_variants'}
VARIANTS = ('all_single_outages', 'longest_ending_inside', 'longest_ending_inside_exactly_k')


def _sets(p):
    return p.T, [int(v) for v in p.L_min], [int(v) for v in p.L_max]


def _candidates(p, name: str):
    T, lmin, lmax = _sets(p)
    if name == 'all_single_outages':
        return sc.outage_footprints(T, lmin, lmax), False
    if name == 'longest_any_start':
        return sc.longest_any_start(T, lmax), False
    return sc.longest_inside(T, lmax), name.endswith('exactly_k')


# ---------------------------------------------------------------- counts

def stage_counts(out_dir: str, max_list_k: int) -> dict:
    p = base_instance(2)
    T, lmin, lmax = _sets(p)
    singles = {'all': len(sc.outage_footprints(T, lmin, lmax)), 'thm1': len(sc.longest_inside(T, lmax)),
               'splus': len(sc.longest_any_start(T, lmax))}
    rows = []
    for k in (1, 2, 3, 4):
        n_all, n1, n2 = sc.count_all(T, lmin, lmax, k), sc.count_thm1(T, lmax, k), sc.count_splus(T, lmax, k)
        t0 = time.time()
        thm1 = sc.scenario_masks(T, lmin, lmax, k, 'thm1')
        t_thm1 = time.time() - t0
        t0 = time.time()
        splus = sc.scenario_masks(T, lmin, lmax, k, 'splus')
        t_splus = time.time() - t0
        assert len(thm1) == n1 and len(splus) == n2 and set(thm1) <= set(splus)
        row = {'k': k, 'scenarios': n_all, 'thm1': n1, 'splus': n2,
               'thm1_share_pct': 100.0 * n1 / n_all, 'splus_share_pct': 100.0 * n2 / n_all,
               'write_down_thm1_s': t_thm1, 'write_down_splus_s': t_splus,
               'T_at_least_k_times_longest': bool(all(T >= k * L for L in lmax)),
               'k_at_most_sum_of_ceil_T_over_longest': bool(k <= sum(-(-T // L) for L in lmax))}
        if k <= max_list_k:
            t0 = time.time()
            full = sc.scenario_masks(T, lmin, lmax, k, 'all')
            t_list = time.time() - t0
            assert len(full) == n_all and set(splus) <= set(full)
            t0 = time.time()
            filtered = pairwise_filter(full)
            t_filter = time.time() - t0
            row.update({'list_all_s': t_list, 'pairwise_filter_s': t_filter,
                        'filter_equals_thm1': set(filtered) == set(thm1)})
            assert row['filter_equals_thm1']
        rows.append(row)
        print('[counts]', {a: (round(b, 3) if isinstance(b, float) else b) for a, b in row.items()}, flush=True)
    out = {'instance': f'lunar example: T = {T}, shortest outages {lmin}, longest outages {lmax}',
           'single_outages': singles, 'rows': rows, 'environment': environment_record()}
    write_json(os.path.join(out_dir, 'counts.json'), out)
    return out


def tree_nodes(bits: Sequence[int], M: int, T: int) -> int:
    """Nodes of the scenario tree of the multi-stage problem: one per month and
    per distinct outage matrix up to that month (as in run_ol_nonanticipative.build_tree)."""
    nodes, upto = 0, 0
    for t in range(T):
        upto |= sum(1 << (m * T + t) for m in range(M))
        nodes += len({b & upto for b in bits})
    return nodes


def stage_treesize(out_dir: str, max_list_k: int) -> dict:
    """Size of the scenario tree for every scenario, for the set of Theorem 2
    and for the set of Theorem 1; no linear program is built."""
    p = base_instance(2)
    T, lmin, lmax = _sets(p)
    rows = []
    for k in (1, 2, 3, 4):
        row = {'k': k}
        for kind in ('all', 'splus', 'thm1'):
            if kind == 'all' and k > max_list_k:
                row[kind] = None       # the whole set is not listed for this k
                continue
            t0 = time.time()
            bits = sc.scenario_masks(T, lmin, lmax, k, kind)
            row[kind] = {'scenarios': len(bits), 'tree_nodes': tree_nodes(bits, len(lmax), T), 'count_s': time.time() - t0}
            del bits
        rows.append(row)
        print('[treesize]', {a: (b if not isinstance(b, dict) else (b['scenarios'], b['tree_nodes'])) for a, b in row.items()},
              flush=True)
    out = {'instance': f'lunar example: T = {T}, shortest outages {lmin}, longest outages {lmax}',
           'tree': 'one node per month and per distinct outage matrix up to that month',
           'rows': rows, 'environment': environment_record()}
    write_json(os.path.join(out_dir, 'tree_sizes.json'), out)
    return out


# ---------------------------------------------------------------- MILP separation

def _run_variants(p, k: int, names: Sequence[str], repeats: int, time_limit_s: float, tag: str) -> Dict[str, List[dict]]:
    runs: Dict[str, List[dict]] = {}
    for name in names:
        events, exact = _candidates(p, name)
        for i in range(repeats):
            print(f'[{tag}] k={k} {name} ({len(events)} 0-1 variables) run {i + 1}/{repeats}', flush=True)
            r = ccg_milp(p, events, k, exact, time_limit_s=time_limit_s)
            runs.setdefault(name, []).append(r)
            msg = (f'{r["rounds"]} rounds, design LP {r["design_lp_s_total"]:.1f}s, MILP {r["milp_s_total"]:.1f}s, '
                   f'wall {r["wall_s"]:.1f}s, cost {r["total_cost"]:.2f}') if r['finished'] else \
                  f'NOT FINISHED after {r["rounds"]} rounds, {r["wall_s"]:.0f}s'
            print(f'[{tag}] k={k} {name}: {msg}', flush=True)
            if not r['finished']:
                break                                   # an unfinished run is not repeated
    return runs


def _summary(runs: Dict[str, List[dict]]) -> Dict[str, dict]:
    out = {}
    for name, rs in runs.items():
        done = [r for r in rs if r['finished']]
        s = {'runs': len(rs), 'finished_runs': len(done), 'zero_one_variables': rs[0]['binaries']}
        if done:
            s.update({'rounds': done[0]['rounds'], 'total_cost': done[0]['total_cost'], 'q': done[0]['q'], 'C': done[0]['C'],
                      'wall_s_all_runs': [r['wall_s'] for r in done], 'median_wall_s': float(np.median([r['wall_s'] for r in done])),
                      'median_milp_s': float(np.median([r['milp_s_total'] for r in done])),
                      'median_design_lp_s': float(np.median([r['design_lp_s_total'] for r in done]))})
        else:
            s.update({'rounds_before_stop': rs[0]['rounds'], 'wall_s_before_stop': rs[0]['wall_s']})
        out[name] = s
    return out


def stage_milpsep(out_dir: str, ks: Sequence[int], repeats: int, time_limit_s: float, only: str,
                  variants: Sequence[str] = ()) -> None:
    names = list(variants) if variants else [n for n in VARIANTS if not only or only in n.split('|') or only == n]
    for k in ks:
        p = base_instance(k)
        runs = _run_variants(p, k, names, repeats, time_limit_s, 'milpsep')
        summ = _summary(runs)
        costs = [s['total_cost'] for s in summ.values() if 'total_cost' in s]
        out = {'instance': f'lunar example, demand from month 2, at most {k} outages',
               'time_limit_per_milp_s': time_limit_s, 'summary': summ,
               'same_cost_in_all_finished_variants': bool(costs and max(costs) - min(costs) <= TOL_DOLLARS),
               'runs': runs, 'environment': environment_record()}
        name = f'milp_separation_k{k}' + (f'_{only}' if only else '') + '.json'
        write_json(os.path.join(out_dir, name), out)
        print(f'[milpsep] k={k} summary:', json.dumps({a: {x: y for x, y in b.items() if x not in ("q", "C")}
                                                        for a, b in summ.items()}, default=str), flush=True)


# ---------------------------------------------------------------- operate the optimal design in every scenario

_EV = None
_ARGS = None


def _init_worker(k: int, C, q) -> None:
    global _EV, _ARGS
    p = base_instance(k)
    _EV, _ARGS = RecourseEvaluator(p), (p.M, p.T, np.asarray(C, dtype=float), np.asarray(q, dtype=float))


def _eval_chunk(masks: Sequence[int]):
    M, T, C, q = _ARGS
    worst, worst_mask = -1.0, None
    for b in masks:
        delta = np.array([[(b >> (m * T + t)) & 1 for t in range(T)] for m in range(M)], dtype=int)
        r = _EV.evaluate(C, q, delta)
        if r.status != 'optimal':
            raise RuntimeError('operating problem failed')
        if r.objective > worst:
            worst, worst_mask = float(r.objective), b
    return worst, worst_mask, len(masks)


def stage_verify(out_dir: str, ks: Sequence[int], jobs: int, time_limit_s: float) -> None:
    from multiprocessing import Pool
    for k in ks:
        p = base_instance(k)
        T, lmin, lmax = _sets(p)
        events, exact = _candidates(p, 'longest_ending_inside')
        design = ccg_milp(p, events, k, exact, time_limit_s=time_limit_s)
        if not design['finished']:
            raise RuntimeError('optimal design not found')
        masks = sc.scenario_masks(T, lmin, lmax, k, 'all')
        assert len(masks) == sc.count_all(T, lmin, lmax, k)
        chunks = [masks[i:i + 2000] for i in range(0, len(masks), 2000)]
        t0, worst, worst_mask, n = time.time(), -1.0, None, 0
        with Pool(jobs, initializer=_init_worker, initargs=(k, design['C'], design['q'])) as pool:
            for w, b, c in pool.imap_unordered(_eval_chunk, chunks):
                n += c
                if w > worst:
                    worst, worst_mask = w, b
                if n % 200000 < 2000:
                    print(f'[verify] k={k}: {n:,} of {len(masks):,} scenarios, worst so far {worst:.2f}, '
                          f'{time.time() - t0:.0f}s', flush=True)
        out = {'k': k, 'scenarios_checked': n, 'design': {'C': design['C'], 'q': design['q'], 'total_cost': design['total_cost']},
               'worst_operating_cost_all_scenarios': worst,
               'worst_operating_cost_theorem1_scenarios': design['worst_operating_cost'],
               'difference': worst - design['worst_operating_cost'],
               'worst_scenario_events': [list(e) for e in sc.events_of(worst_mask, p.M, T)],
               'wall_s': time.time() - t0, 'processes': jobs, 'environment': environment_record()}
        write_json(os.path.join(out_dir, f'verify_k{k}.json'), out)
        print(json.dumps({a: b for a, b in out.items() if a != 'environment'}, indent=1), flush=True)


def stage_worst_floor(out_dir: str, ks: Sequence[int]) -> dict:
    """The worst scenario of the optimal two-stage design (last round of milpsep,
    official record) operated once more with unlimited capacity on every line
    and at every depot. Demand that is still unmet is unmet under every design."""
    from run_ol_revision import UNLIMITED_TONS, delta_of
    rows = []
    for k in ks:
        d = json.load(open(os.path.join(PROJECT_ROOT, 'results', STAGES['milpsep'], f'milp_separation_k{k}.json')))
        run = d['runs']['longest_ending_inside'][0]
        events = [tuple(e) for e in run['round_log'][-1]['worst_events']]
        p = base_instance(k)
        ev = RecourseEvaluator(p)
        delta = delta_of(events, p.M, p.T)
        opt = ev.evaluate(run['C'], run['q'], delta)
        big = ev.evaluate([UNLIMITED_TONS] * p.J, [UNLIMITED_TONS] * p.M, delta)
        assert abs(opt.objective - run['worst_operating_cost']) <= TOL_DOLLARS, k
        rows.append({'k': k, 'worst_scenario_line_first_month_months': [[m + 1, a + 1, l] for m, a, l in events],
                     'optimal_design': {'operating_cost': opt.objective, 'unmet_tons': opt.unmet_tons,
                                        'unmet_by_commodity': list(opt.unmet_by_commodity)},
                     'unlimited_capacity': {'operating_cost': big.objective, 'unmet_tons': big.unmet_tons,
                                            'unmet_by_commodity': list(big.unmet_by_commodity)},
                     'total_cost_of_optimal_design': run['total_cost']})
        print('[worstfloor]', json.dumps(rows[-1]), flush=True)
    out = {'what': f'capacity {UNLIMITED_TONS:,.0f} tons per month on every line and at every depot', 'rows': rows,
           'environment': environment_record()}
    write_json(os.path.join(out_dir, 'worst_scenario_unlimited_capacity.json'), out)
    return out


def stage_same_first_period(out_dir: str, ks: Sequence[int], repeats: int, time_limit_s: float, method: str,
                            tree_ks: Sequence[int]) -> dict:
    """The reduction that needs no proof, as a baseline: every outage is replaced
    by the longest outage of its line with the same first period, which leaves
    the scenarios with at most k longest outages, each beginning in any period.
    Two-stage: column-and-constraint generation with one 0-1 variable for each
    of these longest outages, next to the ones that end within the horizon.
    Multi-stage: the tree linear program over this set, next to the set of the
    theorem (exactly k)."""
    from run_ol_nonanticipative import solve_tree_lp
    out = {'what': 'at most k longest outages, each beginning in any period of the horizon',
           'two_stage': {}, 'multi_stage': {}, 'environment': environment_record()}
    path = os.path.join(out_dir, 'lunar.json')
    for k in ks:
        p = base_instance(k)
        runs = _run_variants(p, k, ('longest_any_start', 'longest_ending_inside'), repeats, time_limit_s, 'samefirst')
        out['two_stage'][str(k)] = {'summary': _summary(runs), 'time_limit_per_milp_s': time_limit_s}
        write_json(path, out)
    for k in tree_ks:
        p = base_instance(k)
        T, lmin, lmax = _sets(p)
        exact = sc.scenario_masks(T, lmin, lmax, k, 'splus')
        upto = {0}
        for j in range(1, k + 1):
            upto |= set(sc.scenario_masks(T, lmin, lmax, j, 'splus'))
        rec = {}
        for name, bits in (('at_most_k', sorted(upto)), ('exactly_k', exact)):
            times, r = [], None
            for i in range(repeats if k <= 2 else 1):
                r = solve_tree_lp(p, bits, share=True, time_limit_s=time_limit_s, method=method)
                if r.get('status', 0) != 0:
                    raise RuntimeError(f'{name} k={k}: {r.get("message")}')
                times.append(r['solve_s'])
                print(f"[samefirst] tree k={k} {name}: {len(bits)} scenarios, {r['nodes']} nodes, {r['solve_s']:.1f}s, "
                      f"cost {r['total_cost']:.2f}", flush=True)
            rec[name] = {'scenarios': len(bits), 'nodes': r['nodes'], 'variables': r['variables'], 'total_cost': r['total_cost'],
                         'q': r['q'], 'solve_s_all_runs': times, 'median_solve_s': float(np.median(times)), 'method': method}
        assert abs(rec['at_most_k']['total_cost'] - rec['exactly_k']['total_cost']) <= 1e-6 * rec['exactly_k']['total_cost']
        out['multi_stage'][str(k)] = rec
        write_json(path, out)
    return out


def _longest_at_most_k(p, k: int) -> List[int]:
    """Matrices of the scenarios with at most k longest outages, each beginning in any period."""
    T, lmin, lmax = _sets(p)
    bits = {0}
    for j in range(1, k + 1):
        bits |= set(sc.scenario_masks(T, lmin, lmax, j, 'splus'))
    return sorted(bits)


def stage_family_same_first(out_dir: str, instances: int, repeats: int, time_limit_s: float, method: str,
                            reuse: str = '') -> dict:
    """The baseline of stage samefirst on the random examples (the seeds of famselect).
    Two-stage: column-and-constraint generation with one 0-1 variable for each longest
    outage beginning in any period, and for each longest outage ending within the
    horizon (Proposition 1), in the same session. Multi-stage: the tree linear program
    over at most k longest outages beginning in any period, and over exactly k
    (Theorem 1). k = 2.

    If the given method fails on either tree of an instance (HiGHS 1.2.0's dual simplex
    returned "HiGHS Status 4: Error" on one tree on 2026-11-12), both trees of that
    instance are solved again with the interior-point method, so that the two times of
    an instance always come from the same method; the failed attempt is recorded.
    `reuse`: a results file of this stage whose finished rows are kept, not solved again."""
    from run_ol_nonanticipative import solve_tree_lp
    k = 2
    methods = [method] + [m for m in ('highs-ipm',) if m != method]
    old = {r['seed']: r for r in json.load(open(reuse))['rows']} if reuse else {}
    rows = []
    path = os.path.join(out_dir, 'family.json')
    for seed in select_random_instances(instances)[0]:
        if seed in old:
            rows.append(dict(old[seed], reused_from=os.path.relpath(os.path.abspath(reuse), PROJECT_ROOT)))
            print(f'[famsamefirst] seed {seed}: kept from {reuse}', flush=True)
            continue
        p, desc = random_instance(seed)
        T, lmin, lmax = _sets(p)
        runs = _run_variants(p, k, ('longest_any_start', 'longest_ending_inside'), repeats, time_limit_s,
                             f'famsamefirst seed {seed}')
        summ = _summary(runs)
        a, b = summ['longest_any_start'], summ['longest_ending_inside']
        assert a['rounds'] == b['rounds'] and abs(a['total_cost'] - b['total_cost']) <= TOL_DOLLARS, seed
        sets = (('at_most_k', _longest_at_most_k(p, k)), ('exactly_k', sc.scenario_masks(T, lmin, lmax, k, 'splus')))
        failed, trees = [], None
        for m in methods:
            trees = {}
            for name, bits in sets:
                r = solve_tree_lp(p, bits, share=True, time_limit_s=time_limit_s, method=m)
                if r.get('status', 0) != 0:
                    failed.append({'tree': name, 'method': m, 'message': str(r.get('message')), 'solve_s': r.get('solve_s')})
                    print(f"[famsamefirst] seed {seed} tree {name}: {m} failed ({r.get('message')}); "
                          f"both trees again with the next method", flush=True)
                    trees = None
                    break
                trees[name] = {'scenarios': len(bits), 'nodes': r['nodes'], 'variables': r['variables'],
                               'total_cost': r['total_cost'], 'solve_s': r['solve_s'], 'method': m}
                print(f"[famsamefirst] seed {seed} tree {name}: {len(bits)} scenarios, {r['nodes']} nodes, "
                      f"{r['solve_s']:.1f}s ({m}), cost {r['total_cost']:.2f}", flush=True)
            if trees is not None:
                break
        if trees is None:
            raise RuntimeError(f'seed {seed}: every method failed: {failed}')
        assert abs(trees['at_most_k']['total_cost'] - trees['exactly_k']['total_cost']) <= 1e-6 * trees['exactly_k']['total_cost']
        rows.append({'seed': seed, 'L_max': desc['L_max'], 'two_stage': summ, 'multi_stage': trees,
                     'multi_stage_failed_attempts': failed,
                     'two_stage_time_ratio_any_start_over_ending_inside': a['median_wall_s'] / b['median_wall_s']})
        write_json(path, {'what': 'at most k longest outages, each beginning in any period, on the random examples',
                          'k': k, 'time_limit_s': time_limit_s, 'methods': methods, 'rows': rows,
                          'environment': environment_record()})
    return {'rows': rows}


TREE_SETS = ('at_most_k_longest', 'theorem_1', 'proposition_1')


def _tree_sets(p, k: int):
    T, lmin, lmax = _sets(p)
    return (('at_most_k_longest', _longest_at_most_k(p, k)), ('theorem_1', sc.scenario_masks(T, lmin, lmax, k, 'splus')),
            ('proposition_1', sc.scenario_masks(T, lmin, lmax, k, 'thm1')))


def _solve_trees(p, sets, repeats: int, time_limit_s: float, methods: Sequence[str], tag: str) -> dict:
    """Solve the tree linear program of every set, `repeats` times, all with one method. If the
    method fails on any of them, all are solved again with the next method; failures are recorded."""
    from run_ol_nonanticipative import solve_tree_lp
    failed = []
    for m in methods:
        out = {}
        for name, bits in sets:
            runs = []
            for i in range(repeats):
                r = solve_tree_lp(p, bits, share=True, time_limit_s=time_limit_s, method=m)
                if r.get('status', 0) != 0:
                    failed.append({'tree': name, 'method': m, 'message': str(r.get('message')), 'solve_s': r.get('solve_s')})
                    print(f"[{tag}] tree {name}: {m} failed ({r.get('message')}); all trees again with the next method",
                          flush=True)
                    break
                runs.append(r)
                print(f"[{tag}] tree {name} run {i + 1}/{repeats}: {len(bits)} scenarios, {r['nodes']} nodes, "
                      f"{r['iterations']} iterations, {r['solve_s']:.1f}s ({m}), cost {r['total_cost']:.2f}", flush=True)
            if len(runs) < repeats:
                out = None
                break
            out[name] = {'scenarios': len(bits), 'nodes': runs[0]['nodes'], 'variables': runs[0]['variables'],
                         'rows': runs[0]['rows'], 'nonzeros': runs[0]['nonzeros'], 'total_cost': runs[0]['total_cost'],
                         'q': runs[0]['q'], 'C': runs[0]['C'], 'method': m,
                         'iterations_all_runs': [r['iterations'] for r in runs],
                         'solve_s_all_runs': [r['solve_s'] for r in runs],
                         'median_solve_s': float(np.median([r['solve_s'] for r in runs]))}
        if out is not None:
            return {'trees': out, 'failed_attempts': failed}
    raise RuntimeError(f'{tag}: every method failed: {failed}')


def stage_trees(out_dir: str, ks: Sequence[int], repeats: int, time_limit_s: float, method: str) -> dict:
    """The lunar example: the tree linear program of the multi-stage problem over the scenarios
    with at most k longest outages beginning in any period, over those of Theorem 1 and over
    those of Proposition 1, in one session, with simplex iterations. `repeats` runs for k = 2,
    one run for larger k."""
    methods = [method] + [m for m in ('highs-ipm',) if m != method]
    out = {'instance': 'lunar example, demand from month 2', 'methods': methods, 'runs': {},
           'environment': environment_record()}
    path = os.path.join(out_dir, 'lunar.json')
    for k in ks:
        p = base_instance(k)
        rec = _solve_trees(p, _tree_sets(p, k), repeats if k <= 2 else 1, time_limit_s, methods, f'trees k={k}')
        costs = [v['total_cost'] for n, v in rec['trees'].items() if n != 'proposition_1']
        assert max(costs) - min(costs) <= 1e-6 * max(costs), (k, costs)
        out['runs'][str(k)] = rec
        write_json(path, out)
    return out


def stage_family_trees(out_dir: str, instances: int, time_limit_s: float, method: str) -> dict:
    """The random examples (seeds of famselect), k = 2: the three tree linear programs of stage
    trees, one run each, in one session, with simplex iterations."""
    methods = [method] + [m for m in ('highs-ipm',) if m != method]
    rows = []
    path = os.path.join(out_dir, 'family.json')
    for seed in select_random_instances(instances)[0]:
        p, desc = random_instance(seed)
        rec = _solve_trees(p, _tree_sets(p, 2), 1, time_limit_s, methods, f'famtrees seed {seed}')
        t = rec['trees']
        assert abs(t['at_most_k_longest']['total_cost'] - t['theorem_1']['total_cost']) <= 1e-6 * t['theorem_1']['total_cost']
        rows.append(dict(rec, seed=seed, L_max=desc['L_max']))
        write_json(path, {'k': 2, 'methods': methods, 'rows': rows, 'environment': environment_record()})
    return {'rows': rows}


def stage_variants_regret(out_dir: str, time_limit_s: float) -> dict:
    """For each late-demand variant in which the tree over the scenarios of Proposition 1 gives a
    lower optimal cost than the tree over those of Theorem 1 (results/40_late_demand_variants/
    variants.json): the cost of the design found with the scenarios of Proposition 1 when it is
    operated in the multi-stage problem over the scenarios of Theorem 1 (which give the cost over
    all scenarios), against the optimal cost there."""
    from run_ol_nonanticipative import scenario_bits, solve_tree_lp
    src = os.path.join(PROJECT_ROOT, 'results', STAGES['variants'], 'variants.json')
    rows = []
    for v in json.load(open(src))['rows']:
        if v['thm1_set_understates_by'] <= TOL_DOLLARS:
            continue
        p = variant_instance(v['first_demand_month'], v['demand_factor'], v['holding_factor'])
        r = solve_tree_lp(p, scenario_bits(p, 2, 'splus'), share=True, time_limit_s=time_limit_s,
                          fixed_C=v['C_multi_stage_on_thm1_set'], fixed_q=v['q_multi_stage_on_thm1_set'])
        if r.get('status', 0) != 0:
            raise RuntimeError(f"{v['variant']}: {r.get('message')}")
        row = {'variant': v['variant'], 'first_demand_month': v['first_demand_month'], 'demand_factor': v['demand_factor'],
               'holding_factor': v['holding_factor'], 'multi_stage_total_cost': v['multi_stage_total_cost'],
               'cost_reported_with_proposition_1_set': v['multi_stage_total_cost_on_thm1_set'],
               'true_cost_of_that_design': r['total_cost'],
               'extra_cost_of_that_design': r['total_cost'] - v['multi_stage_total_cost'],
               'extra_cost_pct': 100.0 * (r['total_cost'] - v['multi_stage_total_cost']) / v['multi_stage_total_cost'],
               'q': v['q_multi_stage_on_thm1_set'], 'C': v['C_multi_stage_on_thm1_set'], 'solve_s': r['solve_s']}
        assert row['extra_cost_of_that_design'] >= -1e-6 * v['multi_stage_total_cost'], row
        rows.append(row)
        print(f"[variantsregret] {v['variant']}: reported {row['cost_reported_with_proposition_1_set']:.2f}, true cost of that "
              f"design {r['total_cost']:.2f}, optimum {v['multi_stage_total_cost']:.2f}, extra {row['extra_cost_of_that_design']:.2f}",
              flush=True)
        write_json(os.path.join(out_dir, 'regret.json'), {'source': os.path.relpath(src, PROJECT_ROOT), 'rows': rows,
                                                          'environment': environment_record()})
    return {'rows': rows}


# ---------------------------------------------------------------- random examples

def stage_family(out_dir: str, instances: int, repeats: int, time_limit_s: float, select: bool = True,
                 variants: Sequence[str] = ()) -> dict:
    rows = []
    names = tuple(variants) if variants else ('all_single_outages', 'longest_ending_inside')
    for seed in select_random_instances(instances, select)[0]:
        p, desc = random_instance(seed)
        T, lmin, lmax = _sets(p)
        runs = _run_variants(p, 2, names, repeats, time_limit_s, f'family seed {seed}')
        summ = _summary(runs)
        row = dict(desc)
        row.update({'scenarios': sc.count_all(T, lmin, lmax, 2), 'thm1': sc.count_thm1(T, lmax, 2),
                    'splus': sc.count_splus(T, lmax, 2), 'summary': summ})
        a, b = summ['all_single_outages'], summ['longest_ending_inside']
        if all('total_cost' in v for v in summ.values()):
            row['same_cost'] = bool(max(v['total_cost'] for v in summ.values())
                                    - min(v['total_cost'] for v in summ.values()) <= TOL_DOLLARS)
            row['same_rounds'] = bool(len({v['rounds'] for v in summ.values()}) == 1)
            row['time_ratio'] = a['median_wall_s'] / b['median_wall_s']
            if 'longest_any_start' in summ:
                row['time_ratio_any_start_over_ending_inside'] = summ['longest_any_start']['median_wall_s'] / b['median_wall_s']
        rows.append(row)
        write_json(os.path.join(out_dir, 'family.json'), {'time_limit_per_milp_s': time_limit_s, 'rows': rows,
                                                           'environment': environment_record()})
    return {'rows': rows}


def stage_family_select(out_dir: str, instances: int, select: bool) -> dict:
    """Which seeds make up the random examples (run_ol_revision.select_random_instances),
    with the demand that no design can serve for every seed tried."""
    seeds, tried = select_random_instances(instances, select)
    for r in tried:
        print(f"[famselect] seed {r['seed']}: unmet without outages and with unlimited capacity "
              f"{r['unmet_tons']:.2f} t of {r['total_demand_tons']:.0f}; {'used' if r['used'] else 'not used'}", flush=True)
    out = {'rule': 'seeds are tried in the order 0, 1, 2, ...; a seed is used if all of its demand can be served '
                   'when no line fails and capacity is unlimited' if select else 'seeds 0..n-1, whatever their demand',
           'instances': instances, 'seeds_used': seeds, 'seeds_tried': len(tried),
           'seeds_not_used': [r['seed'] for r in tried if not r['used']], 'tried': tried,
           'environment': environment_record()}
    write_json(os.path.join(out_dir, 'seed_selection.json'), out)
    print(f"[famselect] {len(seeds)} seeds used of {len(tried)} tried: {seeds}", flush=True)
    return out


# ---------------------------------------------------------------- variants with late demand

VARIANT_GRID = [(a, f, h) for a in (13, 19, 22) for f in (2.0, 3.0) for h in (1.0, 100.0, 300.0, 1000.0)]


def variant_instance(first_month: int, demand_factor: float, holding_factor: float):
    """The lunar example with no demand before `first_month` (1-based), the
    demand from then on multiplied by `demand_factor`, and the holding costs
    multiplied by `holding_factor`. At most two outages."""
    p = base_instance(2)
    D = np.array(p.D_demand, dtype=float)
    D[:, :, :first_month - 1] = 0.0
    D[:, :, first_month - 1:] *= demand_factor
    p.D_demand = D
    p.h_depot = np.asarray(p.h_depot, dtype=float) * holding_factor
    return p


def stage_variants(out_dir: str, time_limit_s: float, only: str = '') -> dict:
    from run_ol_nonanticipative import scenario_bits, solve_tree_lp
    rows = []
    for (a, f, h) in VARIANT_GRID:
        name = f'from{a}_x{f:g}_h{h:g}'
        if only and only != name:
            continue
        p = variant_instance(a, f, h)
        thm1, splus = scenario_bits(p, 2, 'thm1'), scenario_bits(p, 2, 'splus')
        two = solve_tree_lp(p, thm1, share=False, time_limit_s=time_limit_s)
        multi = solve_tree_lp(p, splus, share=True, time_limit_s=time_limit_s)
        multi_on_thm1 = solve_tree_lp(p, thm1, share=True, time_limit_s=time_limit_s)
        if not (two['status'] == multi['status'] == multi_on_thm1['status'] == 0):
            raise RuntimeError(f'{name}: a linear program was not solved')
        row = {'variant': name, 'first_demand_month': a, 'demand_factor': f, 'holding_factor': h,
               'total_demand_tons': float(np.asarray(p.D_demand).sum()),
               'two_stage_total_cost': two['total_cost'], 'multi_stage_total_cost': multi['total_cost'],
               'multi_stage_total_cost_on_thm1_set': multi_on_thm1['total_cost'],
               'thm1_set_understates_by': multi['total_cost'] - multi_on_thm1['total_cost'],
               'thm1_set_understates_by_pct': 100.0 * (multi['total_cost'] - multi_on_thm1['total_cost']) / multi['total_cost'],
               'multi_minus_two_stage': multi['total_cost'] - two['total_cost'],
               'q_two_stage': two['q'], 'q_multi_stage': multi['q'], 'q_multi_stage_on_thm1_set': multi_on_thm1['q'],
               'C_multi_stage': multi['C'], 'C_multi_stage_on_thm1_set': multi_on_thm1['C'],
               'solve_s': [two['solve_s'], multi['solve_s'], multi_on_thm1['solve_s']]}
        rows.append(row)
        print(f"[variants] {name}: two-stage {two['total_cost']:.2f}, multi-stage {multi['total_cost']:.2f}, "
              f"multi-stage on the Theorem 1 set {multi_on_thm1['total_cost']:.2f} "
              f"(short by {row['thm1_set_understates_by']:.2f}, {row['thm1_set_understates_by_pct']:.4f}%); "
              f"q {np.round(multi['q'], 1).tolist()} against {np.round(multi_on_thm1['q'], 1).tolist()}", flush=True)
        write_json(os.path.join(out_dir, 'variants.json'), {'rows': rows, 'environment': environment_record()})
    return {'rows': rows}


# ---------------------------------------------------------------- two-stage and multi-stage, more outages

MILP_LIMIT_S = 1800.0


def stage_learns_ccg(out_dir: str, k: int, time_limit_s: float, method: str, jobs: int, reuse: str = '') -> dict:
    """The lunar example as a two-stage and as a multi-stage problem when the
    two-stage problem is too large for one linear program (three outages:
    26,223 scenarios of 24 months). Its optimal design then comes from
    column-and-constraint generation with the scenarios of Theorem 1, and the
    multi-stage design is operated two-stage scenario by scenario. The run
    names match those of stage `learns` where the meaning is the same.

    `reuse` names a results file of the same instance; a finished run found
    there under the same name, solved with the same method and, where a
    design is fixed, for the same design is copied instead of being solved
    again, and the copy says where it came from."""
    from multiprocessing import Pool
    from run_ol_nonanticipative import scenario_bits, solve_tree_lp
    p = base_instance(k)
    tag = '' if method == 'highs' else '_' + method.replace('highs-', '')
    path = os.path.join(out_dir, f'lunar_k{k}_ccg{tag}.json')
    out = {'instance': f'lunar example, demand from month 2, at most {k} outages, outages may run past month T',
           'environment': environment_record(), 'runs': {}}
    old = json.load(open(reuse)) if reuse else {'runs': {}}
    if reuse and old['instance'] != out['instance']:
        raise RuntimeError('--reuse: that file holds another instance')
    sets = {kind: scenario_bits(p, k, kind) for kind in ('thm1', 'splus')}
    out['counts'] = {kind: len(v) for kind, v in sets.items()}
    print('[learnsccg] counts', out['counts'], flush=True)

    def keep(name, compute, needs_method=True, design=None):
        r = old['runs'].get(name)
        same_design = r is not None and (design is None or (np.allclose(r['C'], design[0], rtol=0, atol=1e-9)
                                                             and np.allclose(r['q'], design[1], rtol=0, atol=1e-9)))
        if r is not None and r.get('status', 0) == 0 and same_design and (not needs_method or r.get('method') == method):
            r = dict(r, reused_from=os.path.relpath(os.path.abspath(reuse), PROJECT_ROOT),
                     reused_run_started=old['environment']['timestamp'])
        else:
            r = compute()
        out['runs'][name] = r
        write_json(path, out)
        if r.get('status', 0) != 0:
            raise RuntimeError(f"{name}: {r.get('message')}")
        took = r.get('solve_s', r.get('wall_s'))
        print(f"[learnsccg] {name}: total cost {r['total_cost']:.2f}, q {np.round(r['q'], 2).tolist()}, "
              f"C {np.round(r['C'], 1).tolist()}, {took:.1f}s" + (' (reused)' if 'reused_from' in r else ''), flush=True)
        return r

    def tree(name, kind, share, **kw):
        def compute():
            r = solve_tree_lp(p, sets[kind], share=share, time_limit_s=time_limit_s, method=method, **kw)
            r['method'] = method
            return r
        return keep(name, compute, design=(kw['fixed_C'], kw['fixed_q']) if kw else None)

    def two_stage():
        events, exact = _candidates(p, 'longest_ending_inside')
        r = ccg_milp(p, events, k, exact, time_limit_s=MILP_LIMIT_S)
        if not r['finished']:
            return {'status': 1, 'message': 'column-and-constraint generation did not finish'}
        return {'status': 0, 'method': 'column-and-constraint generation with the longest outages that end within the horizon',
                'zero_one_variables': r['binaries'], 'rounds': r['rounds'], 'wall_s': r['wall_s'],
                'total_cost': r['total_cost'], 'worst_operating_cost': r['worst_operating_cost'],
                'investment_cost': r['total_cost'] - r['worst_operating_cost'], 'C': r['C'], 'q': r['q']}

    a = keep('knows_scenario__column_and_constraint_generation', two_stage, needs_method=False)
    b = tree('learns_as_outages_happen__splus_set', 'splus', True)
    c = tree('learns_as_outages_happen__thm1_set', 'thm1', True)
    d = tree('design_for_knowing__operated_learning__splus_set', 'splus', True, fixed_C=a['C'], fixed_q=a['q'])

    def operate_two_stage():
        masks = sets['thm1']
        chunks = [masks[i:i + 500] for i in range(0, len(masks), 500)]
        t0, worst, worst_mask, n = time.time(), -1.0, None, 0
        with Pool(jobs, initializer=_init_worker, initargs=(k, b['C'], b['q'])) as pool:
            for w, m, cnt in pool.imap_unordered(_eval_chunk, chunks):
                n += cnt
                if w > worst:
                    worst, worst_mask = w, m
        assert n == len(masks)
        return {'status': 0, 'method': 'the operating problem solved scenario by scenario', 'scenarios': n,
                'worst_operating_cost': worst, 'investment_cost': b['investment_cost'],
                'total_cost': b['investment_cost'] + worst, 'C': b['C'], 'q': b['q'],
                'worst_scenario_events': [list(e) for e in sc.events_of(worst_mask, p.M, p.T)],
                'wall_s': time.time() - t0, 'processes': jobs}

    e = keep('design_for_learning__operated_knowing__thm1_set_scenario_by_scenario', operate_two_stage, needs_method=False,
             design=(b['C'], b['q']))

    tol = 1e-6 * a['total_cost']
    out['summary'] = {
        'two_stage_total_cost': a['total_cost'], 'multi_stage_total_cost': b['total_cost'],
        'multi_stage_minus_two_stage': b['total_cost'] - a['total_cost'],
        'multi_stage_on_thm1_set_total_cost': c['total_cost'],
        'thm1_set_understates_by': b['total_cost'] - c['total_cost'],
        'two_stage_design_operated_multi_stage_extra_cost': d['total_cost'] - b['total_cost'],
        'multi_stage_design_operated_two_stage_extra_cost': e['total_cost'] - a['total_cost'],
        'tolerance_dollars': tol,
        'checks': {
            'multi_stage_at_least_two_stage': bool(b['total_cost'] >= a['total_cost'] - tol),
            'thm1_set_value_between_the_two': bool(a['total_cost'] - tol <= c['total_cost'] <= b['total_cost'] + tol),
            'two_stage_design_no_better_for_multi_stage': bool(d['total_cost'] >= b['total_cost'] - tol),
            'multi_stage_design_between_for_two_stage': bool(a['total_cost'] - tol <= e['total_cost'] <= b['total_cost'] + tol)}}
    write_json(path, out)
    print('[learnsccg] summary', json.dumps(out['summary'], indent=1), flush=True)
    return out


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=list(STAGES))
    ap.add_argument('--ks', default='', help='milpsep, verify: numbers of outages, e.g. 2,3,4')
    ap.add_argument('--k', type=int, default=2, help='learns: number of outages')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--instances', type=int, default=30)
    ap.add_argument('--max-list-k', type=int, default=3, help='counts, treesize: list the whole scenario set up to this k')
    ap.add_argument('--time-limit', type=float, default=1800.0, help='seconds per MILP, or per tree LP')
    ap.add_argument('--only', default='', help="milpsep, variants: run only this variant; learns: 'splus' solves only the tree over the scenarios of Theorem 2")
    ap.add_argument('--jobs', type=int, default=8, help='verify, learnsccg: processes')
    ap.add_argument('--reuse', default='', help='learnsccg, famsamefirst: results file whose finished runs are copied, not solved again')
    ap.add_argument('--with-all', action='store_true', help='learns: also try the tree over every scenario')
    ap.add_argument('--only-all', action='store_true', help='learns: only the tree over every scenario')
    ap.add_argument('--method', default='highs', choices=['highs', 'highs-ds', 'highs-ipm'])
    ap.add_argument('--out-suffix', default='', help='smoke tests: append to the results directory name')
    ap.add_argument('--variants', default='', help='milpsep, family: comma-separated variants, e.g. '
                    'all_single_outages,longest_any_start,longest_ending_inside')
    ap.add_argument('--all-seeds', action='store_true',
                    help='family, learnsfam, famselect: seeds 0..n-1 whatever their demand, as before 2026-11-12')
    args = ap.parse_args()
    out_dir = os.path.join(PROJECT_ROOT, 'results', STAGES[args.stage] + args.out_suffix)
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()
    if args.stage == 'counts':
        stage_counts(out_dir, args.max_list_k)
    elif args.stage == 'treesize':
        stage_treesize(out_dir, args.max_list_k)
    elif args.stage == 'milpsep':
        stage_milpsep(out_dir, [int(v) for v in (args.ks or '2,3,4').split(',')], args.repeats, args.time_limit, args.only,
                      [v for v in args.variants.split(',') if v])
    elif args.stage == 'worstfloor':
        stage_worst_floor(out_dir, [int(v) for v in (args.ks or '2,3,4').split(',')])
    elif args.stage == 'verify':
        stage_verify(out_dir, [int(v) for v in (args.ks or '2,3').split(',')], args.jobs, args.time_limit)
    elif args.stage == 'family':
        stage_family(out_dir, args.instances, args.repeats, args.time_limit, not args.all_seeds,
                     [v for v in args.variants.split(',') if v])
    elif args.stage == 'famselect':
        stage_family_select(out_dir, args.instances, not args.all_seeds)
    elif args.stage == 'learns':
        from run_ol_nonanticipative import stage_lunar
        stage_lunar(out_dir, args.k, args.with_all or args.only_all, args.time_limit, args.method, args.only_all,
                    repeats=args.repeats, only_splus=(args.only == 'splus'))
    elif args.stage == 'learnsccg':
        stage_learns_ccg(out_dir, args.k, args.time_limit, args.method, args.jobs, args.reuse)
    elif args.stage == 'learnsfam':
        from run_ol_nonanticipative import stage_family as stage_learning_family
        stage_learning_family(out_dir, args.instances, args.time_limit, not args.all_seeds)
    elif args.stage == 'samefirst':
        stage_same_first_period(out_dir, [int(v) for v in (args.ks or '2,3,4').split(',')], args.repeats, args.time_limit,
                                args.method, [int(v) for v in (args.only or '2,3').split(',')])
    elif args.stage == 'famsamefirst':
        stage_family_same_first(out_dir, args.instances, args.repeats, args.time_limit, args.method, args.reuse)
    elif args.stage == 'trees':
        stage_trees(out_dir, [int(v) for v in (args.ks or '2,3').split(',')], args.repeats, args.time_limit, args.method)
    elif args.stage == 'famtrees':
        stage_family_trees(out_dir, args.instances, args.time_limit, args.method)
    elif args.stage == 'variantsregret':
        stage_variants_regret(out_dir, args.time_limit)
    elif args.stage == 'variants':
        stage_variants(out_dir, args.time_limit, args.only)
    print(f'done in {time.time() - started:.0f}s -> {out_dir}')


if __name__ == '__main__':
    main()
