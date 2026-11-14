"""Experiments for the major revision of the Optimization Letters manuscript.

Started 2026-11-10 after the simulated review
(paper/optimization_letters/reviews/SIM_REVIEW_20261110.md). Each stage writes
into its own new directory under results/ and reuses the recorded model code
(lp_solver.solve_robust_model for the design LP, revision_common.RecourseEvaluator
for the operating problem). The example used here has no demand in month 1,
the month in which no design can deliver; stage `month2` shows that this only
removes a constant of $3.5 billion.

  month2   results/27_demand_from_month_2/
           Optimal design with and without month-1 demand.
  ccgstd   results/28_ccg_standard/
           Column-and-constraint generation started from the no-outage
           scenario, adding the worst scenario in each round. The check in
           each round runs over all scenarios or over the kept scenarios
           (k longest outages). Repeated, with design-LP and check times apart.
  kscale   results/29_k_scale/
           For k = 1..4 outages per scenario: number of scenarios, number
           kept, time to write the kept scenarios down, and, where the
           scenario set can be listed, time to list it and filter it by
           pairwise comparison. Optimal design for k = 3 by ccg over the kept
           scenarios.
  family   results/30_instance_family/
           Randomly generated examples (costs, demand, initial stock, longest
           outage lengths): rounds, deciding scenarios and where they sit,
           work with the two kinds of check.

  python3 src/run_ol_revision.py --stage ccgstd --repeats 5 --repeats-all 3
  python3 src/run_ol_revision.py --stage kscale
  python3 src/run_ol_revision.py --stage family --instances 30 --all-oracle 5
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import sys
import time
from functools import lru_cache
from itertools import product
from math import comb
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from instance_generator import DisruptionScenario, InstanceParameters, generate_small_instance  # noqa: E402
from lp_solver import solve_robust_model  # noqa: E402
from revision_common import RecourseEvaluator, environment_record, write_json  # noqa: E402

TOL_DOLLARS = 1.0
Event = Tuple[int, int, int]          # (line, start period 0-based, length)


# ---------------------------------------------------------------- instances

def base_instance(k: int = 2) -> InstanceParameters:
    """The lunar example with demand from month 2 on and at most k outages."""
    p = copy.deepcopy(generate_small_instance())
    p.D_demand = np.array(p.D_demand, dtype=float)
    p.D_demand[:, :, 0] = 0.0
    p.Gamma = int(k)
    return p


def random_instance(seed: int) -> Tuple[InstanceParameters, Dict]:
    """A random example around the lunar one (k = 2, T = 24, three lines)."""
    rng = np.random.default_rng(seed)
    p = base_instance(2)
    f = {
        'transport_cost': rng.uniform(0.5, 1.5, p.M), 'line_cost': rng.uniform(0.5, 1.5, p.M),
        'depot_cost': rng.uniform(0.5, 1.5, p.J), 'holding_cost': float(rng.uniform(0.5, 1.5)),
        'demand_level': rng.uniform(0.7, 1.3, p.K), 'initial_stock': float(rng.uniform(0.5, 3.0)),
    }
    p.c_mode = np.asarray(p.c_mode, dtype=float) * f['transport_cost']
    p.v_mode = np.asarray(p.v_mode, dtype=float) * f['line_cost']
    p.f_depot = np.asarray(p.f_depot, dtype=float) * f['depot_cost']
    p.h_depot = np.asarray(p.h_depot, dtype=float) * f['holding_cost']
    level = np.array([200.0, 100.0]) * f['demand_level']
    D = np.zeros((p.D, p.K, p.T))
    D[0, 0, :], D[0, 1, :] = level[0], level[1]
    peaks0 = sorted(int(t) for t in rng.choice(np.arange(1, p.T), size=4, replace=False))
    peaks1 = sorted(int(t) for t in rng.choice(np.arange(1, p.T), size=3, replace=False))
    D[0, 0, peaks0], D[0, 1, peaks1] = 2.0 * level[0], 2.5 * level[1]
    D[:, :, 0] = 0.0
    p.D_demand = D
    p.s_initial = np.asarray(p.s_initial, dtype=float) * f['initial_stock']
    lmax = [(6, 6, 2), (4, 4, 2), (5, 5, 2), (6, 6, 4), (3, 3, 3), (6, 4, 2)][int(rng.integers(0, 6))]
    p.L_max = np.array(lmax)
    p.L_min = np.minimum(np.array([2, 2, 1]), p.L_max)
    desc = {k_: (v.tolist() if hasattr(v, 'tolist') else v) for k_, v in f.items()}
    desc.update({'seed': seed, 'L_max': list(lmax), 'peak_periods_commodity_1': [t + 1 for t in peaks0],
                 'peak_periods_commodity_2': [t + 1 for t in peaks1],
                 'total_demand_tons': float(D.sum())})
    return p, desc


UNLIMITED_TONS = 1.0e6     # per month; far above the total demand of any example (about 10,000 tons)


def unservable_demand(p: InstanceParameters) -> Dict:
    """The example operated with no outage and with unlimited capacity on every
    line and at every depot. Capacity and outages enter the operating problem
    only through upper bounds, so the demand left unmet here is unmet under
    every design and in every scenario (a demand peak before the first
    shipment can arrive, or above what the destination can receive)."""
    ev = RecourseEvaluator(p)
    zero = np.zeros((p.M, p.T))
    r = ev.evaluate([UNLIMITED_TONS] * p.J, [UNLIMITED_TONS] * p.M, zero)
    twice = ev.evaluate([2 * UNLIMITED_TONS] * p.J, [2 * UNLIMITED_TONS] * p.M, zero)
    if r.status != 'optimal' or twice.status != 'optimal':
        raise RuntimeError(f'operating problem without outages: {r.status} {r.message}')
    assert abs(twice.objective - r.objective) <= 1e-9 * max(1.0, abs(r.objective))    # the bound does not bind
    return {'unmet_tons': float(r.unmet_tons), 'unmet_by_commodity': [float(v) for v in r.unmet_by_commodity],
            'operating_cost_dollars': float(r.objective)}


def select_random_instances(n: int, select: bool = True, tol_tons: float = 1e-6) -> Tuple[List[int], List[Dict]]:
    """Seeds of the n random examples that are used, and the record of every seed
    tried. Seeds are tried in the order 0, 1, 2, ...; a seed is used if all of
    its demand can be served when no line fails and capacity is unlimited.
    With select=False the seeds are 0..n-1 whatever their demand (the choice
    made until 2026-11-12, when 13 of those 30 turned out to carry demand that
    no design can serve)."""
    seeds: List[int] = []
    tried: List[Dict] = []
    seed = 0
    while len(seeds) < n:
        p, desc = random_instance(seed)
        rec = {'seed': seed, 'total_demand_tons': desc['total_demand_tons']}
        rec.update(unservable_demand(p))
        rec['used'] = bool(not select or rec['unmet_tons'] <= tol_tons)
        tried.append(rec)
        if rec['used']:
            seeds.append(seed)
        seed += 1
    return seeds, tried


# ---------------------------------------------------------------- scenarios

def compositions(k: int, M: int) -> Iterator[Tuple[int, ...]]:
    if M == 1:
        yield (k,)
        return
    for a in range(k + 1):
        for rest in compositions(k - a, M - 1):
            yield (a,) + rest


def count_kept(T: int, L: Sequence[int], k: int) -> int:
    """Formula of Theorem 1 (iii)."""
    total = 0
    for ks in compositions(k, len(L)):
        prod = 1
        for km, l in zip(ks, L):
            prod *= comb(T - km * l + km, km)
        total += prod
    return total


def count_scenarios(T: int, lmin: Sequence[int], lmax: Sequence[int], k: int) -> int:
    """Number of scenarios with at most k outages (same-line outages do not overlap)."""
    def ways(lo, hi, j):
        @lru_cache(None)
        def f(pos, left):
            if left == 0:
                return 1
            if pos > T:
                return 0
            tot = f(pos + 1, left)
            for l in range(lo, hi + 1):
                if pos + l - 1 <= T:
                    tot += f(pos + l, left - 1)
            return tot
        return f(1, j)
    a = [[ways(int(lmin[m]), int(lmax[m]), j) for j in range(k + 1)] for m in range(len(lmax))]
    return sum(int(np.prod([a[m][js[m]] for m in range(len(lmax))], dtype=object))
               for js in product(range(k + 1), repeat=len(lmax)) if sum(js) <= k)


def _placements(T: int, L: int, n: int) -> Iterator[Tuple[int, ...]]:
    """Start periods (0-based) of n non-overlapping blocks of length L."""
    def rec(start, left):
        if left == 0:
            yield ()
            return
        for a in range(start, T - left * L + 1):
            for rest in rec(a + L, left - 1):
                yield (a,) + rest
    yield from rec(0, n)


def kept_scenarios(params: InstanceParameters, k: int) -> List[List[Event]]:
    """Scenarios made of exactly k longest outages, written down directly."""
    T, M, L = params.T, params.M, [int(v) for v in params.L_max]
    out = []
    for ks in compositions(k, M):
        per_line = [list(_placements(T, L[m], ks[m])) for m in range(M)]
        for combo in product(*per_line):
            out.append([(m, a, L[m]) for m in range(M) for a in combo[m]])
    return out


def _line_sets(T: int, lo: int, hi: int, j: int) -> Iterator[Tuple[Tuple[int, int], ...]]:
    """All sets of j non-overlapping outages (start 0-based, length) on one line."""
    def rec(start, left):
        if left == 0:
            yield ()
            return
        for a in range(start, T):
            for l in range(lo, hi + 1):
                if a + l <= T:
                    for rest in rec(a + l, left - 1):
                        yield ((a, l),) + rest
    yield from rec(0, j)


def all_scenarios(params: InstanceParameters, k: int) -> Iterator[List[Event]]:
    """Every scenario with at most k outages."""
    T, M = params.T, params.M
    lo, hi = [int(v) for v in params.L_min], [int(v) for v in params.L_max]
    per_line = [[list(_line_sets(T, lo[m], hi[m], j)) for j in range(k + 1)] for m in range(M)]
    for js in product(range(k + 1), repeat=M):
        if sum(js) > k:
            continue
        for combo in product(*[per_line[m][js[m]] for m in range(M)]):
            yield [(m, a, l) for m in range(M) for (a, l) in combo[m]]


def bits_of(events: Sequence[Event], T: int) -> int:
    b = 0
    for m, a, l in events:
        b |= ((1 << l) - 1) << (m * T + a)
    return b


def delta_of(events: Sequence[Event], M: int, T: int) -> np.ndarray:
    return DisruptionScenario(list(events)).to_delta_matrix(M, T)


def pairwise_filter(bit_list: Sequence[int]) -> List[int]:
    """Matrices that no other matrix contains, by comparison (numpy on two words)."""
    uniq = sorted(set(bit_list), key=lambda b: (-bin(b).count('1'), b))
    mask = (1 << 64) - 1
    lo = np.fromiter((b & mask for b in uniq), dtype=np.uint64, count=len(uniq))
    hi = np.fromiter((b >> 64 for b in uniq), dtype=np.uint64, count=len(uniq))
    keep_lo = np.empty(len(uniq), dtype=np.uint64)
    keep_hi = np.empty(len(uniq), dtype=np.uint64)
    kept_idx, n = [], 0
    for i in range(len(uniq)):
        a, b = lo[i], hi[i]
        if n and bool(np.any(((keep_lo[:n] & a) == a) & ((keep_hi[:n] & b) == b))):
            continue
        keep_lo[n], keep_hi[n] = a, b
        n += 1
        kept_idx.append(i)
    return [uniq[i] for i in kept_idx]


# ---------------------------------------------------------------- ccg

def _solve_design(params, scenarios):
    with contextlib.redirect_stdout(io.StringIO()):
        return solve_robust_model(params, scenarios)


def ccg(params: InstanceParameters, oracle_events: Sequence[Sequence[Event]], max_rounds: int = 80,
        keep_costs: bool = False) -> Dict:
    """Start from the no-outage scenario; each round solve the design LP on the
    scenarios collected so far, solve the operating problem of that design in
    every oracle scenario, and add the worst one."""
    M, T = params.M, params.T
    deltas = [delta_of(e, M, T) for e in oracle_events]
    ev = RecourseEvaluator(params)
    active: List[DisruptionScenario] = [DisruptionScenario([])]
    rounds, t_start = [], time.time()
    costs = None
    for r in range(1, max_rounds + 1):
        t0 = time.time()
        sol = _solve_design(params, active)
        t_design = time.time() - t0
        if sol.lp_status != 'optimal':
            raise RuntimeError(f'design LP failed in round {r}: {sol.lp_status}')
        t1 = time.time()
        res = [ev.evaluate(sol.C_depot, sol.q_mode, d) for d in deltas]
        t_check = time.time() - t1
        if any(x.status != 'optimal' for x in res):
            raise RuntimeError(f'operating problem failed in round {r}')
        costs = np.array([x.objective for x in res])
        w = int(np.argmax(costs))
        converged = costs[w] <= sol.worst_case_cost + TOL_DOLLARS
        rounds.append({'round': r, 'scenarios_in_design_lp': len(active), 'design_lp_s': t_design,
                       'check_s': t_check, 'operating_problems': len(deltas),
                       'reported_worst_operating_cost': float(sol.worst_case_cost),
                       'true_worst_operating_cost': float(costs[w]), 'worst_events': [list(e) for e in oracle_events[w]],
                       'converged': bool(converged)})
        if converged:
            break
        active.append(DisruptionScenario(list(oracle_events[w])))
    else:
        raise RuntimeError('ccg did not converge')
    worst = float(costs.max())
    deciding = [i for i in range(len(costs)) if costs[i] >= worst - TOL_DOLLARS]
    unmet = [res[i].unmet_tons for i in deciding]
    out = {
        'rounds': len(rounds), 'operating_problems_total': int(sum(x['operating_problems'] for x in rounds)),
        'design_lp_s_total': float(sum(x['design_lp_s'] for x in rounds)),
        'check_s_total': float(sum(x['check_s'] for x in rounds)), 'wall_s': time.time() - t_start,
        'total_cost': float(sol.total_cost), 'investment_cost': float(sol.investment_cost),
        'worst_operating_cost': worst, 'C': [float(v) for v in sol.C_depot], 'q': [float(v) for v in sol.q_mode],
        'deciding_scenarios': [[list(e) for e in oracle_events[i]] for i in deciding],
        'worst_unmet_tons': float(max(unmet)), 'round_log': rounds,
    }
    if keep_costs:
        out['costs'] = costs
    return out


# ---------------------------------------------------------------- MILP separation

def separation_milp(ev: RecourseEvaluator, C, q, events: Sequence[Event], k: int, exactly_k: bool,
                    time_limit_s: float = 3600.0) -> Dict:
    """Worst scenario for a fixed design as one mixed-integer program: the dual
    of the operating problem, one 0-1 variable per candidate outage in
    `events`, no two chosen outages on the same line and period, and at most
    (or exactly) k chosen. Same formulation as revision_common.worst_case_milp,
    with the candidate outages and the form of the budget as arguments."""
    from scipy import sparse
    from scipy.optimize import Bounds, LinearConstraint, milp
    MILLION = 1e6
    params, tpl = ev.params, ev.template
    M, T = params.M, params.T
    Lam = float(np.max(params.p_unmet)) / MILLION
    n_ub, n_eq = tpl.A_ub.shape[0], tpl.A_eq.shape[0]
    n_w, n_z = M * T, len(events)
    o_ub, o_eq, o_w, o_z = 0, n_ub, n_ub + n_eq, n_ub + n_eq + n_w
    n = o_z + n_z
    rows, cols = [], []
    for e, (m, a, l) in enumerate(events):
        for t in range(a, a + l):
            rows.append(m * T + t); cols.append(e)
    cover = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_w, n_z))
    C = np.asarray(C, dtype=float); q = np.asarray(q, dtype=float)
    obj = np.zeros(n)
    obj[o_ub:o_ub + n_ub] = ev.rhs(C, q, np.zeros((M, T), dtype=int))
    obj[o_eq:o_eq + n_eq] = tpl.b_eq
    for m in range(M):
        obj[o_w + m * T: o_w + (m + 1) * T] = q[m]
    cons = []
    A_dual = sparse.hstack([tpl.A_ub.T, tpl.A_eq.T, sparse.csr_matrix((tpl.A_ub.shape[1], n_w + n_z))]).tocsr()
    cons.append(LinearConstraint(A_dual, -np.inf, tpl.c / MILLION))
    B = sparse.lil_matrix((n_w, n))
    for m in range(M):
        for t in range(T):
            B[m * T + t, o_w + m * T + t] = 1.0
            B[m * T + t, o_ub + tpl.mode_rows[m, t]] = 1.0
    cons.append(LinearConstraint(B.tocsr(), -np.inf, 0.0))
    cons.append(LinearConstraint(sparse.hstack([sparse.csr_matrix((n_w, n_ub + n_eq)), sparse.identity(n_w),
                                                -Lam * cover]).tocsr(), -np.inf, 0.0))
    cons.append(LinearConstraint(sparse.hstack([sparse.csr_matrix((n_w, o_z)), cover]).tocsr(), -np.inf, 1.0))
    budget = sparse.hstack([sparse.csr_matrix((1, o_z)), sparse.csr_matrix(np.ones((1, n_z)))]).tocsr()
    cons.append(LinearConstraint(budget, float(k) if exactly_k else -np.inf, float(k)))
    lb = np.full(n, -np.inf); ub = np.full(n, np.inf)
    ub[o_ub:o_ub + n_ub] = 0.0
    for m in range(M):
        lb[o_ub + tpl.mode_rows[m, :]] = -Lam
    lb[o_w:o_w + n_w] = 0.0; ub[o_w:o_w + n_w] = Lam
    lb[o_z:] = 0.0; ub[o_z:] = 1.0
    integrality = np.zeros(n); integrality[o_z:] = 1
    t0 = time.time()
    res = milp(-obj, constraints=cons, bounds=Bounds(lb, ub), integrality=integrality,
               options={'time_limit': time_limit_s, 'mip_rel_gap': 1e-9})
    elapsed = time.time() - t0
    if res.x is None or int(res.status) != 0:
        return {'status': int(res.status), 'message': str(res.message), 'time_s': elapsed, 'binaries': n_z}
    chosen = [tuple(map(int, events[e])) for e in range(n_z) if res.x[o_z + e] > 0.5]
    primal = ev.evaluate(C, q, delta_of(chosen, M, T))
    return {'status': 0, 'bound': float(-res.fun * MILLION), 'cost': float(primal.objective), 'events': chosen,
            'time_s': elapsed, 'binaries': n_z}


def ccg_milp(params: InstanceParameters, events: Sequence[Event], k: int, exactly_k: bool,
             max_rounds: int = 80, time_limit_s: float = 3600.0) -> Dict:
    """Column-and-constraint generation in which the worst scenario of each
    round is found by separation_milp instead of by checking a list."""
    ev = RecourseEvaluator(params)
    active: List[DisruptionScenario] = [DisruptionScenario([])]
    rounds, t_start = [], time.time()
    for r in range(1, max_rounds + 1):
        t0 = time.time()
        sol = _solve_design(params, active)
        t_design = time.time() - t0
        if sol.lp_status != 'optimal':
            raise RuntimeError(f'design LP failed in round {r}: {sol.lp_status}')
        sep = separation_milp(ev, sol.C_depot, sol.q_mode, events, k, exactly_k, time_limit_s)
        if sep['status'] != 0:
            rounds.append({'round': r, 'design_lp_s': t_design, 'milp_s': sep['time_s'], 'failed': sep['message']})
            return {'finished': False, 'rounds': len(rounds), 'wall_s': time.time() - t_start, 'round_log': rounds,
                    'binaries': sep['binaries']}
        converged = sep['cost'] <= sol.worst_case_cost + TOL_DOLLARS
        rounds.append({'round': r, 'scenarios_in_design_lp': len(active), 'design_lp_s': t_design,
                       'milp_s': sep['time_s'], 'reported_worst_operating_cost': float(sol.worst_case_cost),
                       'milp_bound': sep['bound'], 'true_worst_operating_cost': sep['cost'],
                       'worst_events': [list(e) for e in sep['events']], 'converged': bool(converged)})
        print(f'    round {r}: design LP {t_design:.1f}s, MILP {sep["time_s"]:.1f}s, '
              f'worst {sep["cost"]:.2f} vs reported {sol.worst_case_cost:.2f}', flush=True)
        if converged:
            break
        active.append(DisruptionScenario(list(sep['events'])))
    else:
        raise RuntimeError('ccg did not converge')
    return {'finished': True, 'rounds': len(rounds), 'binaries': sep['binaries'],
            'design_lp_s_total': float(sum(x['design_lp_s'] for x in rounds)),
            'milp_s_total': float(sum(x['milp_s'] for x in rounds)), 'wall_s': time.time() - t_start,
            'total_cost': float(sol.total_cost), 'worst_operating_cost': float(sep['cost']),
            'C': [float(v) for v in sol.C_depot], 'q': [float(v) for v in sol.q_mode], 'round_log': rounds}


def stage_milpsep(out_dir: str, ks: Sequence[int], repeats: int, time_limit_s: float, only: str = '') -> dict:
    out = {'instance': 'lunar example, demand from month 2', 'runs': {}, 'environment': environment_record()}
    for k in ks:
        p = base_instance(k)
        every = [(m, a, l) for m in range(p.M) for l in range(int(p.L_min[m]), int(p.L_max[m]) + 1)
                 for a in range(p.T - l + 1)]
        longest = [(m, a, int(p.L_max[m])) for m in range(p.M) for a in range(p.T - int(p.L_max[m]) + 1)]
        variants = (('all_outages_at_most_k', every, False), ('longest_outages_at_most_k', longest, False),
                    ('longest_outages_exactly_k', longest, True))
        for name, events, exact in variants:
            if only and only not in name:
                continue
            for i in range(repeats):
                print(f'[milpsep] k={k} {name} ({len(events)} binaries) run {i + 1}/{repeats}', flush=True)
                r = ccg_milp(p, events, k, exact, time_limit_s=time_limit_s)
                out['runs'].setdefault(f'k{k}', {}).setdefault(name, []).append(r)
                msg = (f'{r["rounds"]} rounds, design LP {r["design_lp_s_total"]:.1f}s, MILP {r["milp_s_total"]:.1f}s, '
                       f'wall {r["wall_s"]:.1f}s, cost {r["total_cost"]:.2f}') if r['finished'] else 'NOT FINISHED'
                print(f'[milpsep] k={k} {name}: {msg}', flush=True)
                write_json(os.path.join(out_dir, 'milp_separation.json'), out)
    return out



def _strip(d: Dict) -> Dict:
    return {k: v for k, v in d.items() if k not in ('round_log', 'costs')}


# ---------------------------------------------------------------- stages

def stage_month2(out_dir: str) -> dict:
    from revision_common import build_library, ccg_loop
    from run_proano_revision import FROZEN
    params = generate_small_instance()
    startup_penalty = float((np.asarray(params.D_demand)[:, :, 0] * np.asarray(params.p_unmet)).sum())
    lib = build_library(base_instance(2))
    with contextlib.redirect_stdout(io.StringIO()):
        res = ccg_loop(lib, [0], lib.maximal_ids, max_iterations=60, log=lambda *a, **k: None)
    out = {'change': 'D_demand[:, :, 0] = 0 (no demand in month 1); everything else as generate_small_instance()',
           'startup_penalty_removed_dollars': startup_penalty, 'total_cost': res['total_cost'],
           'recorded_optimum_with_month_1_demand': FROZEN['exact_total_cost'],
           'difference_from_recorded_minus_constant': res['total_cost'] - (FROZEN['exact_total_cost'] - startup_penalty),
           'environment': environment_record()}
    write_json(os.path.join(out_dir, 'month2_summary.json'), out)
    return out


def stage_ccgstd(out_dir: str, repeats: int, repeats_all: int) -> dict:
    p = base_instance(2)
    kept = kept_scenarios(p, 2)
    everything = list(all_scenarios(p, 2))
    assert len(kept) == count_kept(p.T, p.L_max, 2) and len(everything) == count_scenarios(p.T, p.L_min, p.L_max, 2)
    runs = {'kept': [], 'all': []}
    for name, oracle, n in (('kept', kept, repeats), ('all', everything, repeats_all)):
        for i in range(n):
            r = ccg(p, oracle)
            runs[name].append(r)
            print(f'[ccgstd] {name} run {i + 1}/{n}: {r["rounds"]} rounds, {r["operating_problems_total"]} '
                  f'operating problems, {r["wall_s"]:.1f}s, cost {r["total_cost"]:.2f}', flush=True)
            write_json(os.path.join(out_dir, 'ccg_standard_partial.json'),
                       {k: [_strip(x) for x in v] for k, v in runs.items()})
    def med(name, key):
        return float(np.median([x[key] for x in runs[name]])) if runs[name] else None
    out = {
        'instance': 'lunar example, demand from month 2, at most 2 outages per scenario',
        'scenarios': len(everything), 'kept': len(kept),
        'summary': {name: {'runs': len(runs[name]), 'rounds': runs[name][0]['rounds'] if runs[name] else None,
                           'operating_problems_total': runs[name][0]['operating_problems_total'] if runs[name] else None,
                           'median_wall_s': med(name, 'wall_s'), 'median_design_lp_s': med(name, 'design_lp_s_total'),
                           'median_check_s': med(name, 'check_s_total'),
                           'wall_s_all_runs': [x['wall_s'] for x in runs[name]],
                           'total_cost': runs[name][0]['total_cost'] if runs[name] else None,
                           'q': runs[name][0]['q'] if runs[name] else None} for name in runs},
        'same_design': bool(runs['kept'] and runs['all'] and
                            abs(runs['kept'][0]['total_cost'] - runs['all'][0]['total_cost']) <= TOL_DOLLARS and
                            np.allclose(runs['kept'][0]['q'], runs['all'][0]['q'], atol=1e-6)),
        'first_run_logs': {name: runs[name][0]['round_log'] if runs[name] else None for name in runs},
        'environment': environment_record(),
    }
    write_json(os.path.join(out_dir, 'ccg_standard.json'), out)
    return out


def stage_kscale(out_dir: str, max_list_k: int, solve_k3: bool) -> dict:
    rows = []
    for k in (1, 2, 3, 4):
        p = base_instance(k)
        T, L = p.T, [int(v) for v in p.L_max]
        n_all, n_kept = count_scenarios(T, p.L_min, p.L_max, k), count_kept(T, L, k)
        t0 = time.time()
        kept = kept_scenarios(p, k)
        kept_bits = {bits_of(e, T) for e in kept}
        t_direct = time.time() - t0
        assert len(kept) == n_kept == len(kept_bits)
        row = {'k': k, 'scenarios': n_all, 'kept': n_kept, 'kept_share_pct': 100.0 * n_kept / n_all,
               'write_down_kept_s': t_direct, 'T_at_least_k_times_longest': bool(all(T >= k * l for l in L))}
        if k <= max_list_k:
            t1 = time.time()
            bit_list = [bits_of(e, T) for e in all_scenarios(p, k)]
            t_list = time.time() - t1
            assert len(bit_list) == n_all
            t2 = time.time()
            filtered = pairwise_filter(bit_list)
            t_filter = time.time() - t2
            row.update({'list_all_s': t_list, 'distinct_matrices': len(set(bit_list)),
                        'pairwise_filter_s': t_filter, 'filter_equals_direct': set(filtered) == kept_bits})
            assert row['filter_equals_direct']
        rows.append(row)
        print('[kscale]', {a: (round(b, 3) if isinstance(b, float) else b) for a, b in row.items()}, flush=True)
        write_json(os.path.join(out_dir, 'k_scale_partial.json'), {'rows': rows})
    out = {'instance': 'lunar example (T = 24, longest outages 6, 6, 2), demand from month 2', 'rows': rows}
    if solve_k3:
        p3 = base_instance(3)
        r = ccg(p3, kept_scenarios(p3, 3))
        out['k3_optimum_by_ccg_over_kept'] = _strip(r)
        out['k3_round_log'] = r['round_log']
        print(f'[kscale] k=3 optimum: {r["rounds"]} rounds, {r["wall_s"]:.0f}s, cost {r["total_cost"]:.2f}, q {r["q"]}', flush=True)
    out['environment'] = environment_record()
    write_json(os.path.join(out_dir, 'k_scale.json'), out)
    return out


def stage_family(out_dir: str, instances: int, all_oracle: int) -> dict:
    rows = []
    for seed in range(instances):
        p, desc = random_instance(seed)
        T, L = p.T, [int(v) for v in p.L_max]
        kept = kept_scenarios(p, 2)
        r = ccg(p, kept)
        starts = [min(a for (_, a, _) in sc) + 1 for sc in r['deciding_scenarios'] if sc]
        row = dict(desc)
        row.update({'scenarios': count_scenarios(T, p.L_min, p.L_max, 2), 'kept': len(kept),
                    'rounds': r['rounds'], 'operating_problems_kept': r['operating_problems_total'],
                    'operating_problems_if_all': r['rounds'] * count_scenarios(T, p.L_min, p.L_max, 2),
                    'wall_s_kept': r['wall_s'], 'total_cost': r['total_cost'], 'investment_cost': r['investment_cost'],
                    'q': r['q'], 'C': r['C'], 'worst_unmet_tons': r['worst_unmet_tons'],
                    'worst_unmet_share_pct': 100.0 * r['worst_unmet_tons'] / desc['total_demand_tons'],
                    'deciding_scenarios': r['deciding_scenarios'], 'n_deciding': len(r['deciding_scenarios']),
                    'earliest_start_period_of_deciding': starts,
                    'a_deciding_scenario_starts_in_period_1': bool(any(s == 1 for s in starts))})
        if seed < all_oracle:
            ra = ccg(p, list(all_scenarios(p, 2)))
            row.update({'wall_s_all': ra['wall_s'], 'rounds_all': ra['rounds'],
                        'operating_problems_all_measured': ra['operating_problems_total'],
                        'same_cost_with_all_scenarios': bool(abs(ra['total_cost'] - r['total_cost']) <= TOL_DOLLARS)})
        rows.append(row)
        print(f'[family] seed {seed}: L={L} kept={len(kept)} rounds={r["rounds"]} deciding={len(r["deciding_scenarios"])} '
              f'starts={starts} unmet={r["worst_unmet_tons"]:.1f}t wall={r["wall_s"]:.1f}s'
              + (f' | all: {row["wall_s_all"]:.1f}s same={row["same_cost_with_all_scenarios"]}' if seed < all_oracle else ''),
              flush=True)
        write_json(os.path.join(out_dir, 'family_partial.json'), {'rows': rows})
    out = {'description': 'random examples around the lunar one; see random_instance()', 'rows': rows,
           'environment': environment_record()}
    write_json(os.path.join(out_dir, 'family.json'), out)
    return out


def stage_k3verify(out_dir: str) -> dict:
    """Operate the k = 3 optimal design (found over the kept scenarios) in every
    one of the 2,076,358 scenarios and compare the worst cost."""
    src = json.load(open(os.path.join(PROJECT_ROOT, 'results', '29_k_scale', 'k_scale.json')))['k3_optimum_by_ccg_over_kept']
    p = base_instance(3)
    ev = RecourseEvaluator(p)
    C, q = src['C'], src['q']
    worst, n, above, t0 = -1.0, 0, 0, time.time()
    worst_events = None
    for events in all_scenarios(p, 3):
        r = ev.evaluate(C, q, delta_of(events, p.M, p.T))
        if r.status != 'optimal':
            raise RuntimeError('operating problem failed')
        n += 1
        if r.objective > worst:
            worst, worst_events = r.objective, [list(e) for e in events]
        if r.objective > src['worst_operating_cost'] + TOL_DOLLARS:
            above += 1
        if n % 100000 == 0:
            print(f'[k3verify] {n:,} scenarios, worst so far {worst:.2f}, {time.time() - t0:.0f}s', flush=True)
    out = {'scenarios_checked': n, 'worst_operating_cost_all_scenarios': worst,
           'worst_operating_cost_kept_scenarios': src['worst_operating_cost'],
           'difference': worst - src['worst_operating_cost'], 'scenarios_above_kept_worst': above,
           'worst_events': worst_events, 'wall_s': time.time() - t0, 'environment': environment_record()}
    write_json(os.path.join(out_dir, 'k3_verify.json'), out)
    print(json.dumps({k: v for k, v in out.items() if k != 'environment'}, indent=1))
    return out


STAGES = {'milpsep': '32_milp_separation', 'month2': '27_demand_from_month_2', 'ccgstd': '28_ccg_standard', 'kscale': '29_k_scale',
          'family': '30_instance_family', 'k3verify': '31_k3_all_scenarios'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=list(STAGES))
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--repeats-all', type=int, default=3)
    ap.add_argument('--max-list-k', type=int, default=3, help='kscale: list and filter the whole scenario set up to this k')
    ap.add_argument('--no-k3-solve', action='store_true')
    ap.add_argument('--instances', type=int, default=30)
    ap.add_argument('--all-oracle', type=int, default=5, help='family: also run the check over all scenarios on the first N')
    ap.add_argument('--ks', default='2,3,4', help='milpsep: numbers of outages')
    ap.add_argument('--time-limit', type=float, default=3600.0, help='milpsep: seconds per MILP')
    ap.add_argument('--only', default='', help='milpsep: run only the variants whose name contains this')
    ap.add_argument('--out-suffix', default='', help='smoke tests: append to the results directory name')
    args = ap.parse_args()
    out_dir = os.path.join(PROJECT_ROOT, 'results', STAGES[args.stage] + args.out_suffix)
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()
    if args.stage == 'milpsep':
        stage_milpsep(out_dir, [int(v) for v in args.ks.split(',')], args.repeats, args.time_limit, args.only)
    elif args.stage == 'month2':
        out = stage_month2(out_dir)
        print(json.dumps({k: v for k, v in out.items() if k != 'environment'}, indent=1, default=str))
    elif args.stage == 'ccgstd':
        out = stage_ccgstd(out_dir, args.repeats, args.repeats_all)
        print(json.dumps(out['summary'], indent=1, default=str))
    elif args.stage == 'kscale':
        stage_kscale(out_dir, args.max_list_k, not args.no_k3_solve)
    elif args.stage == 'family':
        stage_family(out_dir, args.instances, args.all_oracle)
    elif args.stage == 'k3verify':
        stage_k3verify(out_dir)
    print(f'done in {time.time() - started:.0f}s -> {out_dir}')


if __name__ == '__main__':
    main()
