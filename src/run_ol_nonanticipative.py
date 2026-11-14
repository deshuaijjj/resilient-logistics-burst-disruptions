"""Design for an operator who learns of outages as they happen (2026-11-11).

The recorded model (lp_solver.build_lp_model_sparse) lets the operator choose
the whole operating plan of a scenario at once. Here the operator sees at the
start of month t which lines are out in month t and then chooses the launches
and deliveries of month t; two scenarios that agree up to month t get the same
decisions up to month t. For a finite scenario set this is one linear program
on the tree of scenario prefixes. Constraints and costs are those of the
recorded model, written per tree node.

Scenario set used here (definition of 2026-11-11): at most k outages; an
outage of line m starts in any month a <= T and lasts l months,
L_min[m] <= l <= L_max[m]; its part after month T is outside the model.

  splus   exactly k longest outages starting in any month (sufficient for the
          operator who learns as he goes; checks/check_nonanticipative_theorem.py)
  thm1    exactly k longest outages that end by month T (Theorem 1; sufficient
          for the operator who knows the scenario in advance)
  all     every scenario

  python3 src/run_ol_nonanticipative.py --stage validate
  python3 src/run_ol_nonanticipative.py --stage lunar --k 2
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

import ol_scenarios  # noqa: E402
from instance_generator import DisruptionScenario, InstanceParameters  # noqa: E402
from lp_solver import solve_robust_model  # noqa: E402
from revision_common import environment_record, write_json  # noqa: E402
from run_ol_revision import base_instance  # noqa: E402


# ---------------------------------------------------------------- scenario sets

def scenario_bits(p: InstanceParameters, k: int, kind: str) -> List[int]:
    """Distinct outage matrices as bit masks (bit m*T+t set: line m out in month t);
    the three sets are defined in ol_scenarios."""
    return ol_scenarios.scenario_masks(p.T, [int(v) for v in p.L_min], [int(v) for v in p.L_max], k, kind)


def delta_from_bits(b: int, M: int, T: int) -> np.ndarray:
    return np.array([[(b >> (m * T + t)) & 1 for t in range(T)] for m in range(M)], dtype=int)


# ---------------------------------------------------------------- tree LP

def build_tree(bits: Sequence[int], M: int, T: int, share: bool = True):
    """Nodes are (month, prefix of the outage matrix up to that month). With
    share=False every scenario keeps its own nodes (operator knows the scenario)."""
    upto = []
    acc = 0
    for t in range(T):
        acc |= sum(1 << (m * T + t) for m in range(M))
        upto.append(acc)
    index: Dict = {}
    node_t: List[int] = []
    node_parent: List[int] = []
    node_out: List[int] = []           # M-bit mask of lines out in the node's month
    paths = np.zeros((len(bits), T), dtype=np.int64)
    for s, b in enumerate(bits):
        parent = -1
        for t in range(T):
            key = (t, b & upto[t]) if share else (t, s)
            n = index.get(key)
            if n is None:
                n = len(node_t)
                index[key] = n
                node_t.append(t); node_parent.append(parent)
                node_out.append(sum(((b >> (m * T + t)) & 1) << m for m in range(M)))
            paths[s, t] = n
            parent = n
    return np.array(node_t), np.array(node_parent), np.array(node_out), paths


def solve_tree_lp(p: InstanceParameters, bits: Sequence[int], share: bool = True,
                  fixed_C: Optional[Sequence[float]] = None, fixed_q: Optional[Sequence[float]] = None,
                  time_limit_s: float = 7200.0, method: str = 'highs') -> Dict:
    M, J, D, K, T = p.M, p.J, p.D, p.K, p.T
    t0 = time.time()
    node_t, node_parent, node_out, paths = build_tree(bits, M, T, share)
    N = len(node_t)
    has_child = np.zeros(N, dtype=bool)
    has_child[node_parent[node_parent >= 0]] = True
    tau_y = np.ceil(np.asarray(p.tau_mode_depot)).astype(int)       # (M, J)
    tau_x = np.ceil(np.asarray(p.tau_depot_dest)).astype(int)       # (J, D)

    def ancestor(n: int, steps: int) -> int:
        while steps > 0 and n >= 0:
            n = int(node_parent[n]); steps -= 1
        return n

    n_y, n_s, n_x, n_u = M * J * K, J * K, J * D * K, D * K
    per = n_y + n_s + n_x + n_u + 1
    base = J + M + 1
    iC = lambda j: j
    iq = lambda m: J + m
    ETA = J + M
    iy = lambda n, m, j, k: base + n * per + (m * J + j) * K + k
    is_ = lambda n, j, k: base + n * per + n_y + j * K + k
    ix = lambda n, j, d, k: base + n * per + n_y + n_s + (j * D + d) * K + k
    iu = lambda n, d, k: base + n * per + n_y + n_s + n_x + d * K + k
    iW = lambda n: base + n * per + n_y + n_s + n_x + n_u
    n_vars = base + N * per

    ur, uc, uv, ub = [], [], [], []
    er, ec, ev, eb = [], [], [], []

    def add_ub(entries, rhs):
        r = len(ub)
        for col, val in entries:
            ur.append(r); uc.append(col); uv.append(val)
        ub.append(rhs)

    def add_eq(entries, rhs):
        r = len(eb)
        for col, val in entries:
            er.append(r); ec.append(col); ev.append(val)
        eb.append(rhs)

    for j in range(J):
        add_ub([(iC(j), -1.0)], -float(np.sum(p.s_initial[j, :])))
    c_mode, h, pen = np.asarray(p.c_mode, float), np.asarray(p.h_depot, float), np.asarray(p.p_unmet, float)
    for n in range(N):
        t = int(node_t[n]); par = int(node_parent[n])
        # inventory balance
        for j in range(J):
            for k in range(K):
                ent = [(is_(n, j, k), 1.0)]
                if par >= 0:
                    ent.append((is_(par, j, k), -1.0))
                for m in range(M):
                    a = ancestor(n, int(tau_y[m, j]))
                    if a >= 0:
                        ent.append((iy(a, m, j, k), -1.0))
                for d in range(D):
                    ent.append((ix(n, j, d, k), 1.0))
                add_eq(ent, float(p.s_initial[j, k]) if t == 0 else 0.0)
        # storage capacity
        for j in range(J):
            add_ub([(iC(j), -1.0)] + [(is_(n, j, k), 1.0) for k in range(K)], 0.0)
        # line capacity
        for m in range(M):
            up = 0.0 if (int(node_out[n]) >> m) & 1 else 1.0
            add_ub([(iq(m), -up)] + [(iy(n, m, j, k), 1.0) for j in range(J) for k in range(K)], 0.0)
        # reception limit and demand
        for d in range(D):
            arrive = [(j, ancestor(n, int(tau_x[j, d]))) for j in range(J)]
            ent = [(ix(a, j, d, k), 1.0) for j, a in arrive if a >= 0 for k in range(K)]
            if ent:
                add_ub(ent, float(p.R_max[d, t]))
            for k in range(K):
                add_eq([(ix(a, j, d, k), 1.0) for j, a in arrive if a >= 0] + [(iu(n, d, k), 1.0)],
                       float(p.D_demand[d, k, t]))
        # cost from this node on: W_n >= cost of the month + W_child
        stage = ([(iy(n, m, j, k), c_mode[m]) for m in range(M) for j in range(J) for k in range(K)]
                 + [(is_(n, j, k), h[j, k]) for j in range(J) for k in range(K)]
                 + [(iu(n, d, k), pen[d, k]) for d in range(D) for k in range(K)])
        if not has_child[n]:
            add_ub(stage + [(iW(n), -1.0)], 0.0)
        if par < 0:
            add_ub([(iW(n), 1.0), (ETA, -1.0)], 0.0)
    # for every child: cost of the parent's month + W_child <= W_parent
    stage_cache: Dict[int, list] = {}
    for n in range(N):
        par = int(node_parent[n])
        if par < 0:
            continue
        if par not in stage_cache:
            stage_cache[par] = ([(iy(par, m, j, k), c_mode[m]) for m in range(M) for j in range(J) for k in range(K)]
                                + [(is_(par, j, k), h[j, k]) for j in range(J) for k in range(K)]
                                + [(iu(par, d, k), pen[d, k]) for d in range(D) for k in range(K)])
        add_ub(stage_cache[par] + [(iW(n), 1.0), (iW(par), -1.0)], 0.0)

    A_ub = sparse.csr_matrix((uv, (ur, uc)), shape=(len(ub), n_vars))
    A_eq = sparse.csr_matrix((ev, (er, ec)), shape=(len(eb), n_vars))
    cost = np.zeros(n_vars)
    cost[:J] = np.asarray(p.f_depot, float)
    cost[J:J + M] = np.asarray(p.v_mode, float) * T
    cost[ETA] = 1.0
    bounds = [(0, None)] * n_vars
    if fixed_C is not None:
        for j in range(J):
            bounds[j] = (float(fixed_C[j]), float(fixed_C[j]))
    if fixed_q is not None:
        for m in range(M):
            bounds[J + m] = (float(fixed_q[m]), float(fixed_q[m]))
    t_build = time.time() - t0
    t1 = time.time()
    res = linprog(cost, A_ub=A_ub, b_ub=np.array(ub), A_eq=A_eq, b_eq=np.array(eb), bounds=bounds,
                  method=method, options={'time_limit': time_limit_s})
    t_solve = time.time() - t1
    out = {'scenarios': len(bits), 'nodes': int(N), 'variables': int(n_vars), 'rows': int(len(ub) + len(eb)),
           'nonzeros': int(A_ub.nnz + A_eq.nnz), 'build_s': t_build, 'solve_s': t_solve, 'status': int(res.status),
           'message': str(res.message), 'iterations': int(getattr(res, 'nit', -1) if getattr(res, 'nit', None) is not None else -1)}
    if res.status == 0:
        x = res.x
        C, q = x[:J], x[J:J + M]
        out.update({'total_cost': float(res.fun), 'worst_operating_cost': float(x[ETA]),
                    'investment_cost': float(res.fun - x[ETA]), 'C': [float(v) for v in C], 'q': [float(v) for v in q]})
        # unmet demand along the worst path
        unmet = np.zeros(len(bits))
        ucols = np.array([[iu(n, d, k) for d in range(D) for k in range(K)] for n in range(N)])
        node_unmet = x[ucols].sum(axis=1)
        unmet = node_unmet[paths].sum(axis=1)
        out['largest_unmet_tons_on_a_path'] = float(unmet.max())
    return out


# ---------------------------------------------------------------- stages

def stage_validate(out_dir: str) -> dict:
    """The tree LP without shared nodes must reproduce the recorded design LP."""
    p = base_instance(2)
    bits = scenario_bits(p, 2, 'thm1')
    rng = np.random.default_rng(7)
    pick = [bits[i] for i in sorted(rng.choice(len(bits), 40, replace=False))]
    scen = []
    for b in pick:
        d = delta_from_bits(b, p.M, p.T)
        ev = []
        for m in range(p.M):
            t = 0
            while t < p.T:
                if d[m, t]:
                    a = t
                    while t < p.T and d[m, t]:
                        t += 1
                    ev.append((m, a, t - a))
                else:
                    t += 1
        scen.append(DisruptionScenario(ev))
    with contextlib.redirect_stdout(io.StringIO()):
        ref = solve_robust_model(p, scen)
    own = solve_tree_lp(p, pick, share=False)
    na = solve_tree_lp(p, pick, share=True)
    out = {'scenarios': len(pick), 'recorded_model_total_cost': float(ref.total_cost),
           'tree_lp_without_sharing_total_cost': own['total_cost'],
           'difference': own['total_cost'] - float(ref.total_cost),
           'tree_lp_with_sharing_total_cost': na['total_cost'],
           'sharing_minus_no_sharing': na['total_cost'] - own['total_cost'],
           'counts': {kind: len(scenario_bits(p, 2, kind)) for kind in ('thm1', 'splus', 'all')}}
    write_json(os.path.join(out_dir, 'validate.json'), out)
    print(json.dumps(out, indent=1))
    return out


def stage_lunar(out_dir: str, k: int, with_all: bool, time_limit_s: float, method: str = 'highs',
                only_all: bool = False, repeats: int = 1, only_splus: bool = False) -> dict:
    p = base_instance(k)
    tag = ('_all' if only_all else '_splus' if only_splus else '') + ('' if method == 'highs' else '_' + method.replace('highs-', ''))
    out = {'instance': f'lunar example, demand from month 2, at most {k} outages, outages may run past month T',
           'environment': environment_record(), 'runs': {}}
    kinds = ('splus',) if only_splus else ('thm1', 'splus', 'all') if with_all else ('thm1', 'splus')
    sets = {kind: scenario_bits(p, k, kind) for kind in kinds}
    out['counts'] = {kind: len(v) for kind, v in sets.items()}
    print('[lunar] counts', out['counts'], flush=True)

    def run(name, bits, share, **kw):
        r = solve_tree_lp(p, bits, share=share, time_limit_s=time_limit_s, method=method, **kw)
        r['method'] = method
        r['solve_s_all_runs'] = [r['solve_s']]
        for _ in range(repeats - 1 if r['status'] == 0 else 0):     # repeat only for the timing
            again = solve_tree_lp(p, bits, share=share, time_limit_s=time_limit_s, method=method, **kw)
            assert abs(again['total_cost'] - r['total_cost']) <= 1e-6 * max(1.0, abs(r['total_cost']))
            r['solve_s_all_runs'].append(again['solve_s'])
        r['median_solve_s'] = float(np.median(r['solve_s_all_runs']))
        out['runs'][name] = r
        msg = (f"cost {r['total_cost']:.2f}, operating {r['worst_operating_cost']:.2f}, q {np.round(r['q'], 2).tolist()}, "
               f"C {np.round(r['C'], 1).tolist()}, unmet {r['largest_unmet_tons_on_a_path']:.2f}") if r['status'] == 0 else r['message']
        print(f"[lunar] {name}: {r['scenarios']} scenarios, {r['nodes']} nodes, build {r['build_s']:.1f}s, "
              f"solve {r['solve_s']:.1f}s; {msg}", flush=True)
        write_json(os.path.join(out_dir, f'lunar_k{k}{tag}.json'), out)
        return r

    if only_splus:
        run('learns_as_outages_happen__splus_set', sets['splus'], True)
        return out
    if only_all:
        run('learns_as_outages_happen__splus_set', sets['splus'], True)
        run('learns_as_outages_happen__all_scenarios', sets['all'], True)
        return out
    a = run('knows_scenario__thm1_set', sets['thm1'], False)
    b = run('learns_as_outages_happen__splus_set', sets['splus'], True)
    run('learns_as_outages_happen__thm1_set', sets['thm1'], True)
    run('design_for_knowing__operated_learning__splus_set', sets['splus'], True, fixed_C=a['C'], fixed_q=a['q'])
    run('design_for_learning__operated_knowing__thm1_set', sets['thm1'], False, fixed_C=b['C'], fixed_q=b['q'])
    if with_all:
        run('learns_as_outages_happen__all_scenarios', sets['all'], True)
    return out


def stage_family(out_dir: str, instances: int, time_limit_s: float, select: bool = True) -> dict:
    """The random examples of run_ol_revision (random_instance, select_random_instances),
    each solved for the operator who knows the scenario and for the one who
    learns as outages happen."""
    from run_ol_revision import random_instance, select_random_instances
    rows = []
    for seed in select_random_instances(instances, select)[0]:
        p, desc = random_instance(seed)
        thm1, splus = scenario_bits(p, 2, 'thm1'), scenario_bits(p, 2, 'splus')
        a = solve_tree_lp(p, thm1, share=False, time_limit_s=time_limit_s)
        b = solve_tree_lp(p, splus, share=True, time_limit_s=time_limit_s)
        c = solve_tree_lp(p, thm1, share=True, time_limit_s=time_limit_s)
        d = solve_tree_lp(p, splus, share=True, time_limit_s=time_limit_s, fixed_C=a['C'], fixed_q=a['q'])
        row = {'seed': seed, 'L_max': desc['L_max'], 'thm1_set': len(thm1), 'splus_set': len(splus),
               'knows_total_cost': a['total_cost'], 'learns_total_cost': b['total_cost'],
               'learns_total_cost_on_thm1_set': c['total_cost'],
               'design_for_knowing_operated_learning_total_cost': d['total_cost'],
               'learns_minus_knows': b['total_cost'] - a['total_cost'],
               'thm1_set_understates_by': b['total_cost'] - c['total_cost'],
               'knows_q': a['q'], 'learns_q': b['q'], 'knows_C': a['C'], 'learns_C': b['C'],
               'solve_s': {'knows': a['solve_s'], 'learns': b['solve_s'], 'learns_thm1': c['solve_s']},
               'nodes': {'knows': a['nodes'], 'learns': b['nodes']}}
        rows.append(row)
        print(f"[family] seed {seed}: knows {a['total_cost']:.2f}, learns {b['total_cost']:.2f} "
              f"(+{row['learns_minus_knows']:.2f}), on Theorem 1 set {c['total_cost']:.2f} "
              f"(short by {row['thm1_set_understates_by']:.2f}), knowing-design operated learning {d['total_cost']:.2f}", flush=True)
        write_json(os.path.join(out_dir, 'family.json'), {'rows': rows, 'environment': environment_record()})
    return {'rows': rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['validate', 'lunar', 'family'])
    ap.add_argument('--instances', type=int, default=30)
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--with-all', action='store_true', help='lunar: also solve the tree over every scenario')
    ap.add_argument('--time-limit', type=float, default=7200.0)
    ap.add_argument('--method', default='highs', choices=['highs', 'highs-ds', 'highs-ipm'])
    ap.add_argument('--only-all', action='store_true', help='lunar: only the two runs that compare S+ with every scenario')
    ap.add_argument('--out-suffix', default='')
    args = ap.parse_args()
    out_dir = os.path.join(PROJECT_ROOT, 'results', '33_learns_as_he_goes' + args.out_suffix)
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()
    if args.stage == 'validate':
        stage_validate(out_dir)
    elif args.stage == 'family':
        stage_family(out_dir, args.instances, args.time_limit)
    else:
        stage_lunar(out_dir, args.k, args.with_all or args.only_all, args.time_limit, args.method, args.only_all)
    print(f'done in {time.time() - started:.0f}s -> {out_dir}')


if __name__ == '__main__':
    main()
