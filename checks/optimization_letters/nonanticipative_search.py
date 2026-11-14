"""Does Theorem 1 survive when the operator learns of outages as they happen?

For a fixed design, compare the best worst-case operating cost of a
non-anticipative operator over
    U        all scenarios with at most k outages (lengths lmin..L on each line),
    exact    the scenarios made of exactly k longest outages (U* of Theorem 1),
    atmost   the scenarios made of at most k longest outages,
each in two versions: an outage must end by period T, or it may start in any
period and is cut off at period T.
The operator sees, at the start of period t, which lines are out in period t,
and decides the shipments of period t; decisions of two scenarios that agree
up to period t are the same. This is solved exactly as one linear program on
the scenario tree. U contains U*, so V(U) >= V(U*); a strict inequality is a
counterexample to the theorem for such an operator.

The small model: M lines with capacity q[m] per period (lost when the line is
out), K commodities kept in stock at the destination, shipping cost c[m][k],
holding cost h[k], penalty p[k] for unmet demand d[k][t]. An outage enters
only through the capacity right-hand sides, as in (A1)-(A3).

Run:  python3 checks/nonanticipative_search.py [--trials N]
"""
import argparse
import itertools
import json
import os

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

HERE = os.path.dirname(os.path.abspath(__file__))


def scenario_sets(M, T, lmin, L, k, truncate=False):
    """Distinct outage matrices (as tuples of tuples).

    Returns (U, exact, atmost): all scenarios with at most k outages; those
    made of exactly k longest outages; those made of at most k longest
    outages. With truncate=True an outage may start in any period and its
    part beyond period T is cut off; otherwise it must end by period T.
    """
    def starts(l):
        return range(T) if truncate else range(T - l + 1)
    outs = [(m, a, l) for m in range(M) for l in range(lmin[m], L[m] + 1) for a in starts(l)]

    def matrix(combo):
        d = [[0] * T for _ in range(M)]
        for m, a, l in combo:
            for t in range(a, min(a + l, T)):
                d[m][t] = 1
        return tuple(tuple(r) for r in d)

    def ok(combo):
        for (m1, a1, l1), (m2, a2, l2) in itertools.combinations(combo, 2):
            if m1 == m2 and not (a1 + l1 <= a2 or a2 + l2 <= a1):
                return False
        return True

    U, exact, atmost = {matrix(())}, set(), {matrix(())}
    for j in range(1, k + 1):
        for combo in itertools.combinations(outs, j):
            if not ok(combo):
                continue
            U.add(matrix(combo))
            if all(l == L[m] for m, _, l in combo):
                atmost.add(matrix(combo))
                if j == k:
                    exact.add(matrix(combo))
    return sorted(U), sorted(exact), sorted(atmost)


def solve_tree(scen, inst, clairvoyant=False):
    """Best worst-case operating cost over `scen` for an operator who knows the
    outage pattern up to the current period (or the whole pattern)."""
    M, K, T = inst['M'], inst['K'], inst['T']
    q, c, h, p, d = inst['q'], inst['c'], inst['h'], inst['p'], inst['d']
    index = {}

    def var(key):
        if key not in index:
            index[key] = len(index)
        return index[key]

    def hist(s, t):
        return (s, t) if clairvoyant else tuple(tuple(row[:t + 1]) for row in scen[s])

    eta = var(('eta',))
    rows, cols, vals, rhs = [], [], [], []      # A_ub x <= b_ub
    erows, ecols, evals, erhs = [], [], [], []  # A_eq x  = b_eq
    nub = neq = 0
    seen_cap, seen_bal = set(), set()
    for s in range(len(scen)):
        for t in range(T):
            H = hist(s, t)
            for m in range(M):
                if (H, m) in seen_cap:
                    continue
                seen_cap.add((H, m))
                for kk in range(K):
                    rows.append(nub); cols.append(var(('y', H, m, kk))); vals.append(1.0)
                rhs.append(q[m] * (1 - scen[s][m][t])); nub += 1
            for kk in range(K):
                if (H, kk) in seen_bal:
                    continue
                seen_bal.add((H, kk))
                # I_t - I_{t-1} - sum_m y + (-u) = -d   (I: stock, u: unmet)
                erows.append(neq); ecols.append(var(('I', H, kk))); evals.append(1.0)
                if t > 0:
                    erows.append(neq); ecols.append(var(('I', hist(s, t - 1), kk))); evals.append(-1.0)
                for m in range(M):
                    erows.append(neq); ecols.append(var(('y', H, m, kk))); evals.append(-1.0)
                erows.append(neq); ecols.append(var(('u', H, kk))); evals.append(-1.0)
                erhs.append(-d[kk][t]); neq += 1
        # cost of scenario s <= eta
        for t in range(T):
            H = hist(s, t)
            for kk in range(K):
                for m in range(M):
                    rows.append(nub); cols.append(var(('y', H, m, kk))); vals.append(c[m][kk])
                rows.append(nub); cols.append(var(('I', H, kk))); vals.append(h[kk])
                rows.append(nub); cols.append(var(('u', H, kk))); vals.append(p[kk])
        rows.append(nub); cols.append(eta); vals.append(-1.0)
        rhs.append(0.0); nub += 1
    n = len(index)
    A_ub = coo_matrix((vals, (rows, cols)), shape=(nub, n)).tocsr()
    A_eq = coo_matrix((evals, (erows, ecols)), shape=(neq, n)).tocsr()
    obj = np.zeros(n); obj[eta] = 1.0
    res = linprog(obj, A_ub=A_ub, b_ub=rhs, A_eq=A_eq, b_eq=erhs, bounds=[(0, None)] * n, method='highs')
    if not res.success:
        raise RuntimeError(res.message)
    return float(res.fun)


def random_instance(rng, M, K, T, dist=0):
    """dist 0: uniform data. dist 1: demand late in the horizon, costly
    holding and moderate penalties, which makes early commitment matter."""
    if dist == 0:
        return {'M': M, 'K': K, 'T': T,
                'q': [int(v) for v in rng.integers(1, 5, M)],
                'c': [[int(v) for v in rng.integers(0, 6, K)] for _ in range(M)],
                'h': [int(v) for v in rng.integers(0, 4, K)],
                'p': [int(v) for v in rng.integers(8, 30, K)],
                'd': [[int(v) for v in rng.integers(0, 6, T)] for _ in range(K)]}
    d = [[int(v) if t >= T // 2 else 0 for t, v in enumerate(rng.integers(0, 6, T))] for _ in range(K)]
    return {'M': M, 'K': K, 'T': T,
            'q': [int(v) for v in rng.integers(1, 4, M)],
            'c': [[int(v) for v in rng.integers(0, 8, K)] for _ in range(M)],
            'h': [int(v) for v in rng.integers(1, 6, K)],
            'p': [int(v) for v in rng.integers(4, 16, K)],
            'd': d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dist', type=int, default=0)
    ap.add_argument('--big', action='store_true', help='add configurations with more outages than lines')
    ap.add_argument('--out', default='nonanticipative_search.json')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    configs = [  # (M, K, T, lmin, L, k)
        (1, 1, 4, (1,), (2,), 1), (1, 1, 4, (1,), (2,), 2), (1, 2, 5, (1,), (2,), 2),
        (2, 1, 3, (1, 1), (2, 2), 1), (2, 2, 3, (1, 1), (2, 2), 1), (2, 2, 4, (1, 1), (2, 2), 1),
        (2, 2, 4, (1, 1), (2, 2), 2), (2, 2, 4, (1, 1), (3, 2), 1), (3, 2, 3, (1, 1, 1), (2, 2, 2), 1),
    ]
    if args.big:
        configs += [(2, 1, 6, (1, 1), (2, 2), 3), (2, 2, 5, (1, 1), (2, 2), 2), (2, 1, 5, (1, 1), (3, 2), 1),
                    (3, 1, 4, (1, 1, 1), (2, 2, 2), 2)]
    summary, examples = [], []
    for truncate in (False, True):
        for (M, K, T, lmin, L, k) in configs:
            U, exact, atmost = scenario_sets(M, T, lmin, L, k, truncate)
            cnt = {'exact': 0, 'atmost': 0, 'clair': 0}
            big = {'exact': 0.0, 'atmost': 0.0}
            for trial in range(args.trials):
                inst = random_instance(rng, M, K, T, args.dist)
                vU = solve_tree(U, inst)
                v = {'exact': solve_tree(exact, inst), 'atmost': solve_tree(atmost, inst)}
                cU = solve_tree(U, inst, clairvoyant=True)
                cS = solve_tree(exact, inst, clairvoyant=True)
                if abs(cU - cS) > 1e-6:
                    cnt['clair'] += 1
                for name in ('exact', 'atmost'):
                    if vU > v[name] + 1e-6:
                        cnt[name] += 1
                        big[name] = max(big[name], vU - v[name])
                        tag = [truncate, name, M, K, T, list(lmin), list(L), k]
                        if len([e for e in examples if e['tag'] == tag]) < 1:
                            examples.append({'tag': tag, 'instance': inst, 'V_all': vU, 'V_subset': v[name],
                                             'clairvoyant_all': cU, 'clairvoyant_exact': cS})
            row = {'truncate': truncate, 'M': M, 'K': K, 'T': T, 'lmin': list(lmin), 'L': list(L), 'k': k,
                   'scenarios': len(U), 'exact': len(exact), 'atmost': len(atmost), 'trials': args.trials,
                   'gap_vs_exact': cnt['exact'], 'gap_vs_atmost': cnt['atmost'],
                   'largest_gap_exact': big['exact'], 'largest_gap_atmost': big['atmost'],
                   'clairvoyant_mismatches': cnt['clair']}
            summary.append(row)
            print(row, flush=True)
    out = {'summary': summary, 'examples': examples}
    json.dump(out, open(os.path.join(HERE, args.out), 'w'), indent=1)
    print('examples with a gap:', len(examples))


if __name__ == '__main__':
    main()
