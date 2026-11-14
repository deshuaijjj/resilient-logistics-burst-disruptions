"""Checks for the result on an operator who learns of outages as they happen.

Scenario set (used from 2026-11-11 on): at most k outages; an outage of line m
starts in some period a <= T and lasts l periods, lmin_m <= l <= L_m; the part
of an outage after period T is outside the model. Two outages of one line do
not overlap.

Claim (non-anticipative operator). Let S+ be the scenarios made of exactly k
longest outages, each starting in any period 1..T. For every design and all
data satisfying (A1)-(A3), the best worst-case operating cost over all
scenarios equals that over S+.

Claim (operator who knows the scenario). The scenarios not contained in
another one are those made of exactly k longest outages that end by period T,
as in Theorem 1; their number is unchanged by the new definition.

Seven checks:
  oneline   with a single line and T >= k L, the construction in the proof of
            the covering lemma goes period by period, so the scenarios of
            Theorem 1 are enough for this operator too; checked on every
            scenario of many single-line settings, and by tree linear
            programs on random instances.
  lunar     the rule used in the proof, run on every scenario of the lunar
            example (at most two and at most three outages).
  cover     the rule used in the proof, run on every scenario of many small
            settings: it gives exactly k longest outages, contains the actual
            scenario, and its first t periods depend only on the first t
            periods of the actual scenario.
  coverk    the same checks for every number of outages from 1 up to the
            bound sum_m ceil(T / L_m) of the theorem, on the settings with at
            most 12 line-periods (cover stops at four outages).
  counts    number of scenarios in S+ against the closed form, and the
            maximal scenarios against Theorem 1.
  values    scenario-tree linear programs on random small instances: value
            over all scenarios against value over S+ (and over Theorem 1's
            scenarios, where a gap is expected).
  needed    for small settings, which scenarios of S+ can be removed without
            changing the value on any of the random instances tried.

Run:  python3 checks/check_nonanticipative_theorem.py [cover|coverk|counts|values|needed|lunar|oneline] ...
"""
import argparse
import itertools
import json
import os
from math import comb

import numpy as np

from nonanticipative_search import random_instance, solve_tree

HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root(start):
    """The repository root: the first directory above this file that holds src/ and results/."""
    d = start
    while True:
        if os.path.isdir(os.path.join(d, "src")) and os.path.isdir(os.path.join(d, "results")):
            return d
        up = os.path.dirname(d)
        if up == d:
            raise RuntimeError("repository root (with src/ and results/) not found above " + start)
        d = up


# ---------------------------------------------------------------- scenario sets

def matrix_of(combo, M, T):
    d = [[0] * T for _ in range(M)]
    for m, a, l in combo:
        for t in range(a, min(a + l, T)):
            d[m][t] = 1
    return tuple(tuple(r) for r in d)


def disjoint(combo):
    return all(not (m1 == m2 and not (a1 + l1 <= a2 or a2 + l2 <= a1))
               for (m1, a1, l1), (m2, a2, l2) in itertools.combinations(combo, 2))


def all_scenarios(M, T, lmin, L, k):
    """Distinct outage matrices of the scenario set (outages may run past T)."""
    outs = [(m, a, l) for m in range(M) for l in range(lmin[m], L[m] + 1) for a in range(T)]
    U = {matrix_of((), M, T)}
    for j in range(1, k + 1):
        for combo in itertools.combinations(outs, j):
            if disjoint(combo):
                U.add(matrix_of(combo, M, T))
    return sorted(U)


def longest_scenarios(M, T, L, k, inside):
    """Exactly k longest outages; inside=True: every outage ends by period T."""
    outs = [(m, a, L[m]) for m in range(M) for a in (range(T - L[m] + 1) if inside else range(T))]
    S = set()
    for combo in itertools.combinations(outs, k):
        if disjoint(combo):
            S.add(matrix_of(combo, M, T))
    return sorted(S)


def compositions(k, M):
    if M == 1:
        yield (k,)
        return
    for first in range(k + 1):
        for rest in compositions(k - first, M - 1):
            yield (first,) + rest


def count_splus(T, L, k):
    return sum(int(np.prod([comb(max(0, T - (km - 1) * (Lm - 1)), km) if km else 1 for km, Lm in zip(c, L)]))
               for c in compositions(k, len(L)))


def count_theorem1(T, L, k):
    return sum(int(np.prod([comb(max(0, T - km * Lm + km), km) for km, Lm in zip(c, L)])) for c in compositions(k, len(L)))


# ---------------------------------------------------------------- the rule of the proof

def causal_cover(delta, L, k):
    """Online rule. Reads the actual outage matrix one period at a time and
    returns the starts of k longest outages (line, start period 0-based)."""
    M, T = len(delta), len(delta[0])
    end = [-1] * M          # last period (0-based) covered by an imagined outage of line m
    left = k
    blocks = []
    for t in range(T):
        for m in range(M):
            if delta[m][t] and end[m] < t:       # line is out and not yet covered: start a longest outage now
                blocks.append((m, t)); end[m] = t + L[m] - 1; left -= 1
                if left < 0:
                    raise AssertionError('more than k outages needed')
        # number of longest outages that can still start after period t
        room = sum(-(-max(0, (T - 1) - max(t, end[m])) // L[m]) for m in range(M))
        if left > room:
            due = [m for m in range(M) if end[m] < t and (T - 1 - t) % L[m] == 0]
            need = left - room
            if need > len(due):
                raise AssertionError('not enough lines for spare outages')
            for m in due[:need]:
                blocks.append((m, t)); end[m] = t + L[m] - 1; left -= 1
    if left != 0:
        raise AssertionError(f'{left} outages not placed')
    return blocks


def check_cover(max_T=8, verbose=True):
    settings = n_scen = 0
    for M in (1, 2, 3):
        for T in range(2, max_T + 1):
            for L in itertools.product(range(1, 5), repeat=M):
                for lmin in itertools.product(range(1, 4), repeat=M):
                    if any(a > b for a, b in zip(lmin, L)):
                        continue
                    for k in (1, 2, 3, 4):
                        if k > sum(-(-T // Lm) for Lm in L):          # needs k <= sum_m ceil(T / L_m)
                            continue
                        outs = M * T * max(b - a + 1 for a, b in zip(lmin, L))
                        if comb(outs, k) > 60000:
                            continue
                        U = all_scenarios(M, T, lmin, L, k)
                        seen = {}
                        for d in U:
                            blocks = causal_cover(d, L, k)
                            assert len(blocks) == k
                            assert disjoint([(m, a, L[m]) for m, a in blocks])
                            img = matrix_of([(m, a, L[m]) for m, a in blocks], M, T)
                            assert all(img[m][t] >= d[m][t] for m in range(M) for t in range(T)), 'not contained'
                            for t in range(T):
                                key = (t, tuple(row[:t + 1] for row in d))
                                val = tuple(row[:t + 1] for row in img)
                                assert seen.setdefault(key, val) == val, 'depends on the future'
                        settings += 1; n_scen += len(U)
        if verbose:
            print(f'cover: lines <= {M}: {settings} settings, {n_scen} scenarios, all pass', flush=True)
    return {'settings': settings, 'scenarios': n_scen}


def check_cover_every_k(max_cells=12, verbose=True):
    """The rule of the proof for every k from 1 to sum_m ceil(T / L_m), on
    every scenario of the settings with at most `max_cells` line-periods.
    The scenario sets come from src/ol_scenarios.py, which lists the distinct
    matrices directly. With three lines the shortest durations are all 1, all
    equal to the longest, or 1 on the middle line only."""
    import sys
    sys.path.append(os.path.join(_repo_root(HERE), 'src'))
    import ol_scenarios as sc
    settings = n_scen = at_bound = 0
    for M in (1, 2, 3):
        for T in range(2, max_cells // M + 1):
            for L in itertools.product(range(1, 5), repeat=M):
                if M < 3:
                    lmins = [lm for lm in itertools.product(range(1, 5), repeat=M) if all(a <= b for a, b in zip(lm, L))]
                else:
                    lmins = sorted({(1, 1, 1), tuple(L), (L[0], 1, L[2])})
                bound = sum(-(-T // Lm) for Lm in L)
                for lmin in lmins:
                    for k in range(1, bound + 1):
                        U = sc.scenario_masks(T, lmin, L, k, 'all')
                        Sp = set(sc.scenario_masks(T, lmin, L, k, 'splus'))
                        assert len(Sp) == count_splus(T, L, k)
                        images = []
                        for b in U:
                            d = tuple(tuple((b >> (m * T + t)) & 1 for t in range(T)) for m in range(M))
                            blocks = causal_cover(d, L, k)
                            assert len(blocks) == k
                            img = sc.bits_of([(m, a, L[m]) for m, a in blocks], T)
                            assert img in Sp, 'not a scenario of S+'
                            assert img & b == b, 'not contained'
                            images.append(img)
                        upto = 0
                        for t in range(T):
                            upto |= sum(1 << (m * T + t) for m in range(M))
                            seen = {}
                            for b, img in zip(U, images):
                                assert seen.setdefault(b & upto, img & upto) == img & upto, 'depends on the future'
                        settings += 1; n_scen += len(U); at_bound += (k == bound)
        if verbose:
            print(f'coverk: lines <= {M}: {settings} settings ({at_bound} with k at the bound), '
                  f'{n_scen} scenarios, all pass', flush=True)
    return {'settings': settings, 'settings_with_k_at_the_bound': at_bound, 'scenarios': n_scen}


def one_line_cover(row, L, k):
    """The construction of the covering lemma, read one period at a time:
    block i starts in the first period that is out and not yet covered, or in
    period T - (k - i + 1) L + 1 if that comes first. Returns the 0-based
    first periods of the k blocks."""
    T = len(row)
    starts, end = [], -1
    for t in range(T):
        i = len(starts)                       # blocks placed so far
        if i == k or t <= end:
            continue
        deadline = T - (k - i) * L            # 0-based form of T - (k - i + 1) L + 1 for block i + 1
        if row[t] or t == deadline:
            starts.append(t); end = t + L - 1
    return starts


def check_one_line(trials=200, seed=20261111, max_T=12):
    settings = n_scen = 0
    for T in range(2, max_T + 1):
        for L in range(1, 6):
            for lmin in range(1, L + 1):
                for k in (1, 2, 3, 4):
                    if T < k * L or comb(T * (L - lmin + 1), k) > 60000:
                        continue
                    U = all_scenarios(1, T, (lmin,), (L,), k)
                    seen = {}
                    for d in U:
                        starts = one_line_cover(d[0], L, k)
                        assert len(starts) == k and starts[-1] + L <= T, 'not k longest outages inside the horizon'
                        assert all(b >= a + L for a, b in zip(starts, starts[1:]))
                        img = matrix_of([(0, a, L) for a in starts], 1, T)
                        assert all(img[0][t] >= d[0][t] for t in range(T)), 'not contained'
                        for t in range(T):
                            key, val = (t, d[0][:t + 1]), img[0][:t + 1]
                            assert seen.setdefault(key, val) == val, 'depends on the future'
                    settings += 1; n_scen += len(U)
    print(f'oneline: {settings} settings, {n_scen} scenarios, all pass', flush=True)
    rng = np.random.default_rng(seed)
    rows = []
    for (K, T, lmin, L, k) in [(1, 5, 1, 2, 2), (2, 6, 2, 3, 2), (1, 6, 1, 3, 1), (2, 7, 1, 2, 3), (1, 8, 2, 4, 2)]:
        U = all_scenarios(1, T, (lmin,), (L,), k)
        S1 = longest_scenarios(1, T, (L,), k, inside=True)
        gaps = 0
        for i in range(trials):
            inst = random_instance(rng, 1, K, T, i % 2)
            if abs(solve_tree(U, inst) - solve_tree(S1, inst)) > 1e-6:
                gaps += 1
        rows.append({'K': K, 'T': T, 'lmin': lmin, 'L': L, 'k': k, 'scenarios': len(U), 'theorem1_set': len(S1),
                     'trials': trials, 'gap_with_theorem1_set': gaps})
        print('oneline values:', rows[-1], flush=True)
    return {'settings': settings, 'scenarios': n_scen, 'values': rows}


def check_lunar(ks=(2, 3)):
    """The rule of the proof on every scenario of the lunar example: the
    scenario it builds is one of S+, contains the actual scenario, and its
    first t periods depend only on the first t periods of the actual one."""
    import sys
    sys.path.append(os.path.join(_repo_root(HERE), 'src'))
    import ol_scenarios as sc
    M, T, lmin, L = 3, 24, (2, 2, 1), (6, 6, 2)
    rows = []
    for k in ks:
        U = sc.scenario_masks(T, lmin, L, k, 'all')
        Sp = set(sc.scenario_masks(T, lmin, L, k, 'splus'))
        images = []
        for b in U:
            d = tuple(tuple((b >> (m * T + t)) & 1 for t in range(T)) for m in range(M))
            blocks = causal_cover(d, L, k)
            img = sc.bits_of([(m, a, L[m]) for m, a in blocks], T)
            assert img in Sp, 'not a scenario of S+'
            assert img & b == b, 'not contained'
            images.append(img)
        upto = 0
        for t in range(T):
            upto |= sum(1 << (m * T + t) for m in range(M))
            seen = {}
            for b, img in zip(U, images):
                assert seen.setdefault(b & upto, img & upto) == img & upto, 'depends on the future'
        rows.append({'k': k, 'scenarios': len(U), 'S_plus': len(Sp),
                     'scenarios_of_S_plus_the_rule_arrives_at': len(set(images))})
        print('lunar:', rows[-1], flush=True)
    return rows


# ---------------------------------------------------------------- counts

def check_counts(max_T=9):
    n = 0
    for M in (1, 2, 3):
        for T in range(2, max_T + 1):
            for L in itertools.product(range(1, 5), repeat=M):
                for k in (1, 2, 3):
                    if comb(M * T, k) > 40000:
                        continue
                    S = longest_scenarios(M, T, L, k, inside=False)
                    assert len(S) == count_splus(T, L, k), (M, T, L, k, len(S), count_splus(T, L, k))
                    if all(T >= k * Lm for Lm in L):
                        lmin = tuple(1 for _ in L)
                        if comb(M * T * max(L), k) <= 40000:
                            U = all_scenarios(M, T, lmin, L, k)
                            bits = [sum(1 << (m * T + t) for m in range(M) for t in range(T) if d[m][t]) for d in U]
                            maximal = {b for b in bits if not any(b != c and b & c == b for c in bits)}
                            inside = longest_scenarios(M, T, L, k, inside=True)
                            ib = {sum(1 << (m * T + t) for m in range(M) for t in range(T) if d[m][t]) for d in inside}
                            assert maximal == ib, (M, T, L, k)
                            assert len(ib) == count_theorem1(T, L, k)
                    n += 1
    print(f'counts: {n} settings pass', flush=True)
    return {'settings': n}


# ---------------------------------------------------------------- values

VALUE_SETTINGS = [  # (M, K, T, lmin, L, k)
    (1, 1, 5, (1,), (2,), 2), (1, 2, 6, (2,), (3,), 2), (2, 1, 3, (1, 1), (2, 2), 1), (2, 2, 4, (1, 1), (3, 2), 1),
    (2, 2, 4, (1, 1), (2, 2), 2), (2, 1, 6, (1, 1), (2, 2), 3), (2, 1, 5, (1, 2), (3, 2), 2), (2, 2, 6, (2, 1), (3, 2), 2),
    (2, 1, 5, (2, 2), (3, 3), 2), (3, 1, 4, (1, 1, 1), (2, 2, 2), 2), (3, 1, 5, (2, 1, 2), (3, 2, 3), 2),
    (2, 1, 7, (1, 1), (3, 2), 3), (1, 1, 7, (1,), (3,), 3), (2, 1, 4, (1, 1), (2, 2), 4), (3, 2, 3, (1, 1, 1), (2, 2, 2), 1),
]


def check_values(trials, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for (M, K, T, lmin, L, k) in VALUE_SETTINGS:
        U = all_scenarios(M, T, lmin, L, k)
        Sp = longest_scenarios(M, T, L, k, inside=False)
        S1 = longest_scenarios(M, T, L, k, inside=True)
        gap_p = gap_1 = 0
        big_1 = 0.0
        for i in range(trials):
            inst = random_instance(rng, M, K, T, i % 2)
            vU, vP = solve_tree(U, inst), solve_tree(Sp, inst)
            if abs(vU - vP) > 1e-6:
                gap_p += 1
            if S1:
                v1 = solve_tree(S1, inst)
                if vU > v1 + 1e-6:
                    gap_1 += 1; big_1 = max(big_1, vU - v1)
        row = {'M': M, 'K': K, 'T': T, 'lmin': list(lmin), 'L': list(L), 'k': k, 'scenarios': len(U),
               'S_plus': len(Sp), 'theorem1_set': len(S1), 'trials': trials, 'gap_with_S_plus': gap_p,
               'gap_with_theorem1_set': gap_1, 'largest_gap_theorem1_set': big_1}
        rows.append(row)
        print(row, flush=True)
    print('values: instances', sum(r['trials'] for r in rows), 'with a gap for S+', sum(r['gap_with_S_plus'] for r in rows),
          '; with a gap for the Theorem 1 set', sum(r['gap_with_theorem1_set'] for r in rows))
    return rows


# ---------------------------------------------------------------- needed

def check_needed(trials, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for (M, K, T, lmin, L, k) in [(2, 1, 3, (1, 1), (2, 2), 1), (2, 1, 4, (1, 1), (2, 2), 1), (2, 2, 4, (1, 1), (3, 2), 1),
                                  (1, 1, 5, (1,), (2,), 2), (2, 1, 4, (1, 1), (2, 2), 2)]:
        Sp = longest_scenarios(M, T, L, k, inside=False)
        inside = set(longest_scenarios(M, T, L, k, inside=True))
        needed = [False] * len(Sp)
        for i in range(trials):
            inst = random_instance(rng, M, K, T, i % 2)
            v = solve_tree(Sp, inst)
            for j in range(len(Sp)):
                if not needed[j] and solve_tree(Sp[:j] + Sp[j + 1:], inst) < v - 1e-6:
                    needed[j] = True
            if all(needed):
                break
        row = {'M': M, 'K': K, 'T': T, 'L': list(L), 'k': k, 'S_plus': len(Sp), 'theorem1_set': len(inside),
               'shown_needed': int(sum(needed)), 'instances_tried': i + 1,
               'not_shown_needed': [Sp[j] for j in range(len(Sp)) if not needed[j]],
               'not_shown_needed_in_theorem1_set': int(sum(1 for j in range(len(Sp)) if not needed[j] and Sp[j] in inside))}
        rows.append(row)
        print({a: b for a, b in row.items() if a != 'not_shown_needed'}, flush=True)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', nargs='*', default=['cover', 'counts', 'values'])
    ap.add_argument('--trials', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20261111)
    ap.add_argument('--max-T', type=int, default=8)
    ap.add_argument('--max-cells', type=int, default=12, help='coverk: settings with at most this many line-periods')
    args = ap.parse_args()
    out = {}
    if 'lunar' in args.what:
        out['lunar'] = check_lunar()
    if 'oneline' in args.what:
        out['oneline'] = check_one_line(args.trials, args.seed)
    if 'cover' in args.what:
        out['cover'] = check_cover(args.max_T)
    if 'coverk' in args.what:
        out['coverk'] = dict(check_cover_every_k(args.max_cells), max_line_periods=args.max_cells)
    if 'counts' in args.what:
        out['counts'] = check_counts()
    if 'values' in args.what:
        out['values'] = check_values(args.trials, args.seed)
    if 'needed' in args.what:
        out['needed'] = check_needed(args.trials, args.seed)
    name = 'check_nonanticipative_theorem_' + '_'.join(args.what) + '.json'
    json.dump(out, open(os.path.join(HERE, name), 'w'), indent=1)
