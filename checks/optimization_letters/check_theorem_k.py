"""Brute-force check of the k-outage version of Theorem 1 (suggested by a
simulated referee on 2026-11-10; not yet in the manuscript).

Claim: with at most k outages per scenario and T >= k * L_m for every line,
the containment-maximal matrices are exactly those of the scenarios made of
exactly k longest outages, every scenario is contained in one of them, and
their number is
    sum over (k_1,...,k_M) with k_1+...+k_M = k of
        prod_m C(T - k_m * L_m + k_m, k_m).
"""
from itertools import combinations, product
from math import comb


def compositions(k, M):
    if M == 1:
        yield (k,); return
    for a in range(k + 1):
        for rest in compositions(k - a, M - 1):
            yield (a,) + rest


def formula(T, L, k):
    tot = 0
    for ks in compositions(k, len(L)):
        p = 1
        for km, l in zip(ks, L):
            p *= comb(T - km * l + km, km)
        tot += p
    return tot


def check(T, lmin, L, k):
    M = len(L)
    outs = [(m, a, l) for m in range(M) for l in range(lmin[m], L[m] + 1) for a in range(1, T - l + 2)]
    cells = {o: frozenset((o[0], t) for t in range(o[1], o[1] + o[2])) for o in outs}
    U, Ustar = {frozenset()}, set()
    for j in range(1, k + 1):
        for combo in combinations(outs, j):
            ok = True
            for a, b in combinations(combo, 2):
                if a[0] == b[0] and (cells[a] & cells[b]):
                    ok = False; break
            if not ok:
                continue
            mat = frozenset().union(*(cells[o] for o in combo))
            U.add(mat)
            if j == k and all(o[2] == L[o[0]] for o in combo):
                Ustar.add(mat)
    Ul = list(U)
    maximal = {d for d in Ul if not any(d < e for e in Ul)}
    covered = all(any(d <= e for e in Ustar) for d in Ul)
    return maximal == Ustar, covered, len(Ustar) == formula(T, L, k), len(U), len(Ustar)


bad = n = 0
for k in (1, 2, 3):
    for M in (1, 2, 3):
        for T in range(2, 10):
            for L in product(range(1, 4), repeat=M):
                if any(T < k * l for l in L):
                    continue
                if k == 3 and M == 3 and T > 7:
                    continue          # keep the enumeration small
                for lmin in {tuple(1 for _ in L), tuple(L)}:
                    r = check(T, lmin, L, k)
                    n += 1
                    if not all(r[:3]):
                        bad += 1
                        print('FAIL', dict(k=k, M=M, T=T, lmin=lmin, L=L), r)
print(f'cases checked: {n}, failures: {bad}')
print('example of the paper with k = 2:', formula(24, (6, 6, 2), 2), ' with k = 3:', formula(24, (6, 6, 2), 3), ' k = 1:', formula(24, (6, 6, 2), 1))
