"""Test the construction in the appendix proof of the covering lemma on every
input for small T (exhaustive), not only the existence of a cover.

Construction (as in sections/appendix.tex, version of 2026-11-12): with
c_i = T - (k - i + 1) L + 1, block i starts in the smallest period of P after
block i - 1, or in period c_i if that comes first or if no period of P is
left.

Two properties are checked for every input:
  cover     the k blocks are disjoint, lie in 1..T and cover P;
  causal    whether a block starts in period t is decided by which of the
            periods 1..t are in P (inputs that agree on 1..t give the same
            block starts up to t).

Run:  python3 checks/check_lemma2_construction.py
"""
from itertools import combinations


def construct(T, L, k, P):
    """P: sorted list of periods (1-based). Returns the starts s_1..s_k."""
    c = [T - (k - i + 1) * L + 1 for i in range(1, k + 1)]
    s = []
    for i in range(k):
        end_prev = s[-1] + L - 1 if s else 0
        later = [t for t in P if t > end_prev]
        s.append(min(later[0], c[i]) if later else c[i])
    return s


def all_block_sets(T, L, k):
    blocks = [(a, a + l - 1) for l in range(1, L + 1) for a in range(1, T - l + 2)]
    yield ()
    for j in range(1, k + 1):
        for combo in combinations(blocks, j):
            if all(x[1] < y[0] or y[1] < x[0] for x, y in combinations(combo, 2)):
                yield combo


def main():
    n = bad_cover = bad_causal = settings = 0
    for k in (1, 2, 3, 4):
        for L in (1, 2, 3):
            for T in range(k * L, min(k * L + 5, 13)):
                settings += 1
                seen = {}
                for blocks in all_block_sets(T, L, k):
                    P = sorted(set(t for a, e in blocks for t in range(a, e + 1)))
                    s = construct(T, L, k, P)
                    W = [set(range(x, x + L)) for x in s]
                    ok = (s[0] >= 1 and s[-1] + L - 1 <= T
                          and all(s[i + 1] >= s[i] + L for i in range(k - 1))
                          and set(P) <= set().union(*W))
                    n += 1
                    if not ok:
                        bad_cover += 1
                        if bad_cover <= 5:
                            print('FAIL cover', dict(T=T, L=L, k=k, blocks=blocks, s=s))
                    for t in range(1, T + 1):
                        key = (t, tuple(p for p in P if p <= t))
                        val = tuple(x for x in s if x <= t)
                        if seen.setdefault(key, val) != val:
                            bad_causal += 1
                            if bad_causal <= 5:
                                print('FAIL causal', dict(T=T, L=L, k=k, t=t, P=P, s=s, other=seen[key]))
    print(f'settings: {settings}, inputs checked: {n}, cover failures: {bad_cover}, '
          f'period-by-period failures: {bad_causal}')
    assert bad_cover == 0 and bad_causal == 0


if __name__ == '__main__':
    main()
