"""Outage scenario sets of the Optimization Letters manuscript.

Definition used from 2026-11-11 on. A scenario says, for every line m and
period t, whether the line is out. It is stored as an integer whose bit
m*T + t is set when line m is out in period t (periods are 0-based here and
1-based in the manuscript). A scenario is produced by at most k outages. An
outage of line m starts in some period of the horizon and lasts between
L_min[m] and L_max[m] periods; whatever lies after the last period is outside
the model. Two outages of one line do not overlap; they may follow each other
directly.

Three sets:
  all     every scenario
  thm1    exactly k longest outages, each ending within the horizon
          (enough when the operator knows the scenario in advance; Theorem 1)
  splus   exactly k longest outages, each starting within the horizon
          (enough when the operator learns of outages as they happen; Theorem 2)

An outage is written as an event (line, first period, number of periods inside
the horizon), so an outage cut off by the end of the horizon appears with the
length that is left of it.

Run this file to cross-check the functions against each other on small cases:
  python3 src/ol_scenarios.py
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from math import comb
from typing import Dict, Iterator, List, Sequence, Tuple

Event = Tuple[int, int, int]
KINDS = ('all', 'thm1', 'splus')


def compositions(k: int, M: int) -> Iterator[Tuple[int, ...]]:
    """All ways to give k outages to M lines."""
    if M == 1:
        yield (k,)
        return
    for first in range(k + 1):
        for rest in compositions(k - first, M - 1):
            yield (first,) + rest


# ---------------------------------------------------------------- single outages

def outage_footprints(T: int, lmin: Sequence[int], lmax: Sequence[int]) -> List[Event]:
    """Every single outage as it appears inside the horizon, without repeats."""
    out: List[Event] = []
    for m in range(len(lmax)):
        seen = set()
        for a in range(T):
            for l in range(int(lmin[m]), int(lmax[m]) + 1):
                n = min(l, T - a)
                if (a, n) not in seen:
                    seen.add((a, n))
                    out.append((m, a, n))
    return out


def longest_inside(T: int, lmax: Sequence[int]) -> List[Event]:
    """Longest outages that end within the horizon (single outages of Theorem 1)."""
    return [(m, a, int(lmax[m])) for m in range(len(lmax)) for a in range(T - int(lmax[m]) + 1)]


def longest_any_start(T: int, lmax: Sequence[int]) -> List[Event]:
    """Longest outages that start in any period (single outages of Theorem 2)."""
    return [(m, a, min(int(lmax[m]), T - a)) for m in range(len(lmax)) for a in range(T)]


# ---------------------------------------------------------------- counts

def count_thm1(T: int, lmax: Sequence[int], k: int) -> int:
    """Number of scenarios made of exactly k longest outages ending within the horizon."""
    total = 0
    for ks in compositions(k, len(lmax)):
        prod = 1
        for km, L in zip(ks, lmax):
            prod *= comb(max(0, T - km * int(L) + km), km)
        total += prod
    return total


def count_splus(T: int, lmax: Sequence[int], k: int) -> int:
    """Number of scenarios made of exactly k longest outages starting within the horizon."""
    total = 0
    for ks in compositions(k, len(lmax)):
        prod = 1
        for km, L in zip(ks, lmax):
            prod *= comb(max(0, T - (km - 1) * (int(L) - 1)), km) if km else 1
        total += prod
    return total


def _least_outages(run: int, lo: int, hi: int, touches_end: bool):
    """Least number of outages that produce `run` consecutive out-periods, or
    None if no number does. A run that reaches the last period may be closed
    by an outage that continues after the horizon."""
    need = -(-run // hi)
    if touches_end or need * lo <= run:
        return need
    return None


def line_pattern_counts(T: int, lo: int, hi: int, kmax: int) -> List[int]:
    """For one line: the number of out-patterns whose least number of outages
    is j, for j = 0..kmax."""
    @lru_cache(None)
    def ways(pos: int, left: int) -> int:
        if pos >= T:
            return 1 if left == 0 else 0
        total = ways(pos + 1, left)                    # in service in period pos
        for run in range(1, T - pos + 1):              # out in periods pos .. pos+run-1
            touches_end = pos + run == T
            need = _least_outages(run, lo, hi, touches_end)
            if -(-run // hi) > left:
                break
            if need is None:
                continue
            if touches_end:
                total += 1 if need == left else 0
            else:
                total += ways(pos + run + 1, left - need)
        return total
    return [ways(0, j) for j in range(kmax + 1)]


def count_all(T: int, lmin: Sequence[int], lmax: Sequence[int], k: int) -> int:
    """Number of scenarios (distinct outage matrices) with at most k outages."""
    M = len(lmax)
    per_line = [line_pattern_counts(T, int(lmin[m]), int(lmax[m]), k) for m in range(M)]
    total = 0
    for js in itertools.product(range(k + 1), repeat=M):
        if sum(js) <= k:
            prod = 1
            for m in range(M):
                prod *= per_line[m][js[m]]
            total += prod
    return total


# ---------------------------------------------------------------- listing

def line_patterns(T: int, lo: int, hi: int, kmax: int) -> Dict[int, int]:
    """For one line: every out-pattern that needs at most kmax outages, as
    {bits over the T periods: least number of outages}."""
    out: Dict[int, int] = {}

    def rec(pos: int, bits: int, used: int) -> None:
        if pos >= T:
            out[bits] = used
            return
        rec(pos + 1, bits, used)
        for run in range(1, T - pos + 1):
            touches_end = pos + run == T
            if used - (-run // hi) > kmax:
                break
            need = _least_outages(run, lo, hi, touches_end)
            if need is None:
                continue
            grown = bits | (((1 << run) - 1) << pos)
            if touches_end:
                out[grown] = used + need
            else:
                rec(pos + run + 1, grown, used + need)

    rec(0, 0, 0)
    return out


def _block_starts(T: int, L: int, n: int, inside: bool) -> Iterator[Tuple[int, ...]]:
    """First periods of n longest outages of one line that do not overlap."""
    last = T - L if inside else T - 1

    def rec(start: int, left: int):
        if left == 0:
            yield ()
            return
        for a in range(start, last - (left - 1) * L + 1):
            for rest in rec(a + L, left - 1):
                yield (a,) + rest

    yield from rec(0, n)


def longest_scenarios(T: int, lmax: Sequence[int], k: int, kind: str) -> List[List[Event]]:
    """The scenarios of Theorem 1 (kind 'thm1') or Theorem 2 (kind 'splus'),
    written down directly as lists of events."""
    if kind not in ('thm1', 'splus'):
        raise ValueError(kind)
    M, L = len(lmax), [int(v) for v in lmax]
    out: List[List[Event]] = []
    for ks in compositions(k, M):
        per_line = [list(_block_starts(T, L[m], ks[m], kind == 'thm1')) for m in range(M)]
        for combo in itertools.product(*per_line):
            out.append([(m, a, min(L[m], T - a)) for m in range(M) for a in combo[m]])
    return out


def bits_of(events: Sequence[Event], T: int) -> int:
    b = 0
    for m, a, n in events:
        b |= ((1 << min(n, T - a)) - 1) << (m * T + a)
    return b


def events_of(bits: int, M: int, T: int) -> List[Event]:
    """The runs of out-periods of a scenario, one event per run."""
    out: List[Event] = []
    for m in range(M):
        row = (bits >> (m * T)) & ((1 << T) - 1)
        t = 0
        while t < T:
            if (row >> t) & 1:
                a = t
                while t < T and (row >> t) & 1:
                    t += 1
                out.append((m, a, t - a))
            else:
                t += 1
    return out


def scenario_masks(T: int, lmin: Sequence[int], lmax: Sequence[int], k: int, kind: str) -> List[int]:
    """Sorted bit masks of the scenarios of one of the three sets."""
    M = len(lmax)
    if kind in ('thm1', 'splus'):
        return sorted(bits_of(ev, T) for ev in longest_scenarios(T, lmax, k, kind))
    if kind != 'all':
        raise ValueError(kind)
    by_need = []
    for m in range(M):
        pats = line_patterns(T, int(lmin[m]), int(lmax[m]), k)
        groups: List[List[int]] = [[] for _ in range(k + 1)]
        for bits, need in pats.items():
            groups[need].append(bits)
        by_need.append(groups)
    out: List[int] = []
    for js in itertools.product(range(k + 1), repeat=M):
        if sum(js) > k:
            continue
        for combo in itertools.product(*[by_need[m][js[m]] for m in range(M)]):
            b = 0
            for m in range(M):
                b |= combo[m] << (m * T)
            out.append(b)
    out.sort()
    return out


# ---------------------------------------------------------------- self-check

def _brute_force(T: int, lmin: Sequence[int], lmax: Sequence[int], k: int, kind: str) -> List[int]:
    """The same sets from the definition, by trying every choice of outages."""
    M = len(lmax)
    if kind == 'all':
        events = [(m, a, l) for m in range(M) for l in range(int(lmin[m]), int(lmax[m]) + 1) for a in range(T)]
        sizes: Sequence[int] = range(0, k + 1)
    elif kind == 'splus':
        events = [(m, a, int(lmax[m])) for m in range(M) for a in range(T)]
        sizes = [k]
    else:
        events = [(m, a, int(lmax[m])) for m in range(M) for a in range(T - int(lmax[m]) + 1)]
        sizes = [k]
    out = set()
    for j in sizes:
        for combo in itertools.combinations(events, j):
            if all(not (m1 == m2 and not (a1 + l1 <= a2 or a2 + l2 <= a1))
                   for (m1, a1, l1), (m2, a2, l2) in itertools.combinations(combo, 2)):
                out.add(bits_of(combo, T))
    return sorted(out)


def self_check() -> Dict[str, int]:
    settings = 0
    for M in (1, 2, 3):
        for T in range(2, 8 if M < 3 else 6):
            for lmax in itertools.product(range(1, 5), repeat=M):
                for lmin in itertools.product(range(1, 4), repeat=M):
                    if any(a > b for a, b in zip(lmin, lmax)):
                        continue
                    for k in (1, 2, 3):
                        if comb(M * T * max(b - a + 1 for a, b in zip(lmin, lmax)), k) > 30000:
                            continue
                        full = _brute_force(T, lmin, lmax, k, 'all')
                        assert scenario_masks(T, lmin, lmax, k, 'all') == full, (M, T, lmin, lmax, k)
                        assert count_all(T, lmin, lmax, k) == len(full), (M, T, lmin, lmax, k)
                        for kind, count in (('thm1', count_thm1), ('splus', count_splus)):
                            direct = scenario_masks(T, lmin, lmax, k, kind)
                            assert direct == _brute_force(T, lmin, lmax, k, kind), (kind, M, T, lmax, k)
                            assert len(direct) == len(set(direct)) == count(T, lmax, k), (kind, M, T, lmax, k)
                            assert set(direct) <= set(full), (kind, M, T, lmin, lmax, k)
                        singles = {bits_of([e], T) for e in outage_footprints(T, lmin, lmax)}
                        assert singles == set(_brute_force(T, lmin, lmax, 1, 'all')) - {0}
                        settings += 1
    return {'settings': settings}


if __name__ == '__main__':
    print('self-check passed:', self_check())
    T, lmin, lmax = 24, (2, 2, 1), (6, 6, 2)
    print('lunar example: single outages', len(outage_footprints(T, lmin, lmax)),
          '| longest, ending inside', len(longest_inside(T, lmax)),
          '| longest, any start', len(longest_any_start(T, lmax)))
    for k in (1, 2, 3, 4):
        print(f'  k={k}: all {count_all(T, lmin, lmax, k):,}  thm1 {count_thm1(T, lmax, k):,}  splus {count_splus(T, lmax, k):,}')
