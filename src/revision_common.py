"""Shared helpers for the September 2026 revision experiments (results/12-18).

Everything here reuses the recorded model code paths (``instance_generator``,
``recourse_certifier``, ``lp_solver``, ``run_dominance_pruned_oracle``) and adds
only what the revision needs: dual extraction from the fixed-design recourse LP,
Kelley/Benders cutting planes on the five first-stage variables, the single-level
worst-case MILP (dualize-and-linearize), a CCG loop with a configurable
separation set, and a vectorized extensive-form builder for the complete
distinct-matrix library.

Units: the model is in U.S. dollars.  The MILP oracle is solved in millions of
dollars for numerical conditioning and converted back on output.  Every
function documents which unit it returns.

No function in this module writes to results/08-11.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from instance_generator import (  # noqa: E402
    DisruptionScenario,
    InstanceParameters,
    generate_burst_scenarios,
    generate_small_instance,
)
from recourse_certifier import (  # noqa: E402
    RecourseTemplate,
    build_recourse_template,
    certify_fixed_design,
    deterministic_even_sample_indices,
    first_stage_cost,
)
from run_dominance_pruned_oracle import (  # noqa: E402
    CanonicalMatrix,
    canonicalize_scenarios,
    hash_canonical_records,
    maximal_matrices,
    scenario_bits,
)

MILLION = 1e6


def popcount(bits: int) -> int:
    """Number of set bits.  ``int.bit_count`` is Python 3.10+; this is portable
    and returns the identical value for every nonnegative integer."""
    return bin(bits).count('1')


# Frozen reference values from results/08-11 (U.S. dollars unless noted).
FROZEN = {
    'event_set_count': 29916,
    'unique_matrix_count': 29207,
    'maximal_matrix_count': 1648,
    'canonical_sha256': 'ac68a411a92f5d401c8b0e7f94b5aca8eb1188776ec4817edf2e306df4e02e0e',
    'maximal_sha256': '010f5b5df4b27a886c0f36c69bf064090a907cb4b0b2f6920b3007da4dbe5b7a',
    'exact_total_cost': 3581093473.035439,
    'exact_worst_recourse': 3512076000.0,
    'exact_C': [600.0, 600.0],
    'exact_q': [350.0, 188.18695428865271, 208.33333333333334],
    'sampled_in_sample_total_cost': 3578972950.9803924,
    'sampled_full_total_cost': 8278684856.582633,
    'sampled_full_worst_recourse': 8209664554.061625,
    'sampled_C': [600.0, 600.0],
    'sampled_q': [418.45938375345594, 300.0, 130.0],
    'sampled_worst_unmet_tons': 770.0,
    'nominal_row_full_worst_recourse': 44336850666.666665,
    'nominal_row_q': [358.3333333333333, 0.0, 0.0],
    'reference_worst_ids': [1295, 1220],
}


# --------------------------------------------------------------------------- #
# Environment, hashing, and I/O
# --------------------------------------------------------------------------- #

def environment_record() -> Dict:
    """Machine and software record stored next to every new artifact."""
    record = {
        'timestamp': _dt.datetime.now().astimezone().isoformat(),
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'thread_env': {k: os.environ.get(k) for k in (
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
    }
    try:
        from scipy.optimize._highspy import _core  # type: ignore
        record['highs_version'] = getattr(_core, 'HIGHS_VERSION', None) or getattr(_core, '__version__', None)
    except Exception:  # pragma: no cover - version reporting is best effort
        record['highs_version'] = None
    try:
        record['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
            text=True).strip()
    except Exception:  # pragma: no cover
        record['git_commit'] = None
    if platform.system() == 'Darwin':
        try:
            record['hardware_model'] = subprocess.check_output(
                ['sysctl', '-n', 'hw.model'], text=True).strip()
            record['memory_bytes'] = int(subprocess.check_output(
                ['sysctl', '-n', 'hw.memsize'], text=True).strip())
        except Exception:  # pragma: no cover
            pass
    return record


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as stream:
        json.dump(data, stream, indent=2, default=_json_default)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):      # numpy ints, floats, bools
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f'not JSON serializable: {type(value)}')


# --------------------------------------------------------------------------- #
# Library
# --------------------------------------------------------------------------- #

@dataclass
class Library:
    params: InstanceParameters
    scenarios: List[DisruptionScenario]
    canonical: List[CanonicalMatrix]
    maximal: List[CanonicalMatrix]
    bits: List[int]
    single_events: List[Tuple[int, int, int]]
    prune_time_s: float

    @property
    def maximal_ids(self) -> List[int]:
        return sorted(record.representative_id for record in self.maximal)

    @property
    def canonical_ids(self) -> List[int]:
        return sorted(record.representative_id for record in self.canonical)

    def delta(self, scenario_id: int) -> np.ndarray:
        return self.scenarios[scenario_id].to_delta_matrix(self.params.M, self.params.T)

    def summary(self) -> Dict:
        return {
            'event_set_count': len(self.scenarios),
            'unique_matrix_count': len(self.canonical),
            'maximal_matrix_count': len(self.maximal),
            'single_event_count': len(self.single_events),
            'canonical_sha256': hash_canonical_records(self.canonical),
            'maximal_sha256': hash_canonical_records(self.maximal),
            'prune_time_s': self.prune_time_s,
        }


def single_events(params: InstanceParameters) -> List[Tuple[int, int, int]]:
    """Feasible single events in the generator's order (mode, start, duration)."""
    events = []
    for m in range(params.M):
        for start in range(params.T):
            for duration in range(int(params.L_min[m]), int(params.L_max[m]) + 1):
                if start + duration <= params.T:
                    events.append((m, start, duration))
    return events


def build_library(params: Optional[InstanceParameters] = None) -> Library:
    params = params or generate_small_instance()
    scenarios = generate_burst_scenarios(params, max_scenarios=None)
    start = time.time()
    canonical = canonicalize_scenarios(scenarios, params.M, params.T)
    maximal = maximal_matrices(canonical)
    prune_time = time.time() - start
    bits = [scenario_bits(s, params.M, params.T) for s in scenarios]
    return Library(params, scenarios, canonical, maximal, bits, single_events(params), prune_time)


def maximal_within(library: Library, ids: Sequence[int]) -> Tuple[List[int], int]:
    """Maximal representatives of a subset of encodings (exact for that subset)."""
    grouped: Dict[int, List[int]] = {}
    for sid in ids:
        grouped.setdefault(library.bits[sid], []).append(int(sid))
    records = [
        CanonicalMatrix(bits=b, severity=popcount(b), representative_id=v[0], event_set_ids=tuple(v))
        for b, v in grouped.items()
    ]
    return [r.representative_id for r in maximal_matrices(records)], len(grouped)


def event_cover_matrix(library: Library) -> sparse.csr_matrix:
    """Cell-by-event 0/1 matrix: row (m*T+t), column e, entry 1 if event e disrupts cell."""
    params = library.params
    n_cells, n_events = params.M * params.T, len(library.single_events)
    rows, cols = [], []
    for e, (m, start, duration) in enumerate(library.single_events):
        for t in range(start, start + duration):
            rows.append(m * params.T + t)
            cols.append(e)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_cells, n_events))


def random_sample_ids(num_scenarios: int, sample_size: int, seed: int) -> List[int]:
    """Nominal plus ``sample_size-1`` non-nominal encodings drawn uniformly without replacement."""
    rng = np.random.default_rng(seed)
    draw = rng.choice(num_scenarios - 1, size=sample_size - 1, replace=False)
    return [0] + sorted(int(1 + i) for i in draw)


# --------------------------------------------------------------------------- #
# Fixed-design recourse evaluation with duals
# --------------------------------------------------------------------------- #

@dataclass
class RecourseEval:
    objective: float               # dollars
    unmet_tons: float
    unmet_by_commodity: List[float]
    status: str
    solve_time_s: float
    y_ub: Optional[np.ndarray] = None
    y_eq: Optional[np.ndarray] = None
    message: str = ''


class RecourseEvaluator:
    """Fixed-design recourse LP with adjustable (C, q) right-hand sides and duals.

    The constraint matrices come from ``recourse_certifier.build_recourse_template``
    so the LP is identical to the recorded evaluator; only the right-hand sides
    that carry C, q, and Delta are rewritten per call.
    """

    def __init__(self, params: InstanceParameters):
        self.params = params
        M, J, D, K, T = params.M, params.J, params.D, params.K, params.T
        placeholder_C = np.sum(params.s_initial, axis=1).astype(float)
        self.template: RecourseTemplate = build_recourse_template(params, placeholder_C, np.ones(M))
        self.inv_rows = np.arange(T * J).reshape(T, J)            # (t, j)
        self.mode_rows = self.template.mode_rows                    # (m, t)
        self.rec_rows = np.arange(T * J + T * M, T * J + T * M + T * D)
        self.u_offset, self.n_u = self.template.offsets[3], self.template.sizes[3]
        self.n_lp = 0

    def rhs(self, C: Sequence[float], q: Sequence[float], Delta: np.ndarray) -> np.ndarray:
        params = self.params
        b_ub = self.template.b_ub_base.copy()
        for t in range(params.T):
            for j in range(params.J):
                b_ub[self.inv_rows[t, j]] = C[j]
        for m in range(params.M):
            b_ub[self.mode_rows[m, :]] = q[m] * (1 - Delta[m, :])
        return b_ub

    def evaluate(self, C, q, Delta: np.ndarray, want_duals: bool = False) -> RecourseEval:
        start = time.time()
        b_ub = self.rhs(C, q, Delta)
        res = linprog(self.template.c, A_ub=self.template.A_ub, b_ub=b_ub,
                      A_eq=self.template.A_eq, b_eq=self.template.b_eq,
                      bounds=self.template.bounds, method='highs')
        self.n_lp += 1
        elapsed = time.time() - start
        if not res.success:
            return RecourseEval(float('inf'), float('inf'), [float('inf')] * self.params.K,
                                'failed', elapsed, message=res.message)
        u = res.x[self.u_offset:self.u_offset + self.n_u].reshape(self.params.D, self.params.K, self.params.T)
        out = RecourseEval(float(res.fun), float(u.sum()), [float(v) for v in u.sum(axis=(0, 2))],
                           'optimal', elapsed)
        if want_duals:
            out.y_ub = np.asarray(res.ineqlin.marginals)
            out.y_eq = np.asarray(res.eqlin.marginals)
        return out

    def dual_objective(self, C, q, Delta: np.ndarray, y_ub: np.ndarray, y_eq: np.ndarray) -> float:
        """b_ub' y_ub + b_eq' y_eq for the given right-hand sides (dollars)."""
        return float(self.rhs(C, q, Delta) @ y_ub + self.template.b_eq @ y_eq)

    def cut(self, Delta: np.ndarray, y_ub: np.ndarray, y_eq: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Benders/Kelley cut Q(C,q;Delta) >= const + gC.C + gq.q (dollars)."""
        params = self.params
        gC = np.array([y_ub[self.inv_rows[:, j]].sum() for j in range(params.J)])
        gq = np.array([(y_ub[self.mode_rows[m, :]] * (1 - Delta[m, :])).sum() for m in range(params.M)])
        const = float(self.template.b_ub_base[self.rec_rows] @ y_ub[self.rec_rows] + self.template.b_eq @ y_eq)
        return const, gC, gq

    def capacity_marginals(self, Delta: np.ndarray, y_ub: np.ndarray) -> Dict:
        """Mode-capacity multipliers lambda_mt and the subgradient sum_t (1-Delta) lambda (dollars per t/month)."""
        lam = y_ub[self.mode_rows]
        return {
            'lambda_mt': lam.tolist(),
            'min_lambda': float(lam.min()),
            'subgradient_wrt_q': [float((lam[m, :] * (1 - Delta[m, :])).sum()) for m in range(self.params.M)],
        }


# --------------------------------------------------------------------------- #
# Kelley / Benders cutting planes on the first stage
# --------------------------------------------------------------------------- #

@dataclass
class CutModel:
    A: List[np.ndarray] = field(default_factory=list)   # rows: [gC, gq, -1]
    b: List[float] = field(default_factory=list)        # rhs: -const


def _master(params: InstanceParameters, cuts: CutModel, objective: Optional[np.ndarray] = None,
            objective_cap: Optional[float] = None):
    J, M = params.J, params.M
    base = np.concatenate([params.f_depot.astype(float), params.v_mode.astype(float) * params.T, [1.0]])
    C_lb = np.sum(params.s_initial, axis=1)
    bounds = [(float(C_lb[j]), None) for j in range(J)] + [(0.0, None)] * M + [(0.0, None)]
    A, b = list(cuts.A), list(cuts.b)
    if objective_cap is not None:
        A.append(base)
        b.append(objective_cap)
    obj = base if objective is None else objective
    if A:
        return linprog(obj, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method='highs')
    return linprog(obj, bounds=bounds, method='highs')


def kelley_cutting_plane(evaluator: RecourseEvaluator, library: Library, separation_ids: Sequence[int],
                         tol_abs: float = 1.0, max_iters: int = 200, cuts_per_iter: int = 1,
                         log=print) -> Dict:
    """Kelley/Benders cutting planes on (C, q) with separation over ``separation_ids``.

    Returns dollars.  ``tol_abs`` is the absolute UB-LB tolerance in dollars.
    """
    params = evaluator.params
    J, M = params.J, params.M
    cuts = CutModel()
    iterations = []
    UB, best = np.inf, None
    n_lp_start = evaluator.n_lp
    t0 = time.time()
    LB = -np.inf
    for it in range(1, max_iters + 1):
        m_start = time.time()
        mres = _master(params, cuts)
        if not mres.success:
            raise RuntimeError(f'master failed at iteration {it}: {mres.message}')
        C, q, LB = mres.x[:J], mres.x[J:J + M], float(mres.fun)
        master_time = time.time() - m_start
        s_start = time.time()
        vals = []
        for sid in separation_ids:
            ev = evaluator.evaluate(C, q, library.delta(sid), want_duals=True)
            if ev.status != 'optimal':
                raise RuntimeError(f'recourse LP failed for scenario {sid}: {ev.message}')
            vals.append((sid, ev))
        sep_time = time.time() - s_start
        vals.sort(key=lambda r: -r[1].objective)
        worst_sid, worst = vals[0]
        total = float(first_stage_cost(params, C, q) + worst.objective)
        if total < UB:
            UB, best = total, (C.copy(), q.copy(), worst_sid, worst.unmet_tons)
        gap = UB - LB
        iterations.append({
            'iteration': it, 'lower_bound': LB, 'upper_bound': UB, 'gap': gap,
            'C': C.tolist(), 'q': q.tolist(), 'cuts_before': len(cuts.A),
            'worst_scenario_id': int(worst_sid), 'worst_recourse': worst.objective,
            'master_time_s': master_time, 'separation_time_s': sep_time,
            'separation_lps': len(separation_ids),
        })
        log(f'  kelley it={it:3d} LB={LB/MILLION:.6f}M UB={UB/MILLION:.6f}M gap={gap/MILLION:.3e}M '
            f'q={np.round(q, 4).tolist()} cuts={len(cuts.A)}')
        if gap <= tol_abs:
            break
        seen = set()
        for sid, ev in vals[:cuts_per_iter]:
            const, gC, gq = evaluator.cut(library.delta(sid), ev.y_ub, ev.y_eq)
            key = tuple(np.round(np.concatenate([gC, gq, [const]]) / MILLION, 9))
            if key in seen:
                continue
            seen.add(key)
            cuts.A.append(np.concatenate([gC, gq, [-1.0]]))
            cuts.b.append(-const)
    C_best, q_best, worst_sid, worst_unmet = best
    # Optimal-face bounds of the converged master: the true optimal set is contained in it.
    face = []
    for i in range(J + M):
        e = np.zeros(J + M + 1); e[i] = 1.0
        lo = _master(params, cuts, objective=e, objective_cap=LB + tol_abs)
        hi = _master(params, cuts, objective=-e, objective_cap=LB + tol_abs)
        face.append([float(lo.fun), float(-hi.fun)])
    return {
        'iterations': iterations,
        'num_iterations': len(iterations),
        'num_cuts': len(cuts.A),
        'recourse_lps': evaluator.n_lp - n_lp_start,
        'lower_bound': LB, 'upper_bound': UB, 'gap': UB - LB,
        'C': C_best.tolist(), 'q': q_best.tolist(),
        'investment_cost': float(first_stage_cost(params, C_best, q_best)),
        'worst_scenario_id': int(worst_sid), 'worst_unmet_tons': float(worst_unmet),
        'optimal_face_minmax': face,
        'optimal_face_max_width': float(max(hi - lo for lo, hi in face)),
        'wall_time_s': time.time() - t0,
        'tolerance_abs': tol_abs, 'cuts_per_iteration': cuts_per_iter,
    }


# --------------------------------------------------------------------------- #
# Single-level worst-case MILP (dualize-and-linearize)
# --------------------------------------------------------------------------- #

def worst_case_milp(evaluator: RecourseEvaluator, library: Library, C, q,
                    big_m_M: Optional[float] = None, bound_lambda: bool = True,
                    time_limit_s: float = 1800.0, mip_rel_gap: float = 1e-9) -> Dict:
    """max_{pi in Pi, z} dual objective with Delta = cover z, budget Gamma, cell packing.

    Solved in millions of dollars.  ``big_m_M`` defaults to max p / 1e6, the bound
    of the lemma in Appendix A.  Returns dollars in ``worst_recourse``.
    """
    params, tpl = evaluator.params, evaluator.template
    M, T = params.M, params.T
    P_M = float(np.max(params.p_unmet)) / MILLION
    Lam = P_M if big_m_M is None else float(big_m_M)
    n_ub, n_eq = tpl.A_ub.shape[0], tpl.A_eq.shape[0]
    n_w, n_z = M * T, len(library.single_events)
    o_ub, o_eq, o_w, o_z = 0, n_ub, n_ub + n_eq, n_ub + n_eq + n_w
    n = o_z + n_z
    cover = event_cover_matrix(library)
    C = np.asarray(C, dtype=float); q = np.asarray(q, dtype=float)

    # Right-hand sides with Delta = 0 carry q_m in the mode rows.
    b_ub0 = evaluator.rhs(C, q, np.zeros((M, T), dtype=int))
    obj = np.zeros(n)
    obj[o_ub:o_ub + n_ub] = b_ub0
    obj[o_eq:o_eq + n_eq] = tpl.b_eq
    for m in range(M):
        obj[o_w + m * T: o_w + (m + 1) * T] = q[m]

    cons = []
    # (a) dual feasibility: A_ub' y_ub + A_eq' y_eq <= c   (costs in $M)
    A_dual = sparse.hstack([tpl.A_ub.T, tpl.A_eq.T, sparse.csr_matrix((tpl.A_ub.shape[1], n_w + n_z))]).tocsr()
    cons.append(LinearConstraint(A_dual, -np.inf, tpl.c / MILLION))
    # (b) w_mt + lambda_mt <= 0
    B = sparse.lil_matrix((n_w, n))
    for m in range(M):
        for t in range(T):
            B[m * T + t, o_w + m * T + t] = 1.0
            B[m * T + t, o_ub + tpl.mode_rows[m, t]] = 1.0
    cons.append(LinearConstraint(B.tocsr(), -np.inf, 0.0))
    # (c) w_mt - Lam * sum_e cover[cell, e] z_e <= 0
    Cm = sparse.hstack([sparse.csr_matrix((n_w, n_ub + n_eq)), sparse.identity(n_w), -Lam * cover]).tocsr()
    cons.append(LinearConstraint(Cm, -np.inf, 0.0))
    # (d) cell packing: sum_e cover[cell, e] z_e <= 1
    Dm = sparse.hstack([sparse.csr_matrix((n_w, o_z)), cover]).tocsr()
    cons.append(LinearConstraint(Dm, -np.inf, 1.0))
    # (e) budget: sum_e z_e <= Gamma
    Em = sparse.hstack([sparse.csr_matrix((1, o_z)), sparse.csr_matrix(np.ones((1, n_z)))]).tocsr()
    cons.append(LinearConstraint(Em, -np.inf, float(params.Gamma)))

    lb = np.full(n, -np.inf); ub = np.full(n, np.inf)
    ub[o_ub:o_ub + n_ub] = 0.0
    if bound_lambda:
        for m in range(M):
            lb[o_ub + tpl.mode_rows[m, :]] = -Lam
    lb[o_w:o_w + n_w] = 0.0; ub[o_w:o_w + n_w] = Lam
    lb[o_z:] = 0.0; ub[o_z:] = 1.0
    integrality = np.zeros(n); integrality[o_z:] = 1

    start = time.time()
    res = milp(-obj, constraints=cons, bounds=Bounds(lb, ub), integrality=integrality,
               options={'time_limit': time_limit_s, 'mip_rel_gap': mip_rel_gap})
    elapsed = time.time() - start
    if res.x is None:
        return {'status': int(res.status), 'message': res.message, 'time_s': elapsed, 'big_m_M': Lam}
    z = res.x[o_z:]
    chosen = [list(map(int, library.single_events[e])) for e in range(n_z) if z[e] > 0.5]
    lam = res.x[o_ub:o_ub + n_ub][tpl.mode_rows]
    Delta = np.zeros((M, T), dtype=int)
    for (m, s0, L) in chosen:
        Delta[m, s0:s0 + L] = 1
    # Independent re-solve of the primal recourse LP at the selected pattern.
    primal = evaluator.evaluate(C, q, Delta)
    return {
        'status': int(res.status), 'message': res.message,
        'worst_recourse': float(-res.fun * MILLION),
        'primal_recheck_recourse': primal.objective,
        'primal_recheck_unmet_tons': primal.unmet_tons,
        'selected_events': chosen,
        'selected_delta_cells': int(Delta.sum()),
        'min_lambda_M': float(lam.min()),
        'big_m_M': Lam, 'bound_lambda': bound_lambda,
        'binaries': n_z, 'time_s': elapsed,
        'mip_gap': float(getattr(res, 'mip_gap', np.nan)),
    }


# --------------------------------------------------------------------------- #
# CCG with a configurable separation set
# --------------------------------------------------------------------------- #

def ccg_loop(library: Library, initial_ids: Sequence[int], separation_ids: Optional[Sequence[int]],
             gap_tol_abs: float = 1.0, gap_tol_rel: float = 1e-8, max_iterations: int = 10,
             log=print) -> Dict:
    """Finite-library CCG mirroring ``run_burst_ccg.run_scenario_generation_ccg``.

    ``separation_ids=None`` separates over all encodings; otherwise over the given
    subset (e.g., the maximal representatives).  Returns dollars.
    """
    from lp_solver import solve_robust_model
    params, scenarios = library.params, library.scenarios
    active = sorted(set(int(i) for i in initial_ids))
    active_set = set(active)
    iterations, final = [], None
    t0 = time.time()
    sep_count = len(scenarios) if separation_ids is None else len(separation_ids)
    for it in range(1, max_iterations + 1):
        m_start = time.time()
        master = solve_robust_model(params, [scenarios[i] for i in active])
        if master.lp_status != 'optimal':
            raise RuntimeError(f'CCG master failed at iteration {it}: {master.lp_status}')
        master_wall = time.time() - m_start
        s_start = time.time()
        cert, _ = certify_fixed_design(
            params=params, scenarios=scenarios, C_depot=master.C_depot, q_mode=master.q_mode,
            design_name=f'ccg_iter_{it}', sampled_total_cost=master.total_cost,
            sampled_worst_operating_cost=master.worst_case_cost,
            scenario_ids=separation_ids, progress_interval=0)
        sep_wall = time.time() - s_start
        LB, UB = master.total_cost, cert.full_total_cost
        gap = UB - LB
        worst_id = cert.worst.scenario_id
        new = worst_id not in active_set
        tol = max(gap_tol_abs, gap_tol_rel * max(abs(UB), 1.0))
        converged = (not new) and gap <= tol
        iterations.append({
            'iteration': it, 'master_scenarios': len(active), 'added_scenario_id': worst_id if new else None,
            'lower_bound': LB, 'upper_bound': UB, 'gap': gap,
            'C': master.C_depot.tolist(), 'q': master.q_mode.tolist(),
            'worst_scenario_id': int(worst_id), 'worst_unmet_tons': cert.worst.unmet_tons,
            'master_solve_time_s': master.solve_time, 'master_wall_time_s': master_wall,
            'separation_time_s': sep_wall, 'separation_lps': sep_count, 'converged': converged,
        })
        log(f'  ccg it={it} LB={LB/MILLION:.6f}M UB={UB/MILLION:.6f}M gap={gap/MILLION:.3e}M worst={worst_id} new={new}')
        final = (master, cert)
        if converged:
            break
        if new:
            active_set.add(worst_id); active = sorted(active_set)
        else:
            break
    master, cert = final
    return {
        'iterations': iterations, 'num_iterations': len(iterations), 'converged': bool(iterations[-1]['converged']),
        'active_ids_final': active, 'separation_scenarios_per_iteration': sep_count,
        'recourse_lps': sep_count * len(iterations), 'master_solves': len(iterations),
        'total_cost': master.total_cost, 'worst_recourse': master.worst_case_cost,
        'C': master.C_depot.tolist(), 'q': master.q_mode.tolist(),
        'investment_cost': master.investment_cost,
        'final_upper_bound': cert.full_total_cost, 'worst_scenario_id': int(cert.worst.scenario_id),
        'worst_unmet_tons': cert.worst.unmet_tons, 'wall_time_s': time.time() - t0,
        'gap_tol_abs': gap_tol_abs, 'gap_tol_rel': gap_tol_rel, 'max_iterations': max_iterations,
    }


# --------------------------------------------------------------------------- #
# Vectorized extensive form over an arbitrary scenario list
# --------------------------------------------------------------------------- #

def build_extensive_form(evaluator: RecourseEvaluator, library: Library, scenario_ids: Sequence[int]):
    """Block-angular EF: variables [C(J), q(M), eta, recourse blocks]; rows built with kron.

    Same LP as ``lp_solver.build_lp_model_sparse`` up to row ordering.  Returns
    (c, A_ub, b_ub, A_eq, b_eq, bounds, dims).
    """
    params, tpl = evaluator.params, evaluator.template
    M, J, D, K, T = params.M, params.J, params.D, params.K, params.T
    N = len(scenario_ids)
    n_rec = tpl.A_ub.shape[1]
    n_first = J + M + 1
    n_vars = n_first + N * n_rec
    # objective
    c = np.zeros(n_vars)
    c[:J] = params.f_depot; c[J:J + M] = params.v_mode * T; c[J + M] = 1.0
    # equality rows: kron(I_N, A_eq)
    A_eq = sparse.hstack([sparse.csr_matrix((N * tpl.A_eq.shape[0], n_first)),
                          sparse.kron(sparse.identity(N, format='csr'), tpl.A_eq, format='csr')]).tocsr()
    b_eq = np.tile(tpl.b_eq, N)
    # inequality rows per scenario: [epigraph (1), template ub rows (144)] -> 145 rows
    n_ub_blk = 1 + tpl.A_ub.shape[0]
    epi_row = sparse.csr_matrix(tpl.c.reshape(1, -1))
    blk = sparse.vstack([epi_row, tpl.A_ub]).tocsr()
    A_rec = sparse.kron(sparse.identity(N, format='csr'), blk, format='csr')
    # linking columns: eta in epigraph rows (-1); C_j in inv rows (-1); q_m in mode rows (-(1-Delta))
    rows, cols, vals = [], [], []
    b_ub = np.zeros(N * n_ub_blk)
    for s, sid in enumerate(scenario_ids):
        Delta = library.delta(sid)
        base = s * n_ub_blk
        rows.append(base); cols.append(J + M); vals.append(-1.0)
        for t in range(T):
            for j in range(J):
                rows.append(base + 1 + evaluator.inv_rows[t, j]); cols.append(j); vals.append(-1.0)
        for m in range(M):
            for t in range(T):
                rows.append(base + 1 + tpl.mode_rows[m, t]); cols.append(J + m); vals.append(-(1.0 - Delta[m, t]))
        b_ub[base + 1 + evaluator.rec_rows] = tpl.b_ub_base[evaluator.rec_rows]
    A_link = sparse.csr_matrix((vals, (rows, cols)), shape=(N * n_ub_blk, n_first))
    A_ub = sparse.hstack([A_link, A_rec]).tocsr()
    # first-stage feasibility rows: -C_j <= -sum_k s0
    C_lb = np.sum(params.s_initial, axis=1)
    A_fs = sparse.csr_matrix((-np.ones(J), (np.arange(J), np.arange(J))), shape=(J, n_vars))
    A_ub = sparse.vstack([A_fs, A_ub]).tocsr()
    b_ub = np.concatenate([-C_lb.astype(float), b_ub])
    bounds = [(0, None)] * n_vars
    dims = {'variables': n_vars, 'inequalities': A_ub.shape[0], 'equalities': A_eq.shape[0],
            'nonzeros': int(A_ub.nnz + A_eq.nnz), 'scenarios': N}
    return c, A_ub, b_ub, A_eq, b_eq, bounds, dims


def solve_extensive_form(evaluator: RecourseEvaluator, library: Library, scenario_ids: Sequence[int],
                         method: str = 'highs', time_limit_s: Optional[float] = None, log=print) -> Dict:
    params = evaluator.params
    J, M = params.J, params.M
    t0 = time.time()
    c, A_ub, b_ub, A_eq, b_eq, bounds, dims = build_extensive_form(evaluator, library, scenario_ids)
    build_time = time.time() - t0
    log(f'  EF built: {dims} in {build_time:.1f}s')
    options = {'disp': False}
    if time_limit_s is not None:
        options['time_limit'] = float(time_limit_s)
    s0 = time.time()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method, options=options)
    solve_time = time.time() - s0
    out = {'dims': dims, 'build_time_s': build_time, 'solve_time_s': solve_time, 'method': method,
           'status': int(res.status), 'message': res.message, 'success': bool(res.success)}
    if res.success:
        out.update({'total_cost': float(res.fun), 'C': res.x[:J].tolist(), 'q': res.x[J:J + M].tolist(),
                    'eta': float(res.x[J + M]),
                    'investment_cost': float(first_stage_cost(params, res.x[:J], res.x[J:J + M]))})
    return out


# --------------------------------------------------------------------------- #
# Frozen design vectors
# --------------------------------------------------------------------------- #

def frozen_designs(project_root: str = PROJECT_ROOT) -> Dict[str, Dict]:
    """Design vectors of the recorded artifacts, read from results/08 and results/11 when present."""
    designs = {
        'sampled': {'C': list(FROZEN['sampled_C']), 'q': list(FROZEN['sampled_q']), 'source': 'FROZEN constants'},
        'exact': {'C': list(FROZEN['exact_C']), 'q': list(FROZEN['exact_q']), 'source': 'FROZEN constants'},
    }
    try:
        import pandas as pd
        e08 = pd.read_csv(os.path.join(project_root, 'results', '08_full_burst_certification',
                                       'full_certification_summary.csv'))
        row = e08[e08['design_label'] == 'burst'].iloc[0]
        designs['sampled'] = {'C': [float(row['C_depot_0']), float(row['C_depot_1'])],
                              'q': [float(row['q_mode_0']), float(row['q_mode_1']), float(row['q_mode_2'])],
                              'source': 'results/08_full_burst_certification/full_certification_summary.csv (burst row)'}
    except Exception as exc:  # pragma: no cover
        designs['sampled']['source'] += f' (results/08 unreadable: {exc})'
    try:
        with open(os.path.join(project_root, 'results', '11_dominance_pruned_oracle',
                               'dominance_pruned_result.json')) as stream:
            e11 = json.load(stream)
        designs['exact'] = {'C': [float(v) for v in e11['C_depot']], 'q': [float(v) for v in e11['q_mode']],
                            'source': 'results/11_dominance_pruned_oracle/dominance_pruned_result.json'}
    except Exception as exc:  # pragma: no cover
        designs['exact']['source'] += f' (results/11 unreadable: {exc})'
    return designs
