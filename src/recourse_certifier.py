"""
Fixed-design recourse evaluation over burst disruption scenarios.

The robust design scripts optimize first-stage capacities over sampled scenario
sets. This module freezes those first-stage decisions and evaluates each burst
scenario independently, which is much smaller than building one monolithic
extensive form for the full scenario library.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from instance_generator import DisruptionScenario, InstanceParameters


@dataclass
class RecourseResult:
    """Evaluation result for one fixed-design recourse LP."""

    scenario_id: int
    events: List
    operating_cost: float
    unmet_tons: float
    unmet_by_commodity: List[float]
    status: str
    solve_time_s: float
    message: str = ""


@dataclass
class CertificationResult:
    """Full-library certificate for a fixed first-stage design."""

    design_name: str
    investment_cost: float
    sampled_total_cost: Optional[float]
    sampled_worst_operating_cost: Optional[float]
    full_worst_operating_cost: float
    full_total_cost: float
    certification_gap: Optional[float]
    certification_gap_pct: Optional[float]
    operating_gap: Optional[float]
    worst: RecourseResult
    evaluated_scenarios: int
    eval_time_s: float
    status: str


@dataclass
class RecourseTemplate:
    """Scenario-independent pieces of the fixed-design recourse LP."""

    params: InstanceParameters
    C_depot: np.ndarray
    q_mode: np.ndarray
    c: np.ndarray
    A_ub: sparse.csr_matrix
    b_ub_base: np.ndarray
    A_eq: sparse.csr_matrix
    b_eq: np.ndarray
    bounds: List[tuple]
    mode_rows: np.ndarray
    offsets: tuple[int, int, int, int]
    sizes: tuple[int, int, int, int]


def first_stage_cost(params: InstanceParameters, C: np.ndarray, q: np.ndarray) -> float:
    """Return depot plus contracted-mode investment cost."""
    return float(np.sum(params.f_depot * C) + np.sum(params.v_mode * q * params.T))


def scenario_key(scenario: DisruptionScenario) -> tuple:
    """Stable key for comparing scenarios by event content."""
    return tuple(tuple(int(v) for v in event) for event in scenario.events)


def deterministic_even_sample_indices(num_scenarios: int, sample_size: int) -> List[int]:
    """Keep scenario 0 and evenly sample the remaining full-library indices."""
    if sample_size >= num_scenarios:
        return list(range(num_scenarios))
    if sample_size <= 1:
        return [0]
    indices = np.linspace(1, num_scenarios - 1, sample_size - 1, dtype=int)
    return [0] + [int(i) for i in indices]


def _events_for_json(events: List) -> List[List[int]]:
    return [[int(v) for v in event] for event in events]


def build_recourse_template(
    params: InstanceParameters,
    C_depot: Sequence[float],
    q_mode: Sequence[float],
) -> RecourseTemplate:
    """Build the constant recourse LP matrices for a fixed first-stage design."""
    M, J, D, K, T = params.M, params.J, params.D, params.K, params.T
    C = np.asarray(C_depot, dtype=float)
    q = np.asarray(q_mode, dtype=float)
    if C.shape != (J,):
        raise ValueError(f'C_depot must have shape {(J,)}, got {C.shape}')
    if q.shape != (M,):
        raise ValueError(f'q_mode must have shape {(M,)}, got {q.shape}')

    initial_required = np.sum(params.s_initial, axis=1)
    if np.any(C + 1e-7 < initial_required):
        raise ValueError(
            'Fixed depot capacity is below initial inventory: '
            f'C={C.tolist()}, required={initial_required.tolist()}'
        )

    n_y = M * J * K * T
    n_s = J * K * T
    n_x = J * D * K * T
    n_u = D * K * T
    y_offset = 0
    s_offset = y_offset + n_y
    x_offset = s_offset + n_s
    u_offset = x_offset + n_x
    n_vars = n_y + n_s + n_x + n_u

    c = np.zeros(n_vars)
    for t in range(T):
        for m in range(M):
            for j in range(J):
                for k in range(K):
                    idx = y_offset + (((m * J + j) * K + k) * T + t)
                    c[idx] = params.c_mode[m]
        for j in range(J):
            for k in range(K):
                idx = s_offset + ((j * K + k) * T + t)
                c[idx] = params.h_depot[j, k]
        for d in range(D):
            for k in range(K):
                idx = u_offset + ((d * K + k) * T + t)
                c[idx] = params.p_unmet[d, k]

    n_ub = (T * J) + (T * M) + (T * D)
    n_eq = (T * J * K) + (T * D * K)
    A_ub = sparse.lil_matrix((n_ub, n_vars), dtype=float)
    b_ub = np.zeros(n_ub)
    A_eq = sparse.lil_matrix((n_eq, n_vars), dtype=float)
    b_eq = np.zeros(n_eq)
    mode_rows = np.zeros((M, T), dtype=int)

    ub_row = 0
    eq_row = 0

    # Inventory dynamics
    for t in range(T):
        for j in range(J):
            for k in range(K):
                idx_s_curr = s_offset + ((j * K + k) * T + t)
                A_eq[eq_row, idx_s_curr] = 1.0
                if t > 0:
                    idx_s_prev = s_offset + ((j * K + k) * T + (t - 1))
                    A_eq[eq_row, idx_s_prev] = -1.0
                else:
                    b_eq[eq_row] = params.s_initial[j, k]

                for m in range(M):
                    tau = int(np.ceil(params.tau_mode_depot[m, j]))
                    t_launch = t - tau
                    if t_launch >= 0:
                        idx_y = y_offset + (((m * J + j) * K + k) * T + t_launch)
                        A_eq[eq_row, idx_y] = -1.0

                for d in range(D):
                    idx_x = x_offset + (((j * D + d) * K + k) * T + t)
                    A_eq[eq_row, idx_x] = 1.0

                eq_row += 1

    # Demand equality
    for t in range(T):
        for d in range(D):
            for k in range(K):
                for j in range(J):
                    tau = int(np.ceil(params.tau_depot_dest[j, d]))
                    t_depart = t - tau
                    if t_depart >= 0:
                        idx_x = x_offset + (((j * D + d) * K + k) * T + t_depart)
                        A_eq[eq_row, idx_x] = 1.0
                idx_u = u_offset + ((d * K + k) * T + t)
                A_eq[eq_row, idx_u] = 1.0
                b_eq[eq_row] = params.D_demand[d, k, t]
                eq_row += 1

    # Inventory capacity
    for t in range(T):
        for j in range(J):
            for k in range(K):
                idx_s = s_offset + ((j * K + k) * T + t)
                A_ub[ub_row, idx_s] = 1.0
            b_ub[ub_row] = C[j]
            ub_row += 1

    # Mode capacity; RHS is scenario dependent.
    for t in range(T):
        for m in range(M):
            mode_rows[m, t] = ub_row
            for j in range(J):
                for k in range(K):
                    idx_y = y_offset + (((m * J + j) * K + k) * T + t)
                    A_ub[ub_row, idx_y] = 1.0
            b_ub[ub_row] = q[m]
            ub_row += 1

    # Destination reception capacity
    for t in range(T):
        for d in range(D):
            for j in range(J):
                tau = int(np.ceil(params.tau_depot_dest[j, d]))
                t_depart = t - tau
                if t_depart >= 0:
                    for k in range(K):
                        idx_x = x_offset + (((j * D + d) * K + k) * T + t_depart)
                        A_ub[ub_row, idx_x] = 1.0
            b_ub[ub_row] = params.R_max[d, t]
            ub_row += 1

    return RecourseTemplate(
        params=params,
        C_depot=C,
        q_mode=q,
        c=c,
        A_ub=A_ub.tocsr(),
        b_ub_base=b_ub,
        A_eq=A_eq.tocsr(),
        b_eq=b_eq,
        bounds=[(0, None)] * n_vars,
        mode_rows=mode_rows,
        offsets=(y_offset, s_offset, x_offset, u_offset),
        sizes=(n_y, n_s, n_x, n_u),
    )


def solve_recourse_for_scenario(
    template: RecourseTemplate,
    scenario_id: int,
    scenario: DisruptionScenario,
    method: str = 'highs',
) -> RecourseResult:
    """Solve one fixed-design recourse LP."""
    params = template.params
    start = time.time()
    b_ub = template.b_ub_base.copy()
    Delta = scenario.to_delta_matrix(params.M, params.T)
    for m in range(params.M):
        for t in range(params.T):
            b_ub[template.mode_rows[m, t]] = template.q_mode[m] * (1 - Delta[m, t])

    result = linprog(
        template.c,
        A_ub=template.A_ub,
        b_ub=b_ub,
        A_eq=template.A_eq,
        b_eq=template.b_eq,
        bounds=template.bounds,
        method=method,
        options={'disp': False},
    )
    solve_time = time.time() - start
    if not result.success:
        return RecourseResult(
            scenario_id=scenario_id,
            events=_events_for_json(scenario.events),
            operating_cost=float('inf'),
            unmet_tons=float('inf'),
            unmet_by_commodity=[float('inf')] * params.K,
            status='failed',
            solve_time_s=solve_time,
            message=result.message,
        )

    _, _, _, u_offset = template.offsets
    _, _, _, n_u = template.sizes
    u = result.x[u_offset:u_offset + n_u].reshape(params.D, params.K, params.T)
    unmet_by_commodity = np.sum(u, axis=(0, 2))
    return RecourseResult(
        scenario_id=scenario_id,
        events=_events_for_json(scenario.events),
        operating_cost=float(result.fun),
        unmet_tons=float(np.sum(u)),
        unmet_by_commodity=[float(x) for x in unmet_by_commodity],
        status='optimal',
        solve_time_s=solve_time,
        message='',
    )


def certify_fixed_design(
    params: InstanceParameters,
    scenarios: Sequence[DisruptionScenario],
    C_depot: Sequence[float],
    q_mode: Sequence[float],
    design_name: str = 'design',
    sampled_total_cost: Optional[float] = None,
    sampled_worst_operating_cost: Optional[float] = None,
    scenario_ids: Optional[Iterable[int]] = None,
    method: str = 'highs',
    progress_interval: int = 500,
    save_scenario_costs_path: Optional[str] = None,
) -> tuple[CertificationResult, Optional[pd.DataFrame]]:
    """Evaluate a fixed design on a scenario set and return the worst case."""
    template = build_recourse_template(params, C_depot, q_mode)
    if scenario_ids is None:
        ids = list(range(len(scenarios)))
    else:
        ids = [int(i) for i in scenario_ids]

    start = time.time()
    worst: Optional[RecourseResult] = None
    rows = [] if save_scenario_costs_path else None
    for count, scenario_id in enumerate(ids, start=1):
        recourse = solve_recourse_for_scenario(
            template,
            scenario_id,
            scenarios[scenario_id],
            method=method,
        )
        if (
            worst is None
            or recourse.operating_cost > worst.operating_cost + 1e-7
            or (
                abs(recourse.operating_cost - worst.operating_cost) <= 1e-7
                and recourse.scenario_id < worst.scenario_id
            )
        ):
            worst = recourse

        if rows is not None:
            rows.append({
                'scenario_id': recourse.scenario_id,
                'events': json.dumps(recourse.events),
                'operating_cost': recourse.operating_cost,
                'operating_cost_M': recourse.operating_cost / 1e6,
                'unmet_tons': recourse.unmet_tons,
                'status': recourse.status,
                'solve_time_s': recourse.solve_time_s,
            })
            if count % progress_interval == 0:
                pd.DataFrame(rows).to_csv(save_scenario_costs_path, index=False)

        if progress_interval and count % progress_interval == 0:
            print(
                f'  {design_name}: evaluated {count:,}/{len(ids):,}; '
                f'current worst scenario {worst.scenario_id} '
                f'(${worst.operating_cost / 1e6:,.2f}M)'
            )

    assert worst is not None
    if rows is not None:
        scenario_costs = pd.DataFrame(rows)
        scenario_costs.to_csv(save_scenario_costs_path, index=False)
    else:
        scenario_costs = None

    investment = first_stage_cost(params, template.C_depot, template.q_mode)
    full_total = investment + worst.operating_cost
    gap = None if sampled_total_cost is None else full_total - sampled_total_cost
    gap_pct = None
    if gap is not None and sampled_total_cost and np.isfinite(sampled_total_cost):
        gap_pct = 100.0 * gap / sampled_total_cost
    operating_gap = None
    if sampled_worst_operating_cost is not None:
        operating_gap = worst.operating_cost - sampled_worst_operating_cost
    status = 'optimal' if worst.status == 'optimal' else 'failed'

    cert = CertificationResult(
        design_name=design_name,
        investment_cost=investment,
        sampled_total_cost=sampled_total_cost,
        sampled_worst_operating_cost=sampled_worst_operating_cost,
        full_worst_operating_cost=worst.operating_cost,
        full_total_cost=full_total,
        certification_gap=gap,
        certification_gap_pct=gap_pct,
        operating_gap=operating_gap,
        worst=worst,
        evaluated_scenarios=len(ids),
        eval_time_s=time.time() - start,
        status=status,
    )
    return cert, scenario_costs


def certification_to_summary_row(
    cert: CertificationResult,
    C_depot: Sequence[float],
    q_mode: Sequence[float],
    source_path: str = '',
    design_sample_size: Optional[int] = None,
) -> dict:
    """Flatten a certification result for CSV output."""
    C = np.asarray(C_depot, dtype=float)
    q = np.asarray(q_mode, dtype=float)
    return {
        'design_label': cert.design_name,
        'design_source': source_path,
        'design_sample_size': design_sample_size,
        'certification_scenarios': cert.evaluated_scenarios,
        'investment_cost_M': cert.investment_cost / 1e6,
        'optimization_total_cost_M': (
            None if cert.sampled_total_cost is None else cert.sampled_total_cost / 1e6
        ),
        'optimization_worst_operating_cost_M': (
            None if cert.sampled_worst_operating_cost is None
            else cert.sampled_worst_operating_cost / 1e6
        ),
        'certified_total_cost_M': cert.full_total_cost / 1e6,
        'certified_worst_operating_cost_M': cert.full_worst_operating_cost / 1e6,
        'cost_gap_M': None if cert.certification_gap is None else cert.certification_gap / 1e6,
        'relative_gap_pct': cert.certification_gap_pct,
        'operating_gap_M': None if cert.operating_gap is None else cert.operating_gap / 1e6,
        'C_depot_0': C[0] if len(C) > 0 else 0.0,
        'C_depot_1': C[1] if len(C) > 1 else 0.0,
        'q_mode_0': q[0] if len(q) > 0 else 0.0,
        'q_mode_1': q[1] if len(q) > 1 else 0.0,
        'q_mode_2': q[2] if len(q) > 2 else 0.0,
        'worst_scenario_id': cert.worst.scenario_id,
        'num_worst_events': len(cert.worst.events),
        'worst_events_json': json.dumps(cert.worst.events),
        'worst_unmet_tons': cert.worst.unmet_tons,
        'eval_time_s': cert.eval_time_s,
        'lp_status': cert.status,
    }


def certification_to_jsonable(cert: CertificationResult) -> dict:
    """Return a JSON-serializable representation of a certification result."""
    data = asdict(cert)
    return data
