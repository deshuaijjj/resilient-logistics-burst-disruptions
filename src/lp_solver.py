"""
Deep-Space Logistics: LP Solver for Robust Optimization Model

This module solves the two-stage robust model using scenario enumeration.
Uses scipy.optimize.linprog for LP solving.

Model:
    min  sum_j f_j * C_j + sum_m v_m * q_m * T + eta
    s.t. eta >= Q(C, q; Delta^i)  for all scenarios i
         Q = recourse cost under scenario Delta^i
         Constraints (C1)-(C6)
"""

import numpy as np
from scipy.optimize import linprog
from scipy import sparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import time
from instance_generator import InstanceParameters, DisruptionScenario


@dataclass
class Solution:
    """Solution structure"""
    # First-stage decisions
    C_depot: np.ndarray      # Depot capacity (J,)
    q_mode: np.ndarray       # Mode contracted capacity (M,)

    # Objective components
    investment_cost: float
    worst_case_cost: float
    total_cost: float

    # Worst scenario info
    worst_scenario_id: int
    worst_scenario_events: List

    # Recourse decisions (for worst scenario)
    y_transport: Optional[np.ndarray] = None  # (M, J, K, T)
    s_inventory: Optional[np.ndarray] = None  # (J, K, T)
    x_delivery: Optional[np.ndarray] = None   # (J, D, K, T)
    u_unmet: Optional[np.ndarray] = None      # (D, K, T)

    # Computational info
    solve_time: float = 0.0
    num_scenarios: int = 0
    lp_status: str = ""


def build_lp_model(params: InstanceParameters,
                   scenarios: List[DisruptionScenario],
                   fixed_C: Optional[np.ndarray] = None,
                   fixed_q: Optional[np.ndarray] = None) -> Tuple:
    """
    Build LP model in scipy.optimize.linprog format

    Args:
        params: Problem instance
        scenarios: List of disruption scenarios
        fixed_C: If provided, fix depot capacity (for redundancy-only baseline)
        fixed_q: If provided, fix mode capacity (for inventory-only baseline)

    Returns:
        (c, A_ub, b_ub, A_eq, b_eq, bounds) for linprog
    """
    M, J, D, K, T = params.M, params.J, params.D, params.K, params.T
    N_scen = len(scenarios)

    # Decision variables indexing:
    # [C_depot (J), q_mode (M), eta (1),
    #  for each scenario s:
    #    y_mjkt (M*J*K*T), s_jkt (J*K*T), x_jdkt (J*D*K*T), u_dkt (D*K*T)]

    n_C = J
    n_q = M
    n_eta = 1
    n_y = M * J * K * T
    n_s = J * K * T
    n_x = J * D * K * T
    n_u = D * K * T
    n_recourse_per_scen = n_y + n_s + n_x + n_u

    n_vars = n_C + n_q + n_eta + N_scen * n_recourse_per_scen

    # Objective: min sum_j f_j * C_j + sum_m v_m * q_m * T + eta
    c = np.zeros(n_vars)

    offset = 0
    c[offset:offset+J] = params.f_depot
    offset += J

    c[offset:offset+M] = params.v_mode * T
    offset += M

    c[offset] = 1.0  # eta
    offset += 1

    # Recourse variables have cost embedded in eta constraints
    # (their costs go into recourse objective Q)

    print(f"Building LP with {n_vars} variables, {N_scen} scenarios...")

    # Bounds: all non-negative, except C and q might have upper bounds
    bounds = [(0, None)] * n_vars

    # Fix C or q if needed for baselines
    if fixed_C is not None:
        for j in range(J):
            bounds[j] = (fixed_C[j], fixed_C[j])

    if fixed_q is not None:
        for m in range(M):
            bounds[J + m] = (fixed_q[m], fixed_q[m])

    # Constraints
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []

    # First-stage physical feasibility: depot capacity must be able to hold
    # inventory that is already present before any recourse action is taken.
    for j in range(J):
        row = np.zeros(n_vars)
        row[j] = -1.0
        A_ub.append(row)
        b_ub.append(-float(np.sum(params.s_initial[j, :])))

    # For each scenario, add constraints
    for s_id, scenario in enumerate(scenarios):
        Delta = scenario.to_delta_matrix(M, T)

        # Compute variable offset for this scenario
        scen_offset = n_C + n_q + n_eta + s_id * n_recourse_per_scen
        y_offset = scen_offset
        s_offset = y_offset + n_y
        x_offset = s_offset + n_s
        u_offset = x_offset + n_x

        # (1) eta >= Q(Delta) constraint
        # Q = sum_t (sum_mjk c_m * y_mjkt + sum_jk h_jk * s_jkt + sum_dk p_dk * u_dkt)
        row = np.zeros(n_vars)
        row[n_C + n_q] = 1  # eta

        # Subtract Q components
        for t in range(T):
            for m in range(M):
                for j in range(J):
                    for k in range(K):
                        idx = y_offset + (((m*J + j)*K + k)*T + t)
                        row[idx] = -params.c_mode[m]

            for j in range(J):
                for k in range(K):
                    idx = s_offset + ((j*K + k)*T + t)
                    row[idx] = -params.h_depot[j, k]

            for d in range(D):
                for k in range(K):
                    idx = u_offset + ((d*K + k)*T + t)
                    row[idx] = -params.p_unmet[d, k]

        # CRITICAL FIX: Negate row to get eta >= recourse_cost (not <=)
        A_ub.append(-row)  # Change sign: eta >= c*y + h*s + p*u
        b_ub.append(0)

        # (2) Inventory dynamics (C1): s_jkt = s_jk,t-1 + arrivals - departures
        for t in range(T):
            for j in range(J):
                for k in range(K):
                    row = np.zeros(n_vars)

                    # s_jkt
                    idx_s_curr = s_offset + ((j*K + k)*T + t)
                    row[idx_s_curr] = 1

                    # - s_jk,t-1 (or initial inventory if t=0)
                    rhs = 0
                    if t > 0:
                        idx_s_prev = s_offset + ((j*K + k)*T + (t-1))
                        row[idx_s_prev] = -1
                    else:
                        # At t=0, use initial inventory
                        rhs = params.s_initial[j, k]

                    # - arrivals: sum_m y_mjk,t-tau
                    for m in range(M):
                        tau = int(np.ceil(params.tau_mode_depot[m, j]))
                        t_launch = t - tau
                        if t_launch >= 0:
                            idx_y = y_offset + (((m*J + j)*K + k)*T + t_launch)
                            row[idx_y] = -1

                    # + departures: sum_d x_jdkt
                    for d in range(D):
                        idx_x = x_offset + (((j*D + d)*K + k)*T + t)
                        row[idx_x] = 1

                    A_eq.append(row)
                    b_eq.append(rhs)

        # (3) Inventory capacity (C2): sum_k s_jkt <= C_j
        for t in range(T):
            for j in range(J):
                row = np.zeros(n_vars)
                row[j] = -1  # -C_j

                for k in range(K):
                    idx_s = s_offset + ((j*K + k)*T + t)
                    row[idx_s] = 1  # +s_jkt

                A_ub.append(row)
                b_ub.append(0)

        # (4) Mode capacity (C3): sum_jk y_mjkt <= q_m * (1 - Delta_mt)
        for t in range(T):
            for m in range(M):
                row = np.zeros(n_vars)
                row[J + m] = -(1 - Delta[m, t])  # -q_m * (1-Delta_mt)

                for j in range(J):
                    for k in range(K):
                        idx_y = y_offset + (((m*J + j)*K + k)*T + t)
                        row[idx_y] = 1  # +y_mjkt

                A_ub.append(row)
                b_ub.append(0)

        # (5) Reception capacity (C4): sum_jk x_jdk,t-tau <= R_max_dt
        for t in range(T):
            for d in range(D):
                row = np.zeros(n_vars)

                for j in range(J):
                    tau = int(np.ceil(params.tau_depot_dest[j, d]))
                    t_depart = t - tau
                    if t_depart >= 0:
                        for k in range(K):
                            idx_x = x_offset + (((j*D + d)*K + k)*T + t_depart)
                            row[idx_x] = 1

                A_ub.append(row)
                b_ub.append(params.R_max[d, t])

        # (6) Demand satisfaction (C5): sum_j x_jdk,t-tau + u_dkt = D_dkt
        for t in range(T):
            for d in range(D):
                for k in range(K):
                    row = np.zeros(n_vars)

                    for j in range(J):
                        tau = int(np.ceil(params.tau_depot_dest[j, d]))
                        t_depart = t - tau
                        if t_depart >= 0:
                            idx_x = x_offset + (((j*D + d)*K + k)*T + t_depart)
                            row[idx_x] = 1

                    idx_u = u_offset + ((d*K + k)*T + t)
                    row[idx_u] = 1

                    A_eq.append(row)
                    b_eq.append(params.D_demand[d, k, t])

    # Convert to numpy arrays
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    A_eq = np.array(A_eq) if A_eq else None
    b_eq = np.array(b_eq) if b_eq else None

    print(f"  Variables: {n_vars}")
    print(f"  Inequality constraints: {len(b_ub) if b_ub is not None else 0}")
    print(f"  Equality constraints: {len(b_eq) if b_eq is not None else 0}")

    return c, A_ub, b_ub, A_eq, b_eq, bounds, (n_C, n_q, n_eta, n_recourse_per_scen)


def build_lp_model_sparse(params: InstanceParameters,
                          scenarios: List[DisruptionScenario],
                          fixed_C: Optional[np.ndarray] = None,
                          fixed_q: Optional[np.ndarray] = None) -> Tuple:
    """
    Build the same LP as build_lp_model, but using sparse matrices.

    Scenario enumeration creates a block-angular model with very low row density.
    Keeping A_ub/A_eq dense makes moderate samples exceed memory, while scipy's
    HiGHS interface accepts CSR sparse matrices directly.
    """
    M, J, D, K, T = params.M, params.J, params.D, params.K, params.T
    N_scen = len(scenarios)

    n_C = J
    n_q = M
    n_eta = 1
    n_y = M * J * K * T
    n_s = J * K * T
    n_x = J * D * K * T
    n_u = D * K * T
    n_recourse_per_scen = n_y + n_s + n_x + n_u
    n_vars = n_C + n_q + n_eta + N_scen * n_recourse_per_scen

    c = np.zeros(n_vars)
    offset = 0
    c[offset:offset+J] = params.f_depot
    offset += J
    c[offset:offset+M] = params.v_mode * T
    offset += M
    c[offset] = 1.0

    bounds = [(0, None)] * n_vars
    if fixed_C is not None:
        for j in range(J):
            bounds[j] = (fixed_C[j], fixed_C[j])
    if fixed_q is not None:
        for m in range(M):
            bounds[J + m] = (fixed_q[m], fixed_q[m])

    n_ub_first_stage = J
    n_ub_per_scen = 1 + (T * J) + (T * M) + (T * D)
    n_eq_per_scen = (T * J * K) + (T * D * K)
    n_ub_total = n_ub_first_stage + N_scen * n_ub_per_scen
    n_eq_total = N_scen * n_eq_per_scen

    print(f"Building sparse LP with {n_vars} variables, {N_scen} scenarios...")

    A_ub = sparse.lil_matrix((n_ub_total, n_vars), dtype=float)
    b_ub = np.zeros(n_ub_total)
    A_eq = sparse.lil_matrix((n_eq_total, n_vars), dtype=float)
    b_eq = np.zeros(n_eq_total)

    ub_row = 0
    eq_row = 0

    # First-stage physical feasibility: depot capacity must be at least the
    # inventory already stored at each depot at the start of the horizon.
    for j in range(J):
        A_ub[ub_row, j] = -1.0
        b_ub[ub_row] = -float(np.sum(params.s_initial[j, :]))
        ub_row += 1

    for s_id, scenario in enumerate(scenarios):
        Delta = scenario.to_delta_matrix(M, T)

        scen_offset = n_C + n_q + n_eta + s_id * n_recourse_per_scen
        y_offset = scen_offset
        s_offset = y_offset + n_y
        x_offset = s_offset + n_s
        u_offset = x_offset + n_x

        # eta >= Q(Delta) -> Q(Delta) - eta <= 0
        A_ub[ub_row, n_C + n_q] = -1.0
        for t in range(T):
            for m in range(M):
                for j in range(J):
                    for k in range(K):
                        idx = y_offset + (((m * J + j) * K + k) * T + t)
                        A_ub[ub_row, idx] = params.c_mode[m]
            for j in range(J):
                for k in range(K):
                    idx = s_offset + ((j * K + k) * T + t)
                    A_ub[ub_row, idx] = params.h_depot[j, k]
            for d in range(D):
                for k in range(K):
                    idx = u_offset + ((d * K + k) * T + t)
                    A_ub[ub_row, idx] = params.p_unmet[d, k]
        ub_row += 1

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

        # Inventory capacity
        for t in range(T):
            for j in range(J):
                A_ub[ub_row, j] = -1.0
                for k in range(K):
                    idx_s = s_offset + ((j * K + k) * T + t)
                    A_ub[ub_row, idx_s] = 1.0
                ub_row += 1

        # Mode capacity under disruption
        for t in range(T):
            for m in range(M):
                A_ub[ub_row, J + m] = -(1 - Delta[m, t])
                for j in range(J):
                    for k in range(K):
                        idx_y = y_offset + (((m * J + j) * K + k) * T + t)
                        A_ub[ub_row, idx_y] = 1.0
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

        # Demand satisfaction
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

    A_ub = A_ub.tocsr()
    A_eq = A_eq.tocsr()

    print(f"  Variables: {n_vars}")
    print(f"  Inequality constraints: {n_ub_total}")
    print(f"  Equality constraints: {n_eq_total}")
    print(f"  Nonzeros: A_ub={A_ub.nnz}, A_eq={A_eq.nnz}")

    return c, A_ub, b_ub, A_eq, b_eq, bounds, (n_C, n_q, n_eta, n_recourse_per_scen)


def solve_robust_model(params: InstanceParameters,
                       scenarios: List[DisruptionScenario],
                       fixed_C: Optional[np.ndarray] = None,
                       fixed_q: Optional[np.ndarray] = None,
                       method: str = 'highs',
                       use_sparse: bool = True) -> Solution:
    """
    Solve the robust optimization model

    Args:
        params: Problem instance
        scenarios: List of disruption scenarios
        fixed_C: Fix depot capacity (for baselines)
        fixed_q: Fix mode capacity (for baselines)
        method: LP solver method ('highs', 'interior-point', etc.)

    Returns:
        Solution object
    """
    print("\n" + "="*60)
    print(f"Solving Robust Model ({len(scenarios)} scenarios)")
    print("="*60)

    start_time = time.time()

    # Build LP
    builder = build_lp_model_sparse if use_sparse else build_lp_model
    c, A_ub, b_ub, A_eq, b_eq, bounds, dims = builder(
        params, scenarios, fixed_C, fixed_q
    )
    n_C, n_q, n_eta, n_recourse_per_scen = dims

    # Solve LP
    print("\nSolving LP...")
    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method=method, options={'disp': True}
    )

    solve_time = time.time() - start_time

    if not result.success:
        print(f"\nWARNING: LP solver failed: {result.message}")
        return Solution(
            C_depot=np.zeros(params.J),
            q_mode=np.zeros(params.M),
            investment_cost=np.inf,
            worst_case_cost=np.inf,
            total_cost=np.inf,
            worst_scenario_id=-1,
            worst_scenario_events=[],
            solve_time=solve_time,
            num_scenarios=len(scenarios),
            lp_status=result.message
        )

    # Extract solution
    x_opt = result.x

    C_depot = x_opt[:n_C]
    q_mode = x_opt[n_C:n_C+n_q]
    eta = x_opt[n_C+n_q]

    investment_cost = np.sum(params.f_depot * C_depot) + np.sum(params.v_mode * q_mode * params.T)
    worst_case_cost = eta
    total_cost = result.fun

    print(f"\nSolution found:")
    print(f"  Total cost: ${total_cost:,.0f}")
    print(f"  Investment cost: ${investment_cost:,.0f}")
    print(f"  Worst-case operating cost: ${worst_case_cost:,.0f}")
    print(f"  Depot capacity: {C_depot}")
    print(f"  Mode capacity: {q_mode}")
    print(f"  Solve time: {solve_time:.2f}s")

    # Find worst scenario (the one with highest recourse cost)
    worst_scen_id = 0
    worst_recourse = -np.inf

    for s_id in range(len(scenarios)):
        scen_offset = n_C + n_q + n_eta + s_id * n_recourse_per_scen
        y_offset = scen_offset
        s_offset = y_offset + params.M * params.J * params.K * params.T
        x_offset = s_offset + params.J * params.K * params.T
        u_offset = x_offset + params.J * params.D * params.K * params.T

        # Compute recourse cost for this scenario
        recourse = 0
        for t in range(params.T):
            for m in range(params.M):
                for j in range(params.J):
                    for k in range(params.K):
                        idx = y_offset + (((m*params.J + j)*params.K + k)*params.T + t)
                        recourse += params.c_mode[m] * x_opt[idx]

            for j in range(params.J):
                for k in range(params.K):
                    idx = s_offset + ((j*params.K + k)*params.T + t)
                    recourse += params.h_depot[j, k] * x_opt[idx]

            for d in range(params.D):
                for k in range(params.K):
                    idx = u_offset + ((d*params.K + k)*params.T + t)
                    recourse += params.p_unmet[d, k] * x_opt[idx]

        if recourse > worst_recourse:
            worst_recourse = recourse
            worst_scen_id = s_id

    print(f"\nWorst scenario: #{worst_scen_id}")
    print(f"  Events: {scenarios[worst_scen_id].events}")
    print(f"  Recourse cost: ${worst_recourse:,.0f}")

    worst_offset = n_C + n_q + n_eta + worst_scen_id * n_recourse_per_scen
    n_y = params.M * params.J * params.K * params.T
    n_s = params.J * params.K * params.T
    n_x = params.J * params.D * params.K * params.T
    n_u = params.D * params.K * params.T
    y_transport = x_opt[worst_offset:worst_offset + n_y].reshape(
        params.M, params.J, params.K, params.T
    )
    s_inventory = x_opt[worst_offset + n_y:worst_offset + n_y + n_s].reshape(
        params.J, params.K, params.T
    )
    x_delivery = x_opt[worst_offset + n_y + n_s:worst_offset + n_y + n_s + n_x].reshape(
        params.J, params.D, params.K, params.T
    )
    u_unmet = x_opt[worst_offset + n_y + n_s + n_x:worst_offset + n_y + n_s + n_x + n_u].reshape(
        params.D, params.K, params.T
    )
    print(f"  Unmet demand: {np.sum(u_unmet):,.2f} tons")

    return Solution(
        C_depot=C_depot,
        q_mode=q_mode,
        investment_cost=investment_cost,
        worst_case_cost=worst_case_cost,
        total_cost=total_cost,
        worst_scenario_id=worst_scen_id,
        worst_scenario_events=scenarios[worst_scen_id].events,
        y_transport=y_transport,
        s_inventory=s_inventory,
        x_delivery=x_delivery,
        u_unmet=u_unmet,
        solve_time=solve_time,
        num_scenarios=len(scenarios),
        lp_status="optimal"
    )


if __name__ == "__main__":
    from instance_generator import generate_small_instance, generate_burst_scenarios

    print("Deep-Space Logistics: LP Solver Test")
    print("="*60)

    # Generate instance
    params = generate_small_instance()

    # Generate scenarios (limit to small number for testing)
    scenarios = generate_burst_scenarios(params, max_scenarios=50)

    # Solve
    solution = solve_robust_model(params, scenarios)

    print("\n" + "="*60)
    print("Solver test complete!")
    print("="*60)
