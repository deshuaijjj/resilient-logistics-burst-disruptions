"""Focused tests for the September 2026 revision helpers on a small instance.

The instance keeps the study model's structure (3 modes, 2 depots, 2
commodities, Gamma = 2) but uses an 8-month horizon so that the complete
library (a few hundred encodings) can be enumerated in seconds.
"""

import os
import sys
from dataclasses import replace

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from instance_generator import generate_small_instance  # noqa: E402
from lp_solver import solve_robust_model  # noqa: E402
from recourse_certifier import certify_fixed_design  # noqa: E402
from revision_common import (  # noqa: E402
    RecourseEvaluator, build_library, kelley_cutting_plane, maximal_within, solve_extensive_form,
    worst_case_milp,
)


@pytest.fixture(scope='module')
def small_lib():
    base = generate_small_instance()
    T = 8
    R_max = np.full((base.D, T), 700.0)
    D_demand = np.zeros((base.D, base.K, T))
    D_demand[0, 0, :] = 200
    D_demand[0, 1, :] = 100
    D_demand[0, 0, 3] = 400
    params = replace(base, T=T, R_max=R_max, D_demand=D_demand,
                     L_min=np.array([2, 2, 1]), L_max=np.array([3, 3, 2]), Gamma=2)
    lib = build_library(params)
    assert len(lib.scenarios) < 5000
    return lib


@pytest.fixture(scope='module')
def exact_design(small_lib):
    """Exact design of the small instance from the complete-library extensive form."""
    sol = solve_robust_model(small_lib.params, small_lib.scenarios)
    assert sol.lp_status == 'optimal'
    return sol


def test_maximal_within_matches_library(small_lib):
    ids, n_distinct = maximal_within(small_lib, range(len(small_lib.scenarios)))
    assert n_distinct == len(small_lib.canonical)
    assert sorted(ids) == small_lib.maximal_ids


def test_strong_duality_and_cut(small_lib, exact_design):
    ev = RecourseEvaluator(small_lib.params)
    C, q = exact_design.C_depot, exact_design.q_mode
    for sid in small_lib.maximal_ids[:15]:
        Delta = small_lib.delta(sid)
        r = ev.evaluate(C, q, Delta, want_duals=True)
        assert r.status == 'optimal'
        dual = ev.dual_objective(C, q, Delta, r.y_ub, r.y_eq)
        assert abs(dual - r.objective) <= 1e-6 * max(1.0, abs(r.objective))
        const, gC, gq = ev.cut(Delta, r.y_ub, r.y_eq)
        assert abs(const + gC @ C + gq @ q - r.objective) <= 1e-6 * max(1.0, abs(r.objective))
        assert np.all(r.y_ub <= 1e-9)
        # the cut is a valid lower bound at a perturbed design
        C2, q2 = C + 50.0, q * 0.9
        r2 = ev.evaluate(C2, q2, Delta)
        assert const + gC @ C2 + gq @ q2 <= r2.objective + 1e-6 * max(1.0, abs(r2.objective))


def test_maximal_set_value_equals_complete_library(small_lib, exact_design):
    C, q = exact_design.C_depot, exact_design.q_mode
    full, _ = certify_fixed_design(params=small_lib.params, scenarios=small_lib.scenarios, C_depot=C, q_mode=q,
                                   design_name='full', progress_interval=0)
    red, _ = certify_fixed_design(params=small_lib.params, scenarios=small_lib.scenarios, C_depot=C, q_mode=q,
                                  design_name='max', scenario_ids=small_lib.maximal_ids, progress_interval=0)
    assert abs(full.full_worst_operating_cost - red.full_worst_operating_cost) <= 1e-6


def test_worst_case_milp_matches_enumeration(small_lib, exact_design):
    ev = RecourseEvaluator(small_lib.params)
    designs = [(exact_design.C_depot, exact_design.q_mode),
               (np.array([600.0, 600.0]), np.array([300.0, 100.0, 50.0]))]
    for C, q in designs:
        ref, _ = certify_fixed_design(params=small_lib.params, scenarios=small_lib.scenarios, C_depot=C, q_mode=q,
                                      design_name='ref', progress_interval=0)
        res = worst_case_milp(ev, small_lib, C, q, time_limit_s=300)
        assert 'worst_recourse' in res, res
        # one U.S. dollar on objectives of order 1e9 dollars (MILP solved in $M)
        assert abs(res['worst_recourse'] - ref.full_worst_operating_cost) <= 1.0
        assert abs(res['primal_recheck_recourse'] - res['worst_recourse']) <= 1.0


def test_kelley_matches_extensive_form(small_lib, exact_design):
    ev = RecourseEvaluator(small_lib.params)
    res = kelley_cutting_plane(ev, small_lib, small_lib.maximal_ids, tol_abs=1.0, max_iters=300, log=lambda *_: None)
    assert res['gap'] <= 1.0
    assert abs(res['upper_bound'] - exact_design.total_cost) <= 1.0
    assert res['optimal_face_max_width'] >= 0.0


def test_vectorized_extensive_form_matches_lp_solver(small_lib, exact_design):
    ev = RecourseEvaluator(small_lib.params)
    res = solve_extensive_form(ev, small_lib, small_lib.canonical_ids, method='highs', log=lambda *_: None)
    assert res['success']
    assert abs(res['total_cost'] - exact_design.total_cost) <= 1.0
