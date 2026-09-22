"""
Deep-Space Logistics Resilience Design: Core Model and Instance Generator

This module implements:
1. Scenario generation for burst disruption uncertainty set
2. LP formulation for two-stage robust optimization
3. Baseline strategies comparison
4. Result visualization
"""

import numpy as np
import itertools
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json


@dataclass
class InstanceParameters:
    """Problem instance parameters"""
    # Sets
    M: int  # Number of transportation modes
    J: int  # Number of orbital depots
    D: int  # Number of destinations (typically 1)
    K: int  # Number of commodity types
    T: int  # Planning horizon (months)

    # Cost parameters
    f_depot: np.ndarray  # Depot capacity construction cost ($/ton), shape (J,)
    v_mode: np.ndarray   # Mode capacity contracting cost ($/ton/month), shape (M,)
    c_mode: np.ndarray   # Mode transportation cost ($/ton), shape (M,)
    h_depot: np.ndarray  # Inventory holding cost ($/ton/month), shape (J, K)
    p_unmet: np.ndarray  # Unmet demand penalty ($/ton), shape (D, K)

    # Physical parameters
    tau_mode_depot: np.ndarray  # Transport time mode->depot (months), shape (M, J)
    tau_depot_dest: np.ndarray  # Transport time depot->dest (months), shape (J, D)
    R_max: np.ndarray           # Reception capacity at destination (ton/month), shape (D, T)
    D_demand: np.ndarray        # Demand (ton), shape (D, K, T)
    s_initial: np.ndarray       # Initial inventory at depots (ton), shape (J, K)

    # Disruption parameters
    L_min: np.ndarray  # Minimum disruption duration (months), shape (M,)
    L_max: np.ndarray  # Maximum disruption duration (months), shape (M,)
    Gamma: int         # Maximum number of disruption events


def generate_small_instance() -> InstanceParameters:
    """
    Generate a small test instance based on NASA Artemis mission estimates

    Returns:
        Small instance with T=24, M=3, J=2, K=2
    """
    M, J, D, K, T = 3, 2, 1, 2, 24

    # Cost parameters (based on MCM problem + NASA estimates)
    f_depot = np.array([50000, 60000])  # LEO cheaper than Lunar orbit
    v_mode = np.array([100, 150, 300])  # Elevator1, Elevator2, Rocket
    c_mode = np.array([1000, 1200, 5000])  # Per ton launch cost

    h_depot = np.array([
        [20, 30],  # LEO depot: propellant, equipment
        [30, 40]   # Lunar orbit depot
    ])

    # Unmet demand penalty: set to 10,000x transportation cost
    # This makes unmet demand very expensive but not absurdly dominant
    p_unmet = np.array([[10_000_000, 15_000_000]])  # $10-15M per ton unmet

    # Physical parameters
    tau_mode_depot = np.array([
        [0.3, 0.5],  # Elevator1 to LEO, Lunar orbit
        [0.3, 0.5],  # Elevator2
        [0.2, 0.3]   # Rocket (faster but expensive)
    ])

    tau_depot_dest = np.array([
        [0.5],  # LEO to Lunar surface
        [0.2]   # Lunar orbit to surface (shorter)
    ])

    # Reception capacity: starts low, ramps up as base construction progresses
    R_max = np.zeros((D, T))
    R_max[0, :6] = 500      # Month 1-6: limited capacity
    R_max[0, 6:12] = 1000   # Month 7-12: medium capacity
    R_max[0, 12:] = 1500    # Month 13+: full capacity

    # Demand: periodic supply needs
    D_demand = np.zeros((D, K, T))
    D_demand[0, 0, :] = 200  # Propellant: constant need
    D_demand[0, 1, :] = 100  # Equipment: constant need
    # Add spikes for construction phases
    D_demand[0, 0, [3, 9, 15, 21]] = 400  # Major construction phases
    D_demand[0, 1, [6, 12, 18]] = 250     # Equipment delivery phases

    # Initial inventory (pre-positioned supplies)
    # Set enough to cover first 2 months of demand at each depot
    s_initial = np.zeros((J, K))
    s_initial[0, 0] = 400  # Depot 0: 400 tons propellant (2 months worth)
    s_initial[0, 1] = 200  # Depot 0: 200 tons equipment
    s_initial[1, 0] = 400  # Depot 1: 400 tons propellant
    s_initial[1, 1] = 200  # Depot 1: 200 tons equipment

    # Disruption parameters
    L_min = np.array([2, 2, 1])  # Elevator maintenance: 2+ months, Rocket: 1+ month
    L_max = np.array([6, 6, 2])  # Maximum disruption length
    Gamma = 2  # Allow up to 2 disruption events

    return InstanceParameters(
        M=M, J=J, D=D, K=K, T=T,
        f_depot=f_depot, v_mode=v_mode, c_mode=c_mode,
        h_depot=h_depot, p_unmet=p_unmet,
        tau_mode_depot=tau_mode_depot, tau_depot_dest=tau_depot_dest,
        R_max=R_max, D_demand=D_demand, s_initial=s_initial,
        L_min=L_min, L_max=L_max, Gamma=Gamma
    )


class DisruptionScenario:
    """Represents a burst disruption scenario"""

    def __init__(self, events: List[Tuple[int, int, int]]):
        """
        Args:
            events: List of (mode, start_time, duration) tuples
        """
        self.events = events

    def to_delta_matrix(self, M: int, T: int) -> np.ndarray:
        """
        Convert event list to Delta_{mt} binary matrix

        Returns:
            Delta matrix of shape (M, T), where Delta[m,t]=1 means mode m disrupted at time t
        """
        Delta = np.zeros((M, T), dtype=int)
        for mode, start, duration in self.events:
            for t in range(start, min(start + duration, T)):
                Delta[mode, t] = 1
        return Delta

    def __repr__(self):
        return f"DisruptionScenario(events={self.events})"


def generate_burst_scenarios(params: InstanceParameters,
                             max_scenarios: int = None) -> List[DisruptionScenario]:
    """
    Generate all feasible burst disruption scenarios

    Args:
        params: Instance parameters
        max_scenarios: Maximum number of scenarios to generate (None = all)

    Returns:
        List of DisruptionScenario objects
    """
    scenarios = []
    single_events = []

    # Generate feasible single events.
    for m in range(params.M):
        for t_start in range(params.T):
            for L in range(params.L_min[m], params.L_max[m] + 1):
                if t_start + L <= params.T:
                    single_events.append((m, t_start, L))

    if params.Gamma <= 0:
        print("Generated 0 single-event scenarios")
        print("Generated 0 two-event scenarios")
        print("Generated 0 three-event scenarios")
        print("Total scenarios: 1")
        return [DisruptionScenario([])]

    scenarios.extend(DisruptionScenario([event]) for event in single_events)

    print(f"Generated {len(scenarios)} single-event scenarios")

    def events_compatible(e1, e2) -> bool:
        """Events on different modes are compatible; same-mode events must not overlap."""
        m1, t1, L1 = e1
        m2, t2, L2 = e2
        if m1 != m2:
            return True
        end1 = t1 + L1
        end2 = t2 + L2
        return end1 <= t2 or end2 <= t1

    # Generate two-event scenarios (Gamma=2) if allowed
    two_event_scenarios = []
    compatible_pairs = []
    if params.Gamma >= 2:
        for i, e1 in enumerate(single_events):
            for j in range(i + 1, len(single_events)):
                e2 = single_events[j]
                if events_compatible(e1, e2):
                    compatible_pairs.append((i, j))
                    two_event_scenarios.append(DisruptionScenario([e1, e2]))

        print(f"Generated {len(two_event_scenarios)} two-event scenarios")
        scenarios.extend(two_event_scenarios)
    else:
        print("Generated 0 two-event scenarios")

    if params.Gamma >= 3:
        three_event_scenarios = []
        for i, j in compatible_pairs:
            e1 = single_events[i]
            e2 = single_events[j]
            for k in range(j + 1, len(single_events)):
                e3 = single_events[k]
                if events_compatible(e1, e3) and events_compatible(e2, e3):
                    three_event_scenarios.append(DisruptionScenario([e1, e2, e3]))

        print(f"Generated {len(three_event_scenarios)} three-event scenarios")
        scenarios.extend(three_event_scenarios)
    else:
        print("Generated 0 three-event scenarios")

    if params.Gamma > 3:
        print("Warning: Scenario generator currently enumerates up to three-event scenarios")

    # Add nominal scenario (no disruption)
    scenarios.insert(0, DisruptionScenario([]))

    # Limit number of scenarios if requested
    if max_scenarios and len(scenarios) > max_scenarios:
        print(f"Warning: Limiting to {max_scenarios} scenarios (from {len(scenarios)})")
        non_nominal = scenarios[1:]
        sample_size = min(max_scenarios - 1, len(non_nominal))
        indices = np.linspace(0, len(non_nominal) - 1, sample_size, dtype=int)
        sampled = [scenarios[0]] + [non_nominal[i] for i in indices]
        scenarios = sampled

    print(f"Total scenarios: {len(scenarios)}")
    return scenarios


def generate_independent_scenarios(params: InstanceParameters,
                                   Gamma_total: int) -> List[DisruptionScenario]:
    """
    Generate scenarios for independent budgeted uncertainty (baseline comparison)

    Args:
        params: Instance parameters
        Gamma_total: Total budget of disrupted time-mode pairs

    Returns:
        List of scenarios (sampled, as full enumeration is huge)
    """
    # For independent model, each (m,t) can be disrupted independently
    # Total number disrupted <= Gamma_total
    # This is combinatorially huge, so we sample worst-case scenarios

    scenarios = [DisruptionScenario([])]  # Nominal

    # Sample strategy: create scenarios with Gamma_total disruptions
    # spread across critical periods (high demand times)
    np.random.seed(42)

    for _ in range(50):  # Generate 50 sample scenarios
        events = []
        disrupted_count = 0

        while disrupted_count < Gamma_total:
            m = np.random.randint(0, params.M)
            t = np.random.randint(0, params.T)

            # Check if already disrupted
            if any(e[0] == m and e[1] == t for e in events):
                continue

            events.append((m, t, 1))  # Single-period disruption
            disrupted_count += 1

        scenarios.append(DisruptionScenario(events))

    return scenarios


def save_instance(params: InstanceParameters, filename: str):
    """Save instance to JSON file"""
    data = {
        'M': params.M, 'J': params.J, 'D': params.D, 'K': params.K, 'T': params.T,
        'f_depot': params.f_depot.tolist(),
        'v_mode': params.v_mode.tolist(),
        'c_mode': params.c_mode.tolist(),
        'h_depot': params.h_depot.tolist(),
        'p_unmet': params.p_unmet.tolist(),
        'tau_mode_depot': params.tau_mode_depot.tolist(),
        'tau_depot_dest': params.tau_depot_dest.tolist(),
        'R_max': params.R_max.tolist(),
        'D_demand': params.D_demand.tolist(),
        'L_min': params.L_min.tolist(),
        'L_max': params.L_max.tolist(),
        'Gamma': params.Gamma
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Instance saved to {filename}")


if __name__ == "__main__":
    print("=" * 60)
    print("Deep-Space Logistics: Instance Generator")
    print("=" * 60)

    # Generate small instance
    print("\n[1] Generating small instance...")
    params = generate_small_instance()

    print(f"\nInstance size:")
    print(f"  Modes (M): {params.M}")
    print(f"  Depots (J): {params.J}")
    print(f"  Destinations (D): {params.D}")
    print(f"  Commodities (K): {params.K}")
    print(f"  Time periods (T): {params.T}")
    print(f"  Disruption budget (Γ): {params.Gamma}")

    # Generate scenarios
    print("\n[2] Generating burst disruption scenarios...")
    scenarios = generate_burst_scenarios(params, max_scenarios=500)

    print("\n[3] Sample scenarios:")
    for i, s in enumerate(scenarios[:5]):
        Delta = s.to_delta_matrix(params.M, params.T)
        n_disrupted = np.sum(Delta)
        print(f"  Scenario {i}: {len(s.events)} events, {n_disrupted} disrupted (m,t) pairs")
        if s.events:
            print(f"    Events: {s.events[:3]}{'...' if len(s.events) > 3 else ''}")

    # Save instance and scenarios
    print("\n[4] Saving data...")
    save_instance(params, '../data/small_instance.json')

    # Save scenarios
    scenario_data = {
        'num_scenarios': len(scenarios),
        'scenarios': [
            {
                'id': i,
                'events': s.events,
                'n_disrupted_pairs': int(np.sum(s.to_delta_matrix(params.M, params.T)))
            }
            for i, s in enumerate(scenarios)
        ]
    }

    with open('../data/burst_scenarios.json', 'w') as f:
        json.dump(scenario_data, f, indent=2)

    print(f"  - Instance: ../data/small_instance.json")
    print(f"  - Scenarios: ../data/burst_scenarios.json")

    print("\n" + "=" * 60)
    print("Instance generation complete!")
    print("=" * 60)
