"""
Generate scenario statistics without solving LP

Output: results/scenario_statistics.csv
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.append(SRC_DIR)

from instance_generator import generate_small_instance, generate_burst_scenarios
import pandas as pd
import numpy as np

def main():
    print("Generating complete burst scenario statistics...")

    # Generate instance
    params = generate_small_instance()
    print(f"Instance: T={params.T}, M={params.M}, Gamma={params.Gamma}")

    # Generate ALL burst scenarios
    print("\nEnumerating all burst scenarios (no max limit)...")
    all_scenarios = generate_burst_scenarios(params, max_scenarios=None)

    print(f"Total scenarios: {len(all_scenarios)}")

    # Statistics
    single_event = [s for s in all_scenarios if len(s.events) == 1]
    double_event = [s for s in all_scenarios if len(s.events) == 2]

    durations = []
    modes_affected = []
    for s in all_scenarios:
        for event in s.events:
            m, t_start, L = event
            durations.append(L)
            modes_affected.append(m)

    # Event statistics by mode
    mode_counts = {m: 0 for m in range(params.M)}
    for m in modes_affected:
        mode_counts[m] += 1

    # Duration statistics
    duration_counts = {}
    for L in durations:
        duration_counts[L] = duration_counts.get(L, 0) + 1

    # Summary
    stats = {
        'total_scenarios': len(all_scenarios),
        'single_event_scenarios': len(single_event),
        'double_event_scenarios': len(double_event),
        'total_events': len(durations),
        'avg_duration': np.mean(durations) if durations else 0,
        'min_duration': min(durations) if durations else 0,
        'max_duration': max(durations) if durations else 0,
        'mode_0_events': mode_counts.get(0, 0),
        'mode_1_events': mode_counts.get(1, 0),
        'mode_2_events': mode_counts.get(2, 0),
    }

    # Add duration distribution
    for L in sorted(duration_counts.keys()):
        stats[f'duration_{L}_count'] = duration_counts[L]

    # Save to CSV
    df = pd.DataFrame([stats])
    output_dir = os.path.join(PROJECT_ROOT, 'results', '01_scenario_statistics')
    os.makedirs(output_dir, exist_ok=True)
    output_path = (
        f'{output_dir}/small_T{params.T}_M{params.M}_J{params.J}_K{params.K}_'
        f'Gamma{params.Gamma}_full_enumeration.csv'
    )
    df.to_csv(output_path, index=False)

    print(f"\n=== Scenario Statistics ===")
    print(f"Total scenarios: {stats['total_scenarios']}")
    print(f"Single-event: {stats['single_event_scenarios']}")
    print(f"Double-event: {stats['double_event_scenarios']}")
    print(f"Avg duration: {stats['avg_duration']:.2f} months")
    print(f"\nDuration distribution:")
    for L in sorted(duration_counts.keys()):
        print(f"  L={L}: {duration_counts[L]} events")
    print(f"\nMode distribution:")
    for m in range(params.M):
        print(f"  Mode {m}: {mode_counts[m]} events")

    print(f"\nSaved to: {output_path}")

if __name__ == '__main__':
    main()
