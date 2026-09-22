import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from instance_generator import DisruptionScenario  # noqa: E402
from run_dominance_pruned_oracle import (  # noqa: E402
    canonicalize_scenarios,
    is_subset_bits,
    maximal_matrices,
    scenario_bits,
)


class DominancePruningTests(unittest.TestCase):
    def test_scenario_bit_encoding_and_subset(self):
        small = DisruptionScenario([(0, 0, 1)])
        large = DisruptionScenario([(0, 0, 2)])
        small_bits = scenario_bits(small, 2, 3)
        large_bits = scenario_bits(large, 2, 3)
        self.assertTrue(is_subset_bits(small_bits, large_bits))
        self.assertFalse(is_subset_bits(large_bits, small_bits))

    def test_duplicates_are_canonicalized(self):
        scenarios = [
            DisruptionScenario([]),
            DisruptionScenario([(0, 0, 1)]),
            DisruptionScenario([(0, 0, 1)]),
        ]
        canonical = canonicalize_scenarios(scenarios, 1, 2)
        self.assertEqual(len(canonical), 2)
        disrupted = next(record for record in canonical if record.severity == 1)
        self.assertEqual(disrupted.event_set_ids, (1, 2))

    def test_maximal_antichain_is_exact(self):
        scenarios = [
            DisruptionScenario([]),
            DisruptionScenario([(0, 0, 1)]),
            DisruptionScenario([(0, 0, 2)]),
            DisruptionScenario([(1, 0, 1)]),
            DisruptionScenario([(0, 0, 2), (1, 0, 1)]),
            DisruptionScenario([(1, 1, 1)]),
        ]
        canonical = canonicalize_scenarios(scenarios, 2, 2)
        maximal = maximal_matrices(canonical)
        maximal_ids = {record.representative_id for record in maximal}
        self.assertEqual(maximal_ids, {4, 5})
        for record in canonical:
            self.assertTrue(any(
                is_subset_bits(record.bits, maximal_record.bits)
                for maximal_record in maximal
            ))


if __name__ == '__main__':
    unittest.main()
