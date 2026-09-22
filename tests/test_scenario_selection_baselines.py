import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from instance_generator import DisruptionScenario  # noqa: E402
from run_scenario_selection_baselines import (  # noqa: E402
    build_disruption_matrix,
    select_maxsum_top_ids,
    select_maxsum_weighted_ids,
    select_random_ids,
    select_scenario_ids,
    selection_summary,
)


class ScenarioSelectionTests(unittest.TestCase):
    def setUp(self):
        self.scenarios = [
            DisruptionScenario([]),
            DisruptionScenario([(0, 0, 1)]),
            DisruptionScenario([(0, 0, 2)]),
            DisruptionScenario([(0, 0, 3)]),
            DisruptionScenario([(1, 0, 3)]),
            DisruptionScenario([(0, 0, 1)]),  # duplicate matrix of ID 1
        ]
        self.matrix = build_disruption_matrix(self.scenarios, 2, 4)

    def test_random_is_seeded_unique_and_keeps_nominal(self):
        first = select_random_ids(len(self.scenarios), 4, seed=17)
        second = select_random_ids(len(self.scenarios), 4, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 0)
        self.assertEqual(len(first), len(set(first)))
        self.assertEqual(len(first), 4)

    def test_maxsum_top_selects_highest_severity(self):
        selected = select_maxsum_top_ids(self.matrix, 3, seed=3)
        self.assertEqual(selected[0], 0)
        self.assertIn(3, selected)
        self.assertIn(4, selected)

    def test_maxsum_weighted_is_seeded_unique(self):
        first = select_maxsum_weighted_ids(self.matrix, 4, seed=9)
        second = select_maxsum_weighted_ids(self.matrix, 4, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 0)
        self.assertEqual(len(first), len(set(first)))

    def test_dispatch_and_summary(self):
        selected = select_scenario_ids('random', self.matrix, 4, seed=5)
        summary = selection_summary(self.matrix, selected)
        self.assertEqual(summary['selected_scenarios'], 4)
        self.assertLessEqual(summary['selected_unique_matrices'], 4)
        self.assertEqual(
            summary['selected_duplicate_matrices'],
            4 - summary['selected_unique_matrices'],
        )

    def test_invalid_method_fails(self):
        with self.assertRaises(ValueError):
            select_scenario_ids('not-a-method', self.matrix, 3, seed=0)

    def test_sample_size_validation(self):
        with self.assertRaises(ValueError):
            select_random_ids(len(self.scenarios), 0, seed=0)
        with self.assertRaises(ValueError):
            select_random_ids(len(self.scenarios), len(self.scenarios) + 1, seed=0)


if __name__ == '__main__':
    unittest.main()
