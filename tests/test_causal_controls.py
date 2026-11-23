"""The causal test's random control must be several directions, without changing the first."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.causal_test import (format_causal_report, random_orthogonal_direction,
                             run_causal_test)


def _setup(n=6, d=16):
    rng = np.random.default_rng(0)
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)
    pair_ids = [f"p{i}" for i in range(n)]
    prompts = {"agent_harmful": [f"h{i}" for i in range(n)],
               "agent_benign": [f"b{i}" for i in range(n)]}
    baseline = [{"variant": v, "pair_id": p, "behavior": "comply"}
                for v in prompts for p in pair_ids]
    return direction, pair_ids, prompts, baseline


class RandomControlTests(unittest.TestCase):
    def test_extra_directions_only_at_selected_doses(self):
        direction, pair_ids, prompts, baseline = _setup()
        used = {}

        def gen(prompt, steer):
            vec, alpha = steer
            used.setdefault(round(alpha, 6), []).append(vec)
            # The restore direction "works" from dose 2 up; random ones never.
            refuse = np.allclose(vec, direction) and alpha >= 2 and prompt.startswith("h")
            return {"text": "I can't help with that." if refuse else "Sure, done.",
                    "proj_before": 0.0, "proj_after": alpha}

        res = run_causal_test(gen, baseline, prompts, pair_ids, direction,
                              dose_unit=1.0, doses=(0.5, 1, 2, 4), seed=42,
                              log=lambda *_: None, n_random_controls=3,
                              extra_control_doses=(1, 4))
        arms = {}
        for r in res["records"]:
            arms.setdefault(r["dose"], set()).add(r["arm"])
        self.assertEqual(arms[0.5], {"restore", "control", "benign"})
        self.assertEqual(arms[1.0], {"restore", "control", "benign", "control_2", "control_3"})
        self.assertEqual(arms[4.0], arms[1.0])
        s = res["summary"]
        self.assertEqual(s["n_random_directions"], [1, 3, 1, 3])
        self.assertEqual(s["restore_rates"], [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(s["random_control_mean"][3], 0.0)
        self.assertIn("random mean", format_causal_report(s, layer=5))

    def test_first_control_unchanged_and_others_differ(self):
        direction, *_ = _setup()
        d42 = random_orthogonal_direction(direction, 42)
        d43 = random_orthogonal_direction(direction, 43)
        self.assertLess(abs(float(d42 @ d43)), 0.9)
        self.assertLess(abs(float(d43 @ direction)), 1e-8)

    def test_single_control_is_the_old_behaviour(self):
        direction, pair_ids, prompts, baseline = _setup()
        res = run_causal_test(lambda p, s: {"text": "Sure."}, baseline, prompts,
                              pair_ids, direction, dose_unit=1.0, doses=(1, 4),
                              log=lambda *_: None)
        self.assertEqual({r["arm"] for r in res["records"]}, {"restore", "control", "benign"})


if __name__ == "__main__":
    unittest.main()
