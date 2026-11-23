"""Theme groups must never be split across a partition, and the v0.3 corpus must load."""
import sys
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loader import load_custom_dataset, load_theme_groups
from src import robust_estimation as re_

V03 = ROOT / "data" / "v03"
GROUPS_CSV = V03 / "theme_groups_67.csv"
DATASET = V03 / "dataset_v03_rerun.jsonl"


def _clusters():
    groups = load_theme_groups(str(GROUPS_CSV))
    order = sorted(groups)                      # the runner sorts pairs by id
    return order, np.array([groups[p] for p in order])


def _assert_whole(test, clusters, blocks):
    """Every cluster lies in exactly one block; blocks cover every row once."""
    seen = np.concatenate(blocks)
    test.assertEqual(sorted(seen.tolist()), list(range(len(clusters))))
    home = defaultdict(set)
    for b, rows in enumerate(blocks):
        for r in rows:
            home[clusters[r]].add(b)
    test.assertTrue(all(len(v) == 1 for v in home.values()))


class ThemeGroupPartitionTests(unittest.TestCase):
    def tearDown(self):
        re_.set_pair_clusters(None)

    def test_map_shape(self):
        order, clusters = _clusters()
        self.assertEqual(len(order), 96)
        self.assertEqual(len(set(clusters)), 67)

    def test_folds_and_halves_keep_groups(self):
        _, clusters = _clusters()
        re_.set_pair_clusters(clusters)
        for seed in range(20):
            rng = np.random.default_rng(seed)
            folds = re_.pair_folds(96, 5, rng)
            _assert_whole(self, clusters, folds)
            sizes = [len(f) for f in folds]
            self.assertLessEqual(max(sizes) - min(sizes), 3)   # largest group is 3
            a, b = re_.pair_halves(96, rng)
            _assert_whole(self, clusters, [a, b])
            self.assertLessEqual(abs(len(a) - len(b)), 3)

    def test_probe_folds_keep_both_arms_and_groups(self):
        _, clusters = _clusters()
        re_.set_pair_clusters(clusters)
        groups = re_.pair_groups(96, 96)
        y = np.r_[np.ones(96), np.zeros(96)]
        folds = re_.grouped_fold_indices(y, 5, 0, groups)
        tests = [t for _, t in folds]
        _assert_whole(self, np.r_[clusters, clusters], tests)
        for train, test in folds:
            self.assertFalse(set(groups[train]) & set(groups[test]))
            self.assertEqual(len(train) + len(test), 192)

    def test_crossfit_folds_follow_groups(self):
        _, clusters = _clusters()
        re_.set_pair_clusters(clusters)
        rng = np.random.default_rng(1)
        acts = {k: rng.normal(size=(96, 8)) for k in
                ("chat_harmful", "chat_benign", "agent_harmful", "agent_benign")}
        res = re_.crossfit_condition_projections(acts, n_folds=5, seed=3)
        fold_by_group = defaultdict(set)
        for c, f in zip(clusters, res.fold_of_pair):
            fold_by_group[c].add(int(f))
        self.assertTrue(all(len(v) == 1 for v in fold_by_group.values()))

    def test_without_map_partitions_are_unchanged(self):
        # Earlier runs must reproduce: same RNG draws as before the change.
        self.assertTrue(np.array_equal(re_.pair_groups(4, 4), np.r_[0:4, 0:4]))
        got = re_.pair_folds(96, 5, np.random.default_rng(7))
        want = np.array_split(np.random.default_rng(7).permutation(96), 5)
        for g, w in zip(got, want):
            self.assertTrue(np.array_equal(g, w))

    def test_length_mismatch_is_an_error(self):
        re_.set_pair_clusters(["a", "a", "b"])
        with self.assertRaises(ValueError):
            re_.pair_folds(4, 2, np.random.default_rng(0))


class DatasetV03Tests(unittest.TestCase):
    def test_loads_and_pairs_are_consistent(self):
        entries = load_custom_dataset(str(DATASET))
        self.assertEqual(len(entries), 384)
        groups = load_theme_groups(str(GROUPS_CSV))
        self.assertEqual({e.pair_id for e in entries}, set(groups))
        by_pair = defaultdict(dict)
        for e in entries:
            by_pair[e.pair_id][e.variant] = e
        for pair, v in by_pair.items():
            self.assertEqual(set(v), {"chat_harmful", "chat_benign",
                                      "agent_harmful", "agent_benign"}, pair)
            # Same tools and system message in both agent arms of a pair.
            self.assertEqual(v["agent_harmful"].tool_definitions,
                             v["agent_benign"].tool_definitions, pair)
            self.assertEqual(v["agent_harmful"].system_prompt,
                             v["agent_benign"].system_prompt, pair)
            # Chat and agent arms share the instruction text.
            self.assertEqual(v["chat_harmful"].prompt, v["agent_harmful"].prompt)
            self.assertEqual(v["chat_benign"].prompt, v["agent_benign"].prompt)

    def test_bad_map_is_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "g.csv"
            p.write_text("pair_id,theme_group_id\na,g1\na,g2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_theme_groups(str(p))


if __name__ == "__main__":
    unittest.main()
