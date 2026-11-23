"""A looping reasoning trace is cut; a long legitimate one keeps its budget."""
import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.causal_test import classify_response, reasoning_loop
from src.intervention import _AnswerBudget, _LOOP_CHECK_EVERY


class _WordTokenizer:
    """id i <-> WORDS[i]; enough of a tokenizer for the stopping criterion."""

    def __init__(self, words):
        self.words = words

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.words[i] for i in ids)


def _criterion(words, opened=True):
    tok = _WordTokenizer(words)
    return _AnswerBudget(n_prompt=0, answer_tokens=64, open_id=0, close_id=1,
                         opened_in_prompt=opened, tokenizer=tok)


def _stops_at(crit, ids):
    for n in range(1, len(ids) + 1):
        if bool(crit(torch.tensor([ids[:n]]), None)[0]):
            return n
    return None


VOCAB = ["<think>", "</think>"] + [f"w{i}" for i in range(2, 3000)]


class LoopGuardTests(unittest.TestCase):
    def test_reasoning_loop_detects_repetition_only(self):
        loop = "Maybe the user wants the report. " * 200
        rng = random.Random(0)
        prose = " ".join(rng.choice(["the", "user", "asks", "for", "a", "report",
                                     "about", "sales", "so", "I", "should", "check",
                                     "which", "tool", "fits", "best", "and", "why"])
                         + str(rng.randint(0, 999)) for _ in range(800))
        self.assertTrue(reasoning_loop(loop))
        self.assertFalse(reasoning_loop(prose))
        self.assertFalse(reasoning_loop("short " * 10))    # under the window

    def test_loop_is_cut_long_reasoning_is_not(self):
        crit = _criterion(VOCAB)
        looping = [5, 6, 7, 8] * 1000                       # repeats a 4-word phrase
        n = _stops_at(crit, looping)
        self.assertIsNotNone(n)
        self.assertEqual(n % _LOOP_CHECK_EVERY, 0)
        self.assertLess(n, 1500)

        rng = random.Random(1)
        varied = [rng.randrange(2, 3000) for _ in range(3000)]
        self.assertIsNone(_stops_at(_criterion(VOCAB), varied))

    def test_answer_budget_still_applies_after_reasoning(self):
        rng = random.Random(2)
        ids = [rng.randrange(2, 3000) for _ in range(500)] + [1] + list(range(2, 200))
        self.assertEqual(_stops_at(_criterion(VOCAB), ids), 501 + 64)

    def test_cut_loop_is_labelled_incomplete(self):
        self.assertEqual(classify_response("<think>" + "again and again " * 300), "incomplete")


if __name__ == "__main__":
    unittest.main()
