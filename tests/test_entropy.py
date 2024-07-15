import unittest

import pandas as pd
from hypothesis import given, assume
from hypothesis import strategies as st
from qumin import entropy
from .strategies import event_probs, events_series


class UtilsTestCase(unittest.TestCase):

    @given(events_series())
    def test_P(self, events):
        p = entropy.P(events)
        # Returns a series
        self.assertTrue(type(p) == pd.Series)
        # Unique events
        self.assertFalse(p.index.duplicated().any())
        # All items from events are in index:
        self.assertEqual(set(p.index), set(events.values))
        # Sums to 1
        self.assertAlmostEqual(p.sum(), 1)
        # All more than 0
        for event, prob in p.items():
            self.assertLess(0, prob)

    def test_cond_P(self):
        # TODO: convert to property based testing
        eventsA = pd.Series([0, 0, 1, 1, 2, 2, 2, 2])
        eventsB = pd.Series([1, 1, 1, 1, 1, 2, 2, 2])
        p = entropy.cond_P(eventsA, eventsB)
        expected = {(1, 2): 0.20000000000000001,
                    (1, 0): 0.40000000000000002,
                    (1, 1): 0.40000000000000002, (2, 2): 1.0}
        self.assertEqual(p.to_dict(), expected)

    @given(events_series(), events_series())
    def test_cond_entropy(self, eventsA, eventsB):
        assume(eventsA.shape == eventsB.shape)
        # slow step-by-step entropy
        cond_p = entropy.cond_P(eventsA, eventsB)
        known_p = entropy.P(eventsB)
        h = sum(known_p * cond_p.groupby(level=0, sort=False).apply(entropy.entropy))
        h2 = entropy.cond_entropy(eventsA, eventsB)
        self.assertAlmostEqual(h2, h)

    @given(event_probs())
    def test_entropy_expansive(self, probs):
        assume((probs.iloc[0] != probs).any())
        h1 = entropy.entropy(probs)
        expanded = probs.copy()
        expanded = pd.concat([expanded, pd.Series(0)])
        h2 = entropy.entropy(expanded)
        self.assertAlmostEqual(h1, h2)

    @given(event_probs())
    def test_entropy_max(self, probs):
        assume((probs.iloc[0] != probs).any())
        h1 = entropy.entropy(probs)
        equiprobs = probs.copy()
        equiprobs.loc[:] = 1/probs.shape[0]
        h2 = entropy.entropy(equiprobs)
        self.assertLess(h1, h2)
    @given(st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=100))
    def test_entropy_increasing(self, i, j):
        probs = pd.Series(1/i, index=range(i))
        j = i+j # ensure j is greater than i
        more_probs = pd.Series(1/j, index=range(j))
        h1 = entropy.entropy(probs)
        h2 = entropy.entropy(more_probs)
        self.assertLess(h1, h2)
    @given(event_probs())
    def test_entropy_symmetrical(self, probs):
        h1 = entropy.entropy(probs)
        shuffled = probs.sample(probs.shape[0]) # Shuffling should use strategy
        h2 = entropy.entropy(shuffled)
        self.assertAlmostEqual(h1, h2)
