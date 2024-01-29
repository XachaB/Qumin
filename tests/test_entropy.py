import unittest
import pandas as pd
from qumin.entropy import utils

class UtilsTestCase(unittest.TestCase):

    def test_P(self):
        events = pd.Series([0,0,1,1,2,2,2,2])
        expected = {0: 0.25, 1: 0.25, 2: 0.5}
        p = utils.P(events)
        self.assertTrue(type(p) == pd.Series)
        self.assertEqual(p.to_dict(),expected)

    def test_cond_P(self):
        eventsA = pd.Series([0,0,1,1,2,2,2,2])
        eventsB = pd.Series([1,1,1,1,1,2,2,2])
        p = utils.cond_P(eventsA,eventsB)
        expected = {(1, 2): 0.20000000000000001,
                    (1, 0): 0.40000000000000002,
                    (1, 1): 0.40000000000000002, (2, 2): 1.0}
        self.assertEqual(p.to_dict(),expected)

    def test_cond_entropy(self):
        eventsA = pd.Series([0,0,1,1,2,2,2,2])
        eventsB = pd.Series([1,1,1,1,1,2,2,2])
        expected = 0.951205059305

        # slow step-by-step entropy
        cond_p = utils.cond_P(eventsA,eventsB)
        known_p = utils.P(eventsB)
        h = sum(known_p * cond_p.groupby(level=0, sort=False).apply(utils.entropy))


        h2 = utils.cond_entropy(eventsA,eventsB)
        self.assertAlmostEqual(h,expected)
        self.assertAlmostEqual(h2,expected)
        self.assertAlmostEqual(h2,h)

    def test_entropy(self):
        p = pd.Series({0: 0.25, 1: 0.25, 2: 0.5})
        h = utils.entropy(p)
        self.assertEqual(h,1.5)
