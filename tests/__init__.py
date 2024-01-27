
from unittest import TestCase
from pandas.testing import assert_frame_equal
import pandas as pd

class TestCaseWithPandas(TestCase):

    def __init__(self, *args,  **kwargs):
        super(TestCaseWithPandas, self).__init__(*args, **kwargs)
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)

    def assertDataframeEqual(self, a, b, msg):
        try:
            assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e