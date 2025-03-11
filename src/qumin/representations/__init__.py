# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Sacha Beniamine.

Utility functions for representations.
"""
import logging
import pandas as pd
from paralex import read_table
log = logging.getLogger("Qumin")


def create_features(md, feature_cols):
    """Read feature and preprocess to be coindexed with paradigms."""
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]
    features = read_table(md.get_table_path('lexemes'))
    features.set_index("lexeme_id", inplace=True)
    features.fillna(value="", inplace=True)
    return features.loc[:, feature_cols]
