# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Sacha Beniamine.

Utility functions for representations.
"""
import logging
import pandas as pd

log = logging.getLogger()


def create_features(md, feature_cols):
    """Read feature and preprocess to be coindexed with paradigms."""
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]
    features = pd.read_csv(md.get_table_path('lexemes'))
    features.set_index("lexeme_id", inplace=True)
    features.fillna(value="", inplace=True)
    return features.loc[:, feature_cols]
