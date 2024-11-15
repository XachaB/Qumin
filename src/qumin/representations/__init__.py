# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Sacha Beniamine.

Utility functions for representations.
"""
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .segments import Inventory, Form
from ..utils import merge_duplicate_columns

log = logging.getLogger()


def create_features(md, feature_cols):
    """Read feature and preprocess to be coindexed with paradigms."""
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]
    features = pd.read_csv(md.get_table_path('lexemes'))
    features.set_index("lexeme_id", inplace=True)
    features.fillna(value="", inplace=True)
    return features.loc[:, feature_cols]


def create_paradigms(dataset, fillna=True, segcheck=False,
                     defective=False, overabundant=False, merge_cols=False,
                     cells=None, sample=None, most_freq=None):
    """Read paradigms data, and prepare it according to a Segment class pool.

    Arguments:
        dataset (str): paralex frictionless Package
            All characters occuring in the paradigms except the first column
            should be inventoried in this class.
        verbose (bool): verbosity switch.
        fillna (bool): Defaults to True. Should #DEF# be replaced by np.NaN ? Otherwise they are filled with empty strings ("").
        segcheck (bool): Defaults to False. Should I check that all the phonological segments in the table are defined in the segments table ?
        defective (bool): Defaults to False. Should I keep rows with defective forms ?
        overabundant (bool): Defaults to False. Should I keep rows with overabundant forms ?
        merge_cols (bool): Defaults to False. Should I merge identical columns (fully syncretic) ?
        cells (List[str]): List of cell names to consider. Defaults to all.

    Returns:
        paradigms (:class:`pandas:pandas.DataFrame`): paradigms table (row are forms, lemmas, cells).
    """

    def get_unknown_segments(row, unknowns):
        """
        Checks whether all segments that appear in the paradigms are known.
        """
        cell, form = row
        known_sounds = set(Inventory._classes) | set(Inventory._normalization) | {"", " "}
        tokens = Inventory._segmenter.split(form)
        for char in tokens:
            if char not in known_sounds:
                unknowns[char].append(form + " " + cell)

    # Reading the paradigms.
    data_file_name = Path(dataset.basepath) / dataset.get_resource("forms").path
    lexemes, cell_col, form_col = ("lexeme", "cell", "phon_form")
    paradigms = pd.read_csv(data_file_name, na_values=["", "#DEF#"], dtype="str", keep_default_na=False,
                            usecols=["form_id", lexemes, cell_col, form_col])
    if not defective:
        defective_lexemes = set(paradigms.loc[paradigms[form_col].isna(), lexemes].unique())
        paradigms = paradigms[~paradigms.loc[:, lexemes].isin(defective_lexemes)]

    if not overabundant:
        paradigms.drop_duplicates(['lexeme', 'cell'], inplace=True)

    if most_freq:
        inflected = paradigms.loc[:, lexemes].unique()
        lexemes_file_name = Path(dataset.basepath) / dataset.get_resource("lexemes").path
        lexemes_df = pd.read_csv(lexemes_file_name, usecols=["lexeme_id", "frequency"])
        # Restrict to lexemes we have kept, if we dropped defectives
        lexemes_df = lexemes_df[lexemes_df.lexeme_id.isin(inflected)]
        selected = set(lexemes_df.sort_values("frequency",
                                              ascending=False
                                              ).iloc[:most_freq, :].loc[:, "lexeme_id"].to_list())
        paradigms = paradigms.loc[paradigms.lexeme.isin(selected), :]

    if sample:
        paradigms = paradigms.sample(sample)

    def check_cells(cells, par_cols):
        unknown_cells = set(cells) - set(par_cols)
        if unknown_cells:
            raise ValueError(f"You specified some cells which aren't in the paradigm : {' '.join(unknown_cells)}")
        return sorted(list(set(par_cols) - set(cells)))

    if not {lexemes, cell_col, form_col} < set(paradigms.columns):
        log.warning("Please use Paralex-style long-form table (http://www.paralex-standard.org).")

    if cells is not None:
        to_drop = check_cells(cells, paradigms[cell_col].unique())
        if len(to_drop) > 0:
            log.info(f"Dropping rows with following cell values: {', '.join(sorted(to_drop))}")
            paradigms = paradigms[(paradigms[cell_col].isin(cells))]

    paradigms.fillna(value="", inplace=True)

    if segcheck:
        log.info("Checking we have definitions for all the phonological segments in this data...")
        unknowns = defaultdict(list)
        paradigms[['cell', 'phon_form']].apply(get_unknown_segments, unknowns=unknowns, axis=1)

        if len(unknowns) > 0:
            alert = "Your paradigm has unknown segments: " + "\n ".join(
                "[{}] (in {} forms:{}) ".format(u, len(unknowns[u]), ", ".join(unknowns[u][:10])) for u in unknowns)
            raise ValueError(alert)

    def parse_cell(row):
        """
        Reads a string representation of a phon_form and returns a Form
        """
        id, segm = row
        if not segm:
            return segm
        return Form(segm, form_id=id)

    paradigms.phon_form = paradigms[['form_id', 'phon_form']].apply(parse_cell, axis=1)

    if merge_cols:
        log.info("Merging identical columns...")
        merge_duplicate_columns(paradigms, sep="#")

    if not fillna:
        paradigms = paradigms.replace("", np.NaN)
    paradigms.set_index('form_id', inplace=True)
    paradigms.rename(columns={"phon_form": "form"}, inplace=True)
    log.debug(paradigms)
    return paradigms
