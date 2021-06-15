# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Sacha Beniamine.

Utility functions for representations.
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from ..utils import merge_duplicate_columns
from .segments import  Inventory, Form
import logging
log = logging.getLogger()


def unique_lexemes(series):
    """Rename duplicates in a serie of strings.

    Take a pandas series of strings and output another serie
    where all originally duplicate strings are given numbers,
    so each cell contains a unique string.
    """
    ids = {}

    def unique_id(string, ids):
        if string in ids:
            ids[string] += 1
            return string + "_" + str(ids[string])
        ids[string] = 1
        return string

    return series.apply(unique_id, args=(ids,))


def create_features(data_file_name):
    """Read feature and preprocess to be coindexed with paradigms."""
    features = pd.read_csv(data_file_name)
    # First column has to be lexemes
    lexemes = features.columns[0]
    features[lexemes] = unique_lexemes(features[lexemes])
    features.set_index(lexemes, inplace=True)
    features.fillna(value="", inplace=True)
    return features



def create_paradigms(data_file_name,
                     cols=None, verbose=False, fillna=True,
                     segcheck=False, merge_duplicates=False,
                     defective=False, overabundant=False, merge_cols=False,
                     long=False, col_names=("lexeme","cell","form")):
    """Read paradigms data, and prepare it according to a Segment class pool.

    Arguments:
        data_file_name (str): path to the paradigm csv file.
        All characters occuring in the paradigms except the first column
        should be inventoried in this class.
        cols (list of str): a subset of columns to use from the paradigm file.
        verbose (bool): verbosity switch.
        merge_duplicates (bool): should identical columns be merged ?
        fillna (bool): Defaults to True. Should #DEF# be replaced by np.NaN ? Otherwise they are filled with empty strings ("").
        segcheck (bool): Defaults to False. Should I check that all the phonological segments in the table are defined in the segments table ?
        defective (bool): Defaults to False. Should I keep rows with defective forms ?
        overabundant (bool): Defaults to False. Should I keep rows with overabundant forms ?
        merge_cols (bool): Defaults to False. Should I merge identical columns (fully syncretic) ?
        long (bool): is the dataset in long form ?
        cols (tuple): names of the lexeme, cells and form columns (in this order).
    Returns:
        paradigms (:class:`pandas:pandas.DataFrame`): paradigms table (columns are cells, index are lemmas).
    """

    def get_unknown_segments(forms, unknowns, name):
        for form in forms.split(";"):
            tokens = Inventory._segmenter.split(form)
            for char in tokens:
                if char not in Inventory._classes and char not in {";", ""}:
                    unknowns[char].append(form + " " + name)

    # Reading the paradigms.
    paradigms = pd.read_csv(data_file_name, na_values=["", "#DEF#"], dtype="str", keep_default_na=False)

    if long:
        lexeme_col, cell_col, form_col = col_names
        paradigms = paradigms.pivot_table(values=form_col, index=lexeme_col, columns=cell_col,
                              aggfunc=lambda x: ";".join(set(x)))
    else:
        # If the original file has two identical lexeme rows.
        if "variants" in paradigms.columns:
            log.info("Dropping the columns named 'variants'")
            paradigms.drop("variants", axis=1, inplace=True)

        # First column has to be lexemes
        lexemes = paradigms.columns[0]

    if not defective:
        paradigms.dropna(axis=0, inplace=True)

    if cols:
        cols.append(lexemes)
        try:
            paradigms = paradigms[cols]
        except KeyError as e:
            raise ValueError("The paradigm's columns are: {}".format(paradigms.columns)) from e

    # Lexemes must be unique identifiers
    paradigms[lexemes] = unique_lexemes(paradigms[lexemes])
    paradigms.set_index(lexemes, inplace=True)

    paradigms.fillna(value="", inplace=True)

    if merge_duplicates:
        agenda = list(paradigms.columns)
        while agenda:
            a = agenda.pop(0)
            for i, b in enumerate(agenda):
                if (paradigms[a] == paradigms[b]).all():
                    log.debug("Identical columns %s and %s ", a, b)
                    new = a + " & " + b
                    agenda.pop(i)
                    agenda.append(new)
                    paradigms[new] = paradigms[a]
                    paradigms.drop([a, b], inplace=True, axis=1)
                    break

    if segcheck:
        log.info("Checking we have definitions for all the phonological segments in this data...")
        unknowns = defaultdict(list)
        paradigms.apply(lambda x: x.apply(get_unknown_segments, args=(unknowns, x.name)), axis=1)

        if len(unknowns) > 0:
            alert = "Your paradigm has unknown segments: " + "\n ".join(
                "[{}] (in {} forms:{}) ".format(u, len(unknowns[u]), ", ".join(unknowns[u][:10])) for u in unknowns)
            raise ValueError(alert)

    def parse_cell(cell):
        if not cell:
            return cell
        forms = [Form(f) for f in cell.split(";")]
        if overabundant:
            forms = tuple(sorted(forms))
        else:
            forms = (forms[0], )
        return forms

    paradigms = paradigms.applymap(parse_cell)

    log.info("Merging identical columns...")
    if merge_cols:
        merge_duplicate_columns(paradigms, sep="#")


    if not fillna:
        paradigms = paradigms.replace("", np.NaN)
    return paradigms