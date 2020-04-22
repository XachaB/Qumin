# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Sacha Beniamine.

Utility functions for representations.
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from representations.segments import Segment, restore, restore_string
from utils import merge_duplicate_columns


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
    #  First column has to be lexemes
    lexemes = features.columns[0]
    features[lexemes] = unique_lexemes(features[lexemes])
    features.set_index(lexemes, inplace=True)
    features.fillna(value="", inplace=True)
    return features


def create_paradigms(data_file_name,
                     cols=None, verbose=False, fillna=True,
                     segcheck=False, merge_duplicates=False,
                     defective=False, overabundant=False, merge_cols=False):
    """Read paradigms data, and prepare it according to a Segment class pool.

    Arguments:
        data_file_name (str): path to the paradigm csv file.
        All characters occuring in the paradigms except the first column
        should be inventoried in this class.
        cols (list of str): a subset of columns to use from the paradigm file.
        verbose (bool): verbosity switch.
        merge_duplicates (bool): should identical columns be merged ?
    Returns:
        paradigms (:class:`pandas:pandas.DataFrame`): paradigms
        (columns are cells, index are lemmas).
    """

    def get_unknown_segments(form, unknowns, name):
        for char in form:
            try:
                Segment.get(char)
            except KeyError:
                if char != ";":
                    unknowns[char].append(restore_string(form) + " " + name)

    # Reading the paradigms.
    paradigms = pd.read_csv(data_file_name, na_values="#DEF#", dtype="str")

    if not defective:
        paradigms.dropna(axis=0, inplace=True)

    # If the original file has two identical lexeme rows.
    if "variants" in paradigms.columns:
        print("Dropping variants")
        paradigms.drop("variants", axis=1, inplace=True)

    #  First column has to be lexemes
    lexemes = paradigms.columns[0]

    if cols:
        cols.append(lexemes)
        try:

            paradigms = paradigms[cols]
        except KeyError:
            print("The paradigm's columns are: {}".format(paradigms.columns))
            raise

    # Lexemes must be unique identifiers
    paradigms[lexemes] = unique_lexemes(paradigms[lexemes])
    paradigms.set_index(lexemes, inplace=True)

    paradigms.fillna(value="", inplace=True)

    if merge_duplicates:
        agenda = list(paradigms.columns)
        while agenda:
            a = agenda.pop(0)
            for i, b in enumerate(agenda):
                if all(paradigms[a] == paradigms[b]):
                    print("Identical columns ", a, " and ", b)
                    new = a + " & " + b
                    agenda.pop(i)
                    agenda.append(new)
                    paradigms[new] = paradigms[a]
                    paradigms.drop([a, b], inplace=True, axis=1)
                    break

    paradigms = normalize_dataframe(paradigms, Segment._aliases, Segment._normalization, verbose=verbose)

    if overabundant:  # sorting the overabundant forms.
        paradigms = paradigms.applymap(lambda x: ";".join(sorted(x.split(";"))))
    if not overabundant:  # if ignore overabundance, take first elt everywhere.
        paradigms = paradigms.applymap(lambda x: x.split(";")[0])

    print("Merging identical columns...")
    if merge_cols:
        merge_duplicate_columns(paradigms, sep="#")

    if segcheck:
        print("Checking we have definitions for all the phonological segments in this data...")
        unknowns = defaultdict(list)
        paradigms.apply(lambda x: x.apply(get_unknown_segments, args=(unknowns, x.name)), axis=1)

        if len(unknowns) > 0:
            alert = "Your paradigm has unknown segments: " + "\n ".join(
                "[{}] (in {} forms:{}) ".format(u, len(unknowns[u]), ", ".join(unknowns[u][:10])) for u in unknowns)
            raise ValueError(alert)

    if not fillna:
        paradigms = paradigms.replace("", np.NaN)
    return paradigms


def normalize_dataframe(paradigms, aliases, normalization, verbose=False):
    """Normalize and Simplify a dataframe.

    **aliases**:
        For all sequence of n characters representing a segment,
        replace with a unique character representing this segment.
    **Normalization**:
        for all groups of characters representing the same feature set,
        translate to one unique character.

    **Note**:
        a .translate strategy works for normalization
        but not for aliases,
        since it only maps single characters to single characters.
        The order of operations is important,
        since .translate assumes mapping of 1: 1 chars.

    Arguments:
        df (:class:`pandas:pandas.DataFrame`): A dataframe.
        verbose (bool): verbosity switch.

    Returns:
        new_df (:class:`pandas:pandas.DataFrame`):
            The same dataframe, normalized and simplified.
    """

    def _norm(cell):
        if cell:
            for key in sorted(aliases, key=lambda x: len(x), reverse=True):
                cell = cell.replace(key, aliases[key])
            return cell.translate(normalization)
        return cell

    if verbose:
        print(paradigms[: 3])
    print("\nNormalizing dataframe's segments...\n")

    new_df = paradigms.applymap(_norm)

    if verbose:
        print(new_df[: 3])
    return new_df


def restore_simplified_dataframe(df, verbose=True):
    """Restore all dataframe cells to non simplified strings.

    Arguments:
        df (:class:`pandas:pandas.DataFrame`):
            A dataframe containing strings
            of simplified segments names in its cells.
        verbose (bool): verbosity switch.

    Returns:
        new_df (:class:`pandas:pandas.DataFrame`):
            The same dataframe, containing strings
            of non simplified segments names in its cells

    """
    if verbose:
        print(df[: 3])
        print("\nRestoring simplified dataframe...\n")

    new_df = df.applymap(restore)

    if verbose:
        print(new_df[: 3])
