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
from paralex import read_table
from .segments import Inventory, Form
from ..utils import merge_duplicate_columns

log = logging.getLogger()


def unique_lexemes(series, file_type):
    """Check if there are duplicates in the lexemes list.
    If yes, raise an error.

    Arguments:
        series (:class:`pandas:pandas.Series`): list of lexemes as a Series object.
        file_type (str): used for error messages. Should describe the dataset which is being tested.
    """
    duplicated = list(series[series.duplicated()])

    if duplicated:
        raise ValueError(f"""There are {len(duplicated)} duplicates among lexemes.
            Please check the {file_type} table.
            Duplicates are: {", ".join(duplicated)}""")


def create_features(md, feature_cols):
    """Read feature and preprocess to be coindexed with paradigms."""
    if type(feature_cols) == str:
        feature_cols = [feature_cols]
    features = pd.read_csv(md.get_table_path('lexemes'))
    unique_lexemes(features["lexeme_id"], "lexemes")
    features.set_index("lexeme_id", inplace=True)
    features.fillna(value="", inplace=True)
    return features.loc[:, feature_cols]


def create_paradigms(dataset, fillna=True,
                     segcheck=False, merge_duplicates=False,
                     defective=False, overabundant=False, merge_cols=False,
                     cells=None, sample=None, most_freq=None, pos=None):
    """Read paradigms data, and prepare it according to a Segment class pool.

    Arguments:
        dataset (str): paralex frictionless Package
        verbose (bool): verbosity switch.
        merge_duplicates (bool): should identical columns be merged ?
        fillna (bool): Defaults to True. Should #DEF# be replaced by np.NaN ? Otherwise they are filled with empty strings ("").
        segcheck (bool): Defaults to False. Should I check that all the phonological segments in the table are defined in the segments table ?
        defective (bool): Defaults to False. Should I keep rows with defective forms ?
        overabundant (bool): Defaults to False. Should I keep rows with overabundant forms ?
        merge_cols (bool): Defaults to False. Should I merge identical columns (fully syncretic) ?
        cells (List[str]): List of cell names to consider. Defaults to all.
        pos (List[str]): List of parts of speech to consider. Defaults to all.

    Returns:
        paradigms (:class:`pandas:pandas.DataFrame`): paradigms table (columns are cells, index are lemmas).
    """

    def get_unknown_segments(forms, unknowns, name):
        known_sounds = set(Inventory._classes) | set(Inventory._normalization) | {"", " "}
        for form_id in forms:
            form = form_dic[form_id]
            tokens = Inventory._segmenter.split(form)
            for char in tokens:
                if char not in known_sounds:
                    unknowns[char].append(form + " " + name)

    # Reading the paradigms.
    data_file_name = Path(dataset.basepath) / dataset.get_resource("forms").path
    lexemes, cell_col, form_col = ("lexeme", "cell", "phon_form")
    paradigms = pd.read_csv(data_file_name, na_values=["", "#DEF#"], dtype="str", keep_default_na=False,
                            usecols=["form_id", lexemes, cell_col, form_col])

    if pos:
        if 'lexemes' in dataset.resource_names:
            table = read_table('lexemes', dataset)
            if 'POS' not in table.columns:
                log.warning('No POS column in the lexemes table.')
            else:
                if isinstance(pos, str):
                    pos = [pos]
                paradigms = paradigms[paradigms['lexeme']
                                      .map(table.set_index('lexeme_id').POS)
                                      .isin(pos)]
        else:
            log.warning("No lexemes table. Can't filter based on POS.")

    if not defective:
        defective_lexemes = set(paradigms.loc[paradigms[form_col].isna(), lexemes].unique())
        paradigms = paradigms[~paradigms.loc[:, lexemes].isin(defective_lexemes)]

    if not {lexemes, cell_col, form_col} < set(paradigms.columns):
        log.warning("Please use Paralex-style long-form table (http://www.paralex-standard.org).")

    def check_cells(cells, par_cols):
        unknown_cells = set(cells) - set(par_cols)
        if unknown_cells:
            raise ValueError(f"You specified some cells which aren't in the paradigm : {' '.join(unknown_cells)}")
        return sorted(list(set(par_cols) - set(cells)))

    # Filter cells before pivoting for speed reasons
    if cells is not None:
        to_drop = check_cells(cells, paradigms[cell_col].unique())
        if len(to_drop) > 0:
            log.info(f"Dropping rows with following cell values: {', '.join(sorted(to_drop))}")
            paradigms = paradigms[(paradigms[cell_col].isin(cells))]

    # Get only most frequent lexemes
    if most_freq:
        inflected = paradigms.loc[:, lexemes].unique()
        lexemes_file_name = Path(dataset.basepath) / dataset.get_resource("lexemes").path
        lexemes_df = pd.read_csv(lexemes_file_name, usecols=["lexeme_id", "frequency"])
        # Restrict to lexemes we have kept, if we dropped defectives or cells
        lexemes_df = lexemes_df[lexemes_df.lexeme_id.isin(inflected)]
        selected = set(lexemes_df.sort_values("frequency",
                                              ascending=False
                                              ).iloc[:most_freq, :].loc[:, "lexeme_id"].to_list())
        paradigms = paradigms.loc[paradigms.lexeme.isin(selected), :]

    # Sample paradigms
    if sample:
        paradigms = paradigms.sample(sample)

    paradigms.fillna(value="", inplace=True)
    form_dic = paradigms.set_index('form_id')[form_col].to_dict()

    def aggregator(s):
        form_ids = tuple(form_id for form_id in s if form_dic[form_id] != '')
        return form_ids if len(form_ids) > 0 else ""

    paradigms = paradigms.pivot_table(values='form_id', index=lexemes,
                                      columns=cell_col,
                                      aggfunc=aggregator)
    paradigms.fillna(value="", inplace=True)

    paradigms.reset_index(inplace=True, drop=False)
    log.debug(paradigms)

    # Lexemes must be unique identifiers
    unique_lexemes(paradigms[lexemes], 'paradigms')
    paradigms.set_index(lexemes, inplace=True)

    if merge_duplicates:
        agenda = list(paradigms.columns)
        while agenda:
            a = agenda.pop(0)
            for i, b in enumerate(agenda):
                apar = paradigms[a].apply(lambda x: [form_dic[i] for i in x])
                bpar = paradigms[b].apply(lambda x: [form_dic[i] for i in x])
                if (apar == bpar).all():
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
        forms = [Form(form_dic[f], form_id=f) for f in cell]
        if overabundant:
            forms = tuple(sorted(forms))
        else:
            forms = (forms[0],)
        return forms

    paradigms = paradigms.map(parse_cell)

    log.info("Merging identical columns...")
    if merge_cols:
        merge_duplicate_columns(paradigms, sep="#")

    if not fillna:
        paradigms = paradigms.replace("", np.NaN)

    log.debug(paradigms)
    return paradigms
