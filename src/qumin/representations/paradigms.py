# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Sacha Beniamine and Jules Bouton.

Paradigms class to represent paralex paradigms.
"""

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from .segments import Inventory, Form
import logging
import pandas as pd
from pandas.api.types import union_categoricals

log = logging.getLogger()
tqdm.pandas()


class Paradigms(object):

    default_cols = ("lexeme", "cell", "phon_form")
    dropped = None
    data = None
    cells = None

    def __init__(self, dataset, **kwargs):
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
            pos (List[str]): List of parts of speech to consider. Defaults to all.

        Returns:
            paradigms (:class:`pandas:pandas.DataFrame`): paradigms table (row are forms, lemmas, cells).
        """

        # Reading the paradigms.
        self.dataset = dataset
        data_file_name = Path(dataset.basepath) / dataset.get_resource("forms").path
        self.data = pd.read_csv(data_file_name, na_values=["#DEF#"],
                                dtype=defaultdict(lambda: 'string', {'cell': 'category',
                                                                     'lexeme': 'category'}),
                                keep_default_na=False,
                                usecols=["form_id"] + list(self.default_cols))
        self.preprocess(**kwargs)

    def _get_unknown_segments(self, row, unknowns):
        """
        Checks whether all segments that appear in the paradigms are known.
        """
        cell, form = row
        known_sounds = set(Inventory._classes) | set(Inventory._normalization) | {"", " "}
        tokens = Inventory._segmenter.split(form)
        for char in tokens:
            if char not in known_sounds:
                unknowns[char].append(form + " " + cell)

    def preprocess(self, fillna=True, segcheck=False,
                   defective=False, overabundant=False, merge_cols=False,
                   cells=None, sample=None, most_freq=None, pos=None):
        """
        Preprocess a Paralex paradigms table to meet the requirements of Qumin:
            - Remove
        """
        lexemes, cell_col, form_col = self.default_cols
        paradigms = self.data

        if pos:
            if 'lexemes' in self.dataset.resource_names:
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
            paradigms.drop(paradigms[paradigms.loc[:, lexemes].isin(defective_lexemes)].index,
                           inplace=True)

        if not overabundant:
            paradigms.drop_duplicates([lexemes, cell_col], inplace=True)

        if most_freq:
            inflected = paradigms.loc[:, lexemes].unique()
            lexemes_file_name = \
                Path(self.dataset.basepath) / self.dataset.get_resource("lexemes").path
            lexemes_df = pd.read_csv(lexemes_file_name, usecols=["lexeme_id", "frequency"])
            # Restrict to lexemes we have kept, if we dropped defectives
            lexemes_df = lexemes_df[lexemes_df.lexeme_id.isin(inflected)]
            selected = set(lexemes_df.sort_values("frequency", ascending=False)
                           .iloc[:most_freq, :].loc[:, "lexeme_id"].to_list())
            paradigms.drop(paradigms.loc[~paradigms.lexeme.isin(selected), :],
                           inplace=True)

        if sample:
            paradigms = paradigms.sample(sample)

        if not {lexemes, cell_col, form_col} < set(paradigms.columns):
            log.warning("Please use Paralex-style long-form table (http://www.paralex-standard.org).")

        self._drop_cells(paradigms, cells, 'cell')
        paradigms[form_col] = paradigms[form_col].fillna(value="")

        if segcheck:
            log.info("Checking we have definitions for all the phonological segments in this data...")
            unknowns = defaultdict(list)
            paradigms[[cell_col, form_col]].progress_apply(self._get_unknown_segments,
                                                           unknowns=unknowns, axis=1)

            if len(unknowns) > 0:
                alert = "Your paradigm has unknown segments: " + "\n ".join(
                    "[{}] (in {} forms:{}) ".format(
                        u, len(unknowns[u]), ", ".join(unknowns[u][:10])) for u in unknowns)
                raise ValueError(alert)

        # Create Form() objects from strings representations.
        paradigms[form_col] = paradigms[['form_id', form_col]].apply(
            lambda x: Form(x[form_col], x.form_id), axis=1)

        paradigms.rename(columns={form_col: "form"}, inplace=True)
        paradigms.set_index('form_id', inplace=True)

        if merge_cols:
            self.merge_duplicate_columns(sep="#")
        log.debug(paradigms)
        self.data = paradigms
        self._update_cell()

    def _drop_cells(self, paradigms, cells, column):
        """ Drops cells from a table.
        Performs security check before dropping.

        Arguments:
            paradigms (pandas.DataFrame): the paradigms to alter.
            cells (List[Str]): the list of cells to drop.
            column (str): name of the column which contains the cells
        """

        col_cells = paradigms[column].unique()
        if cells is not None:
            unknown_cells = set(cells) - set(col_cells)
            if unknown_cells:
                raise ValueError(f"You specified some cells which aren't in the paradigm : {' '.join(unknown_cells)}")
            to_drop = set(col_cells) - set(cells)
            if len(to_drop) > 0:
                log.info(f"Dropping rows with following cell values: {', '.join(sorted(to_drop))}")
            paradigms.drop(paradigms[paradigms[column].isin(to_drop)].index,
                           inplace=True)

    def merge_duplicate_columns(self, sep="#", keep_names=True):
        """Merge duplicate columns and return new DataFrame.

        Arguments:
            df (:class:`pandas:pandas.DataFrame`): A dataframe
            sep (str): separator to use when joining columns names.
            keep_names (bool): Whether to keep the names of the original duplicated
                columns by merging them onto the columns we keep.
        """
        log.info("Merging identical columns...")
        names = defaultdict(list)
        col = self.data.cell.unique()
        n_col = len(col)

        for c in tqdm(col):
            hashable = tuple(sorted(self.data.loc[self.data.cell == c, ['form', 'lexeme']]
                                    .apply(tuple, axis=1).to_list()))
            names[hashable].append(c)
        keep = [i[0] for i in names.values()]
        self.dropped = self.data[~self.data.cell.isin(keep)].copy()
        self.data.drop(self.dropped.index, inplace=True)
        if keep_names:
            new_names = {i[0]: sep.join(i) for i in names.values()}
            self.data.cell = self.data.cell\
                .cat.rename_categories(new_names)\
                .cat.remove_unused_categories()
        self._update_cell()
        log.info("Reduced from %s to %s columns", n_col, len(self.cells))

    def unmerge_columns(self, sep="#"):
        """Unmerges columns that were previously merged with
        `Paradigms.merge_duplicate_columns()`.
        """
        names = {col: col.split(sep)[0] for col in self.data.cell.unique() if sep in col}
        self.data.cell = self.data.cell.cat.rename_categories(names)
        # Controlling for category conformity
        uc = union_categoricals([self.data.cell, self.dropped.cell])
        self.data.cell = pd.Categorical(self.data.cell, categories=uc.categories )
        self.dropped.cell = pd.Categorical(self.dropped.cell, categories=uc.categories )
        self.data = pd.concat([self.data, self.dropped])
        self.dropped = None
        self._update_cell()

    def get_empty_pattern_df(self, a, b):
        """
        Returns an oriented dataframe to store
        patterns for two cells.
        """
        new = pd.merge(self.data.loc[self.data.cell == a],
                       self.data.loc[self.data.cell == b],
                       on="lexeme")
        return new

    def _update_cell(self):
        self.cells = list(self.data.cell.unique())
