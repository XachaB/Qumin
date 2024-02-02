# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Jules Bouton.

Functions for frequency management.
"""

import pandas as pd
import logging

log = logging.getLogger()


class Weights():
    """Frequency information for a language. Based on frequency tables in
    Paralex format

    Examples:

        >>> w = Weights('tests/data/frequencies.csv')
        >>> wn = Weights('../Lexiques/Estonien-copie/estonian_frequencies.csv')

    Attributes:

        frequencies (:class:`pandas:pandas.DataFrame`):
            containing forms.

        headers (:class:`pandas:pandas.DataFrame`):
            containing pairwise patterns of alternation.
        """

    def __init__(self, filename,
                 col_names={"lexeme": "lexeme",
                            "cell": "cell",
                            "form": "form"}, default_source=None):
        """Constructor for Weights.

        Arguments:
            filename (str): Path to a frequency file in Paralex Format. Required is the form_id columns. At least one of the following columns should be there : lexeme, cell, form.
            col_names (dict): mapping of names for the following columns : lexeme, cell, form.
        """

        self.col_names = {"lexeme": "lexeme",
                          "cell": "cell",
                          "form": "form"}.update(col_names)
        self.origin = filename
        log.info('Reading frequency table')
        self.weight = pd.read_csv(filename, index_col='freq_id')
        freq_col = self.weight.columns
        if set(col_names.values()) < set(freq_col) & set(col_names.values()) != set():
            raise ValueError(f"These column names don't appear in the frequency table: {set(col_names.values())-set(freq_col)}")

        if "source" not in freq_col:
            self.weight['source'] = 'default'
            self.default_source = 'default'
        elif default_source is None:
            self.default_source = list(self.weight['source'].unique())[0]
            log.info(f"No default source provided for frequencies. Using {self.default_source}")

    def get_freq(self, pairs, source=None, mean=False):
        """
        Return the frequency of an item for a given source

        The frequency of an item is defined as the sum of the frequencies of this item,
        across all rows.

        Examples:

            >>> w = Weights('tests/data/frequencies.csv')
            >>> w.get_freq({'lexeme':'aa'})
            29.0
            >>> w.get_freq({'lexeme':None, 'cell':'par'}, mean=True)
            4.0
            >>> w.get_freq({'lexeme':'bb', 'form':'bbb'})
            nan

        Arguments:
            pairs (dict): a mapping of the following form `{"lexeme": value, "cell": value, "form": value}`. At least one key is required.
            source (str): the name of the source to use. If nothing is provided, the default source is selected.
            mean (bool): Defaults to False. If True, returns a mean instead of a sum.
        """

        # Filter out None values
        mapping = {k: v for k, v in pairs.items() if v is not None}
        if source is None:
            source = self.default_source
        mapping.update({"source": source})

        # Filter using keys from mapping dict
        sublist = self.weight.loc[(self.weight[list(mapping)] == pd.Series(mapping)).all(axis=1)]

        if mean:
            return sublist['value'].mean(skipna=False)
        else:
            return sublist['value'].sum(skipna=False)

    def get_relative_freq(self, filters=None, group_on=None, source=None):
        """
        Return the relative frequency of an item for a given source

        The frequency of an item is defined as the sum of the frequencies of this item,
        across all rows.

        Examples:

            >>> w = Weights('tests/data/frequencies.csv')
            >>> w.get_relative_freq(filters={'lexeme':'bb', 'cell':'nom'}, group_on=["lexeme"])['result']
            freq_id
            5    0.954545
            6    0.045455
            Name: result, dtype: float64
            >>> w.get_relative_freq(filters={'cell':"gen"}, group_on=["cell"])['result']
            freq_id
            2    0.5
            7    0.5
            Name: result, dtype: float64
            >>> w.get_relative_freq(filters={'cell':"par"}, group_on=["cell"])['result']
            freq_id
            3    1.0
            8    0.0
            Name: result, dtype: float64
            >>> w.get_relative_freq(filters={'lexeme':'bb'}, group_on=["lexeme", "cell"])['result']
            freq_id
            5    0.954545
            6    0.045455
            7    1.000000
            8    1.000000
            Name: result, dtype: float64

        Arguments:
            filters (dict): a mapping of the following form `{"lexeme": value, "cell": value, "form": value}`. At least one key is required.
            group_on (List['str']): column on which relative frequencies should be computed
            source (str): the name of the source to use. If nothing is provided, the default source is selected.
        """

        mapping = dict()
        # Filter out None values
        if filters is not None:
            mapping = {k: v for k, v in filters.items() if v is not None}
        if source is None:
            source = self.default_source
        mapping.update({"source": source})

        # Filter using keys from mapping dict
        sublist = self.weight.loc[(self.weight[list(mapping)] == pd.Series(mapping)).all(axis=1)].copy()

        def _compute_rel_freq(x):
            if x.isna().values.any() or x['value'].sum() == 0:
                x['value'] = 1/x['value'].shape[0]
                return x['value']
            else:
                return x['value']/x['value'].sum(skipna=False)

        sublist['result'] = sublist.groupby(by=group_on, group_keys=False).apply(_compute_rel_freq).T

        return sublist


if __name__ == "__main__":
    import doctest
    doctest.testmod()
