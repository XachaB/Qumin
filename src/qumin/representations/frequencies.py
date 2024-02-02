# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Jules Bouton.

Functions for frequency management.
"""

import pandas as pd
import logging

log = logging.getLogger()


class Weights(object):
    """Frequency information for a language. Based on frequency tables in
    Paralex format.

    Examples:

        >>> w = Weights('tests/data/frequencies.csv')

    Attributes:

        frequencies (:class:`pandas:pandas.DataFrame`):
            Table of frequency values read from a Paralex file.

        col_names (List[str]):
            List of column names on which operation are performed. Usually lexeme, cell, form.
        """

    def __init__(self, filename,
                 col_names=["lexeme", "cell", "form"], default_source=None):
        """Constructor for Weights.

        Arguments:
            filename (str): Path to a frequency file in Paralex Format. Required is the form_id columns. At least one of the following columns should be there : lexeme, cell, form.
            col_names (dict): mapping of names for the following columns : lexeme, cell, form.
            default_source (str): name of the source to use if nothing specified. Defaults to a random source.
        """

        self.col_names = col_names
        self.origin = filename
        log.info('Reading frequency table')
        self.weight = pd.read_csv(filename, index_col='freq_id', usecols=col_names+['value', 'freq_id'])
        freq_col = self.weight.columns
        if set(col_names) < set(freq_col) & set(col_names) != set():
            raise ValueError(f"These column names don't appear in the frequency table: {set(col_names)-set(freq_col)}")

        if "source" not in freq_col:
            self.weight['source'] = 'default'
            self.default_source = 'default'
        elif default_source is None:
            self.default_source = list(self.weight['source'].unique())[0]
            log.info(f"No default source provided for frequencies. Using {self.default_source}")

    def get_freq(self, filters={}, group_on=None, source=None, mean=False):
        """
        Return the frequency of an item for a given source

        The frequency of an item is defined as the sum of the frequencies of this item,
        across all rows.

        Examples:

            >>> w = Weights('tests/data/frequencies.csv')
            >>> w.get_freq({'lexeme':'aa'})
            29.0
            >>> w.get_freq({'cell':'par'}, mean=True)
            4.0
            >>> w.get_freq(group_on=['lexeme'])
            lexeme
            aa    29.0
            bb     NaN
            Name: value, dtype: float64

        Arguments:
            filters (dict): a mapping of the following form `{"lexeme": value, "cell": value, "form": value}`. At least one key is required.
            source (str): the name of the source to use. If nothing is provided, the default source is selected.
            mean (bool): Defaults to False. If True, returns a mean instead of a sum.
        """

        # Filter using keys from mapping dict
        sublist = self._filter_weights(filters, source)

        if group_on is None:
            if mean:
                return sublist['value'].mean(skipna=False)
            else:
                return sublist['value'].sum(skipna=False)
        else:
            if mean:
                return sublist.groupby(by=group_on, group_keys=False)['value'].apply(lambda x: x.mean(skipna=False))
            else:
                return sublist.groupby(by=group_on, group_keys=False)['value'].apply(lambda x: x.sum(skipna=False))

    def get_relative_freq(self, filters={}, group_on=None, source=None):
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

        # Filter using keys from mapping dict
        sublist = self._filter_weights(filters, source)

        def _compute_rel_freq(x):
            if x.isna().values.any() or x['value'].sum() == 0:
                x['value'] = 1/x['value'].shape[0]
                return x['value']
            else:
                return x['value']/x['value'].sum(skipna=False)

        sublist['result'] = sublist.groupby(by=group_on, group_keys=False).apply(_compute_rel_freq).T

        return sublist

    def _filter_weights(self, filters, source):
        missing = set(filters.keys())-set(self.col_names)
        if missing:
            log.warning("You passed some column names that don't exist. They will be ignored: %s",
                        ", ".join(missing))
        mapping = {k: v for k, v in filters.items() if v is not None and k not in missing}
        if source is None:
            source = self.default_source
        mapping["source"] = source
        return self.weight.loc[(self.weight[list(mapping)] == pd.Series(mapping)).all(axis=1)].copy()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
