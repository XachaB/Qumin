# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Jules Bouton.

Functions for frequency management.
"""

import pandas as pd
from tqdm import tqdm
import logging
tqdm.pandas()

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

    def __init__(self, md, real_frequencies=False, filters={},
                 col_names=["lexeme", "cell", "form"], default_source=None):
        """Constructor for Weights.

        Arguments:
            filename (str): Path to a frequency file in Paralex Format. Required is the form_id columns. At least one of the following columns should be there : lexeme, cell, form.
            col_names (dict): mapping of names for the following columns : lexeme, cell, form.
            default_source (str): name of the source to use if nothing specified. Defaults to a random source.
        """

        self.col_names = col_names
        paradigms = pd.read_csv(md.get_table_path("forms"), dtype={'freq_id': 'int64'})

        if real_frequencies is False:
            log.info('Building normalized weights for the provided columns...')
            self.weight = paradigms
            self.weight['source'] = 'default'
            self.weight.rename({'phon_form': 'form'}, axis=1, inplace=True)
            self.weight['value'] = pd.NA
            self.default_source = 'default'
            paradigms.set_index('form_id', inplace=True)
        else:
            log.info('Reading frequency table...')
            self.weight = pd.read_csv(md.get_table_path("frequencies"), index_col='freq_id',
                                      usecols=col_names+['value', 'freq_id'])
            freq_col = self.weight.columns
            if set(col_names) < set(freq_col) & set(col_names) != set():
                raise ValueError(f"These column names don't appear in the frequency table: {set(col_names)-set(freq_col)}")

            if "source" not in freq_col:
                self.weight['source'] = 'default'
                self.default_source = 'default'
            elif default_source is None:
                self.default_source = list(self.weight['source'].unique())[0]
                log.info(f"No default source provided for frequencies. Using {self.default_source}")

            # The form_id should be replaced with phon_form values from the Paralex paradigms
            paradigms.set_index('form_id', inplace=True)
            # TODO
            # In a further version, this should be useless and only form_id should be used.
            self.weight.set_index('form', inplace=True)

            # Take the intersection of both indexes
            ixs = self.weight.index.intersection(paradigms.index)
            self.weight['form'] = paradigms.loc[ixs]['phon_form']
            self.weight = self.weight[~self.weight['form'].isna()]
            self.weight.reset_index(inplace=True, names='form_id')
        self.weight.sort_index(inplace=True)
        self._filter_weights(filters, source=False, inplace=True)

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
            # TODO Replicate the same speed improvements as for relative frequencies ?
            if mean:
                return sublist.groupby(by=group_on, group_keys=False)['value'].apply(lambda x: x.mean(skipna=False))
            else:
                return sublist.groupby(by=group_on, group_keys=False)['value'].apply(lambda x: x.sum(skipna=False))

    def get_relative_freq(self, filters={}, group_on=None, source=None):
        """
        Return the relative frequency of an item for a given source

        The frequency of an item is defined as the sum of the frequencies of this item,
        across all rows.

        Note:
            For very large dataframes, computations can be long, be careful.

        Examples:

            >>> w = Weights('tests/data/frequencies.csv')
            >>> w.get_relative_freq(filters={'cell':"gen"}, group_on=["cell"])['result'].values
            array([0.5, 0.5])
            >>> w.get_relative_freq(filters={'lexeme':'bb', 'cell':'nom'}, group_on=["lexeme"])['result'].values
            array([0.95454545, 0.04545455])
            >>> w.get_relative_freq(filters={'cell':"par"}, group_on=["cell"])['result'].values
            array([1., 0.])
            >>> w.get_relative_freq(filters={'lexeme':'bb'}, group_on=["lexeme", "cell"])['result'].values
            array([0.95454545, 0.04545455, 1.        , 1.        ])

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

        # We first get the size of each group
        sublist['result'] = sublist\
            .groupby(group_on, sort=False)['value']\
            .transform("size")

        sublist['result'] = sublist['result'].astype('float64')

        # And if there were some nan values, we give a uniform weight
        # (we could also count nan as a zero)
        # This is the slowest part (not using a C implementation)
        sublist.loc[sublist['result'] != 1, 'result'] = sublist.loc[sublist['result'] != 1]\
            .groupby(group_on, sort=False)['value']\
            .transform(lambda x: 1/x.size if x.isna().any() else 2)

        # For bigger groups we sum
        selector = sublist['result'] == 2
        sublist.loc[selector, 'result'] = sublist.loc[selector]['value']/sublist.loc[selector]\
            .groupby(group_on, sort=False)['value']\
            .transform('sum')

        # TODO this should be replaced by a simpler .transform(sum, skipna=False),
        # However, skipna is not yet implemented for GroupBy.sum
        # Another solution is :
        # sublist.loc[sublist['result'] != 1, 'result'] = sublist.loc[sublist['result'] != 1].groupby(
        #     group_on, sort=False)['value'].transform(lambda x: x/x.sum(skipna=False))
        # sublist.loc[sublist['result'].isna(), 'result'] = \
        #     sublist.loc[sublist['result'].isna()]\
        #     .groupby(group_on, sort=False)['value']\
        #     .transform(lambda x: 1/x.size)

        sublist.reset_index(inplace=True)
        sublist.set_index(self.col_names, inplace=True)
        return sublist

    def _filter_weights(self, filters, source, inplace=False):
        """Filters the dictionary based on a set of filters
        provided as a dictionary by the user."""

        missing = set(filters.keys())-set(self.col_names)
        if missing:
            log.warning("You passed some column names that don't exist. They will be ignored: %s",
                        ", ".join(missing))

        def _listify(x):
            """Ensure that passed values of mapping are list-like objects"""
            if isinstance(x, str):
                x = [x]
            else:
                try:
                    iter(x)
                except TypeError:
                    x = [x]
                else:
                    x = list(x)
            return x

        mapping = {k: _listify(v) for k, v in filters.items() if v is not None and k not in missing}

        if source is None:
            source = self.default_source
        if source is not False:
            mapping["source"] = [source]

        def _selector(mapping):
            """Avoid repetition of this complex line"""
            if mapping:
                return self.weight.loc[self.weight[list(mapping)].isin(mapping).all(axis=1)].copy()
            else:
                return self.weight

        if inplace:
            self.weight = _selector(mapping)
        else:
            return _selector(mapping)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
