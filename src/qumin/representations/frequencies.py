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

        >>> w = Weights('tests/data/forms.csv', frequencies_path='tests/data/frequencies.csv', real_frequencies=True)

    Attributes:

        frequencies (:class:`pandas:pandas.DataFrame`):
            Table of frequency values read from a Paralex file.

        col_names (List[str]):
            List of column names on which operations are performed. Usually lexeme, cell, form.
        """

    def __init__(self, paradigm_path, frequencies_path=False, real_frequencies=False, filters={},
                 col_names=["lexeme", "cell", "form"], default_source=False):
        """Constructor for Weights.

        Arguments:
            paradigm_path (str): path to the paradigms.
            frequencies_path (str): frequencies file required if `real_frequencies=True`
                Should contain one of these columns: lexeme, cell, form.
            real_frequencies (bool): Whether to use real relative frequencies or
                uniform distribution. Default to False.
            filters (Dict[str, str]): A dictionary of column-value pairs to filter the frequencies file.
            col_names (List[str]): mapping of names for the following columns : lexeme, cell, form.
            default_source (str): name of the source to use if several are available
                in the frequency file. Defaults to a random source.
        """

        self.col_names = col_names
        paradigms = pd.read_csv(paradigm_path, dtype={'freq_id': 'int64'})

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
            self.weight = pd.read_csv(frequencies_path, index_col='freq_id',
                                      usecols=col_names+['value', 'freq_id'])
            freq_col = self.weight.columns
            if set(col_names) < set(freq_col) & set(col_names) != set():
                raise ValueError(f"These column names are missing in the frequency table: {set(col_names)-set(freq_col)}")

            if "source" not in freq_col:
                self.weight['source'] = 'default'
                self.default_source = 'default'
            elif default_source is False:
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

        The frequency of an item is defined as the sum of the frequencies of this item
        across all rows.

        Examples:

            >>> w = Weights('tests/data/forms.csv', frequencies_path='tests/data/frequencies.csv', real_frequencies=True)
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
            filters (dict): a mapping of the following form `{"lexeme": value,
                "cell": value, "form": value}`. At least one key is required.
            source (str): the name of the source to use.
                If nothing is provided, the default source is selected.
            mean (bool): Defaults to False. If True, returns a mean instead of a sum.
        """

        # Filter using keys from mapping dict
        sublist = self._filter_weights(filters, source)

        if group_on is None:
            if mean:
                return sublist.value.mean(skipna=False)
            else:
                return sublist.value.sum(skipna=False)
        else:
            # TODO Replicate the same speed improvements as for relative frequencies ?
            # Not necessary for the moment.
            if mean:
                return sublist.groupby(by=group_on, group_keys=False).value.apply(lambda x: x.mean(skipna=False))
            else:
                return sublist.groupby(by=group_on, group_keys=False).value.apply(lambda x: x.sum(skipna=False))

    def get_relative_freq(self, filters={}, group_on=None, source=None):
        """
        Return the relative frequency of an item for a given source

        The frequency of an item is defined as the sum of the frequencies of this item,
        across all rows.

        Note:
            For very large dataframes, computations can be long, be careful.

        Examples:

            >>> w = Weights('tests/data/forms.csv', frequencies_path='tests/data/frequencies.csv', real_frequencies=True)
            >>> w.get_relative_freq(filters={'lexeme':'bb', 'cell':'nom'}, group_on=["lexeme"])['result'].values
            array([0.95454545, 0.04545455])
            >>> w.get_relative_freq(filters={'cell':"gen"}, group_on=["cell"])['result'].values
            array([0.5, 0.5])
            >>> w.get_relative_freq(filters={'cell':"par"}, group_on=["cell"])['result'].values
            array([1., 0.])
            >>> w.get_relative_freq(filters={'lexeme':'bb'}, group_on=["lexeme", "cell"])['result'].values
            array([0.95454545, 0.04545455, 1.        , 1.        ])

        Arguments:
            filters (dict): a mapping of the following form `{"lexeme": value,
                "cell": value, "form": value}`. At least one key is required.
            group_on (List[str]): column on which relative frequencies should be computed
            source (str): the name of the source to use. If nothing is provided,
                the default source is selected.
        """

        # Filter using keys from mapping dict
        sublist = self._filter_weights(filters, source)

        # We first get the size of each group
        sublist['result'] = sublist\
            .groupby(group_on, sort=False).value\
            .transform("size")

        sublist['result'] = sublist.result.astype('float64')

        # If there are any nan values, we give a uniform weight
        sublist['notna'] = True
        sublist.loc[sublist.value.isna(), 'notna'] = False
        nanval = (sublist.result != 1) & ~sublist.groupby(group_on).notna.transform('all')
        sublist.loc[nanval, 'result'] = 1/sublist.loc[nanval, 'result']

        # For bigger groups we sum the frequencies
        selector = sublist['result'] > 1
        sublist.loc[selector, 'result'] = sublist.loc[selector, 'value']/sublist.loc[selector]\
            .groupby(group_on, sort=False)['value']\
            .transform('sum')

        # TODO this should be replaced by a simpler .transform(sum, skipna=False),
        # However, skipna is not yet implemented for GroupBy.sum

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
