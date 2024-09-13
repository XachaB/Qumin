# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Jules Bouton.

Class for frequency management.
"""

import pandas as pd
from tqdm import tqdm
import paralex as px
import frictionless as fl
import logging
tqdm.pandas()

log = logging.getLogger()


class Frequencies(object):
    """Frequency management for a Paralex dataset. Frequencies are built for forms,
    lexemes and cells.

    The parsed frequency columns or tables should conform to the Paralex principles:
        - An empty value means that there is no measure available
        - A zero value means that there is a measure, which is zero

    When aggregating accross rows, any empty cell yields a uniform distribution
    for the whole set of rows, whereas zeros are taken into account. This behaviour
    can be disabled for some functions by passing skipna=True.

    Examples:

        >>> p = fl.Package('../Lexiques/Test/test.package.json')
        >>> f = Frequencies(p, real=False)

    Attributes:
        p (frictionless.Package): package to analyze
        real (bool): Whether the frequencies are real or fake.
        default_source (Dict[str, str]): source used by default for each table.
            Contains either a value for the source field of a Paralex frequency table,
            or the name of the table used to extract the frequency.
        forms (:class:`pandas:pandas.DataFrame`):
            Table of frequency values associated to a form_id.
        lexemes (:class:`pandas:pandas.DataFrame`):
            Table of frequency values associated to a lexeme_id.
        cells (:class:`pandas:pandas.DataFrame`):
            Table of frequency values associated to a cell_id.
        """

    def __init__(self, package, real=True, default_source=False):
        """Constructor for Frequencies. We gather and store frequencies
        for forms, lexemes and cells. Behaviour is the following:
            - If `real` is `False`, we use the paradigms table to generate a Uniform distribution.
            - If not, we try to get a frequency column from the tables: form, lexemes, cell
            - If any of those is missing, we use the frequencies table.

        Arguments:
            p (frictionless.Package): package to analyze
            real (bool): Whether to use real relative frequencies or
                uniform distributions. Defaults to True.
            default_source (Dict[str, str]): name of the source to use when several are available.
        """

        self.p = package
        self.real = real
        self.col_names = ["lexeme", "cell", "form"]
        self.default_source = {"cells": None,
                               "lexemes": None,
                               "forms": None}
        if default_source:
            self.default_source.update(default_source)

        self._read_form_frequencies()
        self._read_other_frequencies("lexemes")
        self._read_other_frequencies("cells")

    def _read_form_frequencies(self, default_source=False):
        """Recover the available information about form frequencies.
        """
        paradigms = px.read_table("forms", self.p)

        if self.real is False:
            log.info('Building normalized frequencies for the paradigm columns...')
            self.forms = paradigms
            self.forms['source'] = 'empty'
            self.forms.rename({'phon_form': 'form'}, axis=1, inplace=True)
            self.forms['value'] = pd.NA
            self.default_source['forms'] = 'empty'

        elif "frequency" in paradigms.columns:
            log.info('Frequencies in in the forms table. Reading them...')
            self.forms = paradigms
            self.forms['source'] = 'forms_table'
            self.forms.rename({'phon_form': 'form', "frequency": "value"}, axis=1, inplace=True)
            self.default_source['forms'] = 'forms_table'

        elif self.p.has_resource("frequencies"):
            log.info('No frequencies in the paradigms table, looking for a frequencies table...')
            freq = px.read_table('frequency', self.p, index_col='freq_id',
                                 usecols=['form', 'value', 'freq_id'])
            freq_col = freq.columns
            if "form" not in freq_col:
                raise ValueError("The form column is required in the frequency table if real=False.")

            if "source" not in freq_col:
                freq['source'] = 'frequencies_table'
                self.default_source['forms'] = 'frequencies_table'
            elif self.default_source['forms'] is None:
                self.default_source['forms'] = list(self.freq['source'].unique())[0]
                log.info(f"No default source provided for frequencies. Using {self.default_source['forms']}")

            # We use the form_id column to match both dataframes
            paradigms.set_index('form_id', inplace=True)
            freq.set_index('form', inplace=True)

            missing_idx = ~paradigms.index.isin(freq.index)
            if missing_idx.any():
                log.warning(f"""The frequencies table does not contain a row for every form_id row. Missing:
                            {paradigms.loc[missing_idx].head()}""")

            paradigms.loc[self.freq.index, 'value'] = freq.value
            self.forms = paradigms.rename({'phon_form': 'form'})
        else:
            raise ValueError("""If no form frequencies are available in the Paralex dataset,
                             real should be set to False""")

        self.forms = self.forms[['form_id', 'cell', 'lexeme', 'form', 'value', 'source']]

        # Check for duplicate overabundant phon_forms and sum the frequencies.
        # This handles cases where the orth_form is different and has two records.
        dup = self.forms.duplicated(subset=self.col_names, keep=False)
        if dup.any():
            self.forms.loc[dup, 'value'] = \
                self.forms.loc[dup].groupby(self.col_names).value.transform(sum)
            self.forms.drop_duplicates(subset=self.col_names, inplace=True)

        self.forms.sort_index(inplace=True)

    def _read_other_frequencies(self, name):
        """
        Recover frequency information for cells and lexemes.

        Arguments:
            name(str): Frequency table to build. Either cells or lexemes.
        """
        table = px.read_table(name, self.p, index_col=name[:-1] + "_id")

        # There are up to 3 different situations:
        # 1. Building a fake uniform frequency distribution.
        if self.real is False:
            log.info(f'{name}: Building normalized frequencies...')
            table['source'] = 'empty'
            table['value'] = pd.NA
            self.default_source[name] = 'empty'

        # 2. Reading frequencies from the given table.
        elif "frequency" in table.columns:
            log.info(f'{name}: Frequencies in the table. Reading them...')
            table['source'] = 'cells_table'
            table.rename({"frequency": "value"}, axis=1, inplace=True)
            self.default_source[name] = name + '_table'

        # 3. Building frequencies from the forms table.
        else:
            log.info(f'{name}: No frequencies in the paradigms table, building from the forms...')
            freq = self.forms.groupby(name[:-1]).value.sum()
            table.loc[freq.index, "value"] = freq.values
            table['source'] = 'forms_table'
            self.default_source[name] = 'forms_table'

        # We save the resulting table
        setattr(self, name, table[['value', 'source']])

    def get_absolute_freq(self, mean=False, group_on=None, skipna=False, **kwargs):
        """
        Return the frequency of an item for a given source

        The frequency of an item is defined as the sum of the frequencies of this item
        across all rows.

        Examples:

            >>> p = fl.Package('../Lexiques/Test/test.package.json')
            >>> f = Frequencies(p, real=True)
            >>> f.get_absolute_freq(filters={'lexeme':'q'}, skipna=True)
            38.0
            >>> f.get_absolute_freq(filters={'lexeme':'q'}, skipna=False)
            nan
            >>> f.get_absolute_freq(filters={'cell':'third'}, mean=True, skipna=True)
            20.0
            >>> f.get_absolute_freq(group_on=['lexeme'])
            lexeme
            k    193.0
            p      NaN
            q      NaN
            s     33.0
            Name: value, dtype: float64

        Arguments:
            mean (bool): Defaults to False. If True, returns a mean instead of a sum.
            skipna(bool): Defaults to False. Skip nan values for sums or means.
        """

        # Filter using keys from mapping dict
        sublist = self._filter_frequencies(**kwargs)

        if group_on is None:
            if mean:
                return sublist.value.mean(skipna=skipna)
            else:
                return sublist.value.sum(skipna=skipna)
        else:
            if mean:
                return sublist.groupby(by=group_on, group_keys=False).value.apply(lambda x: x.mean(skipna=skipna))
            else:
                return sublist.groupby(by=group_on, group_keys=False).value.apply(lambda x: x.sum(skipna=skipna))

    def get_relative_freq(self, group_on=None, **kwargs):
        """
        Returns the relative frequencies of a set of rows according to a set of grouping columns.
        If any of the values is empty, we generate a Uniform distribution for this group.

        Note:
            To avoid long computations, we use C implementations.
            Unfortunately, `skipna` is not yet implemented in `GroupBy.sum`. For this reason,
            we use a more complex pipeline of C functions.

        Todo:
            Replace the pipeline by a much simpler .transform(sum, skipna=False), once possible.

        Examples:

            >>> p = fl.Package('../Lexiques/Test/test.package.json')
            >>> f = Frequencies(p, real=True)
            >>> f.get_relative_freq(filters={'lexeme': 'p', 'cell':'first'}, group_on=["lexeme"])['result'].values
            array([0.05882353, 0.94117647])
            >>> f.get_relative_freq(filters={'lexeme': 's', 'cell':'second'}, group_on=["lexeme"])['result'].values
            array([0., 1.])
            >>> f.get_relative_freq(filters={'cell':"third"}, group_on=["cell"])['result'].values
            array([0.25, 0.25, 0.25, 0.25])
            >>> f.get_relative_freq(filters={'lexeme':'p'}, group_on=["lexeme", "cell"])['result'].values
            array([0.05882353, 0.94117647, 1.        , 1.        ])

        Arguments:
            group_on (List[str]): column on which relative frequencies should be computed
        """

        # Filter using keys from mapping dict
        sublist = self._filter_frequencies(**kwargs)

        # 1. We first get the nb of items in each group
        sublist['result'] = sublist\
            .groupby(group_on, sort=False).value\
            .transform("size")

        sublist['result'] = sublist.result.astype('float64')

        # 2. If there are any NaN values, we give a uniform frequency to the group
        sublist['notna'] = True
        sublist.loc[sublist.value.isna(), 'notna'] = False
        nanval = (sublist.result != 1) & ~sublist.groupby(group_on).notna.transform('all')
        sublist.loc[nanval, 'result'] = 1/sublist.loc[nanval, 'result']

        # 3. If all values are filled and if the group is bigger than one, we sum the frequencies
        selector = sublist['result'] > 1
        sublist.loc[selector, 'result'] = sublist.loc[selector, 'value']/sublist.loc[selector]\
            .groupby(group_on, sort=False).value.transform('sum')

        sublist.reset_index(inplace=True)
        sublist.set_index(self.col_names, inplace=True)
        return sublist

    def _filter_frequencies(self, source=None, filters={}, data="forms", inplace=False):
        """Filters the dataframe based on a set of filters
        provided as a dictionary.

        Arguments:
            filters (dict): a mapping of the following kind `{"lexeme": value,
                "cell": value, "form": value}`.
            data(str): name of one of the three tables (forms, lexemes, cells)
            source (str): the name of the source to use. If nothing is provided,
                the default source is selected.
            inplace (bool): whether the filter should operate in place or not. Defaults to False.
        """

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
            source = self.default_source[data]
        if source is not False:
            mapping["source"] = [source]

        freq = getattr(self, data)

        def _selector(mapping):
            """Avoid repetition of this complex line"""
            if mapping:
                return freq.loc[freq[list(mapping)].isin(mapping).all(axis=1)].copy()
            return freq

        if inplace:
            setattr(self, data, _selector(mapping))
        else:
            return _selector(mapping)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
