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

        >>> p = fl.Package('tests/data/TestPackage/test.package.json')
        >>> Frequencies.initialize(p)
        >>> print(Frequencies.info().to_markdown())
        | Table   | Source      |   Records |   Sum(f) |   Mean(f) |
        |:--------|:------------|----------:|---------:|----------:|
        | forms   | forms_table |        18 |      459 |   28.6875 |
        | lexemes | forms_table |         4 |      459 |  114.75   |
        | cells   | forms_table |         3 |      459 |  153      |


    Attributes:
        p (frictionless.Package): package to analyze
        real (bool): Whether the frequencies are real or fake.
        source (Dict[str, str]): source used by default for each table.
            Contains either a value for the source field of a Paralex frequency table,
            or the name of the table used to extract the frequency.
        forms (:class:`pandas:pandas.DataFrame`):
            Table of frequency values associated to a form_id.
        lexemes (:class:`pandas:pandas.DataFrame`):
            Table of frequency values associated to a lexeme_id.
        cells (:class:`pandas:pandas.DataFrame`):
            Table of frequency values associated to a cell_id.
        """

    p = None
    col_names = ["lexeme", "cell", "form"]
    source = {"cells": None,
                      "lexemes": None,
                      "forms": None}

    @classmethod
    def initialize(cls, package, source=False, **kwargs):
        """Constructor for Frequencies. We gather and store frequencies
        for forms, lexemes and cells. Behaviour is the following:
            - If `real` is `False`, we use the paradigms table to generate a Uniform distribution.
            - If not, we try to get a frequency column from the tables: form, lexemes, cell
            - If any of those is missing, we use the frequencies table.

        Arguments:
            p (frictionless.Package): package to analyze
            source (Dict[str, str]): name of the source to use when several are available.
            **kwargs: keyword arguments for frequency reading methods.
        """

        cls.p = package

        if source:
            cls.source.update(source)

        cls._read_form_frequencies(**kwargs)
        cls._read_other_frequencies("lexemes", **kwargs)
        cls._read_other_frequencies("cells", **kwargs)

    @classmethod
    def _read_form_frequencies(cls, real=True):
        """Recover the available information about form frequencies.

        Arguments:
            real (bool): Whether to use real frequencies or
                empty uniform distributions. Defaults to True.
        """
        paradigms = px.read_table("forms", cls.p)

        if real and "frequency" in paradigms.columns:
            log.info('forms: Frequencies in the table. Reading them.')
            cls.forms = paradigms
            cls.forms['source'] = 'forms_table'
            cls.forms.rename({"frequency": "value"}, axis=1, inplace=True)
            cls.source['forms'] = 'forms_table'

        elif real and cls.p.has_resource("frequencies"):
            log.info('No frequencies in the paradigms table, looking for a frequency table.')
            freq = px.read_table('frequency', cls.p, index_col='freq_id',
                                 usecols=['form', 'value', 'freq_id'])
            freq_col = freq.columns
            if "form" not in freq_col:
                raise ValueError("The form column is required in the frequency table if real=True and no frequencies in the forms table.")

            if "source" not in freq_col:
                freq['source'] = 'frequencies_table'
                cls.source['forms'] = 'frequencies_table'
            elif cls.source['forms'] is None:
                cls.source['forms'] = list(cls.freq['source'].unique())[0]
                log.info(f"No default source provided for frequencies. Using {cls.source['forms']}")

            # We use the form_id column to match both dataframes
            paradigms.set_index('form_id', inplace=True)
            freq.set_index('form', inplace=True)

            missing_idx = ~paradigms.index.isin(freq.index)
            if missing_idx.any():
                log.warning(f"""The frequencies table does not contain a row for every form_id row. Missing:
                            {paradigms.loc[missing_idx].head()}""")

            paradigms.loc[cls.freq.index, 'value'] = freq.value
            cls.forms = paradigms.copy()

        else:
            if real:
                log.warning("""No form frequencies were available in the Paralex dataset.""")
            log.info('forms: Building empty frequencies.')
            cls.forms = paradigms
            cls.forms['source'] = 'empty'
            cls.forms['value'] = pd.NA
            cls.source['forms'] = 'empty'

        # Check for duplicate overabundant phon_forms and sum the frequencies.
        # This handles cases where the orth_form is different and has two records.
        # Paradigms should be read only once, and this code shouldn't be redundant with
        # the main script. This should be fixed elsewhere.
        cls.forms['form'] = cls.forms.phon_form
        dup = cls.forms.duplicated(subset=cls.col_names, keep=False)
        if dup.any():
            cls.forms.loc[dup, 'value'] = \
                cls.forms.loc[dup].groupby(cls.col_names).value.transform(sum)
            cls.forms.drop_duplicates(subset=cls.col_names, inplace=True)

        cls.forms = cls.forms[['form_id', 'cell', 'lexeme', 'value', 'source']]\
            .set_index('form_id').sort_index()
        cls.forms.index.name = "form"

    @classmethod
    def _read_other_frequencies(cls, name, real=True):
        """
        Recover frequency information for cells and lexemes.

        Arguments:
            name(str): Frequency table to build. Either cells or lexemes.
            real (bool): Whether to use real frequencies or
                empty uniform distributions. Defaults to True.
        """
        table = px.read_table(name, cls.p, index_col=name[:-1] + "_id")

        # There are up to 3 different situations:
        # 1. Reading frequencies from the given table.
        if real and "frequency" in table.columns:
            log.info(f'{name}: Frequencies in the table. Reading them.')
            table['source'] = 'cells_table'
            table.rename({"frequency": "value"}, axis=1, inplace=True)
            cls.source[name] = name + '_table'

        # 2. Building frequencies from the forms table.
        elif real and (cls.source['forms'] != "empty"):
            log.info(f'{name}: No frequencies in the {name} table, building from the forms.')
            freq = cls.forms.groupby(name[:-1]).value.sum()
            table.loc[freq.index, "value"] = freq.values
            table['source'] = 'forms_table'
            cls.source[name] = 'forms_table'

        # 3. Building a fake uniform frequency distribution.
        else:
            log.info(f'{name}: Building empty frequencies.')
            table['source'] = 'empty'
            table['value'] = pd.NA
            cls.source[name] = 'empty'

        # We save the resulting table
        table.sort_index(inplace=True)
        table.index.name = name[:-1]
        setattr(cls, name, table[['value', 'source']])

    @classmethod
    def get_absolute_freq(cls, mean=False, group_on=False, skipna=False, **kwargs):
        """
        Return the frequency of an item for a given source

        The frequency of an item is defined as the sum of the frequencies of this item
        across all rows.

        Examples:

            >>> p = fl.Package('tests/data/TestPackage/test.package.json')
            >>> Frequencies.initialize(p, real=True)
            >>> Frequencies.get_absolute_freq(filters={'lexeme':'q'}, group_on="index", skipna=True)
            form
            11    12.0
            12     6.0
            14    20.0
            18     NaN
            Name: value, dtype: float64
            >>> Frequencies.get_absolute_freq(filters={'lexeme':'q'})
            nan
            >>> Frequencies.get_absolute_freq(filters={'cell':'third'}, mean=True, skipna=True)
            20.0
            >>> Frequencies.get_absolute_freq(group_on=['lexeme'])
            lexeme
            k    193.0
            p      NaN
            q      NaN
            s     33.0
            Name: value, dtype: float64

        Arguments:
            group_on (List[str]): columns on for which absolute frequencies should be computed.
                If `False, aggregates across all records.
            mean (bool): Defaults to False. If True, returns a mean instead of a sum.
            skipna(bool): Defaults to False. Skip nan values for sums or means.
        Returns:
            `pandas.Series`: a Series which contains the output values.
                The index is either the original one, or the grouping columns.
        """

        # Filter using keys from mapping dict
        sublist = cls._filter_frequencies(**kwargs)

        if group_on == "index":
            return sublist.value
        elif group_on is False:
            groups = [True] * len(sublist)
        else:
            groups = group_on

        if mean:
            def func(x): return x.mean(skipna=skipna)
        else:
            def func(x): return x.sum(skipna=skipna)

        result = sublist.groupby(by=groups, group_keys=False).value.apply(func)

        if group_on is False:
            return result.iloc[0]
        else:
            return result

    @classmethod
    def get_relative_freq(cls, group_on=False, **kwargs):
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

            >>> p = fl.Package('tests/data/TestPackage/test.package.json')
            >>> Frequencies.initialize(p, real=True)
            >>> Frequencies.get_relative_freq(filters={'lexeme': 'p', 'cell':'first'}, group_on=["lexeme"])['result'].values
            array([0.05882353, 0.94117647])
            >>> Frequencies.get_relative_freq(filters={'lexeme': 's', 'cell':'second'}, group_on=["lexeme"])['result'].values
            array([0., 1.])
            >>> Frequencies.get_relative_freq(filters={'cell':"third"}, group_on=["cell"])['result'].values
            array([0.25, 0.25, 0.25, 0.25])
            >>> Frequencies.get_relative_freq(filters={'lexeme':'p'}, group_on=["lexeme", "cell"])['result'].values
            array([0.05882353, 0.94117647, 1.        , 1.        ])

        Arguments:
            group_on (List[str]): column on which relative frequencies should be computed

        Returns:
            `pandas.DataFrame`: a DataFrame which contains a `result` column with the output value.
                The index is the original one. The grouping columns are also provided.
        """

        # Filter using keys from mapping dict
        sublist = cls._filter_frequencies(**kwargs)

        if group_on is False:
            groups = [True] * len(sublist)
            col_names = list()
        else:
            groups = group_on
            col_names = list(group_on)

        # 1. We first get the nb of items in each group
        sublist['result'] = sublist\
            .groupby(groups, sort=False).value\
            .transform("size")

        sublist.result = sublist.result.astype('float64')

        # 2. If there are any NaN values, we give a uniform frequency to the group
        sublist['notna'] = True
        sublist.loc[sublist.value.isna(), 'notna'] = False
        nanval = (sublist.result != 1) & ~sublist.groupby(groups).notna.transform('all')
        sublist.loc[nanval, 'result'] = 1/sublist.loc[nanval, 'result']

        # 3. If all values are filled and if the group is bigger than one, we sum the frequencies
        selector = sublist.result > 1

        if group_on is False:
            groups = selector

        sublist.loc[selector, 'result'] = sublist.loc[selector, 'value']/sublist.loc[selector]\
            .groupby(groups, sort=False).value.transform('sum')

        return sublist[col_names + ["result"]]

    @classmethod
    def _filter_frequencies(cls, data="forms", source=None, filters={}, inplace=False):
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

        missing = set(filters.keys())-set(cls.col_names)
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
            source = cls.source[data]
        if source is not False:
            mapping["source"] = [source]

        freq = getattr(cls, data).copy()
        idx_name = freq.index.name
        freq.reset_index(inplace=True)

        def _selector(mapping):
            """Avoid repetition of this complex line"""
            if mapping:
                return freq.loc[freq[list(mapping)].isin(mapping).all(axis=1)]\
                    .copy().set_index(idx_name)
            return freq.set_index(idx_name)

        if inplace:
            setattr(cls, data, _selector(mapping))
        else:
            return _selector(mapping)

    @classmethod
    def info(cls):
        """Returns a convenient DataFrame with summary statistics.

        Returns:
            `pandas.DataFrame`: A summary of statistics about this Frequencies handler.
        """
        metrics = []
        for i in ['forms', 'lexemes', 'cells']:
            data = getattr(cls, i)
            metrics.append([i, cls.source[i], len(data),
                            data.value.sum(), data.value.mean()])
        return pd.DataFrame(metrics, columns=['Table', 'Source', 'Records', 'Sum(f)', 'Mean(f)'])\
            .set_index('Table')


if __name__ == "__main__":
    import doctest
    doctest.testmod()