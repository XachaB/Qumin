# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Sacha Beniamine.

Encloses distribution of patterns on paradigms.
"""

import logging
from collections import Counter, defaultdict
from functools import reduce
from itertools import combinations

import pandas as pd

from . import cond_entropy, cond_entropy_slow, entropy, cond_psuccess

log = logging.getLogger("Qumin")


class PatternDistribution(object):
    """Statistical distribution of patterns.

    Attributes:
        patterns (~qumin.representations.patterns.ParadigmPatterns):
            A dict of :class:`pandas.DataFrame`, where each row describes an alternation between
            two cells forms belonging to different cells of the same lexeme.
            The row also contains the correct pattern and the set of applicable patterns.

        data (dict[int, pandas.DataFrame]):
            dict mapping n to a dataframe containing the entropies
            for the distribution :math:`P(c_{1}, ..., c_{n} → c_{n+1})`.

        name (str):
            Name of the dataset.
    """

    def __init__(self, patterns, name, frequencies, features=None, legacy=False):
        """Constructor for PatternDistribution.

        Arguments:
            patterns (~qumin.representations.patterns.ParadigmPatterns):
                A dict of :class:`pandas.DataFrame`,
                where each row describes an alternation between
                forms belonging to two different cells of the same lexeme.
                The row also contains the correct pattern and the set of applicable patterns.
            name (str): dataset name.
            frequencies (~qumin.representations.frequencies.Frequencies):
                The frequencies for the paradigms.

            features:
                optional table of features
        """
        self.name = name
        self.patterns = patterns
        self.frequencies = frequencies

        if features is not None:
            # Add feature names
            features = features.apply(lambda x: x.name + "=" + x.apply(str), axis=0)
            # To tuples
            features = features.map(lambda x: (str(x),))
            self.features_len = features.shape[1]
            self.features = pd.DataFrame.sum(features, axis=1)
        else:
            self.features_len = 0
            self.features = None

        self.data = pd.DataFrame(None,
                                 columns=["predictor",
                                          "predicted",
                                          "measure",
                                          "value",
                                          "n_pairs",
                                          "n_preds",
                                          "dataset"
                                          ])

    def get_results(self, measure=None, n=1):
        """
        Returns computation results from a distribution of patterns.

        Arguments:
            measure (List or str): measure name.
            n (int): number of predictors

        Returns:
            pandas.DataFrame: a DataFrame of results.

        """
        if isinstance(measure, str):
            measure = [measure]
        elif measure is None:
            measure = list(self.data.measure.unique())
        is_measure = self.data.loc[:, "measure"].isin(measure)
        is_one_pred = self.data.loc[:, "n_preds"] == n
        return self.data.loc[is_measure & is_one_pred, :]

    def export_file(self, filename):
        """ Export the :attr:`PatternDistribution.data` DataFrame to file

        Arguments:
            filename: the file's path.
        """

        def join_if_multiple(preds):
            if type(preds) is tuple:
                return "&".join(preds)
            return preds

        data = self.data.copy()
        data.loc[:, "predictor"] = data.loc[:, "predictor"].apply(join_if_multiple)
        if "entropy" in data.columns:
            # Rounding at 10 significant digits, ensuring positive zeros.
            data.loc[:, "entropy"] = data.loc[:, "entropy"].map(lambda x: round(x, 10)) + 0
        data.to_csv(filename, index=False)

    def import_file(self, filename):
        """Read already computed entropies from a file.

        Arguments:
            filename: the file's path.
        """

        def split_if_multiple(preds):
            if "&" in preds:
                return tuple(preds.split("&"))
            return preds

        data = pd.read_csv(filename)
        data.loc[:, "predictor"] = data.loc[:, "predictor"].apply(split_if_multiple)
        self.data = pd.concat(self.data, data)

    def add_features(self, group):
        """
        Adds lexeme features if available to a DataFrame containing a column named "applicable"
        and lexemes as indexes.

        Arguments:
            group (pandas.DataFrame): a dataframe of lexemes and applicable patterns.
        """
        if self.features:
            ret = group.applicable + group.lexeme.map(self.features)
            return ret
        else:
            return group.applicable

    def prepare_data(self, n=1, debug=False, legacy=False):
        """
        Prepares the dataframe to store the results for an entropy computation.

        Attributes:
            n (int): number of predictors to consider
            debug (bool): Whether the computation is a standard one or a debug one.
        Returns:
            pandas.DataFrame: a dataframe with the predictors and the predicted cells,
                as well as some metadata.
        """
        rows = self.patterns.cells
        idx = ["&".join(x) for x in combinations(rows, n)]

        data = pd.DataFrame(index=idx,
                            columns=rows).reset_index(drop=False,
                                                      names="predictor").melt(id_vars="predictor",
                                                                              var_name="predicted",
                                                                              value_name="value")
        suffix = "_debug" if debug else ""
        # drop A -> A cases
        data = data[data.apply(lambda x: x.predicted not in x.predictor.split('&'), axis=1)]
        data.loc[:, "n_pairs"] = None
        data.loc[:, "n_preds"] = n
        if n == 1:
            measures = ["cond_entropy" + suffix]
            if not legacy:
                measures += ['accuracy' + suffix]
            data.loc[:, "measure"] = [measures] * data.shape[0]
        else:
            data.loc[:, "measure"] = "cond_entropy" + suffix
        data.loc[:, "dataset"] = self.name
        data.set_index(['predictor', 'predicted'], inplace=True)
        return data

    def one_pred_metrics(self, legacy=False, debug=False,
                         token_patterns=False, token_predictors=False,
                         token_oa=False):
        r"""Return a :class:`pandas:pandas.DataFrame` with unary entropies and counts of lexemes.

        The result contains entropy :math:`H(c_{1} \to c_{2})`.

        Values are computed for all unordered pairs
        of columns :math:`(c_{1}, c_{2})` where `c_{1} != c_{2}`
        in the :attr:`PatternDistribution.patterns`'s keys.

        Example:
            For two cells c1, c2, entropy of c1 → c2,
            noted :math:`H(c_{1} \to c_{2})` is:

            .. math::

                H( \textrm{patterns}_{c1, c2} | \textrm{classes}_{c1, c2} )

        The probability distribution of the patterns, on which this entropy
        is computed, is established eithor on the type or token frequency of
        the patterns. The probability of the pattern (:math:`p_i`) is the sum
        of the frequencies of all pairs of forms :math:`(x, y)` that instanciate
        this pattern (:math:`S_i`):

        .. math ::

            P(p_i) = \frac{\sum_{(x, y)\in S_i} f_x f_y}
            {\sum_{L \in \mathcal{L}} f_{L_{c_1}} f_{L_{c_2}}}

        Where :math:`\mathcal{L}` is the set of all lexemes and :math:` f_{L_{c_1}}`
        is the frequency of the lexeme :math:`L` in cell :math:`c_1`.

        Arguments:
            debug (bool): Whether to print a debug log. Defaults to False
            token_patterns (bool): Whether to use token frequencies to compute
                pattern probabilities.
            token_predictors (bool): Whether to use token frequencies to compute
                pattern probabilities.
            token_oa (bool): Whether to use token frequencies to compute
                the relative weight of overabundant forms.
            legacy (bool): Whether to use legacy computations.
        """
        log.info("Computing c1 → c2 entropies")
        log.debug("Logging one predictor probabilities")
        log.debug(" P(x → y) = P(x~y | Class(x))")

        # For faster access
        patterns = self.patterns
        data = self.prepare_data(debug=debug, legacy=legacy)

        if not legacy:
            # Prepare frequency data
            tok_freq = self.frequencies.get_absolute_freq(group_on='form')\
                .to_dict()
            typ_freq = self.frequencies.get_relative_freq(group_on=["lexeme", 'cell'],
                                                          uniform_duplicates=not token_oa)\
                .result.to_dict()

        # Compute conditional entropy
        for pair, df in patterns.items():
            # Defective rows can't be kept here.
            selector = df.pattern.notna()
            df = df[selector].copy()

            if not legacy:
                # We compute the probability of the predictors.
                df['f_pred'] = df.form_x.apply(lambda x: x.id).map(
                    tok_freq if token_predictors else typ_freq)

                # We compute the probability of the pairs
                freq = tok_freq if token_patterns else typ_freq
                f_pred = df.f_pred if token_predictors else \
                    df.form_x.apply(lambda x: x.id).map(freq)

                f_out = df.form_y.apply(lambda x: x.id).map(freq)

                # Final result
                df['f_pair'] = f_pred * f_out
            else:
                df['f_pred'] = 1
                df['f_pair'] = 1

            # We compute the number of pairs concerned with this calculation.
            data.loc[pair, "n_pairs"] = sum(selector)

            # We aggregate features and applicable patterns.
            # Lexemes that share these properties belong to similar classes.
            classes = self.add_features(df)

            if legacy and not debug:
                data.loc[pair, "value"] = cond_entropy(df.pattern.apply(lambda x: (x,)),
                                                       classes,
                                                       subset=selector)
            elif not debug:
                data.at[pair, "value"] = [cond_entropy_slow(df), cond_psuccess(df)]
            else:
                data.at[pair, "value"] = self.cond_metrics_log(df,
                                                               classes,
                                                               pair,
                                                               legacy=legacy,
                                                               subset=selector)
        data = data.explode(['value', "measure"])
        if self.data.empty:
            self.data = data.reset_index()
        else:
            self.data = pd.concat([self.data, data.reset_index()])

    def cond_metrics_log(self, group, classes, cells, subset=None, legacy=False):
        """Print a log of the probability distribution for one predictor.

        Writes down the distributions
        :math:`P( patterns_{c1, c2} | classes_{c1, c2} )`
        for all unordered pairs of columns in :attr:`.patterns`.
        Also writes the entropy of the distributions.
        """

        def subclass_summary(subgroup, total):
            """ Produces a nice summary for a subclass"""
            ex = subgroup.iloc[0, :]
            freq = subgroup.f_pair.sum() / total if total is not None else subgroup.shape[0]
            freq = 0 if pd.isna(freq) else freq

            return pd.Series([
                              f"{ex.lexeme}: {ex.form_x} → {ex.form_y}",
                              freq,
                             ],
                             index=["example", 'subclass_size'])

        def success_table(subgroup, patterns):
            ex = subgroup.iloc[0]
            series = {'example': f"{ex.lexeme}: {ex.form_x} → {', '.join(subgroup.form_y.values)}"}
            series.update({patterns.loc[p, 'id']: patterns.loc[p, 'proba'] for p in ex.pattern_pred})
            series['f_pred'] = subgroup.f_pred.sum()
            series['psuccess'] = subgroup.psuccess.sum()
            return pd.Series(series)

        log.debug("\n# Distribution of {}→{} \n".format(cells[0], cells[1]))

        A = group[subset]
        B = classes[subset]
        cond_events = A.groupby(B, sort=False)

        log.debug("Showing distributions for "
                  + str(len(cond_events))
                  + " classes")

        summary = []

        for i, (classe, members) in enumerate(sorted(cond_events,
                                                     key=lambda x: len(x[1]),
                                                     reverse=True)):
            # Group by patterns and build a summary
            p_table = members.groupby('pattern').apply(subclass_summary, members.f_pair.sum())

            # Log features
            if self.features is not None:
                feature_log = (
                    "Features: "
                    + ", ".join(str(x) for x in classe[-self.features_len:]))
                classe = classe[:-self.features_len]

            # List possible patterns that are not used in this class.
            for pattern in classe:
                if pattern not in p_table.index:
                    p_table.loc[str(pattern), :] = ["-", 0]

            # Create an ID for the patterns
            p_table['id'] = range(p_table.shape[0])
            p_table.id = "p_" + p_table.id.astype(str)

            # Group by patterns for the predictor only (i.e. allow for overabundance)
            members['pattern_pred'] = members.groupby(['lexeme', 'form_x'], observed=False)\
                .pattern.transform(lambda x: [tuple(x)]*x.shape[0])

            # Compute the pattern probabilities
            p_table['proba'] = p_table.subclass_size / p_table.subclass_size.sum()
            p_table.fillna(0, inplace=True)
            members['psuccess'] = members.pattern.map(p_table.proba)

            # Get nice table with examples.
            table = members.groupby('pattern_pred')\
                .apply(success_table, patterns=p_table)\
                .reset_index(drop=True)

            # Compute metrics
            ent = 0 + entropy(p_table.proba)
            psuccess = table.f_pred @ table.psuccess / table.f_pred.sum()
            summary.append([members.f_pred.sum(), ent, psuccess])

            # Log the subclass properties
            headers = ("Pattern", "Example",
                       "Frequency", "P(Pattern|class)")
            p_table = p_table.reset_index().set_index("id")
            p_table.columns = headers
            table.rename(columns={"example": "Example",
                                  "f_pred": "Frequency",
                                  "psuccess": "P(success|class)"}, inplace=True)
            table.set_index('Example', inplace=True)

            stats = f"\n## Class n°{i} ({len(members)} members), H={ent:.3f}"
            if not legacy:
                stats += f", P(success)={psuccess:.3f}"
            log.debug(stats)
            if self.features is not None:
                log.debug(feature_log)
            if legacy:
                p_table = p_table.iloc[:, :-1]
            log.debug("\nPatterns found\n\n" + p_table.to_markdown())
            if not legacy:
                log.debug("\nDistribution of the forms\n\n" + table.to_markdown())

        log.debug('\n## Class summary')
        summary = pd.DataFrame(summary, columns=['Frequency',
                                                 'H(pattern|class)',
                                                 'P(success|class)'])
        summary.index.name = "Class"

        if legacy:
            summary = summary.iloc[:, :-1]
        sums = summary.iloc[:, 0].T @ summary.iloc[:, 1::] / summary.iloc[:, 0].sum()
        log.debug(f'\nAv. conditional entropy: H(pattern|class)={sums.iloc[0]}')
        if not legacy:
            log.debug(f'\nAv. success probability: P(success|class)={sums.iloc[1]}')
        log.debug("\n" + summary.to_markdown())
        return sums.values

    def n_preds_entropy(self, n, paradigms, debug=False):
        r"""
        Wrapper to prepare the computation of nary entropies.

        Loops through the cells and runs the computations for every set of predictors.

        Arguments:
            n (int): number of predictors.
            paradigms (pandas.DataFrame): a DataFrame of paradigms
            debug (bool): Whether to run a debug computation with full log. Defaults to False.
        """

        def check_zeros(n):
            log.info("Saving time by listing already known 0 entropies...")
            if n - 1 in self.data.loc[:, "n_preds"]:
                df = self.get_results(measure="cond_entropy", n=n - 1).groupby("predicted")

                if n - 1 == 1:
                    df = df.agg({"predictor": lambda ps: set(frozenset({pred}) for pred in ps)})
                else:
                    df = df.agg({"predictor": lambda ps: set(frozenset(pred) for pred in ps)})
                return df.predictor.to_dict()
            return None

        if n == 1 and not debug:
            return self.one_pred_entropy()
        elif n == 1 and debug:
            return self.one_pred_entropy_log()

        log.info("Computing (c1, ..., c{!s}) → c{!s} entropies".format(n, n + 1))
        log.debug(f"Logging n preds probabilities, with n = {n}")
        log.debug(" P(x, y → z) = P(x~z, y~z | Class(x), Class(y), x~y)")

        data = self.prepare_data(n=n, debug=debug).reset_index(drop=False)

        # Build order agnostic dict
        pat_order = {}
        for a, b in self.patterns:
            pat_order[(a, b)] = (a, b)
            pat_order[(b, a)] = (a, b)

        # Get the measures
        if not debug:
            zeros = check_zeros(n)
            data = data.groupby('predictor').apply(
                self.n_preds_condent,
                paradigms.data, pat_order, zeros, n,
                )
        else:
            data = data.groupby('predictor').apply(
                self.n_preds_condent_log,
                paradigms.data, pat_order, n,
                )

        # Add to previous results
        self.data = pd.concat([self.data, data])

    def n_preds_condent(self, df, paradigms, pat_order, zeros, n):
        r"""
        Computes the probability distribution for n predictors.

        Writes down the distributions:

        .. math::

            P( patterns_{c1, c3}, \; \; patterns_{c2, c3} \; \;  |
               classes_{c1, c3}, \; \; \; \;  classes_{c2, c3},
               \; \;  patterns_{c1, c2} )

        The result contains entropy :math:`H(c_{1}, ..., c_{n} \\to c_{n+1} )`.

        Values are computed for all unordered combinations of
        :math:`(c_{1}, ..., c_{n+1})` in the
        :attr:`paradigms`'s columns.
        Indexes are tuples :math:`(c_{1}, ..., c_{n})`
        and columns are the predicted cells :math:`c_{n+1}`.

        Example:
            For three cells c1, c2, c3, (n=2)
            entropy of c1, c2 → c3,
            noted :math:`H(c_{1}, c_{2} \to c_{3})` is:

        .. math::

            H( patterns_{c1, c3}, \; \; patterns_{c2, c3}\; \;
            | classes_{c1, c3}, \; \; \; \;
            classes_{c2, c3}, \; \;  patterns_{c1, c2} )

        Arguments:
            n (int): number of predictors.
            df (pandas.DataFrame): a DataFrame containing patterns and applicable patterns for pairs of forms.
            paradigms (pandas.DataFrame): a DataFrame of paradigms.
            pat_order (dict): a dictionary to normalize cell names.
            zeros (dict): a dictionary of pairs that lead to an entropy of zero.
            n (int): number of predictors
        """

        def already_zero(predictors, out, zeros):
            for preds_subset in combinations(predictors, n - 1):
                if frozenset(preds_subset) in zeros[out]:
                    return True
            return False

        # For faster access
        patterns = self.patterns
        predictors = df.name.split('&')
        pairs_of_predictors = list(combinations(predictors, 2))
        set_predictors = set(predictors)

        known_patterns = pd.concat([patterns[k]
                                    .set_index('lexeme')
                                    .pattern
                                    for k in pairs_of_predictors],
                                   axis=1)

        predlexemes = known_patterns.notna().all(axis=1)
        known_patterns = known_patterns.map(lambda x: (x,) if not isinstance(x, tuple) else x)\
            .sum(axis=1)

        def row_condent(x):
            """
            Computes the conditional entropy for a given set of predictors
            and a target.

            Arguments:
                x (pandas.Series): a Series containing information for the computation.
            """
            out = x.predicted
            outlexemes = paradigms[(paradigms.cell == out) &
                                   ~(paradigms.form.apply(lambda x: x.is_defective()))]
            selector = predlexemes & predlexemes.index.isin(outlexemes.lexeme)
            x.n_pairs = sum(selector)

            if zeros is not None and already_zero(set_predictors, out, zeros):
                x.value = 0
            else:
                # Under the pattern column, getting intersection of patterns events for each
                # predictor: x~z, y~z
                # Under the applicable column, getting
                # - Known classes Class(x), Class(y)
                # - known patterns x~y
                # - plus all features

                pattern_pairs = [patterns[pat_order[(pred, out)]]
                                 .set_index('lexeme')
                                 [selector][['pattern', 'applicable']]
                                 .map(lambda x: (x,) if not isinstance(x, tuple) else x)
                                 for pred in predictors]
                pattern_pairs = reduce(lambda x, y: x+y, pattern_pairs)
                pattern_pairs.applicable += known_patterns[selector]

                classes = self.add_features(pattern_pairs)

                # Prediction of H(A|B)
                x.value = cond_entropy(pattern_pairs.pattern,
                                       classes,
                                       subset=selector)
            return x

        return df.apply(row_condent, axis=1)

    def n_preds_condent_log(self, df, paradigms, pat_order, n):
        r"""
        Computes the probability distribution for n predictors
        and logs the details of the computations.

        Writes down the distributions:

        .. math::

            P( patterns_{c1, c3}, \; \; patterns_{c2, c3} \; \;  |
               classes_{c1, c3}, \; \; \; \;  classes_{c2, c3},
               \; \;  patterns_{c1, c2} )

        The result contains entropy :math:`H(c_{1}, ..., c_{n} \to c_{n+1} )`.

        Values are computed for all unordered combinations of
        :math:`(c_{1}, ..., c_{n+1})` in the
        :attr:`paradigms`'s columns.
        Indexes are tuples :math:`(c_{1}, ..., c_{n})`
        and columns are the predicted cells :math:`c_{n+1}`.

        Example:
            For three cells c1, c2, c3, (n=2)
            entropy of c1, c2 → c3,
            noted :math:`H(c_{1}, c_{2} \to c_{3})` is:

        .. math::

            H( patterns_{c1, c3}, \; \; patterns_{c2, c3}\; \;
            | classes_{c1, c3}, \; \; \; \;
            classes_{c2, c3}, \; \;  patterns_{c1, c2} )

        Arguments:
            n (int): number of predictors.
            df (pandas.DataFrame): a DataFrame containing patterns and applicable patterns for pairs of forms.
            paradigms (pandas.DataFrame): a DataFrame of paradigms.
            pat_order (dict): a dictionary to normalize cell names.
            n (int): number of predictors
        """

        def count_with_examples(row, counter, examples, paradigms, pred, out):
            lemma, pattern = row
            predictors = "; ".join(paradigms.loc[(paradigms.lexeme == lemma) &
                                                 (paradigms.cell == c)]
                                   .form.values[0]
                                   for c in pred)
            predicted = paradigms.loc[(paradigms.lexeme == lemma) &
                                      (paradigms.cell == out)].form.values[0]
            example = f"{lemma}: ({predictors}) → {predicted}"
            counter[pattern] += 1
            examples[pattern] = example

        def format_patterns(series, string):
            patterns = ("; ".join(str(pattern)
                                  for pattern in pair)
                        for pair in series)
            return string.format(*patterns)

        pred_numbers = list(range(1, n + 1))
        patterns_string = "\n".join(f"{pred}~{n + 1}" + "= {}" for pred in pred_numbers)
        classes_string = "\n    * " + "\n    * ".join(f"Class({pred}, {n + 1})" + "= {}"
                                                      for pred in pred_numbers)
        known_pat_string = "\n    * " "\n    * ".join("{!s}~{!s}".format(*preds) +
                                                      "= {}" for preds
                                                      in combinations(pred_numbers, 2))

        def format_features(features):
            return "\n* Features:\n    * " + "\n    * ".join(str(x) for x in features)

        def formatting_local_patterns(x):
            return format_patterns(x, patterns_string)

        def formatting_known_classes(x):
            return format_patterns(x, classes_string)

        def formatting_known_patterns(x):
            return format_patterns(x, known_pat_string)

        # For faster access
        patterns = self.patterns
        predictors = df.name.split('&')
        pairs_of_predictors = list(combinations(predictors, 2))

        known_patterns = pd.concat([patterns[k]
                                   .set_index('lexeme')
                                   .pattern
                                   .rename('&'.join(k))
                                   for k in pairs_of_predictors],
                                   axis=1)

        predlexemes = known_patterns.notna().all(axis=1)
        known_patterns = known_patterns.map(lambda x: (x,) if not isinstance(x, tuple) else x)

        def row_condent(x, known_patterns):
            patterns = self.patterns
            out = x.predicted
            outlexemes = paradigms[(paradigms.cell == out) &
                                   ~(paradigms.form.apply(lambda x: x.is_defective()))]
            selector = predlexemes & predlexemes.index.isin(outlexemes.lexeme)
            x.n_pairs = sum(selector)
            log.debug(f"\n# Distribution of ({', '.join(predictors)}) → {out} \n")

            known_classes = [patterns[pat_order[(pred, out)]]
                             .set_index('lexeme')
                             [selector].applicable
                             .rename(f"{pred}&{out}")
                             .map(lambda x: (x,) if not isinstance(x, tuple) else x)
                             for pred in predictors]
            known_classes = pd.concat(known_classes, axis=1)

            gold_patterns = [patterns[pat_order[(pred, out)]]
                             .set_index('lexeme')
                             [selector].pattern
                             .rename(f"{pred}&{out}")
                             .map(lambda x: (x,) if not isinstance(x, tuple) else x)
                             for pred in predictors]
            gold_patterns = pd.concat(gold_patterns, axis=1)

            # Getting intersection of patterns events for each predictor:
            # x~z, y~z
            A = gold_patterns.apply(formatting_local_patterns, axis=1)

            # Known classes Class(x), Class(y) and known patterns x~y
            known_classes = known_classes.apply(formatting_known_classes,
                                                axis=1)
            known_patterns = known_patterns.apply(formatting_known_patterns,
                                                  axis=1)

            B = known_classes + known_patterns

            if self.features is not None:
                known_features = self.features[selector].apply(format_features)
                B = B + known_features

            cond_events = A.groupby(B, sort=False)

            log.debug("Showing distributions for "
                      + str(len(cond_events))
                      + " classes")

            summary = []

            for i, (classe, members) in enumerate(sorted(cond_events,
                                                         key=lambda x: len(x[1]),
                                                         reverse=True)):
                log.debug("\n## Class n°%s (%s members).", i, len(members))
                counter = Counter()
                examples = defaultdict()
                members.reset_index().apply(count_with_examples,
                                            args=(counter, examples,
                                                  paradigms,
                                                  predictors, out), axis=1)
                total = sum(list(counter.values()))
                log.debug("* Total: %s", total)

                table = []
                for my_pattern in counter:
                    row = (my_pattern,
                           examples[my_pattern],
                           counter[my_pattern],
                           counter[my_pattern] / total)
                    table.append(row)

                headers = ("Patterns", "Example",
                           "Size", "P(Pattern|class)")
                table = pd.DataFrame(table, columns=headers)
                # Get the slow computation results
                summary.append([table.Size.sum(),
                                0 + entropy(table.iloc[:, -1])])
                log.debug("\n" + table.to_markdown())

            log.debug('\n## Class summary')
            summary = pd.DataFrame(summary, columns=['Size', 'H(pattern|class)'])
            summary.index.name = "Class"
            x.value = (summary.iloc[:, -2] * summary.iloc[:, -1] / summary.iloc[:, -2].sum()).sum()
            log.debug(f'\nAv. conditional entropy: H(pattern|class)={x.value}')
            log.debug("\n" + summary.to_markdown())
            return x

        return df.apply(row_condent, args=[known_patterns], axis=1)
