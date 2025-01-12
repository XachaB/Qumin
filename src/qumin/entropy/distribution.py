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
from tqdm import tqdm

from . import cond_entropy, entropy

log = logging.getLogger("Qumin")


def value_norm(df):
    """ Rounding at 10 significant digits, avoiding negative 0s"""
    return df.map(lambda x: round(x, 10)) + 0


def merge_split_df(dfs):
    merged = {col: reduce(lambda x, y: x + y, [df[col] for df in dfs])
              for col in dfs[0].columns}
    return pd.DataFrame(merged, index=dfs[0].index, columns=dfs[0].columns)


def dfsum(df, **kwargs):
    cols = df.columns
    S = df[cols[0]]
    for col in cols:
        S += df[col]
    return S


class PatternDistribution(object):
    """Statistical distribution of patterns.

    Attributes:
        patterns (:class:`pandas:pandas.DataFrame`):
            A table where each row describes an alternation between
            two cells forms belonging to different cells of the same lexeme.
            The row also contains the correct pattern and the set of applicable patterns.

        entropies (dict[int, pandas.DataFrame]):
            dict mapping n to a dataframe containing the entropies
            for the distribution :math:`P(c_{1}, ..., c_{n} → c_{n+1})`.
    """

    def __init__(self, patterns, name, features=None):
        """Constructor for PatternDistribution.

        Arguments:
            patterns (:class:`pandas:pandas.DataFrame`):
                Table containing forms and patterns.
            name (str): dataset name.
            features:
                optional table of features
        """
        self.name = name
        self.patterns = patterns

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

    def get_results(self, measure="cond_entropy", n=1):
        is_cond_ent = self.data.loc[:, "measure"] == measure
        is_one_pred = self.data.loc[:, "n_preds"] == n
        return self.data.loc[is_cond_ent & is_one_pred, :]

    def export_file(self, filename):
        """ Export the data DataFrame to file

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
            data.loc[:, "entropy"] = value_norm(data.loc[:, "entropy"])
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
        if self.features:
            ret = group.applicable + group.lexeme.map(self.features)
            return ret
        else:
            return group.applicable

    def n_preds_entropy_matrix(self, n, paradigms):
        r"""Return a:class:`pandas:pandas.DataFrame` with nary entropies, and one with counts of lexemes.

        The result contains entropy :math:`H(c_{1}, ..., c_{n} \to c_{n+1} )`.

        Values are computed for all unordered combinations of
        :math:`(c_{1}, ..., c_{n+1})` in the
        :attr:`PatternDistribution.paradigms`'s columns.
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

        def already_zero(predictors, out, zeros):
            for preds_subset in combinations(predictors, n - 1):
                if frozenset(preds_subset) in zeros[out]:
                    return True
            return False

        if n == 1:
            return self.one_pred_entropy()

        log.info("Computing (c1, ..., c{!s}) → c{!s} entropies".format(n, n + 1))

        # For faster access
        patterns = self.patterns
        cells = self.patterns.cells
        zeros = check_zeros(n)
        data = self.prepare_data(n=n).reset_index(drop=False)

        pat_order = {}
        for a, b in patterns:
            pat_order[(a, b)] = (a, b)
            pat_order[(b, a)] = (a, b)

        def calc_condent(x, paradigms):
            # combinations gives us all x, y unordered unique pair for all of
            # the n predictors.
            predictors = x.predictor.split('&')
            pairs_of_predictors = list(combinations(predictors, 2))
            known_patterns = pd.concat([patterns[k]
                                        .set_index('lexeme')
                                        .pattern
                                        for k in pairs_of_predictors],
                                       axis=1)
            set_predictors = set(predictors)
            predlexemes = known_patterns.notna().all(axis=1)
            known_patterns = known_patterns.map(lambda x: (x,) if not isinstance(x, tuple) else x)\
                                           .sum(axis=1)

            for out in (x for x in cells if x not in predictors):

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

        data = data.apply(calc_condent, args=[paradigms.data], axis=1)
        self.data = pd.concat([self.data, data])

    def prepare_data(self, n=1, debug=False):
        """
        Prepares the dataframe to store the results for an entropy computation
        """
        rows = self.patterns.cells
        idx = ["&".join(x) for x in combinations(rows, n)]

        data = pd.DataFrame(index=idx,
                            columns=rows).reset_index(drop=False,
                                                      names="predictor").melt(id_vars="predictor",
                                                                              var_name="predicted",
                                                                              value_name="value")
        data = data[data.predictor != data.predicted]  # drop a -> a cases
        data.loc[:, "n_pairs"] = None
        data.loc[:, "n_preds"] = n
        data.loc[:, "measure"] = "cond_entropy" if not debug else "cond_entropy_debug"
        data.loc[:, "dataset"] = self.name
        data.set_index(['predictor', 'predicted'], inplace=True)

        return data

    def one_pred_entropy(self, debug=False):
        r"""Return a :class:`pandas:pandas.DataFrame` with unary entropies and counts of lexemes.

        The result contains entropy :math:`H(c_{1} \to c_{2})`.

        Values are computed for all unordered combinations
        of :math:`(c_{1}, c_{2})` where `c_{1} != c_{2}`
        in the :attr:`PatternDistribution.paradigms`'s columns.

        Example:
            For two cells c1, c2, entropy of c1 → c2,
            noted :math:`H(c_{1} \to c_{2})` is:

            .. math::

                H( patterns_{c1, c2} | classes_{c1, c2} )
        """
        log.info("Computing c1 → c2 entropies")
        log.debug("Logging one predictor probabilities")
        log.debug(" P(x → y) = P(x~y | Class(x))")

        # For faster access
        patterns = self.patterns
        data = self.prepare_data(debug=debug)

        def calc_condent(group, cells, data):
            """
            Computes the conditional entropy for a pair of cells.

            Arguments:
                group (pandas.DataFrame):
                    Subset of a patterns table for a single pair.
                data (pandas.DataFrame):
                    DataFrame to store computation results.
            """
            # Defective rows can't be kept here.
            selector = group.pattern.notna()

            # We compute the number of pairs concerned with this calculation.
            data.loc[cells, "n_pairs"] = sum(selector)

            # We aggregate features and applicable patterns.
            # Lexemes that share these properties belong to similar classes.
            classes = self.add_features(group)

            if debug:
                data.loc[cells, "value"] = cond_entropy(group.pattern.apply(lambda x: (x,)),
                                                        classes,
                                                        subset=selector)
            else:
                data.loc[cells, "value"] = self.cond_entropy_log(group,
                                                                 classes,
                                                                 cells,
                                                                 subset=selector)

        for pair, df in patterns.items():
            calc_condent(df, pair, data)
        if self.data.empty:
            self.data = data.reset_index()
        else:
            self.data = pd.concat([self.data, data.reset_index()])

    def cond_entropy_log(self, group, classes, cells, subset=None):
        """Print a log of the probability distribution for one predictor.

        Writes down the distributions
        :math:`P( patterns_{c1, c2} | classes_{c1, c2} )`
        for all unordered combinations of two column
        names in :attr:`PatternDistribution.paradigms`.
        Also writes the entropy of the distributions.
        """

        def subclass_summary(subgroup):
            """ Produces a nice summary for a subclass"""
            ex = subgroup.iloc[0, :]
            return pd.Series([
                              f"{ex.lexeme}: {ex.form_x} → {ex.form_y}",
                              subgroup.shape[0]
                             ],
                             index=["example", 'subclass_size'])

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
            table = members.groupby('pattern').apply(subclass_summary)

            if self.features is not None:
                feature_log = (
                    "Features: "
                    + ", ".join(str(x) for x in classe[-self.features_len:]))
                classe = classe[:-self.features_len]

            # List possible patterns that are not used in this class.
            for pattern in classe:
                if pattern not in table.index:
                    table.loc[str(pattern), :] = ["-", 0]

            # Get the slow computation results
            table['proba'] = table.subclass_size / table.subclass_size.sum()
            ent = 0 + entropy(table.proba)
            summary.append([table.subclass_size.sum(), ent])

            # Log the subclass properties
            headers = ("Pattern", "Example",
                       "Size", "P(Pattern|class)")
            table.reset_index(inplace=True)
            table.columns = headers
            log.debug(f"\n## Class n°{i} ({len(members)} members), H={ent}")
            if self.features is not None:
                log.debug(feature_log)
            log.debug("\n" + table.to_markdown())

        log.debug('\n## Class summary')
        summary = pd.DataFrame(summary, columns=['Size', 'H(pattern|class)'])
        summary.index.name = "Class"
        sum_entropy = (summary.iloc[:, -2] * summary.iloc[:, -1] / summary.iloc[:, -2].sum()).sum()
        log.debug(f'\nAv. conditional entropy: H(pattern|class)={sum_entropy}')
        log.debug("\n" + summary.to_markdown())
        return sum_entropy

    def n_preds_distrib_log(self, n):
        r"""Print a log of the probability distribution for n predictors.

        Writes down the distributions:

        .. math::

            P( patterns_{c1, c3}, \; \; patterns_{c2, c3} \; \;  |
               classes_{c1, c3}, \; \; \; \;  classes_{c2, c3},
               \; \;  patterns_{c1, c2} )

        for all unordered combinations of two column names
        in :attr:`PatternDistribution.paradigms`.

        Arguments:
            n (int): number of predictors.
        """

        def count_with_examples(row, counter, examples, paradigms, pred, out):
            lemma, pattern = row
            predictors = "; ".join(paradigms.at[lemma, c] for c in pred)
            example = f"{lemma}: ({predictors}) → {paradigms.at[lemma, out]}"
            counter[pattern] += 1
            examples[pattern] = example

        log.info(f"Printing log of P( (c1, ..., c{n}) → c{n + 1} ).")
        log.debug(f"Logging n preds probabilities, with n = {n}")
        log.debug(" P(x, y → z) = P(x~z, y~z | Class(x), Class(y), x~y)")

        # For faster access
        patterns = self.patterns
        classes = self.classes
        columns = list(self.paradigms.columns)

        pat_order = {}
        for a, b in self.patterns:
            pat_order[(a, b)] = (a, b)
            pat_order[(b, a)] = (a, b)

        indexes = list(combinations(columns, n))

        def format_patterns(series, string):
            patterns = ("; ".join(str(pattern)
                                  for pattern in pair)
                        for pair in series)
            return string.format(*patterns)

        pred_numbers = list(range(1, n + 1))
        patterns_string = "\n".join(f"{pred}~{n + 1}" + "= {}" for pred in pred_numbers)
        classes_string = "\n    * " + "\n    * ".join(f"Class({pred}, {n + 1})" + "= {}" for pred in pred_numbers)
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

        for predictors in tqdm(indexes):
            #  combinations gives us all x, y unordered unique pair for all of
            # the n predictors.
            pairs_of_predictors = list(combinations(predictors, 2))

            predsselector = reduce(lambda x, y: x & y,
                                   (self.hasforms[x] for x in predictors))

            for out in (x for x in columns if x not in predictors):

                log.debug(f"\n# Distribution of ({', '.join(predictors)}) → {out} \n")

                selector = predsselector & self.hasforms[out]

                # Getting intersection of patterns events for each predictor:
                # x~z, y~z
                local_patterns = patterns.loc[
                    selector, [pat_order[(pred, out)] for pred in predictors]]
                A = local_patterns.apply(formatting_local_patterns, axis=1)

                # Known classes Class(x), Class(y) and known patterns x~y
                known_classes = classes.loc[
                    selector, [(pred, out) for pred in predictors]]
                known_classes = known_classes.apply(formatting_known_classes,
                                                    axis=1)

                known_patterns = patterns.loc[selector, pairs_of_predictors]
                known_patterns = known_patterns.apply(formatting_known_patterns, axis=1)

                B = known_classes + known_patterns

                if self.features is not None:
                    known_features = self.features[selector].apply(format_features)
                    B = B + known_features

                cond_events = A.groupby(B, sort=False)

                for i, (classe, members) in enumerate(
                        sorted(cond_events, key=lambda x: len(x[1]), reverse=True)):
                    headers = ("Patterns", "Example",
                               "Size", "P(Pattern|class)")
                    table = []

                    log.debug("\n## Class n°%s (%s members).", i, len(members))
                    counter = Counter()
                    examples = defaultdict()
                    members.reset_index().apply(count_with_examples,
                                                args=(counter, examples,
                                                      self.paradigms,
                                                      predictors, out), axis=1)
                    total = sum(list(counter.values()))
                    log.debug("* Total: %s", total)

                    for my_pattern in counter:
                        row = (my_pattern,
                               examples[my_pattern],
                               counter[my_pattern],
                               counter[my_pattern] / total)
                        table.append(row)

                    log.debug("\n" + pd.DataFrame(table, columns=headers).to_markdown())
