# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Sacha Beniamine.

Encloses distribution of patterns on paradigms.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from prettytable import PrettyTable, ALL
from itertools import combinations

from functools import reduce
from entropy import cond_entropy, entropy, P
import representations.patterns
from tqdm import tqdm
import logging

log = logging.getLogger()


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

        paradigms (:class:`pandas:pandas.DataFrame`):
            containing forms.

        patterns (:class:`pandas:pandas.DataFrame`):
            containing pairwise patterns of alternation.

        classes (:class:`pandas:pandas.DataFrame`):
            containing a representation of applicable patterns
            from one cell to another.
            Index are lemmas.

        entropies (`dict` of `int`::class:`pandas:pandas.DataFrame`):
            dict mapping n to a dataframe containing the entropies
            for the distribution :math:`P(c_{1}, ..., c_{n} → c_{n+1})`.
    """

    def __init__(self, paradigms, patterns, pat_dic, features=None):
        """Constructor for PatternDistribution.

        Arguments:
            paradigms (:class:`pandas:pandas.DataFrame`):
                containing forms.
            patterns (:class:`pandas:pandas.DataFrame`):
                patterns (columns are pairs of cells, index are lemmas).
            patterns (dict):
                dictionnary of pairs of cells to patterns
            features:
                optional table of features
        """
        self.paradigms = paradigms.applymap(lambda x: x[0])
        self.pat_dict = pat_dic
        self.patterns = patterns.applymap(lambda x: (str(x),))

        if features is not None:
            # Add feature names
            features = features.apply(lambda x: x.name + "=" + x.apply(str), axis=0)
            # To tuples
            features = features.applymap(lambda x: (str(x),))
            self.features_len = features.shape[1]
            self.features = pd.DataFrame.sum(features, axis=1)
        else:
            self.features_len = 0
            self.features = None
            self.add_features = lambda x: x

        log.info("Looking for classes of applicable patterns")
        self.classes = representations.patterns.find_applicable(self.paradigms,
                                                                self.pat_dict)

        self.hasforms = {cell: (paradigms[cell] != "") for cell in self.paradigms}
        self.entropies = [None] * 10
        self.effectifs = [None] * 10

    def add_features(self, series):
        return series + self.features[series.index]

    def __str__(self):
        """Return a string summary."""
        string = "Distribution for the cells {}. Total of {} lemmas"
        string += "\nEntropy computed for {} predictors"
        preds = [str(i) for i, ent in enumerate(self.entropies)
                 if ent is not None]
        if not preds:
            preds = ["no"]

        string = string.format(list(self.patterns.columns),
                               str(len(self.paradigms.index)),
                               " or ".join(preds))
        return string

    def read_entropy_from_file(self, filename):
        """Read already computed entropies from a file.

        Arguments:
            filename: the file's path.
        """
        entropies = pd.read_csv(filename, sep="\t", index_col=0)

        if ", " in entropies.index[0]:
            entropies.index = [tuple(y.strip(' "\'')
                                     for y in x.strip("()").split(", "))
                               for x in entropies.index]

            n = len(entropies.index[0])
        else:
            n = 1

        self._register_entropy(n, entropies, None)

    def _register_entropy(self, n, entropy, effectifs):
        """Register an entropy score_matrix for n predictors.

        Arguments:
            n (int): number of predictors
            entropy  (:class:`pandas:pandas.DataFrame`):
                Entropy score_matrix to register.
        """
        entropy = entropy.applymap(lambda x: max(0, x))
        try:
            self.entropies[n] = entropy
            self.effectifs[n] = effectifs
        except IndexError:
            self.entropies.append([None] * n)
            self.effectifs.append([None] * n)
            self.entropies[n] = entropy
            self.effectifs[n] = effectifs

    def n_preds_entropy_matrix(self, n):
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
            zeros = defaultdict(set)

            if self.entropies[n - 1] is not None:
                if n - 1 == 1:
                    df = self.entropies[1].stack()
                    tuples = list(df[df == 0].index)
                    for out, pred in tuples:
                        zeros[out].add(frozenset({pred}))
                else:
                    df = self.entropies[n - 1].stack()
                    tuples = list(df[df == 0].index)
                    for out, pred in tuples:
                        zeros[out].add(frozenset(pred))
            return zeros

        if n == 1:
            return self.entropy_matrix()

        log.info("Computing (c1, ..., c{!s}) → c{!s} entropies".format(n, n + 1))

        # For faster access
        patterns = self.patterns
        classes = self.classes
        columns = list(self.paradigms.columns)

        def already_zero(predictors, out, zeros):
            for preds_subset in combinations(predictors, n - 1):
                if preds_subset in zeros[out]:
                    return True
            return False

        if any((self.entropies[i] is not None for i in range(1, n))):
            log.info("Saving time by listing already known 0 entropies...")
            zeros = check_zeros(n)
        else:
            zeros = None

        pat_order = {}
        for a, b in patterns:
            pat_order[(a, b)] = (a, b)
            pat_order[(b, a)] = (a, b)

        indexes = list(combinations(columns, n))
        entropies = pd.DataFrame(index=indexes,
                                 columns=columns,
                                 dtype="float16")
        effectifs = pd.DataFrame(index=indexes,
                                 columns=columns)
        iterations = len(indexes)

        for predictors in tqdm(indexes):

            # combinations gives us all x, y unordered unique pair for all of
            # the n predictors.
            pairs_of_predictors = list(combinations(predictors, 2))
            known_patterns = patterns[pairs_of_predictors]
            set_predictors = set(predictors)
            predsselector = reduce(lambda x, y: x & y,
                                   (self.hasforms[x] for x in predictors))

            for out in (x for x in columns if x not in predictors):
                if zeros is not None and already_zero(set_predictors, out, zeros):
                    entropies.at[predictors, out] = 0
                else:
                    # Getting intersection of patterns events for each
                    # predictor: x~z, y~z
                    selector = predsselector & self.hasforms[out]
                    local_patterns = patterns[
                        [pat_order[(pred, out)] for pred in predictors]]
                    A = dfsum(local_patterns)

                    # Known classes Class(x), Class(y) and known patterns x~y
                    # plus all features
                    known_classes = classes.loc[
                        selector, [(pred, out) for pred in predictors]]
                    known = known_classes.join(known_patterns[selector])

                    B = self.add_features(dfsum(known))

                    # Prediction of H(A|B)
                    entropies.at[predictors, out] = cond_entropy(A, B, subset=selector)
                    effectifs.at[predictors, out] = sum(selector)


        self._register_entropy(n, entropies, effectifs)
        return entropies, effectifs

    def entropy_matrix(self):
        r"""Return a:class:`pandas:pandas.DataFrame` with unary entropies, and one with counts of lexemes.

        The result contains entropy :math:`H(c_{1} \to c_{2})`.

        Values are computed for all unordered combinations
        of :math:`(c_{1}, c_{2})`
        in the :attr:`PatternDistribution.paradigms`'s columns.
        Indexes are predictor cells :math:`c{1}`
        and columns are the predicted cells :math:`c{2}`.

        Example:
            For two cells c1, c2, entropy of c1 → c2,
            noted :math:`H(c_{1} \to c_{2})` is:

            .. math::

                H( patterns_{c1, c2} | classes_{c1, c2} )
        """
        log.info("Computing c1 → c2 entropies")

        # For faster access
        patterns = self.patterns
        classes = self.classes
        rows = list(self.paradigms.columns)

        entropies = pd.DataFrame(index=rows, columns=rows, dtype="float16")
        effectifs = pd.DataFrame(index=rows, columns=rows)

        for a, b in patterns.columns:
            selector = self.hasforms[a] & self.hasforms[b]
            known_ab = self.add_features(classes[(a, b)])
            known_ba = self.add_features(classes[(b, a)])

            entropies.at[a, b] = cond_entropy(patterns[(a, b)],
                                              known_ab, subset=selector)
            entropies.at[b, a] = cond_entropy(patterns[(a, b)],
                                              known_ba, subset=selector)

            effectifs.at[a, b] = sum(selector)
            effectifs.at[b, a] = sum(selector)

        self._register_entropy(1, entropies, effectifs)
        return entropies, effectifs

    def one_pred_distrib_log(self, sanity_check=False):
        """Print a log of the probability distribution for one predictor.

        Writes down the distributions
        :math:`P( patterns_{c1, c2} | classes_{c1, c2} )`
        for all unordered combinations of two column
        names in :attr:`PatternDistribution.paradigms`.
        Also writes the entropy of the distributions.

        Arguments:
            sanity_check (bool): Use a slower calculation to check that the results are exact.
        """

        def count_with_examples(row, counter, examples, paradigms, cells):
            c1, c2 = cells
            lemma, pattern = row
            example = "{}: {} → {}".format(lemma,
                                           paradigms.at[lemma, c1],
                                           paradigms.at[lemma, c2])
            counter[pattern] += 1
            examples[pattern] = example

        log.info("Printing log for P(c1 → c2).")

        if sanity_check:
            rows = list(self.paradigms.columns)
            entropies_check = pd.DataFrame(index=rows,
                                           columns=rows,
                                           dtype="float16")

        log.debug("Logging one predictor probabilities")
        log.debug(" P(x → y) = P(x~y | Class(x))")

        patterns = self.patterns.applymap(lambda x: x[0])

        for column in patterns:

            for pred, out in [column, column[::-1]]:

                selector = self.hasforms[pred] & self.hasforms[out]
                log.debug("\n# Distribution of {}→{} \n".format(pred, out))

                A = patterns.loc[selector, :][column]
                B = self.add_features(self.classes.loc[selector, :][(pred, out)])
                cond_events = A.groupby(B, sort=False)

                if sanity_check:
                    classes_p = P(B)
                    cond_p = P(cond_events)

                    surprisal = cond_p.groupby(level=0).apply(entropy)
                    slow_ent = min(0, np.float16(sum(classes_p * surprisal)))
                    entropies_check.at[pred, out] = slow_ent
                    log.debug("Entropy from this distribution: %s", slow_ent)

                    if self.entropies[1] is not None:
                        ent = self.entropies[1].at[pred, out]
                        log.debug("Entropy from the score_matrix: %s", ent)

                        if ent != slow_ent and abs(ent - slow_ent) > 1e-5:
                            log.warning("\n# Distribution of {}→{}".format(pred, out))
                            log.warning("Something is wrong"
                                        " in the entropy's calculation. "
                                        "Slow and fast methods produce "
                                        "different results: slow {}, fast {}"
                                        "".format(slow_ent, ent))

                log.debug("Showing distributions for "
                          + str(len(cond_events))
                          + " classes")

                for i, (classe, members) in enumerate(sorted(cond_events,
                                                             key=lambda x: len(x[1]),
                                                             reverse=True)):
                    headers = ("Pattern", "Example",
                               "Size", "P(Pattern|class)")
                    table = PrettyTable(headers,
                                        hrules=ALL)  # TODO: change to remove prettytable
                    # table.set_style(PLAIN_COLUMNS)

                    log.debug("\n## Class n°%s (%s members).", i, len(members))
                    counter = Counter()
                    examples = defaultdict()
                    members.reset_index().apply(count_with_examples,
                                                args=(counter,
                                                      examples,
                                                      self.paradigms,
                                                      (pred, out)),
                                                axis=1)
                    total = sum(list(counter.values()))
                    if self.features is not None:
                        log.debug("Features:"
                                  + " ".join(str(x)
                                             for x in classe[-self.features_len:]))
                        classe = classe[:-self.features_len]

                    for my_pattern in classe:
                        if my_pattern in counter:
                            row = (str(my_pattern),
                                   examples[my_pattern],
                                   counter[my_pattern],
                                   counter[my_pattern] / total)
                        else:
                            row = (str(my_pattern), "-", 0, 0)
                        table.add_row(row)

                    log.debug(table.get_string())

        if sanity_check:
            return entropies_check

    def value_check(self, n):
        """Check that predicting from n predictors isn't harder than with less.

        Check that the value of entropy from n predictors c1, ....cn
        is lower than the entropy from n-1 predictors c1, ..., cn-1
        (for all computed n preds entropies).

        Arguments:
            n: number of predictors.
        """
        if self.entropies[1] is None or self.entropies[n] is None:
            return None

        log.info("Now checking if all entropies with n predictors "
                 "are lower than their counterparts with n-1 predictors.")

        found_wrong = False

        entropies_n = self.entropies[n]
        entropies_one = self.entropies[1]

        for predictors in entropies_n.index:

            for out in entropies_n:
                value_n = entropies_n.at[predictors, out]

                for predictor in predictors:
                    value_one = entropies_one.at[predictor, out]

                    if value_n > value_one and \
                            abs(value_n - value_one) > 1e-5:
                        found_wrong = True
                        log.debug("Found error: H({} → {}) = {}"
                                  "(type = {}) "
                                  " higher than H({} → {}) = {} "
                                  " (type= {})"
                                  "".format(", ".join(predictors),
                                            out,
                                            value_n,
                                            type(value_n),
                                            predictor, out,
                                            value_one,
                                            type(value_one)))

        if found_wrong:
            log.warning("Found errors ! Check logfile or re-run with -d for details.")
        else:
            log.info("Everything is right !")

        return found_wrong

    def n_preds_distrib_log(self, n, sanity_check=False):
        r"""Print a log of the probability distribution for two predictors.

        Writes down the distributions:

        .. math::

            P( patterns_{c1, c3}, \; \; patterns_{c2, c3} \; \;  |
               classes_{c1, c3}, \; \; \; \;  classes_{c2, c3},
               \; \;  patterns_{c1, c2} )

        for all unordered combinations of two column names
        in :attr:`PatternDistribution.paradigms`.

        Arguments:
            n (int): number of predictors.
            sanity_check (bool): Use a slower calculation to check that the results are exact.
        """

        def count_with_examples(row, counter, examples, paradigms, pred, out):
            lemma, pattern = row
            predictors = "; ".join(paradigms.at[lemma, c] for c in pred)
            example = "{}: ({}) → {}".format(lemma,
                                             predictors,
                                             paradigms.at[lemma, out])
            counter[pattern] += 1
            examples[pattern] = example

        log.info("Printing log of "
                 "P( (c1, ..., c{!s}) → c{!s} ).".format(n, n + 1))

        log.debug("Logging n preds probabilities, with n = {}".format(n))
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

        if sanity_check:
            columns = list(self.paradigms.columns)
            entropies_check = pd.DataFrame(index=indexes,
                                           columns=columns,
                                           dtype="float16")

        def format_patterns(series, string):
            patterns = ("; ".join(str(pattern)
                                  for pattern in pair)
                        for pair in series)
            return string.format(*patterns)

        pred_numbers = list(range(1, n + 1))
        patterns_string = "\n".join("{!s}~{!s}".format(pred, n + 1) +
                                    "= {}" for pred in pred_numbers)
        classes_string = "\n    * " + \
                         "\n    * ".join("Class({!s}, {!s})".format(pred, n + 1) +
                                         "= {}" for pred in pred_numbers)
        known_pat_string = "\n    * " + \
                           "\n    * ".join("{!s}~{!s}".format(*preds) +
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

                log.debug(
                    "\n# Distribution of ({}) → {z} \n".format(", ".join(predictors),
                                                               z=out))

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

                if sanity_check:
                    classes_p = P(B)
                    cond_p = P(cond_events)
                    surprisal = cond_p.groupby(level=0).apply(entropy)
                    slow_ent = min(0, np.float16(sum(classes_p * surprisal)))
                    entropies_check.at[predictors, out] = slow_ent
                    log.debug("Entropy from this distribution: %s", slow_ent)

                    if n < len(self.entropies) and self.entropies[n] is not None:
                        ent = self.entropies[n].at[predictors, out]
                        log.debug("Entropy from the score_matrix: %s", ent)
                        if ent != slow_ent and abs(ent - slow_ent) > 1e-5:
                            log.warning("\n# Distribution of ({}, {}) → {z} \n"
                                      .format(*predictors, z=out))
                            log.warning("Something is wrong"
                                        " in the entropy's calculation."
                                        " Slow and fast methods produce"
                                        " different results:"
                                        " slow {}, fast {} "
                                        "".format(slow_ent, ent))

                for i, (classe, members) in enumerate(
                        sorted(cond_events, key=lambda x: len(x[1]), reverse=True)):
                    headers = ("Patterns", "Example",
                               "Size", "P(Pattern|class)")
                    table = PrettyTable(headers, hrules=ALL)
                    # table.set_style(PLAIN_COLUMNS)

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
                        table.add_row(row)

                    log.debug(table.get_string())

        if sanity_check:
            return entropies_check


class SplitPatternDistribution(PatternDistribution):
    """ Implicative entropy distribution for split systems

    Split system entropy is the joint entropy on both systems.
    """

    def __init__(self, paradigms_list, patterns_list, pat_dic_list, names,
                 features=None):
        if features is not None:
            raise NotImplementedError(
                "Split patterns with features is not implemented yet.")
        columns = [tuple(paradigms.columns) for paradigms in paradigms_list]
        assert len(set(columns)) == 1, "Split systems must share same paradigm cells"

        self.distribs = [PatternDistribution(paradigms_list[i],
                                             patterns_list[i],
                                             pat_dic_list[i]) for i in
                         range(len(paradigms_list))]

        self.names = names
        self.paradigms = merge_split_df(paradigms_list)

        patterns_list = [p.applymap(lambda x: (str(x),)) for p in patterns_list]
        self.patterns = merge_split_df(patterns_list)
        log.info("Looking for classes of applicable patterns")
        classes_list = [d.classes for d in self.distribs]

        self.classes = merge_split_df(classes_list)

        # Information on the shape of both dimensions is always available in forms
        for distrib in self.distribs:
            distrib.classes = self.classes

        self.hasforms = {cell: (self.paradigms[cell] != "") for cell in self.paradigms}
        self.entropies = [None] * 10
        self.effectifs = [None] * 10

        # Extra
        self.columns = columns[0]
        self.patterns_list = patterns_list
        self.classes_list = classes_list

    def mutual_information(self, normalize=False):
        """ Information mutuelle entre les deux systèmes."""

        H, _ = self.distribs[0].entropy_matrix()
        Hprime, _ = self.distribs[1].entropy_matrix()
        Hjointe, _ = self.entropy_matrix()

        I = H + Hprime - Hjointe

        if normalize:
            return (2 * I) / (H + Hprime)
        else:
            return I

    def cond_bipartite_entropy(self, target=0, known=1):
        """ Entropie conditionnelle entre les deux systèmes, H(c1->c2\|c1'->c2') ou H(c1'->c2'\|c1->c2)
        """
        # For faster access
        log.info("Computing implicative H({}|{})".format(self.names[target],
                                                      self.names[known]))
        pats = self.patterns_list[target]

        predpats = self.patterns_list[known]

        cols = self.columns

        entropies = pd.DataFrame(index=cols, columns=cols)

        for a, b in pats.columns:
            selector = self.hasforms[a] & self.hasforms[b]
            entropies.at[a, b] = cond_entropy(pats[(a, b)], self.add_features(
                self.classes[(a, b)] + predpats[(a, b)]),
                                              subset=selector)
            entropies.at[b, a] = cond_entropy(pats[(a, b)], self.add_features(
                self.classes[(b, a)] + predpats[(a, b)]),
                                              subset=selector)

        return entropies
