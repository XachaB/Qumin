# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Sacha Beniamine.

Encloses distribution of patterns on paradigms.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from prettytable import PrettyTable, ALL
from itertools import combinations, product
# TODO replace sklearn with some lighter solution
from sklearn.feature_extraction.text import CountVectorizer

from functools import reduce
from . import cond_entropy, entropy, P, cond_entropy_OA, matrix_analysis
from .. import representations
from tqdm import tqdm
import logging

log = logging.getLogger()


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

    def __init__(self, paradigms, patterns, pat_dic, overabundant=False,
                 features=None, frequencies=None):
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
            weights:
                optional frequency information
        """
        if not overabundant:
            # Keep the first form for each cell
            self.paradigms = paradigms.map(lambda x: x[0] if x else x)
        else:
            self.paradigms = paradigms

        if frequencies is not None:
            self.weights = representations.frequencies.Weights(frequencies)
        else:
            self.weights = False
        self.pat_dict = pat_dic
        self.patterns = patterns.map(lambda x: (str(x),))
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
            self.add_features = lambda x: x

        log.info("Looking for classes of applicable patterns")
        if overabundant:
            self.classes = representations.patterns.find_applicable_OA(self.paradigms,
                                                                       self.pat_dict)
        else:
            self.classes = representations.patterns.find_applicable(self.paradigms,
                                                                    self.pat_dict)
        log.debug("Classes:")
        log.debug(self.classes)
        self.hasforms = {cell: (paradigms[cell] != "") for cell in self.paradigms}
        self.entropies = [None] * 10
        self.accuracies = [None] * 10
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

    def _register_entropy(self, n, entropy, effectifs, accuracies=None):
        """Register an entropy score_matrix for n predictors.

        Arguments:
            n (int): number of predictors
            entropy  (:class:`pandas:pandas.DataFrame`):
                Entropy score_matrix to register.
            accuracies  (:class:`pandas:pandas.DataFrame`):
                Accuracy score_matrix to register.
        """
        entropy = value_norm(entropy)
        if accuracies is not None:
            accuracies = value_norm(accuracies)

        try:
            if accuracies is not None:
                self.accuracies[n] = accuracies
            self.entropies[n] = entropy
            self.effectifs[n] = effectifs
        except IndexError:
            self.entropies.append([None] * n)
            self.effectifs.append([None] * n)
            self.entropies[n] = entropy
            self.effectifs[n] = effectifs
            if accuracies is not None:
                self.accuracies.append([None] * n)
                self.accuracies[n] = accuracies

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
                                 columns=columns)
        effectifs = pd.DataFrame(index=indexes,
                                 columns=columns)

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

    def entropy_matrix_OA(self, silent=False, **kwargs):
        r"""Creates a :class:`pandas:pandas.DataFrame`
        with unary entropies, and one with counts of lexemes.

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

        Arguments:
            **kwargs: optional keyword arguments.

        Note:
            As opposed to :func:`entropy_matrix`, this function allows overabundant forms.

        Todo:
            Merge with :func:`entropy_matrix`.
        """

        if not silent:
            log.info("Computing c1 → c2 entropies")

        # For faster access
        patterns = self.patterns
        patterns_dic = {}
        weights_dic = {}
        col_names = set(self.paradigms.columns)

        # For each cell, we have a DF, where the index is composed
        # of (lexeme, wordform) pairs, the columns correspond to
        # the remaining cells, and the values are the applicable rules

        for cell in tqdm(self.paradigms.columns):
            _ = self.paradigms[cell].explode()
            patterns_dic[cell] = pd.DataFrame(columns=list(col_names-{cell}),
                                              index=[_.index, _])
            weights_dic[cell] = patterns_dic[cell].copy()

        if self.weights:
            weights = self.weights.get_relative_freq(group_on=['lexeme', 'cell'])

        def dispatch_patterns(row, a, b, reverse=False):
            """ This function reassigns the patterns to their forms.
            This information was lost during previous steps.
            This step is mandatory to compute source-cell overabundance

            Arguments:
                row (:class:`pandas:pandas.Series`) : a row of the patterns table
                a (str) : first column name.
                b (str) : second column name.
                reverse (bool) : whether to compute for A->B of B->A. The function is not symmetric.

            Note:
                As opposed to :func:`entropy_matrix`, this function allows overabundant forms.

            Todo:
                Merge with :func:`entropy_matrix`."""

            lex = row.lexeme.iloc[0]
            if reverse:
                forms = self.paradigms.at[lex, b], self.paradigms.at[lex, a]
                outname = a
            else:
                forms = self.paradigms.at[lex, a], self.paradigms.at[lex, b]
                outname = b

            nullset = {''}
            if forms != (nullset, nullset):
                if forms[0] == forms[1]:  # If the lists are identical, do not compute product
                    pairs = [(x, x) for x in forms[0]]
                    lpatterns = row[(a, b)][0].split(";")
                elif forms[0] == '':
                    pairs = [('', x) for x in forms[1]]
                    lpatterns = ['' for i in forms[1]]
                elif forms[1] == '':
                    pairs = [(x, '') for x in forms[0]]
                    lpatterns = ['' for i in forms[0]]
                else:
                    pairs = [(pred, out) for pred, out in product(*forms)]
                    lpatterns = row[(a, b)][0].split(";")
                if self.weights:
                    local_weights = [weights.loc[lex, outname, str(out).strip()]['result']
                                     if out != '' else 0 for pred, out in pairs]

            return pd.Series([[p for p, _ in pairs], lpatterns, local_weights])

        def format_patterns(a, b, reverse=False):
            """This is used for reformating DFs"""

            z = patterns[(a, b)].reset_index().apply(
                lambda x: dispatch_patterns(x, a, b, reverse=reverse),
                axis=1).explode([0, 1, 2])
            z = z.reset_index().groupby(['index', 0], dropna=False).agg(
                {1: lambda x: (';'.join(x),),
                 2: lambda x: tuple(x)})

            if reverse:
                pred, out = b, a
            else:
                pred, out = a, b
            z.index = patterns_dic[pred][out].index
            patterns_dic[pred][out] = z[1]
            weights_dic[pred][out+"_w"] = z[2]

        for a, b in tqdm(patterns.columns):
            format_patterns(a, b)
            format_patterns(a, b, reverse=True)

        classes = self.classes
        rows = list(self.paradigms.columns)

        entropies = pd.DataFrame(index=rows, columns=rows)
        accuracies = pd.DataFrame(index=rows, columns=rows)
        effectifs = pd.DataFrame(index=rows, columns=rows)

        for a, b in patterns.columns:
            selector = self.hasforms[a] & self.hasforms[b]
            if selector[selector].size != 0:
                known_ab = self.add_features(classes[a][b])
                known_ba = self.add_features(classes[b][a])

                _ = cond_entropy_OA(pd.concat([patterns_dic[a][b], weights_dic[a][b+"_w"]], axis=1),
                                    known_ab, subset=selector,
                                    **kwargs)

                entropies.at[a, b] = _[0, 1]
                accuracies.at[a, b] = _[0, 0]

                _ = cond_entropy_OA(pd.concat([patterns_dic[b][a], weights_dic[b][a+"_w"]], axis=1),
                                    known_ba, subset=selector,
                                    **kwargs)

                entropies.at[b, a] = _[0, 1]
                accuracies.at[b, a] = _[0, 0]

                effectifs.at[a, b] = sum(selector)
                effectifs.at[b, a] = sum(selector)
        self._register_entropy(1, entropies, effectifs, accuracies=accuracies)

    def one_pred_distrib_log_OA(self, sanity_check=False, detailed=False):
        r"""Print a log of the probability distribution for
        one predictor with overabundance.

        Writes down the distributions
        :math:`P( patterns_{c1, c2} | classes_{c1, c2} )`
        for all unordered combinations of two column
        names in :attr:`PatternDistribution.paradigms`.
        Also writes the entropy of the distributions.

        Arguments:
            sanity_check (bool): Default to False. Use a slower calculation to check that the results are exact.
            detailed (bool): Default to False. Whether to display or not each individual lexeme in the log.

        Note:
            As opposed to :func:`one_pred_distrib_log`, this function allows
            overabundant forms.

        Todo:
            Enable sanity_check here also.
            Handling of overabundant source forms (not correct yet)
        """

        def count_with_examples(row, counter, examples, paradigms, cells):
            c1, c2 = cells
            lemma, pattern = row
            example = "{}: {} → {}".format(lemma,
                                           paradigms.at[lemma, c1],
                                           paradigms.at[lemma, c2])
            counter[pattern] += 1
            examples[pattern] = example

        log.debug("Printing log for P(c1 → c2).")

        if sanity_check:
            rows = list(self.paradigms.columns)
            entropies_check = pd.DataFrame(index=rows,
                                            columns=rows)

        log.debug("Logging one predictor probabilities")
        log.debug(" P(x → y) = P(x~y | Class(x))")

        patterns = self.patterns  # .map(lambda x: x[0])

        for column in patterns:
            for pred, out in [column, column[::-1]]:
                selector = self.hasforms[pred] & self.hasforms[out]
                log.debug("\n# Distribution of {}→{} \n".format(pred, out))
                A = patterns[column][selector]
                B = self.add_features(self.classes[(pred, out)][selector])
                cond_events = A.groupby(B, sort=False)

                # I disable the sanity_check, but I may add it later.
                # if sanity_check:
                #     classes_p = P(B)
                #     cond_p = P(cond_events)

                #     surprisal = cond_p.groupby(levlevelel=0).apply(entropy)
                #     slow_ent = (classes_p * surprisal).sum()
                #     entropies_check.at[pred, out] = slow_ent
                #     print("Entropy from this distribution: ",
                #           slow_ent)

                #     if self.entropies[1] is not None:
                #         ent = self.entropies[1].at[pred, out]
                #         print("Entropy from the score_matrix: ", ent)

                #         if not isclose(ent, slow_ent):
                #             print("\n# Distribution of {}→{}".format(pred, out))
                #             print("WARNING: Something is wrong"
                #                   " in the entropy's calculation. "
                #                   "Slow and fast methods produce "
                #                   "different results: slow {}, fast {}"
                #                   "".format(slow_ent, ent))

                log.debug("Showing distributions for %s classes",
                          len(cond_events))
                population = selector.shape[0]
                results = []
                myform = "{:5.2f}"
                myform2 = ".2f"
                for i, (classe, members) in enumerate(
                        sorted(cond_events,
                               key=lambda x: len(x[1]),
                               reverse=True)):
                    phi = "soft"
                    beta = 10

                    # Tableau des patterns (init).
                    ptable = pd.DataFrame(
                        columns=["ID", "pattern",
                                 "P(p) ["+phi+"," + "beta = "+str(beta)+"]"])

                    # On analyse chaque groupe
                    cv = CountVectorizer(tokenizer=lambda x: x,
                                         lowercase=False,
                                         token_pattern=None)

                    m = cv.fit_transform([_[0].split(";")
                                          for _ in members]).todense()

                    # Compute P(success) for each row.
                    (accuracy, entropy,
                     row_proba, pat_proba) = matrix_analysis(m,
                                                             phi=phi,
                                                             beta=beta)

                    pat_proba_dic = defaultdict(
                        int, {str(e): pat_proba[0, n] for n, e in enumerate(
                                    cv.get_feature_names_out())})
                    # This must return 0 if pattern not available

                    m = np.concatenate((m, row_proba), axis=1)
                    results.append([accuracy, entropy, members.shape[0]])

                    # Log
                    # Table of patterns

                    log.debug("\n## Class n°" + str(i) + " ("+str(len(members))
                              + " members).\n")
                    log.debug("Accuracy : "+myform.format(accuracy))
                    log.debug("Entropy  : "+myform.format(entropy))

                    pat_dic = {}
                    pat_list = []
                    n = 0
                    for n, my_pattern in enumerate(classe):
                        id = "p"+str(n)
                        s = str(my_pattern)
                        pat_dic[s] = id
                        pat_list.append(my_pattern)
                        ptable.loc[len(ptable)] = [id, s, myform.format(pat_proba_dic[s])]
                    ptable = ptable.set_index(["ID"])

                    source_oa = False
                    try:
                        # Table of results
                        titles = [pat_dic[str(e)]
                                  for e in cv.get_feature_names_out()]
                        titles_fmt = [".0f"]+[".0f" for e in
                                              cv.get_feature_names_out()]
                        titles_fmt.extend([myform2, myform2])
                        titles.append("P(success)")
                    except:
                        # TODO brute-force handling of exceptions
                        source_oa = True

                    log.debug("List of patterns for this class is:\n")
                    log.debug("\n" + ptable.to_markdown())

                    # Only if source cell is not overabundant.
                    if not source_oa:
                        display = pd.DataFrame(m, columns=titles,
                                               index=members.index)

                        # Regrouper les sous-catégories similaires
                        # et compter leur fréquence
                        if not detailed:
                            display_words = display.copy().drop_duplicates().reset_index()
                            display = display.value_counts(normalize=True).reset_index()
                            display = display.merge(display_words).set_index("lexeme")
                        counter = Counter()
                        examples = defaultdict()
                        members.reset_index().apply(count_with_examples,
                                                    args=(counter,
                                                          examples,
                                                          self.paradigms,
                                                          (pred, out)),
                                                    axis=1)
                        # total = sum(list(counter.values()))
                        if self.features is not None:
                            log.debug("Features: %s", *classe[-self.features_len:])
                            classe = classe[:-self.features_len]

                        # for my_pattern in classe:
                        #     if my_pattern in counter:
                        #         row = (str(my_pattern),
                        #                examples[my_pattern],
                        #                counter[my_pattern],
                        #                counter[my_pattern] / total)
                            # else:
                                # row = (str(my_pattern), "-", 0, 0)
                            # table.add_row(row)

                        log.debug("\nMatrix of available patterns:\n")
                        log.debug("\n"+display.to_markdown(floatfmt=titles_fmt))
                        log.debug("\n")
                    else:
                        # TODO Handle this differently
                        log.debug("""\n!!! Source cell is overabundant !!!
                            Skip.""")

                results = np.matrix(results)
                global_res = results[:, 2].T@(results[:, 0:2]/population)
                df_res = pd.DataFrame(results, columns=["Accuracy",
                                                        "Entropy",
                                                        "Size"])
                log.debug("\n# Global results")
                log.debug("\nGlobal accuracy is: %s", global_res[0, 0])
                log.debug("Global entropy  is: %s", global_res[0, 1])
                log.debug("\n## Detailed overview of results\n")
                log.debug(df_res.to_markdown(
                    floatfmt=[".3f", ".3f", ".3f", ".0f"]))

        if sanity_check:
            return value_norm(entropies_check)

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

        Note:
            As opposed to :func:`entropy_matrix_OA`, this won't work
            with overabundant forms.
        """
        log.info("Computing c1 → c2 entropies")

        # For faster access
        patterns = self.patterns
        classes = self.classes
        rows = list(self.paradigms.columns)

        entropies = pd.DataFrame(index=rows, columns=rows)
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

    def one_pred_distrib_log(self, sanity_check=False):
        """Print a log of the probability distribution for one predictor.

        Writes down the distributions
        :math:`P( patterns_{c1, c2} | classes_{c1, c2} )`
        for all unordered combinations of two column
        names in :attr:`PatternDistribution.paradigms`.
        Also writes the entropy of the distributions.

        Arguments:
            sanity_check (bool): Use a slower calculation to check that the results are exact.
        Note:
            As opposed to :func:`one_pred_distrib_log_OA`, this won't work
            with overabundant forms.
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
                                           columns=rows)

        log.debug("Logging one predictor probabilities")
        log.debug(" P(x → y) = P(x~y | Class(x))")

        patterns = self.patterns.map(lambda x: x[0])

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
                    slow_ent = (classes_p * surprisal).sum()
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
            return value_norm(entropies_check)

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
                                           columns=columns)

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
                    slow_ent = (classes_p * surprisal).sum()
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
            return value_norm(entropies_check)


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

        patterns_list = [p.map(lambda x: (str(x),)) for p in patterns_list]
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

        self.distribs[0].entropy_matrix()
        self.distribs[1].entropy_matrix()
        self.entropy_matrix()

        H = self.distribs[0].entropies[1]
        Hprime = self.distribs[1].entropies[1]
        Hjointe = self.entropies[1]

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

        return value_norm(entropies)
