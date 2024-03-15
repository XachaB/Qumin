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

from functools import reduce
from . import cond_entropy, entropy, P, matrix_analysis
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
                 features=None, frequencies_file_path=None, paradigms_file_path=None):
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
        Todo:
            Remove paradigms_file_path from arguments.
        """
        if not overabundant:
            # Keep the first form for each cell
            self.paradigms = paradigms.map(lambda x: x[0] if x else x)
        else:
            self.paradigms = paradigms

        self.weights = representations.frequencies.Weights(frequencies_file_path, paradigms_file_path)

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
        self.results = pd.DataFrame(columns=['effectifs'],
                                    index=pd.MultiIndex(levels=[[], [], []],
                                                        codes=[[], [], []],
                                                        names=['pred', 'out', 'params']))

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

    def _add_metric(self, pred, out, column, value, both=False):
        if type(column) is not str and len(column) > 1:
            index = (pred, out, column[1])
            index_r = (out, pred, column[1])
            metric = column[0]
        else:
            index = (pred, out)
            index_r = (out, pred)
            metric = column

        self.results.at[index, metric] = round(value, 10) + 0
        if both:
            self.results.at[index_r, metric] = round(value, 10) + 0

        self.results.sort_index(inplace=True)

    def n_preds_entropy_matrix(self, n):
        r"""Return a:class:`pandas:pandas.DataFrame` with nary entropies,
        and one with counts of lexemes.

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

    def entropy_matrix_OA(self, debug=False, weighting='type', sanity_check=False, **kwargs):
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
            debug (bool): Whether to enable debug logging. Default False.
            weighting (str): Kind of wheighting to use  # TODO
            sanity_check (bool): Whether to perform a slow computation check. Default False.
            **kwargs: optional keyword arguments.

        Note:
            As opposed to :func:`entropy_matrix`, this function allows overabundant forms.
        """

        log.info("Computing c1 → c2 entropies")

        log.info('Building frequency data')
        weights = self.weights.get_relative_freq(group_on=['lexeme', 'cell'])
        patterns_dic, weights_dic = self._patterns_to_long(weights)

        classes = self.classes

        def _pair_entropy(a, b, selector, **kwargs):
            """Produce an entropy analysis for a pair of columns
            """
            known_ab = self.add_features(classes[a][b])
            col_weights = weights.loc[pd.IndexSlice[:, a, :]]
            A, B = self._prepare_OA_data(pd.concat([patterns_dic[a][b], weights_dic[a][b+"_w"]],
                                                   axis=1),
                                         known_ab, subset=selector, weights=col_weights,
                                         weighting=weighting)

            if debug:
                log.debug("# Distribution of {} → {}".format(a, b))
                self.cond_entropy_OA_log(A, B, subset=selector, **kwargs)
            else:
                results_dict = self.cond_entropy_OA(A, B, subset=selector, **kwargs).unstack().to_dict()
                for param, result in results_dict.items():
                    self._add_metric(a, b, param, result)
                self._add_metric(a, b, 'effectifs', sum(selector))

        if debug:
            log.debug("Logging one predictor probabilities")
            log.debug(" P(x → y) = P(x~y | Class(x))")
        log.info('Going through each pair of columns')
        for a, b in tqdm(self.patterns.columns):
            selector = self.hasforms[a] & self.hasforms[b]
            if selector[selector].size != 0:
                _pair_entropy(a, b, selector, **kwargs)
                _pair_entropy(b, a, selector, **kwargs)

    def _patterns_to_long(self, weights=None):
        """This function is used to handle overabundant computations.
        It's main aim is to identify to which form (and thus frequency)
        each pattern corresponds. Indeed, this information is lost in
        the patterns file.

        Here, we rebuild the missing information by unpacking
        forms with the same method as in find_patterns.
        Final data is in long format. Since alignment is lost, we use a dict.
        """

        patterns = self.patterns
        col_names = set(self.paradigms.columns)
        patterns_dic = {}
        weights_dic = {}

        # For each cell, we have a DF, where the index is composed
        # of (lexeme, wordform) pairs, the columns correspond to
        # the remaining cells, and the values are the applicable rules

        for cell in self.paradigms.columns:
            _ = self.paradigms[cell].explode()
            patterns_dic[cell] = pd.DataFrame(columns=list(col_names-{cell}),
                                              index=[_.index, _])
            weights_dic[cell] = patterns_dic[cell].copy()

        def _dispatch_patterns(row, a, b, reverse=False):
            """ This function reassigns the patterns to their forms.
            This information was lost during previous steps.
            This step is mandatory to compute source-cell overabundance

            Arguments:
                row (:class:`pandas:pandas.Series`) : a row of the patterns table
                a (str) : first column name.
                b (str) : second column name.
                reverse (bool) : whether to compute for A->B of B->A. The function is not symmetric
            """
            lex = row.lexeme.iloc[0]
            forms = self.paradigms.at[lex, a], self.paradigms.at[lex, b]

            # This strategy avoids checking if reverse is True.
            rev = 1-int(reverse)
            outname = [a, b]

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
                pat_weights = [weights.loc[lex, outname[rev], str(p[rev]).strip()]['result']
                               if p[rev] != '' else 0 for p in pairs]
            return pd.Series([[p[1-rev] for p in pairs], lpatterns, pat_weights])

        def _format_patterns(a, b, reverse=False):
            """This is used for reformating DFs"""

            # haspattern = patterns[patterns[(a, b)] != ('None',)][(a, b)]
            z = patterns[(a, b)].reset_index().apply(
                lambda x: _dispatch_patterns(x, a, b, reverse=reverse),
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

        log.info('Formatting patterns')
        for a, b in tqdm(patterns.columns):
            _format_patterns(a, b)
            _format_patterns(a, b, reverse=True)

        return patterns_dic, weights_dic

    def _prepare_OA_data(self, A, B, subset=None, weights=None, weighting='type'):
        """This function is used to prepare the data for overabundance analysis.
        It checks if the arguments are right and it
        reorganizes the input DataFrames accordingly.

        Note:
            There are three options for weighting. The following settings are available :
                1. type: Normalized weighting for overabundant patterns and source cells
                2. mixed: Frequency based weighting for overabundant and source cells
                3. token: Consider the frequency of lexemes, both pattern prediction\
                and averaging of entropy/accuracy.

            Note that in cases 2 and 3, forms with a frequency of 0\
            will simply be skipped.
        """
        # TODO Move these to top of file / opening of weights
        if weights is None and weighting in ['mixed', 'token']:
            log.warning('Frequency computation required but no frequencies were provided.')
            log.warning('Falling back to type weighting.')
            weighting = 'type'
        # elif weights is not None and weighting == 'type':
        #     raise ValueError("Type weighting doesn't require any frequencies.")

        def get_weights(A):
            """Provides weights for the source cell.
            Only if token was selected.

            Todo:
                Remove this function and use the frequency API
            """
            if weighting == 'token':
                return A.apply(lambda x: weights.loc[
                    (x.name[0], str(x.name[1]).strip(' ')),
                    'value'], axis=1)
            else:  # Last case is common to mixed and type options. In type option, weights should already be uniform.
                return A.apply(lambda x: weights.loc[
                    (x.name[0], str(x.name[1]).strip(' ')),
                    'result'], axis=1)

        # Subsetting dataframes
        iname = A.index.names[0]
        A = A[subset[A.index.get_level_values(iname)].values].copy()
        B = B[subset[B.index.get_level_values(iname)].values]

        # Getting weights
        # breakpoint()
        A['w'] = get_weights(A)

        return A, B

    def cond_entropy_OA_log(self, A, B, subset=None, weighting='type',
                            **kwargs):
        """
        Print a log of the probability distribution for
        one predictor with overabundance.

        Writes down the distributions
        :math:`P( patterns_{c1, c2} | classes_{c1, c2} )`
        for all unordered combinations of two column
        names in :attr:`PatternDistribution.paradigms`.
        Also writes the entropy of the distributions.

        Print a log of this probability distribution for one predictor with overabundance.

        Arguments:
            A (:class:`pandas.core.series.Series`): A series of data.
            B (:class:`pandas.core.series.Series`): A series of data.
            subset (Optional[iterable]): Only give the distribution for a subset of values.
            weighting (str): which kind of approach should be used for weighting : type, \
            mixed, token.
            **kwargs: optional keyword arguments for :func:`matrix_analysis`.

        Return:
            list[float]: A list of metrics. First the global accuracy, then H(A|B).

        Note:
            This uses exactly the same process as
            :func:`cond_entropy_OA`. It only displays an additional log.)

        Todo:
            Enable sanity_check here also.
        """

        # A : patterns that can in fact be applied to each form
        # B : patterns that are potentially applicable to each form

        grouped_A = A.groupby(B, sort=False)
        results = []
        final_weights = []

        # Each subclass (forms with similar properties) is analyzed.
        for i, (classe, members) in enumerate(
                        sorted(grouped_A,
                               key=lambda x: len(x[1]),
                               reverse=True)):

            group_name = list(members.columns)
            patterns = group_name[0]
            members[patterns] = members[patterns].apply(lambda x: x[0].split(';'))
            weight = np.array(list(members['w']))
            group = members.explode(group_name[0:2])
            pivoted = group.pivot(values=group_name[1],
                                  columns=patterns)
            pat2id = {p: n for n, p in enumerate(pivoted.columns)}
            id2pat = list(pivoted.columns)
            pivoted.rename(pat2id, inplace=True, axis=1)

            matrix = np.nan_to_num(pivoted.to_numpy().astype(float))

            res = matrix_analysis(matrix, weights=weight, full=True, **kwargs)

            # Debuging
            myform = "{:5.2f}"
            log.debug("\n\n## Class n°%s (%s members).\n", str(i), str(len(members)))
            log.debug("Accuracy : "+myform.format(res[0]))
            log.debug("Entropy  : "+myform.format(res[1]))
            ptable = pd.DataFrame(pd.Series(id2pat, name="pattern"))

            ptable.index.name = "ID"
            ptable['frequency'] = res[4]
            ptable['P(p)'] = res[3]
            pivoted['weight'] = weight
            pivoted['P(success)'] = res[2]

            log.debug("Patterns:\n\n"+ptable.to_markdown(index=True)+"\n")
            pivoted.reset_index(inplace=True)
            log.debug("Members:\n\n"+pivoted.to_markdown(index=False)+"\n")
            results.append(list(res[0:2])+[len(members)])
            final_weights += [np.nansum(weight)]

        global_res = (np.array(final_weights)/sum(final_weights))@np.array(results)[:, 0:2]

        df_res = pd.DataFrame(results, columns=["Accuracy",
                                                "Entropy",
                                                "Size"])
        df_res['Weights'] = final_weights
        df_res.index.name = "Class"
        log.debug("\n\n# Global results\n")
        log.debug("Global accuracy is: %s", global_res[0])
        log.debug("Global entropy  is: %s", global_res[1])
        log.debug("\n"+df_res.to_markdown(
                    floatfmt=[".3f", ".3f", ".3f", ".0f"])+"\n")
        return None

    def cond_entropy_OA(self, A, B, subset=None, weighting='type', beta=[1], **kwargs):
        """Writes down the distributions
        :math:`P( patterns_{c1, c2} | classes_{c1, c2} )`
        for all unordered combinations of two column
        names in :attr:`PatternDistribution.paradigms`.
        Also writes the entropy of the distributions,
        and the accuracy of the computation.

        Arguments:
            A (:class:`pandas.core.series.Series`): A series of data.
            B (:class:`pandas.core.series.Series`): A series of data.
            subset (Optional[iterable]): Only give the distribution for a subset of values.
            weighting (str): which kind of approach should be used for weighting : type, \
            mixed, token.
            beta (List(float): values of beta to test if using a softmax computation.
            **kwargs: optional keyword arguments for :func:`matrix_analysis`.

        Return:
            list[float]: A list of metrics. First the global accuracy, then H(A|B).
        """
        population = A['w'].sum(skipna=True)
        grouped_A = A.groupby(B, sort=False)
        results = pd.DataFrame(columns=['accuracies', 'entropies'])

        for b in beta:

            # Each subclass (forms with similar properties) is analyzed.
            def group_analysis(group):
                group_name = list(group.columns)
                pattern = group_name[0]
                group[pattern] = group[pattern].apply(lambda x: x[0].split(';'))
                weight = np.array(list(group['w']))
                group = group.explode(group_name[0:2])
                matrix = np.nan_to_num(
                    group.pivot(
                        values=group_name[1],
                        columns=pattern)
                    .to_numpy()
                    .astype(float))

                return [i*(np.nansum(weight)/population)
                        for i in matrix_analysis(matrix, weights=weight, beta=b, **kwargs)[0:2]]

            results.loc[b] = np.nansum(list(grouped_A.apply(group_analysis)), axis=0)
        return results

    def entropy_matrix(self):
        r"""Return a:class:`pandas:pandas.DataFrame` with unary entropies,
        and one with counts of lexemes.

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

        for a, b in patterns.columns:
            selector = self.hasforms[a] & self.hasforms[b]
            known_ab = self.add_features(classes[(a, b)])
            known_ba = self.add_features(classes[(b, a)])

            self._add_metric(a, b, 'entropies',
                             cond_entropy(patterns[(a, b)],
                                          known_ab, subset=selector))
            self._add_metric(b, a, 'entropies',
                             cond_entropy(patterns[(a, b)],
                                          known_ba, subset=selector))

            self._add_metric(a, b, 'effectifs', sum(selector), both=True)

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
        """ Entropie conditionnelle entre les deux systèmes,
        H(c1->c2\|c1'->c2') ou H(c1'->c2'\|c1->c2)
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
