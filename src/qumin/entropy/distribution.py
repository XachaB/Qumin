# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Sacha Beniamine.

Encloses distribution of patterns on paradigms.
"""

import logging
import numpy as np
from collections import Counter, defaultdict
from functools import reduce
from itertools import combinations, chain, product

import pandas as pd
from tqdm import tqdm

from . import cond_entropy, matrix_analysis
from .. import representations

log = logging.getLogger(__name__)


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

    def __init__(self, paradigms, patterns, classes, name, md, real_frequencies=False, overabundant=False,
                 features=None):
        """Constructor for PatternDistribution.

        Arguments:
            paradigms (:class:`pandas:pandas.DataFrame`):
                containing forms.
            patterns (:class:`pandas:pandas.DataFrame`):
                patterns (columns are pairs of cells, index are lemmas).
            classes (:class:`pandas:pandas.DataFrame`):
                classes of applicable patterns from one cell to another.
            overabundant (bool):
            features:
                optional table of features
            weights:
                optional frequency information
        Todo:
            Remove paradigms_file_path from arguments.
        """
        self.name = name

        if not overabundant:
            # Keep the first form for each cell
            self.paradigms = paradigms.map(lambda x: x[0] if x else x)
        else:
            self.paradigms = paradigms

        self.classes = classes

        self.weights = representations.frequencies.Weights(md.get_table_path('forms'),
                                                           frequencies_path=md.get_table_path('frequencies'),
                                                           real_frequencies=real_frequencies)

        self.patterns = patterns.map(lambda x: (str(x),))
        # TODO check if the version below is really not useful and why
        # self.patterns = patterns.map(lambda x: (str(x),) if type(x) is not tuple else x)

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

        self.hasforms = {cell: (paradigms[cell] != "") for cell in self.paradigms}
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
        measure = [measure] if type(measure) is str else measure
        is_cond_ent = self.data.loc[:, "measure"].isin(measure)
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

    def add_features(self, series):
        return series + self.features[series.index]

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
            log.info("Saving time by listing already known 0 entropies...")
            if n - 1 in self.data.loc[:, "n_preds"]:
                df = self.get_results(measure="cond_entropy", n=n - 1).groupby("predicted")
                if n - 1 == 1:
                    df = df.agg({"predictor": lambda ps: set(frozenset({pred}) for pred in ps)})
                else:
                    df = df.agg({"predictor": lambda ps: set(frozenset(pred) for pred in ps)})
                return df.to_dict(orient="index")
            return None

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

        zeros = check_zeros(n)

        pat_order = {}
        for a, b in patterns:
            pat_order[(a, b)] = (a, b)
            pat_order[(b, a)] = (a, b)

        def calc_condent(predictors):
            # combinations gives us all x, y unordered unique pair for all of
            # the n predictors.
            pairs_of_predictors = list(combinations(predictors, 2))
            known_patterns = patterns[pairs_of_predictors]
            set_predictors = set(predictors)
            predsselector = reduce(lambda x, y: x & y,
                                   (self.hasforms[x] for x in predictors))
            for out in (x for x in columns if x not in predictors):
                selector = predsselector & self.hasforms[out]
                if zeros is not None and already_zero(set_predictors, out, zeros):
                    yield [predictors, out, 0, sum(selector)]
                else:
                    # Getting intersection of patterns events for each
                    # predictor: x~z, y~z
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
                    yield [predictors, out, "cond_entropy", cond_entropy(A, B, subset=selector),
                           sum(selector), len(predictors), self.name]

        rows = chain(*[calc_condent(preds) for preds in combinations(columns, n)])

        self.data = pd.concat([self.data, pd.DataFrame(rows, columns=self.data.columns)])

    def one_pred_entropy_OA(self, cfg, verbose=False):
        r"""Return a:class:`pandas:pandas.DataFrame` with unary entropies and counts of lexemes.

        The result contains entropy :math:`H(c_{1} \to c_{2})`.

        Values are computed for all unordered combinations
        of :math:`(c_{1}, c_{2})` where `c_{1} != c_{2}`
        in the :attr:`PatternDistribution.paradigms`'s columns.

        Example:
            For two cells c1, c2, entropy of c1 → c2,
            noted :math:`H(c_{1} \to c_{2})` is:

            .. math::

                H( patterns_{c1, c2} | classes_{c1, c2} )

        Arguments:
            cfg (dict): Configuration dictionary.
            debug (bool): Whether to enable full logging. Default False.

        Note:
            As opposed to :func:`entropy_matrix`, this function allows overabundant forms.
        """

        log.info("Computing c1 → c2 entropies")

        log.info('Building frequency data')
        weights = self.weights.get_relative_freq(group_on=['lexeme', 'cell'])
        patterns_dic, weights_dic = self._patterns_to_long(weights)

        classes = self.classes
        rows = list(self.paradigms.columns)

        def _pair_entropy(a, b, selector, cfg):
            """Produce an entropy analysis for a pair of columns
            """
            known_ab = self.add_features(classes[a][b])
            col_weights = weights.loc[pd.IndexSlice[:, a, :]]
            A, B = self._prepare_OA_data(pd.concat([patterns_dic[a][b], weights_dic[a][b+"_w"]],
                                                   axis=1),
                                         known_ab, subset=selector, weights=col_weights,
                                         token=cfg.token)

            # If the target is overabundant but unattested, we need to set the weights to 0
            A['w'] = A['w']*A[f'{list(A.columns)[0]}_w'].apply(sum)

            if verbose:
                log.debug("# Distribution of {} → {}".format(a, b))
                self.cond_entropy_OA_log(A, B, cfg, subset=selector)
            else:
                results = self.cond_entropy_OA(A, B, cfg, subset=selector).unstack()
                results.name = "value"
                results = results.to_frame()

                # The size of the sample corresponds to the number of pairs of forms that have a
                # weight for both predictor and target.
                results.loc[:, 'n_pairs'] = A['w'].sum()

                return results

        data = pd.DataFrame(index=rows,
                            columns=rows).reset_index(drop=False,
                                                      names="predictor").melt(id_vars="predictor",
                                                                              var_name="predicted",
                                                                              value_name="value")
        data = data[data.predictor != data.predicted]  # drop a -> a cases
        data.loc[:, "n_pairs"] = None
        data.loc[:, "n_preds"] = 1
        data.loc[:, "dataset"] = self.name
        data.loc[:, "measure"] = "cond_entropy"
        data.loc[:, "parameters"] = None

        def calc_condent(row):
            a, b = row["predictor"], row["predicted"]
            selector = self.hasforms[a] & self.hasforms[b]
            row['measure'] = ['cond_entropy', 'accuracy']
            if selector[selector].size != 0:
                results = _pair_entropy(a, b, selector, cfg)
                if not verbose:
                    results.index.set_names(['measure', 'parameters'], inplace=True)
                    results = results.reset_index()
                    for c in results.columns:
                        row[c] = results[c].to_list()
            else:
                row['value'] = None, None
            return row

        log.info("Printing log for P(c1 → c2).")
        log.debug("Logging one predictor probabilities")
        log.debug(" P(x → y) = P(x~y | Class(x))")
        log.info('Going through each pair of columns')

        data = data.apply(calc_condent, axis=1)
        if not verbose:
            data = data.explode(['measure', 'value', 'parameters', 'n_pairs'])
            self.data = pd.concat([self.data, data])

    def _patterns_to_long(self, weights=None):
        """This function is used to handle overabundant computations.
        It's main aim is to identify to which form (and thus frequency)
        each pattern corresponds. This information is lost in
        the patterns file.

        Here, we rebuild the missing information by unpacking
        forms with the same method as in find_patterns.
        Final data is in long format. Since alignment is lost, we use a dict.
        """

        patterns = self.patterns
        col_names = set(self.paradigms.columns)
        patterns_dic = {}
        weights_dic = {}

        # For each cell, pattern_dic contains a DF, where the index is a list
        # of (lexeme, wordform) tuples, and the columns are the other
        # cells. The values are the applicable rules to produce the other cells,
        # given a wordform.

        for cell in self.paradigms.columns:
            _ = self.paradigms[cell].explode()
            patterns_dic[cell] = pd.DataFrame(columns=list(col_names-{cell}),
                                              index=[_.index, _])
            weights_dic[cell] = patterns_dic[cell].copy()

        def _dispatch_patterns(row, a, b, reverse=False):
            """ This function reassigns the patterns to their forms.
            This information was lost in the patternsPhonsim file. We
            also retrieve weights.

            Arguments:
                row (:class:`pandas:pandas.Series`): a row of the patterns table to dispatch
                a (str): first column name.
                b (str): second column name.
                reverse (bool): whether to compute for A->B of B->A. The function is not symmetric
            Returns:
                pandas.Series: The same row, containing lists. Each lists contains
                in the same order the name of the source/target forms,
                and the list of the patterns for each pair, and the weights associated.
            """
            lex = row.lexeme.iloc[0]
            forms = self.paradigms.at[lex, a], self.paradigms.at[lex, b]

            # This strategy avoids checking all the time if reverse is True.
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

                # At the same time we also associate a weight to each pattern.
                pat_weights = [weights.loc[lex, outname[rev], str(p[rev]).strip()]['result']
                               if p[rev] != '' else 0 for p in pairs]

            return pd.Series([[p[1-rev] for p in pairs], lpatterns, pat_weights])

        def _format_patterns(a, b, reverse=False):
            """This is used for reformating DFs to the correct shape.
            First, the function :func:`_dispatch_patterns` associates
            the patterns to pairs of forms. It returns parallel lists,
            that we explode to obtain proper DataFrames.

            Arguments:
                a (str): first column name.
                b (str): second column name.
                reverse (bool): whether to compute for A->B of B->A. The function is not symmetric
            """

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

    def _prepare_OA_data(self, A, B, subset=None, weights=None, token=False):
        """This function is used to prepare the data for overabundance analysis.
        It checks if the arguments are right and it
        reorganizes the input DataFrames accordingly.

        Note:
            There are three options for weighting. The following settings are available :
                1. type without frequencies: Normalized weighting for overabundant patterns and source cells
                2. type with frequencies: Frequencies are used for overabundant and source cells ratios.
                3. token: Consider the frequency of lexemes, both for pattern prediction\
                and averaging of entropy/accuracy.

            Note that in cases 2 and 3, forms with a frequency of 0\
            will simply be skipped.
        """

        def get_weights(A):
            """Provides weights for the source cell.
            Only if token was selected.

            Todo:
                Remove this function and use the frequency API
            """
            if token:
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
        A['w'] = get_weights(A)

        return A, B

    def cond_entropy_OA_log(self, A, B, cfg, subset=None):
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
            cfg (dict): Configuration file for entropy computations.
            subset (Optional[iterable]): Only give the distribution for a subset of values.

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
            group = group.reset_index().groupby(
                    list(group.index.names) + [patterns]).sum().reset_index(level=patterns)
            pivoted = group.pivot(values=group_name[1],
                                  columns=patterns)

            pat2id = {p: n for n, p in enumerate(pivoted.columns)}
            id2pat = list(pivoted.columns)
            pivoted.rename(pat2id, inplace=True, axis=1)

            matrix = np.nan_to_num(pivoted.to_numpy().astype(float))

            res = matrix_analysis(matrix, cfg, weights=weight, full=True)

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

    def cond_entropy_OA(self, A, B, cfg, subset=None):
        """Writes down the distributions
        :math:`P( patterns_{c1, c2} | classes_{c1, c2} )`
        for all unordered combinations of two column
        names in :attr:`PatternDistribution.paradigms`.
        Also writes the entropy of the distributions,
        and the accuracy of the computation.

        Arguments:
            A (:class:`pandas.core.series.Series`): A series of data.
            B (:class:`pandas.core.series.Series`): A series of data.
            cfg (dict): Configuration file for entropy computations.
            subset (Optional[iterable]): Only give the distribution for a subset of values.

        Return:
            list[float]: A list of metrics. First the global accuracy, then H(A|B).
        """

        population = A['w'].sum(skipna=True)
        grouped_A = A.groupby(B, sort=False)
        results = pd.DataFrame(columns=['accuracy', 'cond_entropy'])

        for b in cfg.beta:

            # Each subclass (forms with similar properties) is analyzed.
            def group_analysis(group):
                group_name = list(group.columns)
                patterns = group_name[0]
                group[patterns] = group[patterns].apply(lambda x: x[0].split(';'))
                weight = np.array(list(group['w']))
                group = group.explode(group_name[0:2])

                # This step is mandatory because in some very rare cases, the same pattern can
                # apply several times to one form. We simply reduce this case, since it's marginal.
                # Yet it could be interesting to find another solution.
                group = group.reset_index().groupby(
                    list(group.index.names) + [patterns]).sum().reset_index(level=patterns)

                # We turn our results into a matrix where rows are predictors and columns patterns.
                matrix = np.nan_to_num(
                    group.pivot(
                        values=group_name[1],
                        columns=patterns)
                    .to_numpy()
                    .astype(float))

                return [i*(np.nansum(weight)/population)
                        for i in matrix_analysis(matrix, cfg, weights=weight, beta=b)[0:2]]
            if cfg.function != "soft":
                param = cfg.function
            else:
                param = cfg.function + ' - ' + str(b)
            results.loc[param] = np.nansum(list(grouped_A.apply(group_analysis)), axis=0)
        return results

    def one_pred_entropy(self):
        r"""Return a:class:`pandas:pandas.DataFrame` with unary entropies and counts of lexemes.

        The result contains entropy :math:`H(c_{1} \to c_{2})`.

        Values are computed for all unordered combinations
        of :math:`(c_{1}, c_{2})` where `c_{1} != c_{2}`
        in the :attr:`PatternDistribution.paradigms`'s columns.

        Example:
            For two cells c1, c2, entropy of c1 → c2,
            noted :math:`H(c_{1} \to c_{2})` is:

            .. math::

                H( patterns_{c1, c2} | classes_{c1, c2} )

        Note:
            As opposed to :func:`one_pred_entropy_OA`, this won't work
            with overabundant forms.
        """
        log.info("Computing c1 → c2 entropies")

        # For faster access
        patterns = self.patterns
        classes = self.classes
        rows = list(self.paradigms.columns)

        data = pd.DataFrame(index=rows,
                            columns=rows).reset_index(drop=False,
                                                      names="predictor").melt(id_vars="predictor",
                                                                              var_name="predicted",
                                                                              value_name="value")
        data = data[data.predictor != data.predicted]  # drop a -> a cases
        data.loc[:, "n_pairs"] = None
        data.loc[:, "n_preds"] = 1
        data.loc[:, "measure"] = "cond_entropy"
        data.loc[:, "dataset"] = self.name

        def calc_condent(row):
            a, b = row["predictor"], row["predicted"]
            selector = self.hasforms[a] & self.hasforms[b]
            row["n_pairs"] = sum(selector)
            known_ab = self.add_features(classes[(a, b)])
            pats = patterns[(a, b)] if (a, b) in patterns else patterns[(b, a)]
            row["value"] = cond_entropy(pats, known_ab, subset=selector)
            return row

        data = data.apply(calc_condent, axis=1)
        self.data = pd.concat([self.data, data])

    def one_pred_distrib_log(self):
        """Print a log of the probability distribution for one predictor.

        Writes down the distributions
        :math:`P( patterns_{c1, c2} | classes_{c1, c2} )`
        for all unordered combinations of two column
        names in :attr:`PatternDistribution.paradigms`.
        Also writes the entropy of the distributions.
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

                log.debug("Showing distributions for "
                          + str(len(cond_events))
                          + " classes")

                for i, (classe, members) in enumerate(sorted(cond_events,
                                                             key=lambda x: len(x[1]),
                                                             reverse=True)):
                    headers = ("Pattern", "Example",
                               "Size", "P(Pattern|class)")
                    table = []

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
                        table.append(row)

                    log.debug("\n" + pd.DataFrame(table,
                                                  columns=headers).to_markdown())

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


class SplitPatternDistribution(PatternDistribution):
    """ Implicative entropy distribution for split systems

    Split system entropy is the joint entropy on both systems.
    """

    def __init__(self, paradigms_list, patterns_list, classes_list, names,
                 features=None):
        columns = [tuple(paradigms.columns) for paradigms in paradigms_list]
        assert len(set(columns)) == 1, "Split systems must share same paradigm cells"

        super().__init__(merge_split_df(paradigms_list),
                         merge_split_df([p.map(lambda x: (str(x),)) for p in patterns_list]),
                         merge_split_df(classes_list),
                         "bipartite:" + "&".join(names),
                         features=features
                         )

        # Add one pattern distribution for each dataset
        self.distribs = [PatternDistribution(paradigms_list[i],
                                             patterns_list[i],
                                             classes_list[i],
                                             name=names[i],
                                             features=features
                                             ) for i in
                         range(len(paradigms_list))]

        # Information on the shape of both dimensions is always available in forms
        for distrib in self.distribs:
            distrib.classes = self.classes

        # Extra
        self.columns = columns[0]
        self.patterns_list = patterns_list
        self.classes_list = classes_list

    def mutual_information(self, normalize=False):
        """ Information mutuelle entre les deux systèmes."""
        self.distribs[0].one_pred_entropy()
        self.distribs[1].one_pred_entropy()
        self.one_pred_entropy()

        index = ["predictor", "predicted"]
        left_ent = self.distribs[0].get_results()
        right_ent = self.distribs[1].get_results()

        # For operations, we need all of these as simple series of values,
        # indexed by predictors & predicted
        H = left_ent.set_index(index).value
        Hprime = right_ent.set_index(index).value
        Hjointe = self.get_results().set_index(index).value

        I = H + Hprime - Hjointe
        NMI = (2 * I) / (H + Hprime)

        # Register results
        I = I.reset_index(drop=False)
        I["measure"] = "mutual_information"
        I["dataset"] = self.name
        I["n_pairs"] = ""

        NMI = NMI.reset_index(drop=False)
        NMI["measure"] = "normalized_mutual_information"
        NMI["dataset"] = self.name
        NMI["n_pairs"] = ""

        self.data = pd.concat([self.data, left_ent, right_ent, I, NMI])
