# -*- coding: utf-8 -*-
# !/usr/bin/python3

import numpy as np
from scipy import special as sp


def P(x, subset=None):
    """Return the probability distribution of elements in a :class:`pandas.core.series.Series`.

    Arguments:
        x (:class:`pandas.core.series.Series`): A series of data.
        subset (iterable): Only give the distribution for a subset of values.

    Returns:
        A :class:`pandas.core.series.Series` which index are x's elements and which values are their probability in x.
    """
    if subset is None:
        return x.value_counts(normalize=True, sort=False)
    else:
        return x[subset].value_counts(normalize=True, sort=False)


def cond_P(A, B, subset=None):
    """Return the conditional probability distribution P(A|B) for elements in two :class:`pandas.core.series.Series`.

    Arguments:
        A (:class:`pandas.core.series.Series`): A series of data.
        B (:class:`pandas.core.series.Series`): A series of data.
        subset (iterable): Only give the distribution for a subset of values.

    Return:
        A :class:`pandas.core.series.Series` whith two indexes.
        The first index is from the elements of B, the second from the elements of A.
        The values are the P(A|B).
    """
    if subset is None:
        cond_events = A.groupby(B, sort=False)
    else:
        cond_events = A[subset].groupby(B[subset], sort=False)
    return P(cond_events)


def cond_entropy(A, B, **kwargs):
    """Calculate the conditional entropy between two series of data points.
       Presupposes that values in the series are of the same type, typically tuples.

    Arguments:
        A (:class:`pandas.core.series.Series`): A series of data.
        B (:class:`pandas.core.series.Series`): A series of data.

    Return:
        H(A|B)
    """
    return entropy(P(A + B, **kwargs)) - entropy(P(B, **kwargs))


def entropy(A):
    """Calculate the entropy for a series of probabilities.

    Arguments:
        A (:class:`pandas.core.series.Series`): A series of data.

    Return:
        H(A)"""
    return -(A * np.log2(A)).sum()


def cross_entropy(A, B):
    """Calculate the entropy for a series of probabilities.

    Arguments:
        A (:class:`pandas.core.series.Series`): A series of data.
        B (:class:`pandas.core.series.Series`): A series of data.

    Return:
        H(A,B)"""
    return -(P(B) * np.log2(P(A))).sum()


def cond_entropy_OA(A, B, subset=None, weights=None, weighting='normal', **kwargs):
    r"""Calculate the conditional entropy between two series of overabundant data points.
    Presupposes that values in the series are of the same type, typically tuples.

    Arguments:
        A (:class:`pandas.core.series.Series`): A series of data.
        B (:class:`pandas.core.series.Series`): A series of data.
        subset (Optional[iterable]): Only give the distribution for a subset of values.
        **kwargs: optional keyword arguments for :func:`matrix_analysis`.

    Return:
        list[float]: A list of metrics. First the global accuracy, then H(A|B).

    Note:
        There are three options for weighting. The following settings are available :
            1. normal: Normalized weighting for overabundant patterns and source cells
            2. frequency: Frequency based weighting for overabundant and source cells
            3. frequency_extended: Consider the frequency of lexemes, both for predicting patterns \
            and averaging entropy/accuracy.

        Note that in cases 2 and 3, forms with a frequency of 0\
        will simply be skipped.
    """

    # A : patterns that can in fact be applied to each form
    # B : patterns that are potentially applicable to each form

    if weights is None and weighting in ['frequency', 'frequency_extended']:
        raise ValueError('Frequency computation required but no weights were provided.')

    def get_weights(A):
        """Provides weights for the source cell.
        Only if frequency_extended was selected.

        Todo:
            Remove this function and use the frequency API
        """
        if weighting == 'frequency_extended':
            def lambfunc(x):
                return weights.loc[(x.name[0], str(x.name[1]).strip(' ')),
                                   'result']

            return A.apply(lambfunc, axis=1)
        else:
            w = A.index.to_frame()
            w.rename_axis(['lex', 'a'], inplace=True)
            c = w.value_counts('lex')
            return w['lexeme'].apply(lambda x: 1/c[x])

    iname = A.index.names[-1]
    A = A.reset_index(level=iname)[subset].set_index(iname, append=True)
    B = B.reset_index(level=iname)[subset].set_index(iname, append=True)
    A['w'] = get_weights(A)

    # We need to align first to add the wordform index to A (patterns)
    colname = B.columns[0]
    grouped_A = A.groupby(B[colname], sort=False)
    population = subset.shape[0]
    results = []

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

        return [i*(np.sum(weight)/population)
                for i in matrix_analysis(matrix, weights=weight,
                                         **kwargs, weighting=weighting)[0:2]]

    results = list(grouped_A.apply(group_analysis))
    sums = np.sum(np.matrix(results), axis=0)

    return sums


def matrix_analysis(matrix, weights=None,
                    phi="soft", beta=1,
                    grad_success=True, cat_pattern=False, weighting=0):
    """Given an overabundance matrix and a function, computes the probability of
    each individual pattern and the accuracy for each lexeme.

    Arguments:
        matrix (:class:`numpy.array`): A matrix of 0 and 1.
        phi (str): One of the following distributions: `norm` (normalized),\
        `soft` (softmax), `uni` (bare uniform).
        beta (float): The value of beta when using `softmax`.
        categorical_success (bool) : whether to consider success as a boolean or as a scalar (between 0-1)

    Return:
        A list of objects: The global accuracy (`float`), the global entropy, H(A|B) (`float`),\
        the probability of each row to be correctly predicted (matrix), \
        the probability of each pattern to be applied (list[float]).

    Todo:
        Add a grid search option ?
    """

    phi_dic = {
        "norm": lambda x: x/np.sum(x, axis=1),
        "soft": lambda x: sp.softmax(x*beta),
        "uni": lambda x: np.matrix([[1/x.shape[1]]*x.shape[1]]),
    }

    # If weights sum to zero, all forms are defective and can be skipped.
    if np.sum(weights) == 0:
        return 0, 0, None, None

    # Compute the frequency of each pattern (using frequency data if available)
    if cat_pattern:
        raise NotImplementedError  # TODO
    else:
        pat_freq = np.average(matrix, axis=0, weights=weights)

    # Compute entropy based on patterns
    entropy = -np.sum(np.where(pat_freq == 0, 0, np.log2(pat_freq))*pat_freq) + 0

    # Apply transformation to pattern probabilities
    phi_pat = phi_dic[phi](pat_freq)

    # Compute probability of success on each row
    if grad_success:
        raise NotImplementedError  # TODO
    else:
        row_proba = matrix@phi_pat.T

    # Compute average probability of success on this subclass
    # There can be some weighting, if available.
    accuracy = np.average(row_proba, weights=weights)

    return accuracy, entropy, row_proba, phi_pat

