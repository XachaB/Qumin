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
        weights (): TODO
        weighting (str): which kind of approach should be used for weighting : normal, \
        frequency, frequency_extended.
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

    grouped_A = A.groupby(B, sort=False)
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
                                         **kwargs)[0:2]]

    results = list(grouped_A.apply(group_analysis))
    return np.nansum(results, axis=0)


def matrix_analysis(matrix, weights=None,
                    phi="soft", beta=1,
                    grad_success=False, cat_pattern=False):
    """Given an overabundance matrix and a function, computes the probability of
    each individual pattern and the accuracy for each lexeme.

    Arguments:
        matrix (:class:`numpy.array`): A matrix of 0 and 1.
        phi (str): One of the following distributions: `norm` (normalized),\
        `soft` (softmax), `uni` (bare uniform).
        beta (float): The value of beta when using `softmax`.
        cat_pattern (bool): whether to build pattern frequencies on categorical information or not. Defaults to False.
        grad_success (bool): whether to consider success as a scalar (between 0-1) or not. Defaults to False.

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
    # If weights is None, then it won't be used by np.average, so nothing to do.
    if np.sum(weights) == 0:
        return 0, 0, None, None

    if cat_pattern or not grad_success:
        bool_matrix = np.array(matrix, dtype=bool)

    # Compute the frequency of each pattern (using frequency data if available)
    if cat_pattern:
        pat_freq = np.average(bool_matrix, axis=0, weights=weights)
        pat_freq = pat_freq/np.sum(pat_freq)
    else:
        pat_freq = np.average(matrix, axis=0, weights=weights)

    # Compute entropy based on patterns
    with np.errstate(divide='ignore'):
        entropy = -np.sum(np.log2(pat_freq, where=pat_freq != 0)*pat_freq) + 0

    # Apply transformation to pattern probabilities
    phi_pat = phi_dic[phi](pat_freq)

    # Compute probability of success on each row
    if grad_success:
        row_accuracy = matrix@phi_pat.T
    else:
        row_accuracy = bool_matrix@phi_pat.T

    # Compute average probability of success on this subclass
    # There can be some weighting, if available.
    accuracy = np.average(row_accuracy, weights=weights)

    return accuracy, entropy, row_accuracy, phi_pat
