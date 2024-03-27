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


def matrix_analysis(matrix, weights=None,
                    function="norm", beta=1, full=False,
                    grad_success=False, cat_pattern=False):
    """Given an overabundance matrix and a function, computes the probability of
    each individual pattern and the accuracy for each lexeme.

    Arguments:
        matrix (:class:`numpy.array`): A matrix of 0 and 1.
        weights: TODO
        function (str): One of the following distributions: `norm` (normalized),\
        `soft` (softmax), `uni` (bare uniform).
        beta (float): The value of beta when using `softmax`.
        full (bool): whether to return all mesures or only accuracy and entropy. Defaults to False.
        cat_pattern (bool): whether to build pattern frequencies on categorical information or not. Defaults to False.
        grad_success (bool): whether to consider success as a scalar (between 0-1) or not. Defaults to False.

    Return:
        A list of objects: The global accuracy (`float`), the global entropy, H(A|B) (`float`),\
        the probability of each row to be correctly predicted (matrix), \
        the probability of each pattern to be applied (List[float]).

    Todo:
        Add a grid search option ?
    """

    phi_dic = {
        "norm": lambda x: x/np.sum(x),
        "soft": lambda x: sp.softmax(x*beta),
        "uni": lambda x: np.matrix([[1/x.shape[1]]*x.shape[1]]),
    }

    # If weights sum to zero, all forms are defective and can be skipped.
    # If weights is None, then it won't be used by np.average, so nothing to do.
    if np.nansum(weights) == 0:
        if full:
            return 0, 0, None, None, 0
        else:
            return 0, 0
    if cat_pattern or not grad_success:
        bool_matrix = np.array(matrix, dtype=bool)

    # Compute the frequency of each pattern (using frequency data if available)
    if cat_pattern:
        pat_freq = np.average(bool_matrix, axis=0, weights=weights)
        pat_freq = pat_freq/np.sum(pat_freq)
    else:
        pat_freq = np.nansum(matrix*weights.reshape(-1, 1), axis=0)/np.nansum(weights)

    # Apply transformation to pattern probabilities
    if np.sum(pat_freq) == 0:  # We should find a general strategy to handle such cases
        phi_pat = pat_freq
    else:
        phi_pat = phi_dic[function](pat_freq)

    # Compute entropy based on patterns
    with np.errstate(divide='ignore'):
        entropy = -np.sum(np.log2(phi_pat, where=phi_pat != 0)*phi_pat) + 0

    # Compute probability of success on each row
    if grad_success:
        row_accuracy = matrix@phi_pat
    else:
        row_accuracy = bool_matrix@phi_pat

    # Compute average probability of success on this subclass
    # There can be some weighting, if available.
    accuracy = np.nansum(row_accuracy*weights)/np.nansum(weights)
    if full:
        return accuracy, entropy, row_accuracy, phi_pat, pat_freq
    else:
        return accuracy, entropy
