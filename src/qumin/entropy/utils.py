# -*- coding: utf-8 -*-
# !/usr/bin/python3

import numpy as np
from scipy import special as sp
# TODO replace sklearn with some lighter solution
from sklearn.feature_extraction.text import CountVectorizer


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


def cond_entropy_OA(A, B, subset=None, **kwargs):
    r"""Calculate the conditional entropy between two series of overabundant data points.
    Presupposes that values in the series are of the same type, typically tuples.

    Arguments:
        A (:class:`pandas.core.series.Series`): A series of data.
        B (:class:`pandas.core.series.Series`): A series of data.
        subset (Optional[iterable]): Only give the distribution for a subset of values.
        **kwargs: optional keyword arguments for :func:`matrix_analysis`.

    Return:
        list[float]: A list of metrics. First the global accuracy, then H(A|B).
    """

    # On regroupe les données de A selon celles de B
    grouped_A = A[subset].groupby(B[subset], sort=False)
    population = subset.shape[0]

    # On analyse chaque groupe
    results = []
    for name, group in grouped_A:
        cv = CountVectorizer(tokenizer=lambda x: x,
                            lowercase=False, token_pattern=None)
        m = cv.fit_transform([_[0].split(";") for _ in group])
        results.append([i*(group.shape[0]/population)
                        for i in matrix_analysis(m.todense(), **kwargs)[0:2]])
    sums = np.sum(np.matrix(results), axis=0)

    return sums


def matrix_analysis(matrix, phi="soft", beta=1, with_proba=False, verbose=False):
    """Given an overabundance matrix and a function, computes the probability of
    each individual pattern and the accuracy for each lexeme.

    Arguments:
        matrix (:class:`numpy.matrix`): A matrix of 0 and 1.
        phi (str): One of the following distributions: `norm` (normalized), `soft` (softmax), `uni` (bare uniform).
        beta (float): The value of beta when using `softmax`.
        with_proba (bool): whether to return an array of results.
        verbose (bool): whether to display the results or not.

    Return:
        A list of objects: The global accuracy (`float`), the global entropy, H(A|B) (`float`), the probability of each row to be correctly predicted (matrix), the probability of each pattern to be applied (list[float]).

    Todo:
        Separate the grid search from the normal behavior.
    """
    def _apply_phi(phi_fn):
        phi_pat = phi_fn(pat_freq)
        row_proba = matrix@phi_pat.T
        accuracy = np.mean(row_proba)
        # +0 to avoid displaying results as "-0"
        entropy = -np.sum(np.log2(row_proba)*1/row_proba.shape[0]) + 0
        if verbose:
            print("Phi_pat:", phi_pat)
            print("Pat_freq:", pat_freq)
            print("Overall probability of success :", accuracy)
            print("Custom entropy :", entropy)
        return accuracy, entropy, row_proba, phi_pat

    phi_dic = {
        "norm": lambda x: x/np.sum(x, axis=1),
        "soft": lambda x: sp.softmax(x*beta),
        "uni": lambda x: np.matrix([[1/x.shape[1]]*x.shape[1]]),
    }

    pat_freq = np.matrix(np.mean(matrix, axis=0))  # Frequency of success of each pattern
    return _apply_phi(phi_dic[phi])
