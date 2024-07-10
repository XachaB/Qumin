# -*- coding: utf-8 -*-
# !/usr/bin/python3

import numpy as np


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
