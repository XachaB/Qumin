# -*- coding: utf-8 -*-
# !/usr/bin/python3

import numpy as np


def P(x, weights=None, subset=None):
    """Return the probability distribution of elements in a :class:`pandas.core.series.Series`.

    Arguments:
        x (:class:`pandas.core.series.Series`): A series of data.
        subset (Iterable): Only give the distribution for a subset of values.

    Returns:
        :class:`pandas.core.series.Series`: A Series which index are x's elements and which values are their probability in x.
    """
    if weights is not None:
        return weights.groupby(x).sum() / weights.sum()
    if subset is None:
        return x.value_counts(normalize=True, sort=False)
    else:
        return x[subset].value_counts(normalize=True, sort=False)


def cond_P(A, B, subset=None):
    """Return the conditional probability distribution P(A|B) for elements in two :class:`pandas.core.series.Series`.

    Arguments:
        A (:class:`pandas.core.series.Series`): A series of data.
        B (:class:`pandas.core.series.Series`): A series of data.
        subset (Iterable): Only give the distribution for a subset of values.

    Return:
        :class:`pandas.core.series.Series`: A Series whith two indexes.
        The first index is from the elements of B, the second from the elements of A.
        The values are the P(A|B).
    """
    if subset is None:
        cond_events = A.groupby(B, sort=False)
    else:
        cond_events = A[subset].groupby(B[subset], sort=False)
    return P(cond_events)


def cond_entropy(A, B, **kwargs):
    """Calculate the conditional entropy of A knowing B, two series of data points.
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


def cond_entropy_slow(df, subset=None):
    """
    Calculate the conditional entropy through a slower method (with iterations across all groups).

    Uses token frequencies to weight the patterns.
    """
    def compute_group_ent(group):
        return entropy(P(group.pattern, weights=group.f_pair)) * group.f_pred.sum()

    return 0 + df.groupby('applicable').apply(compute_group_ent).sum() / df.f_pred.sum()


def cond_psuccess(df):
    """
    Calculate the conditional probability of success.

    Uses token frequencies to weight the patterns.
    """
    def compute_group_psuccess(group):
        proba = P(group.pattern, weights=group.f_pair)
        return group.pattern.map(proba)

    return 0 + (df.groupby("applicable", group_keys=False).apply(compute_group_psuccess)
                * df.f_pred / df.f_pred.sum()).sum()
