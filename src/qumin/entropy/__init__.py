# -*- coding: utf-8 -*-
# !/usr/bin/python3

import numpy as np
from scipy import special as sp


def P(x, subset=None):
    """Return the probability distribution of elements in a :class:`pandas.core.series.Series`.

    Arguments:
        x (:class:`pandas.core.series.Series`): A series of data.
        subset (Iterable): Only give the distribution for a subset of values.

    Returns:
        :class:`pandas.core.series.Series`: A Series which index are x's elements
            and which values are their probability in x.
    """
    if subset is None:
        return x.value_counts(normalize=True, sort=False)
    else:
        return x[subset].value_counts(normalize=True, sort=False)


def cond_P(A, B, subset=None):
    """
    Return the conditional probability distribution P(A|B) for elements
    in two :class:`pandas.core.series.Series`.

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


def extra_cond_metrics(group, mapping="norm", debug=False, cat_success=True, beta=5):
    """
    Computes entropy and probability of success for overabundant distributions,
    with some extra settings and options.

    Arguments:
        group (pandas.DataFrame): The class for which entropy and
            probability of success should be computed.
        mapping (str): Which mapping to use between the pattern frequencies
            and the pattern probabilities. Available options are:
            soft (for softmax), norm (for normalized), uni (for uniform).
        debug (bool): Whether to return results for debug or for normal computation.
        cat_success (bool): Whether to consider a success
            as a categorical or gradiant feature.
        beta (float): Value of beta for softmax computations (if mapping=soft)
    """
    # Here we can apply various functions to map weights to probabilities (softmax, etc).
    func = {
        "norm": lambda x: x / x.sum(),          # Normalize weights.
        "uni": lambda x: 1 / x.shape[0],        # Use a bare Uniform distribution.
        "soft": lambda x: sp.softmax(x*beta)    # Use a softmax function
            }[mapping]

    # We compute a weight for each pair¯
    x_sum = group[~group.duplicated(['form_x', 'lexeme'])].w_x.sum()
    y_sum = group[~group.duplicated(['form_y', 'lexeme'])].w_y.sum()
    group['w'] = ((group.w_x / x_sum) * (group.w_y / y_sum))
    group.w = group.w / group.w.sum()

    # We apply this normalizing function to the overall frequency of the pattern.
    pat_freq = group.groupby('pattern').w.sum()
    pat_proba = func(pat_freq)

    # Different ways of measuring success.
    if cat_success:
        fact = (group.w_y != 0)
    else:
        raise NotImplementedError
        # TODO check this setting
        fact = group.w

    # Returning results.
    results = [0 + entropy(pat_proba),  # Entropy "0+" to ensure positive values.
               (group.pattern.map(pat_proba)
                * fact * group.w_x / x_sum).sum(),  # P(success)
               x_sum,  # Class size
               ]
    if debug:
        results.extend([pat_freq, pat_proba])
    return results


def entropy(A):
    """Calculate the entropy for a series of probabilities.

    Arguments:
        A (:class:`pandas.core.series.Series`): A series of data.

    Return:
        H(A)"""
    return -(A * np.log2(A)).sum()
