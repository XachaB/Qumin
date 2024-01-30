# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Sacha Beniamine.

This module is used to align sequences.
"""
from itertools import zip_longest
import numpy as np


def commonprefix(*args):
    """Given a list of strings, returns the longest common prefix"""
    if not args:
        return None
    shortest, *args = sorted(args, key=len)
    zipped = zip_longest(*args)
    for i, c in enumerate(shortest):
        zipped_i = next(zipped)
        if any([s != c for s in zipped_i]):
            return shortest[:i]
    return shortest


def commonsuffix(*args):
    """Given a list of strings, returns the longest common suffix"""
    return commonprefix(*(x[::-1] for x in args))[::-1]


def _all_min(iterable):
    """Return all minimum elements from an iterable according to the first element of its tuples."""
    minimums = []
    minkeyval = iterable[-1][0]
    for x in iterable:
        keyval = x[0]
        if keyval < minkeyval:
            minimums = [x]
            minkeyval = keyval
        elif keyval == minkeyval:
            minimums.append(x)
    return minimums


def levenshtein_ins_cost(*_):
    return 1


def levenshtein_sub_cost(a, b):
    return int(a != b)


def multi_sub_cost(a, b):
    return int(b not in a)


def align_multi(*strings, **kwargs):
    """ Levenshtein-style alignment over arguments, two by two."""
    if len(strings) == 1:
        return [(elem,) for elem in strings[0]]

    kwargs["fillvalue"] = kwargs.get("fillvalue", "")
    kwargs["insert_cost"] = levenshtein_ins_cost
    kwargs["sub_cost"] = multi_sub_cost

    def flatten_alignment(alignment):
        for a, b in alignment:
            try:
                yield a | {b}
            except TypeError:  # a is the fillvalue
                yield {a, b}

    first_seq = [{s} for s in strings[0]]
    aligned = first_seq

    for i in range(1, len(strings)):
        aligned = list(flatten_alignment(align_auto(aligned, strings[i], **kwargs)[0]))
    return aligned


def align_auto(s1, s2, insert_cost, sub_cost, distance_only=False, fillvalue="", **kwargs):
    """Return all the best alignments of two words according to some edit distance matrix.

    Arguments:
        s1 (str): first word to align
        s2 (str): second word to align
        insert_cost (func): A function which takes one value and returns an insertion cost
        sub_cost (func): A function which takes two values and returns a substitution cost
        distance_only (bool): defaults to False. If True, returns only the best distance. If False, returns an alignment.
        fillvalue: (optional) the value with which to pad when iterable have varying lengths. Default:  "".

    Returns:
        Either an alignment (a `list` of `list` of zipped tuples), or a distance (if `distance_only` is True).
    """
    m = len(s1)
    n = len(s2)
    paths = np.empty((m + 1, n + 1), dtype=list)
    paths[0, 0] = [(0, (0, 0), ("", ""))]

    for i, a in enumerate(s1):
        paths[i + 1, 0] = [(paths[i, 0][0][0] + insert_cost(a), (i, 0), (a, fillvalue))]

    for j, b in enumerate(s2):
        paths[0, j + 1] = [(paths[0, j][0][0] + insert_cost(b), (0, j), (fillvalue, b))]

    for i in range(1, m + 1):
        a = s1[i - 1]

        for j in range(1, n + 1):
            b = s2[j - 1]
            subcost = paths[i - 1, j - 1][0][0] + sub_cost(a, b)
            insb = paths[i, j - 1][0][0] + insert_cost(b)
            insa = paths[i - 1, j][0][0] + insert_cost(a)

            paths[i, j] = _all_min([(subcost, (i - 1, j - 1), (a, b)),
                                    (insb, (i, j - 1), (fillvalue, b)),
                                    (insa, (i - 1, j), (a, fillvalue))])
    if distance_only:
        return paths[-1][-1][0][0]
    else:
        return _multibacktrack(paths)


def _multibacktrack(paths):
    max_i, max_j = paths.shape
    stack = [([],  # empty alignment
              (max_i - 1, max_j - 1),  # start at last cell
              set())  # No cell is forbidden.
             ]
    solutions = []
    while stack:
        current_path, (i, j), visited = stack.pop(0)
        if (i, j) in visited:
            continue  # abandon this path, it is redundant
        else:
            visited.add((i, j))
            if i == 0 and j == 0:
                solutions.append(current_path)
            else:
                # ins_del = {(i - 1, j), (i, j - 1)}
                for step in paths[i, j]:
                    _, idxs, action = step
                    # Share the same visited set unless perfect match
                    new_visited = visited if action[0] != action[1] else set()
                    stack.append(([action] + current_path,
                                  idxs,
                                  new_visited))
    return solutions


def align_left(*args, **kwargs):
    """Align left all arguments (wrapper around zip_longest).

    Examples:
        >>> align_left("mɪs","mas")
        [('m', 'm'), ('ɪ', 'a'), ('s', 's')]
        >>> align_left("mɪs","mɪst")
        [('m', 'm'), ('ɪ', 'ɪ'), ('s', 's'), ('', 't')]
        >>> align_left("mɪs","amɪs")
        [('m', 'a'), ('ɪ', 'm'), ('s', 'ɪ'), ('', 's')]
        >>> align_left("mɪst","amɪs")
        [('m', 'a'), ('ɪ', 'm'), ('s', 'ɪ'), ('t', 's')]


    Arguments:
        *args: any number of iterables >= 2
        fillvalue: the value with which to pad when iterable have varying lengths. Default:  "".

    Returns:
        a `list` of zipped tuples, left aligned.
    """
    if "fillvalue" not in kwargs:
        kwargs["fillvalue"] = ""
    return list(zip_longest(*args, **kwargs))


def align_right(*iterables, **kwargs):
    """Align right all arguments. Zip longest with right alignment.

    Examples:
        >>> align_right("mɪs","mas")
        [('m', 'm'), ('ɪ', 'a'), ('s', 's')]
        >>> align_right("mɪs","mɪst")
        [('', 'm'), ('m', 'ɪ'), ('ɪ', 's'), ('s', 't')]
        >>> align_right("mɪs","amɪs")
        [('', 'a'), ('m', 'm'), ('ɪ', 'ɪ'), ('s', 's')]
        >>> align_right("mɪst","amɪs")
        [('m', 'a'), ('ɪ', 'm'), ('s', 'ɪ'), ('t', 's')]


    Arguments:
        *iterables: any number of iterables >= 2
        fillvalue: the value with which to pad when iterable have varying lengths. Default:  "".

    Returns:
        a `list` of zipped tuples, right aligned.
    """
    if "fillvalue" not in kwargs:
        kwargs["fillvalue"] = ""
    reverse = list(zip_longest(*[x[::-1] for x in iterables], **kwargs))
    return reverse[::-1]


def align_baseline(*args, **kwargs):
    """ Simple alignment intended as an inflectional baseline. (Albright & Hayes 2002)

    single change, either suffixal, or suffixal, or infixal.
    This doesn't work well when there is both a prefix and a suffix.
    Used as a baseline for evaluation of the auto-aligned patterns.

    see "Modeling English Past Tense Intuitions with Minimal Generalization", Albright, A. & Hayes, B.
    *Proceedings of the ACL-02 Workshop on Morphological and Phonological Learning* - Volume 6,
    Association for Computational Linguistics, 2002, 58-69, page 2 :

        "The exact procedure for finding a word-specific
        rule is as follows: given an input pair (X, Y), the
        model first finds the maximal left-side substring
        shared by the two forms (e.g., #mɪs), to create the
        C term (left side context). The model then exam-
        ines the remaining material and finds the maximal
        substring shared on the right side, to create the D
        term (right side context). The remaining material is
        the change; the non-shared string from the first
        form is the A term, and from the second form is the
        B term."

    Examples:
        >>> align_baseline("mɪs","mas")
        [('m', 'm'), ('ɪ', 'a'), ('s', 's')]
        >>> align_baseline("mɪs","mɪst")
        [('m', 'm'), ('ɪ', 'ɪ'), ('s', 's'), ('', 't')]
        >>> align_baseline("mɪs","amɪs")
        [('', 'a'), ('m', 'm'), ('ɪ', 'ɪ'), ('s', 's')]
        >>> align_baseline("mɪst","amɪs")
        [('m', 'a'), ('ɪ', 'm'), ('s', 'ɪ'), ('t', 's')]


    Arguments:
        *args: any number of iterables >= 2
        fillvalue: the value with which to pad when iterable have varying lengths. Default:  "".

    Returns:
        a `list` of zipped tuples.
    """

    fillvalue = kwargs.get("fillvalue", "")
    la = len(args)

    C = commonprefix(*args)
    rest = [x[len(C):] for x in args]
    D = commonsuffix(*rest)

    if C:
        C = list(zip(*[C for _ in range(la)]))
    else:
        C = []
    if D:
        AB = [x[:-len(D)] for x in rest]
        D = list(zip(*[D for _ in range(la)]))
    else:
        AB = rest
        D = []

    AB = list(zip_longest(*AB, fillvalue=fillvalue))

    return C + AB + D
