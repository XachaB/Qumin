# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import numpy as np
import pandas as pd


def find_microclasses(paradigms, patterns, freqs=None):
    """Find microclasses in a paradigm (lines with identical rows).

    This is useful to identify an exemplar of each inflection microclass,
    and limit further computation to the collection of these exemplars.

    Arguments:
        paradigms (pandas.DataFrame):
            a dataframe containing inflectional paradigms.
            rows describe a pattern between forms from a given lexeme for a given cell.
        freqs (pandas.Series): a series of frequencies for each lemma

    Return:
        microclasses (dict).
            classes is a dict. Its keys are exemplars,
            its values are lists of the name of rows identical to the exemplar.
            Each exemplar represents a macroclass. ::

            {"a":["a","A","aa"], "b":["b","B","BBB"]}

    """
    lexemes = pd.Series(index=paradigms.data.lexeme.unique())
    grouped = lexemes.groupby([df.groupby('lexeme', observed=False).pattern.apply(
        lambda x: tuple(sorted([str(p) for p in x if p is not None])))
                               for df in patterns.values()])
    mc = {}

    for name, group in grouped:
        members = list(group.index)
        if freqs is not None:
            freq_subset = freqs[group.index]
            exemplar = freq_subset.index[freq_subset.argmax()]
        else:
            exemplar = min(members, key=lambda string: len(string))
        mc[exemplar] = members

    return mc


def find_min_attribute(tree, attr):
    """Find the minimum value for an attribute in a tree.

    Arguments:
        tree (node.Node): The tree in which to find the minimum attribute.
        attr (str): the attribute's key."""
    agenda = [tree]
    mini = np.inf
    while agenda:
        node = agenda.pop(0)
        if node.children:
            agenda.extend(node.children)
        if attr in node.attributes and float(node.attributes[attr]) < mini:
            mini = node.attributes[attr]

    return mini
