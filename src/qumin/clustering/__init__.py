# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import numpy as np


def find_microclasses(paradigms, freqs=None):
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

    # Reformating to wide format required here.
    data = paradigms.copy()
    data['cells'] = list(zip(data.name_a, data.name_b))
    data.drop(['name_a', 'name_b'], axis=1, inplace=True)
    data.set_index(['cells', 'lexeme', 'form_a', 'form_b'], inplace=True)
    data = data.groupby(['lexeme', 'cells']).pattern.apply(lambda x: tuple(sorted(set(x)))).unstack('cells')

    grouped = data.fillna(0).groupby(list(data.columns))
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
