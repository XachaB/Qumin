# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Sacha Beniamine.

This module is used to generalise patterns contexts.
"""

from copy import deepcopy
from collections import Counter
from representations.segments import Segment
from representations.contexts import Context

def generalize_patterns(patterns, debug=False):
    """Generalize these patterns' context.

    Arguments:
        patterns: an iterable of :class:`Patterns.Pattern`

    Return:
        a new :class:`Patterns.Pattern`.
    """
    if len(patterns) == 1:
        return patterns[0]

    if debug:
        print("generalizing:",patterns)
    new = deepcopy(patterns[0])
    alternations = set(x.to_alt(exhaustive_blanks=False) for x in patterns)
    contexts = [p.context for p in patterns]
    if len(alternations) > 1:
        # Generalize the alternation if there is a generalized formulation
        # And there are different surface alternations
        if patterns[0]._gen_alt != None:
            new._generalize_alt()

    if not new._is_max_gen():
        new.context = Context.merge(contexts, debug=debug)
        new._create_regex()
        new._repr = new._make_str_(features=False)
        new._feat_str = new._make_str_(features=True)
        if debug:
            print("New:",new)

    new.lexemes = set().union(*(p.lexemes for p in patterns))
    return new


def incremental_generalize_patterns(*args):
    """Merge patterns incrementally as long as the pattern has the same coverage.

    Attempt to merge each patterns two by two, and refrain from doing so if the pattern doesn't match all the lexemes that lead to its inference.
    Also attempt to merge together patterns that have not been merged with others.

    Arguments:
        *args : the patterns

    Returns:
        a list of patterns, at best of length 1, at worst of the same length as the input.
    """

    if len(args) == 1:
        return args

    def correct(p, a, b):
        """Return whether the pattern p is correct for the forms a and b and the specified cells."""
        return p.applicable(a, p.cells[0]) and \
               p.applicable(b, p.cells[1]) and \
               p.apply(a, p.cells) == b and \
               p.apply(b, p.cells[::-1]) == a

    exact_alternations = [x.to_alt(exhaustive_blanks=True) for x in args]
    counts = Counter(exact_alternations)
    args = sorted(args,key=lambda x:counts[x.to_alt(exhaustive_blanks=True)],reverse=True)
    merged = [args[0]]
    # print("INCREMENTAL GEN: starting with",args[0])
    for pat in args[1:]:
        # print("\tAdding ...",pat)
        pat_is_merged = False
        for i in range(len(merged)):
            lexemes = merged[i].lexemes
            if not (lexemes.issubset(pat.lexemes) or lexemes.issuperset(pat.lexemes)):
                # print("\t\t+",merged[i],"?")
                new = generalize_patterns([merged[i],pat],debug=False)
                if all(correct(new, a, b) for l, a, b in new.lexemes):
                    merged[i] = new
                    pat_is_merged = True
                    # print("\t\tresult",new)
                    break
            # else:
            #     print("Couldn't merge: ",repr(merged[i]))
            #     print("with:",repr(pat))
            #     print("Not inferring wrong pattern ",repr(new))
            #     print([(a,b) for l, a, b in new.lexemes if not correct(new, a, b)])
        if not pat_is_merged:
            merged.append(pat)

    return merged
