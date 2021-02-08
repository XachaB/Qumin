# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Sacha Beniamine.

This module is used to generalise pats contexts.
"""

from copy import deepcopy
from collections import Counter
from representations.contexts import Context
import logging
log = logging.getLogger()



def generalize_patterns(pats):
    """Generalize these patterns' context.

    Arguments:
        patterns: an iterable of :class:`Patterns.Pattern`

    Return:
        a new :class:`Patterns.Pattern`.
    """
    p0 = pats[0]
    if len(pats) == 1:
        return p0

    log.debug("generalizing: %s", pats)
    new = deepcopy(p0)

    # Generalize the alternation if possible
    if new._gen_alt is not None:
            new._generalize_alt(*pats)

    # Generalize the context if possible
    if not new._is_max_gen():
        new.context = Context.merge([p.context for p in pats])
        new._create_regex()
        new._repr = new._make_str_(features=False)
        new._feat_str = new._make_str_(features=True)

    log.debug("Result: %s", new)

    new.lexemes = set().union(*(p.lexemes for p in pats))
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
    args = sorted(args, key=lambda x: counts[x.to_alt(exhaustive_blanks=True)], reverse=True)
    merged = [args[0]]
    for pat in args[1:]:
        pat_is_merged = False
        for i in range(len(merged)):
            lexemes = merged[i].lexemes
            if not (lexemes.issubset(pat.lexemes) or lexemes.issuperset(pat.lexemes)):
                new = generalize_patterns([merged[i], pat])
                if all(correct(new, a, b) for l, a, b in new.lexemes):
                    merged[i] = new
                    pat_is_merged = True
                    break
        if not pat_is_merged:
            merged.append(pat)

    return merged
