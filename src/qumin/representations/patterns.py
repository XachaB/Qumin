# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""author: Sacha Beniamine.

This module addresses the modeling of inflectional alternation patterns."""

import logging
import re
from collections import defaultdict
from copy import deepcopy
from itertools import groupby, zip_longest, combinations, product
from math import comb
# External tools
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import alignment
from .contexts import Context
from .generalize import generalize_patterns, incremental_generalize_patterns
from .quantity import one, optional, some, kleenestar
from .segments import Inventory, Form, _regex_or
# Our modules
from ..utils import memory_check

log = logging.getLogger("Qumin")

len = len
sorted = sorted
all = all


def _compatible_context_type(*args):
    """Returns whether several contexts with these types can be merged without verification."""
    compat = []
    greedyany = {"greedy", "any"}
    for positions in zip(*args):
        for types in zip(*positions):
            if "precise" in types or "lazy" in types:
                return False
            settypes = set(types)
            if len(settypes) == 1 or settypes == greedyany:
                compat.append(True)
            else:
                compat.append(False)
    return all(compat)


def _get_pattern_matchtype(p, c1, c2):
    """Determine a type of match for an elementary pattern"""
    type_names = {(True, True): "precise",
                  (True, False): "greedy",
                  (False, True): "lazy",
                  (False, False): "any"}
    match_type1 = []
    match_type2 = []
    for i, group in enumerate(p.context):
        match_type1.append([False, False])
        match_type2.append([False, False])
        # print("i,group:",i,group)
        if group.blank:
            ctxt = repr(group)
            # print("left context:",ctxt)
            alt1 = "".join(str(s) for s in p.alternation[c1][i])
            alt2 = "".join(str(s) for s in p.alternation[c2][i])
            match_type1[i][0] = alt1 != "" and alt1 in ctxt
            # print("is",alt1,"in the left context ?",match_type1[i][0])
            match_type2[i][0] = alt2 != "" and alt2 in ctxt
            # print("is",alt2,"in the left context ?",match_type1[i][0])

            if i + 1 < len(p.context):
                ctxt2 = repr(p.context[i + 1])
                # print("right context:",ctxt2)
                match_type1[i][1] = alt1 != "" and alt1 in ctxt2
                # print("is",alt1,"in the right context",ctxt2," ?",match_type1[i][1])
                match_type2[i][1] = alt2 != "" and alt2 in ctxt2
                # print("is",alt2,"in the right context",ctxt2,"?",match_type1[i][1])

    return tuple(type_names[tuple(x)] for x in match_type1), tuple(type_names[tuple(x)] for x in match_type2)


def _replace_alternation(matchgroups, replacements):
    """ Replace all matches in matching groups using replacements.

    Args:
        matches (iterable of str): an iterable of input sequences which match the rule (should cover the entire form)
        replacements (iterable of str|None|tuple): an iterable of replacements.
            Replacements can be:
                - A tuple symbolizing a bijective phonological function
                - None if no replacement is to be made (copy matched characters)
                - characters by which to replace the match
    Returns:
        a space separated string

    Examples:
        In this example,
        - the first match, "t a " is copied as is,
        - the second match, "t " is transformed by consonant voicing
        - the third match, "a " is replaced by "i"

        >>> Inventory.initialize("tests/data/frenchipa.csv")
        >>> matches = ("t a ",  "t ",                   "a ")
        >>> repl    = (None,    (set("ptk"),set("bdg")), "i")
        >>> _replace_alternation(matches, repl)
        't a d i '
    """

    def iter_replacements():
        for chars, repl in zip(matchgroups, replacements):
            chars = chars.strip()
            t = type(repl)
            if repl is None:  # no change
                yield chars
            elif t is str:  # change by substitution
                yield repl
            elif t is tuple:  # change by phonological func
                yield Inventory.get_from_transform(chars, repl)

    return " ".join(iter_replacements()) + " "


def are_all_identical(iterable):
    """Test whether all elements in the iterable are identical."""
    return iterable and len(set(iterable)) == 1


def _iter_alternation(alt):
    """ Group alternations into sequences of segments or phonological transfomations.

    An alternation part is a sequence of strings or frozenset. Each string represents either:
    - A segment
    - A frozenset representing a class of segment (which forms part of a phonological transformation)

    This iterates by grouping contiguous segments together, and classes of segments separately.

    Example:
        >>> Inventory.initialize("tests/data/frenchipa.csv")
        >>> alt_members = _iter_alternation(['a', 'b', 'a', frozenset(('e', 'u'))])
        >>> list(alt_members) == [(True, ['a', 'b', 'a']), (False, frozenset({'e', 'u'}))]
        True

    Args:
        alt (iterable of str or frozenset): An alternation part.

    Yields:
        Iterator of pairs of is_segment, then either a sequence of segments or a frozenset.

    """
    for is_segment, group in groupby(alt, lambda x: Inventory.is_leaf(x)):
        if is_segment:
            yield is_segment, list(group)
        else:
            for x in group:
                yield is_segment, x


class NotApplicable(Exception):
    """Raised when a :class:`Pattern` can't be applied to a form."""
    pass


class Pattern(object):
    r"""Represent the alternation pattern between two forms.

    Applying the pattern to one of the original forms yields the second one.

    As an example, we will use the following alternation
    in a present verb of french:

    ========================== ========================== ==========================
    cells                      Forms                      Transcription
    ========================== ========================== ==========================
    prs.1.sg ⇌ prs.2.pl        j'amène ⇌ vous amenez      amEn ⇌ amənE
    ========================== ========================== ==========================

    Example:
        >>> Inventory.initialize("tests/data/frenchipa.csv")
        >>> cells = ("prs.1.sg", "prs.2.pl")
        >>> forms = (Form("amEn"), Form("amØnE"))
        >>> p = Pattern(cells, forms, aligned=False)
        >>> type(p)
        <class 'qumin.representations.patterns.Pattern'>
        >>> p
        E_ ⇌ Ø_E / am_n_ <0>
        >>> p.apply(Form("amEn"), cells)
        Form(amØnE)
    """

    def __lt__(self, other):
        """Sort on lexicographic order.

        There is no reason to sort patterns,
        but Pandas wants to do it from time to time,
        this is only implemented to avoid Pandas complaining.

        Example:
            >>> Inventory.initialize("tests/data/frenchipa.csv")
            >>> cells = ("prs.1.sg", "prs.2.pl")
            >>> forms = (Form("amEn"), Form("amənE"))
            >>> forms2 = (Form("bEn"), Form("bənE"))
            >>> p1 = Pattern(cells, forms, aligned=False)
            >>> p2 = Pattern(cells, forms2, aligned=False)
            >>> p1 < p2
            True
        """
        return str(self) < str(other)

    def __init__(self, cells, forms, aligned=False):
        """Constructor for Pattern.

        Arguments:
            cells (Iterable): Cells labels (str), in the same order.
            forms (Iterable): Forms (str) to be segmented.
            aligned (bool): whether forms are already aligned. Otherwise, left alignment will be performed.

        Example:
            >>> Inventory.initialize("tests/data/frenchipa.csv")
            >>> cells = ("prs.1.sg", "prs.2.pl")
            >>> forms = (Form("amEn"), Form("amənE"))
            >>> p = Pattern(cells, forms, aligned=False)
            >>> p
            E_ ⇌ Ø_E / am_n_ <0>
            >>> p.score # is zero at initialization
            0
            >>> p.lexemes # is empty at initialization
            set()
            >>> p.alternation
            {'prs.1.sg': [('E',), ('',)], 'prs.2.pl': [('Ø',), ('E',)]}
            >>> p.context # this is a Context
            ((?:a )(?:m )){}((?:n )){}
            >>> p.cells
            ('prs.1.sg', 'prs.2.pl')
            >>> p._repr
            'E_ ⇌ Ø_E / am_n_'
            >>> p._feat_str
            'E_ ⇌ Ø_E / am_n_'
            >>> p._gen_alt == {'prs.1.sg': ((frozenset({'ɑ̃', 'ɛ̃', 'i', 'j', 'E'}),), ('',)),
            ...                'prs.2.pl': ((frozenset({'ɥ', 'ɔ̃', 'y', 'Ø', 'œ̃'}),), ('E',))}
            True
        """
        self.score = 0
        self.lexemes = set()
        self.alternation = {}
        self.context = ()
        self.cells = cells
        if aligned:
            alignment_of_forms = list(forms)
        else:
            alignment_of_forms = list(alignment.align_left(*[f.tokens for f in forms], fillvalue=""))

        self._init_from_alignment(alignment_of_forms)
        self._repr = self._make_str_(features=False)
        self._feat_str = self._make_str_(features=True)
        self._find_generalized_alt()

    def __deepcopy__(self, memo):
        """ Deep copy of this pattern."""
        cls = self.__class__
        copy = cls.__new__(cls, self.cells)
        copy.context = deepcopy(self.context)
        copy.alternation = deepcopy(self.alternation)
        copy.cells = self.cells
        copy.score = self.score
        copy._repr = self._repr
        copy._feat_str = self._feat_str
        copy._gen_alt = deepcopy(self._gen_alt)
        return copy

    @classmethod
    def new_identity(cls, cells):
        """ Identity pattern factory.

        The alternation is empty, and the context is a sequence of any number of allowed segments.

        Args:
            cells: Pair of cell for this pattern.

        Returns:
            Pattern: a new identity pattern.

        Example:

            >>> Inventory.initialize("tests/data/frenchipa.csv")
            >>> print(Pattern.new_identity(('A','B')))
             ⇌  / X*
        """
        p = cls(cells, ("", ""), aligned=True)
        p.context = Context([(Inventory._max, kleenestar)])
        p._repr = p._make_str_(features=False)
        p._feat_str = p._make_str_(features=True)
        return p

    @classmethod
    def _from_str(cls, cells, string):
        """ Parse an exported pattern.

        To be parsed back, patterns need to be exported by `repr()`, not `str()`.

        Note: Phonemes in context classes are now separated by ","

        Args:
            cells (tuple of str): Cells labels (str).
            string (str): pattern given as a string.

        Returns:
            Pattern: a parsed Pattern object.

        Example:
            >>> Inventory.initialize("tests/data/frenchipa.csv")
            >>> p = Pattern._from_str(('A', 'B'), "ɥ ⇌ yj / {E,O,a,b,d,f,g,i,j,k,l,m,n,p,s,t,u,v,w,y,z,Ø,ŋ,œ̃,ɑ̃,ɔ̃,ɛ̃,ɥ,ɲ,ʁ,ʃ,ʒ}*{b,d,f,g,k,l,m,n,p,s,t,v,z,ŋ,ɲ,ʁ,ʃ,ʒ}_E <58>")
            >>> type(p) is Pattern
            True
            >>> str(p)
            'ɥ ⇌ yj / X*C_E'
            >>> p
            ɥ ⇌ yj / {E,O,a,b,d,f,g,i,j,k,l,m,n,p,s,t,u,v,w,y,z,Ø,ŋ,œ̃,ɑ̃,ɔ̃,ɛ̃,ɥ,ɲ,ʁ,ʃ,ʒ}*{b,d,f,g,k,l,m,n,p,s,t,v,z,ŋ,ɲ,ʁ,ʃ,ʒ}_E <58.0>
            >>> p = Pattern._from_str(('A','B'), "E_ ⇌ Ø_E / am_n_ <0>")
            >>> type(p) is Pattern
            True
            >>> p
            E_ ⇌ Ø_E / am_n_ <0.0>
        """
        quantities = {"": one, "?": optional, "+": some, "*": kleenestar}

        simple_segs = sorted((s for s in Inventory._classes if Inventory.is_leaf(s)),
                             key=len, reverse=True)

        seg = r"(?:{})".format("|".join(simple_segs))
        classes = r"(?:\{[^\}]+\})"

        def is_class(s):
            return s is not None and ("," in s) and (s[0], s[-1]) == ("{", "}")

        def get_class(s):
            return frozenset(s[1:-1].split(","))

        def parse_alternation(string, cells):
            regex = r"({classes}|{seg})".format(seg=seg, classes=classes)
            left, right = string.split(" ⇌ ")
            c1, c2 = cells
            alternation = {c1: [], c2: []}

            for segs_l, segs_r in zip_longest(left.split("_"),
                                              right.split("_")):
                segs_l = re.findall(regex, segs_l)
                segs_r = re.findall(regex, segs_r)

                alt_l = []
                alt_r = []

                # Re-align classes:
                i, j = 0, 0
                while i < len(segs_l) and j < len(segs_r):
                    l_class = is_class(segs_l[i]) if i < len(segs_l) else False
                    r_class = is_class(segs_r[j]) if j < len(segs_r) else False
                    if l_class and not r_class:
                        segs_l = [""] + segs_l
                    elif r_class and not l_class:
                        segs_r = [""] + segs_r
                    else:
                        i += 1
                        j += 1

                # prepare alternation
                for sl, sr in zip_longest(segs_l, segs_r):
                    if sr is None:
                        alt_l.append(sl)
                    elif sl is None:
                        alt_r.append(sr)
                    else:
                        l_class = is_class(sl)
                        r_class = is_class(sr)

                        if l_class:
                            alt_l.append(get_class(sl))
                        else:
                            alt_l.append(sl)

                        if r_class:
                            alt_r.append(get_class(sr))
                        else:
                            alt_r.append(sr)

                alternation[c1].append(tuple(alt_l))
                alternation[c2].append(tuple(alt_r))

            return alternation

        def parse_context(string):
            regex = r"({classes}|{seg}|_)([+*?]?)".format(seg=seg, classes=classes)
            for s, q in re.findall(regex, string):
                if (s, q) == ("_", ""):
                    yield "{}"
                elif is_class(s):
                    yield get_class(s), quantities[q]
                else:
                    yield s, quantities[q]

        new = cls.__new__(cls, cells)

        try:
            alt_str, ctxt_str, score_str = re.match(r"(.*) / (.*) ?<([\d.e-]+)>", string).groups()
        except AttributeError as e:
            message = "I can't create a pattern from this: {}. Maybe the pattern has been exported with str and not repr ?".format(
                string)
            raise ValueError(message) from e

        new.context = Context(list(parse_context(ctxt_str)))
        new.alternation = parse_alternation(alt_str, cells)
        new.cells = cells
        new.score = float(score_str)
        new._repr = new._make_str_(features=False)
        new._feat_str = new._make_str_(features=True)
        new._gen_alt = None
        new.lexemes = set()
        return new

    def __eq__(self, other):
        """ Pattern equality: we simply check that they are both Pattern and their full string representation is identical

        Example:
            >>> Inventory.initialize("tests/data/frenchipa.csv")
            >>> p1 = Pattern._from_str(("A", "B"), "E_ ⇌ Ø_E / am_n_ <0>")
            >>> p2 = Pattern(('A','B'), (Form("amEn"), Form("amənE")), aligned=False)
            >>> p1 == p2
            True
            >>> p1 == "E_ ⇌ Ø_E / am_n_ <0>"
            False

        Args:
            other (Pattern): another Pattern

        Returns:
            Whether the two patterns are identical
        """
        return type(self) is Pattern and type(other) is Pattern and str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        """Return a repr string, for ex: _ ⇌ E / abEs_ <0.5>.

        repr() provides an exportable string, which:
        - Lists all sound classes exhaustively
        - Comprises also the score

        This makes it possible to instantiate back a pattern.
        """
        try:
            return '{content} <{score}>'.format(content=self._repr, score=self.score)
        except AttributeError:
            self._repr = self._make_str_(features=False)
            return '{content} <{score}>'.format(content=self._repr, score=self.score)

    def __str__(self):
        """ Return a str representation, for ex: _ ⇌ E / X+_

        str() provides a human readable string which:
        - Represents sounds classes in shorthand
        - Does not include the score
        """
        try:
            return self._feat_str
        except AttributeError:
            self._feat_str = self._make_str_(features=True)
            return self._feat_str

    def is_identity(self):
        """ Checks whether this pattern is an identity pattern.

        Example:
            >>> Inventory.initialize("tests/data/frenchipa.csv")
            >>> p = Pattern.new_identity(("A", "B"))
            >>> p.is_identity()
            True
        """
        return all(self.alternation[x] == [()] for x in self.cells)

    def _make_str_(self, features=True, reverse=False):
        """ Generic string builder used to construct representations.
        """
        alternation = self._format_alt(features=features)
        if reverse:
            alternation = " ⇌ ".join("_".join(alt) for alt in alternation[::-1])
        else:
            alternation = " ⇌ ".join("_".join(alt) for alt in alternation)

        context = self.context.to_str(mode=int(features) + 1)

        return alternation + " / " + context

    def to_alt(self, exhaustive_blanks=True, use_gen=False, **kwargs):
        """ Build a string representing the alternation

        Example:
            >>> Inventory.initialize("tests/data/frenchipa.csv")
            >>> cells = ("prs.1.sg", "prs.2.pl")
            >>> forms = (Form("amEn"), Form("amənE"))
            >>> p = Pattern(cells, forms, aligned=False)
            >>> p.alternation
            {'prs.1.sg': [('E',), ('',)], 'prs.2.pl': [('Ø',), ('E',)]}
            >>> p.to_alt()
            '_E_ ⇌ _Ø_E'
            >>> p.to_alt(exhaustive_blanks=False)
            'E_ ⇌ Ø_E'
            >>> p.to_alt(use_gen=True)
            '_[-arro]_ ⇌ _[+arro]_E'

        Arguments:
            exhaustive_blanks (bool): Whether initial and final contexts should be marked by a filler.
            use_gen (bool): Whether the alternation should use phonological generalizations (when available).

        Returns:
            A string representing the alternation, with contexts positions replaced by the filler "_".
        """
        filler = "_"

        def add_ellipsis(alt, initial, final):
            if alt == [""]:
                return filler
            else:
                flattened = ["".join(str(x) for x in affix) for affix in alt]
                return initial + filler.join(flattened) + final

        initial = "" if (not self.context[0].blank or not exhaustive_blanks) else filler
        final = "" if (self.context[-1].blank or not exhaustive_blanks) else filler

        if use_gen and self._gen_alt:
            tmp_alt = self.alternation
            self.alternation = self._gen_alt

        result = [add_ellipsis(alt, initial, final) for alt in self._format_alt()]

        if use_gen and self._gen_alt:
            self.alternation = tmp_alt
            self._repr = self._make_str_(features=False)
            self._feat_str = self._make_str_(features=True)

        return " ⇌ ".join(result)

    def _iter_alt(self, **kwargs):
        """Generator of formatted alternating material for each cell."""
        for cell in self.cells:
            formatted = []
            for segs in self.alternation[cell]:
                formatted.append("".join(segs))
            yield formatted

    def _init_from_alignment(self, alignment):
        alternation = []
        context = []
        comparables = iter(alignment)
        elements = next(comparables, None)
        while elements is not None:
            while elements is not None and are_all_identical(elements):
                context.append(elements[0])
                elements = next(comparables, None)
            if elements is not None and not are_all_identical(elements):
                altbuffer = [[x] for x in elements]
                context.append("{}")
                elements = next(comparables, None)
                while elements and not are_all_identical(elements):
                    for buffer, new in zip(altbuffer, elements):
                        if buffer[-1] == "":
                            buffer[-1] = new
                        elif new != "":
                            buffer.append(new)
                    elements = next(comparables, None)
                alternation.append(altbuffer)

        alternation = {cell: [tuple(x) for x in alt]
                       for cell, alt
                       in zip_longest(self.cells,
                                      zip(*alternation),
                                      fillvalue=("",))}
        context = [(x, one) if x != "{}" else "{}" for x in context]

        self.alternation, self.context = alternation, Context(context)

    @property
    def _regex(self):
        """Get the regex, ensure its creation."""
        try:
            return self._saved_regex
        except AttributeError:
            self._create_regex()
            return self._saved_regex

    @property
    def _repl(self):
        """Get the replacement string, ensure its creation."""
        try:
            return self._saved_repl
        except AttributeError:
            self._create_regex()
            return self._saved_repl

    @_repl.setter
    def _repl(self, value):
        self._saved_repl = value

    @_regex.setter
    def _regex(self, value):
        self._saved_regex = value

    def _create_regex(self):
        """Create regexes and replacement strings for this pattern.

        Example:
            >>> Inventory.initialize("tests/data/frenchipa.csv")
            >>> cells = ("prs.1.sg", "prs.2.pl")
            >>> forms = (Form("amEn"), Form("amənE"))
            >>> p = Pattern(cells, forms, aligned=False)
            >>> p
            E_ ⇌ Ø_E / am_n_ <0>
            >>> p._repl # Calls _create_regex if needed
            {'prs.1.sg': [None, 'E', None, ''], 'prs.2.pl': [None, 'Ø', None, 'E']}
            >>> p._regex # Calls _create_regex if needed
            {'prs.1.sg': re.compile('^((?:a )(?:m ))((?:E ))((?:n ))()$'), 'prs.2.pl': re.compile('^((?:a )(?:m ))((?:Ø ))((?:n ))((?:E ))$')}

        """
        c1, c2 = self.cells

        # Build alternation as list of zipped segments / transformations
        alternances = []
        for left, right in zip(self.alternation[c1], self.alternation[c2]):
            alternances.append(
                list(zip_longest(_iter_alternation(left),
                                 _iter_alternation(right),
                                 fillvalue=(False, ""))))

        regex = {c1: "", c2: ""}
        repl = {c1: [], c2: []}

        for i, group in enumerate(self.context):
            c = group.to_str(mode=0).format("")
            regex[c1] += c
            regex[c2] += c
            repl[c1].append(None)
            repl[c2].append(None)

            if group.blank:
                # alternation
                # We build one regex group for each continuous sequence of segments and each transformation
                for (is_segments, chars_1), (_, chars_2) in alternances[i]:
                    if is_segments:
                        # Substitution replacement: pass directly the target segments
                        # (this is a string; or None if no replacement)
                        repl[c1].append(" ".join(chars_1))
                        repl[c2].append(" ".join(chars_2))

                        # Regex matches these segments as one
                        regex[c1] += "({})".format("".join(Inventory.regex(x) if x else "" for x in chars_1))
                        regex[c2] += "({})".format("".join(Inventory.regex(x) if x else "" for x in chars_2))
                    else:
                        # Transformation replacement (this is a tuple)
                        repl[c1].append((chars_2, chars_1))
                        repl[c2].append((chars_1, chars_2))

                        # Regex matches these segments as one group
                        regex[c1] += "({})".format(_regex_or(chars_1))
                        regex[c2] += "({})".format(_regex_or(chars_2))

        self._saved_regex = {c: re.compile("^" + regex[c] + "$") for c in regex}
        self._saved_repl = repl

    def _find_generalized_alt(self):
        """See if the alternation can be generalized using phonological operations."""
        c1, c2 = self.cells
        this_alt = {c1: [], c2: []}
        gen_any = False
        for left, right in zip(self.alternation[c1], self.alternation[c2]):
            gen_left = []
            gen_right = []
            for a, b in zip_longest(left, right, fillvalue=""):
                if a != "" and b != "":
                    A, B = Inventory.transformation(a, b)
                else:
                    A, B = "", ""
                if (len(A) > 1 or len(B) > 1):
                    gen_any = True
                    gen_left.append(A)
                    gen_right.append(B)
                else:
                    gen_left.append(a)
                    gen_right.append(b)
            this_alt[c1].append(tuple(gen_left))
            this_alt[c2].append(tuple(gen_right))
        if gen_any:
            self._gen_alt = dict(zip(self.cells, (tuple(this_alt[x]) for x in self.cells)))
            self._create_regex()
        else:
            self._gen_alt = None

    def applicable(self, form, cell):
        """Test if this pattern matches a form, i.e. if the pattern is applicable to the form.

        Arguments:
            form (str): a form.
            cell (str): A cell contained in self.cells.

        Returns:
            `bool`: whether the pattern is applicable to the form from that cell.
        """
        try:
            regex = self._regex[cell]
            return bool(regex.match(form))
        except KeyError as err:
            raise KeyError("Unknown cell {}."
                           " This pattern's cells are {}."
                           "".format(err, " and ".join(self.cells)))

    def apply(self, form, names, raiseOnFail=True):
        """Apply the pattern to a form.

        Arguments:
            form : a form, assumed to belong to the cell `names[0]`.
            names :
                apply to a form of cell `names[0]`
                to produce a form of cell `names[1]` (default:`self.cells`).
                Patterns being non-oriented, it is better to use the names argument.
            raiseOnFail (bool):
                defaults to True. If true, raise an error when the pattern is not applicable to the form.
                If False, return None instead.

        Returns:
            form belonging the opposite cell.

        """
        from_cell, to_cell = names if names else self.cells
        reg = self._regex[from_cell]
        string, nb_subs = reg.subn(lambda x: _replace_alternation(x.groups(""), self._repl[to_cell]), form)
        if nb_subs == 0 and (not self.applicable(form, from_cell)):
            if raiseOnFail:
                raise NotApplicable("The context {} from the pattern {} and cells {} -> {}"
                                    "doesn't match the form \"{}\""
                                    "".format(self._regex[from_cell].pattern, self, from_cell, to_cell, form))
            else:
                return None
        return Form.from_segmented_str(string)

    def _generalize_alt(self, *others):
        """Use the generalized alternation, using features when possible rather than segments."""

        c1, c2 = self.cells
        # At first, alternations are {cell: parts},
        # parts are tuple(positions)
        # positions are tuples of phoneme positions
        # The order of embedding is cell, then parts then positions
        # Ex: {c1: (('a','b'),('c',)), c2: (('A','B'),('C',))}

        # This gets us to parts, then cells, then positions
        # [ (('a','b'), ('A','B')),
        #   (('c',), ('C',)) ]
        others += (self,)
        generalized = list(zip(self._gen_alt[c1], self._gen_alt[c2]))
        specific = list(zip(*(zip(p.alternation[c1], p.alternation[c2]) for p in others)))
        minimal_generalization = []

        # Iterate first over parts
        for i, generalized_part in enumerate(generalized):  # Iterates on alternation parts, between blanks
            # A part looks like: (('a','b'), ('A','B')),
            # We now want positions, then cells: (('a','A'), ('b','B')),
            generalized_part = list(zip(*generalized_part))

            # Arrange by phoneme position, then cell, then patterns
            specific_parts = list(zip(*(zip_longest(*p, fillvalue="")
                                        for p in specific[i])))
            minimal_gen_part = ([], [])
            for j, generalized_pos in enumerate(generalized_part):  # Iterates on each change in the part
                specific_pos = set(specific_parts[j])
                if len(specific_pos) > 1:
                    minimal_gen_part[0].append(generalized_pos[0])
                    minimal_gen_part[1].append(generalized_pos[1])
                else:
                    change = specific_pos.pop()
                    minimal_gen_part[0].append(change[0])
                    minimal_gen_part[1].append(change[1])
            minimal_generalization.append(minimal_gen_part)

        minimal_generalization = zip(*minimal_generalization)
        alternation = dict(zip([c1, c2], minimal_generalization))
        self.alternation = alternation
        self._repr = self._make_str_(features=False)
        self._feat_str = self._make_str_(features=True)

    def _is_max_gen(self):
        maxi_seg = Inventory._max
        return all([x in [(maxi_seg, kleenestar), "{}"] for x in self.context])

    def _format_alt(self, features=True):
        """Get formatted alternating material for each cell."""

        def format_as_chars(left, right):
            return ("{{{}}}".format(",".join(sorted(left))),
                    "{{{}}}".format(",".join(sorted(right))))

        def format_as_features(left, right):
            feats_left, feats_right = Inventory.get_transform_features(left, right)
            feats_left = "[{}]".format(" ".join(sorted(feats_left)))
            feats_right = "[{}]".format(" ".join(sorted(feats_right)))
            chars_left, chars_right = format_as_chars(left, right)
            if len(feats_left) + len(feats_right) <= len(chars_left) + len(chars_right):
                return feats_left, feats_right
            return chars_left, chars_right

        if features:
            format_regular_change = format_as_features
        else:
            format_regular_change = format_as_chars

        c1, c2 = self.cells
        alternation = zip(self.alternation[c1], self.alternation[c2])
        c1_alt = []
        c2_alt = []

        for left, right in alternation:
            formatted_left = ""
            formatted_right = ""
            for seg_left, seg_right in zip_longest(left, right, fillvalue=""):
                if Inventory.is_leaf(seg_left) and Inventory.is_leaf(seg_right):
                    formatted_left += seg_left
                    formatted_right += seg_right
                else:
                    l, r = format_regular_change(seg_left, seg_right)
                    formatted_left += l
                    formatted_right += r
            c1_alt.append(formatted_left)
            c2_alt.append(formatted_right)
        return c1_alt, c2_alt


class ParadigmPatterns(dict):
    """
    This class stores alternation patterns computed for a paradigm.

    Arguments:
        cells (list[str]): cells for which patterns are registered.
    """

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)
        self.cells = []

    def info(self):
        log.debug('Patterns:')
        log.debug(self.__repr__())
        if len(self.keys()) > 0:
            log.debug(list(self.values())[0].head())
        else:
            log.debug('Does not contain any dataframe')

    def find_patterns(self, paradigms, *args, method="edits", disable_tqdm=False, cpus=1, optim_mem=False, **kwargs):
        """Find Patterns in a DataFrame.

        Methods can be:
            - edits (dynamic alignment using levenshtein scores)
            - phon (dynamic alignment using segment similarity scores)

        Patterns are chosen according to their coverage and accuracy among competing patterns,
        and they are merged as much as possible. Their alternation can be generalized.

        This method updates the internal dict and does not return anything.

        The internal dict is of shape dict of tuples to pd.DataFrame,
        where the tuples are pairs of cells and the dataframes hold patterns for pairs of cells.

        Arguments:
            paradigms (:class:`pandas:pandas.DataFrame`): paradigms (columns are cells, index are lemmas).
            method (str): method for scoring best pairwise alignments. Can be "edits" or "phon".
            disable_tqdm (bool): if true, do not show progressbar
            cpus (int): number of CPUs to use for parallelisation (defaults to 1)
            optim_mem (bool): whether to convert patterns to str to use less memory (defaults to False)
        """
        if method == "edits":
            self.insert_cost = alignment.edits_ins_cost
            self.sub_cost = alignment.edits_sub_cost
        elif method == "phon":
            Inventory.init_dissimilarity_matrix(**kwargs)
            self.insert_cost = Inventory.insert_cost
            self.sub_cost = Inventory.sub_cost
        else:
            raise NotImplementedError("Alignment method {} is not implemented."
                                      "Use `phon` or `edits`"
                                      "rather than this function.".format(method))

        self.cells = list(paradigms.data.cell.unique())
        self.optim_mem = optim_mem

        tqdm.pandas(leave=False, disable=disable_tqdm)
        log.info("Looking for analogical patterns...")
        log.info(f"Using {cpus} threads for pattern inference.")
        # Register empty dfs of patterns
        # This is to avoid threads depending on a shared paradigms object !
        for pair in combinations(self.cells, 2):
            dict.__setitem__(self, pair, paradigms.get_empty_pattern_df(*pair))

        with Pool(cpus) as pool:  # Create a multiprocessing Pool
            self.update(tqdm(pool.imap_unordered(self.find_cellpair_patterns,
                                                 list(self)),  # list of dict keys = the pairs
                             total=comb(len(self.cells), 2)))

    def __repr__(self):
        if len(self.cells) == 0:
            return "ParadigmPatterns(empty)"
        return f"ParadigmPatterns({', '.join([a + '~' + b for a, b in dict.keys(self)])})"

    def export(self, md, kind, optim_mem=False):
        """
        Export dataframes to a folder for later use.

        Arguments:
            kind (str): type of patterns (phon or edits).
            optim_mem (bool): Whether to not export human readable patterns too. Defaults to False.
        """

        # Create pattern map
        pattern_list = set()
        for pair in self:
            patterns = self[pair]['pattern'].unique()
            pattern_list.update([repr(pat) for pat in patterns])
        pattern_map = {pat: n for n, pat in enumerate(pattern_list)}
        filename = md.register_file("patterns_map.csv")
        s = pd.Series({n: pat for pat, n in pattern_map.items()}, name="patterns")
        s.to_csv(filename)

        # Save regular patterns
        folder = "patterns"
        md.register_folder(folder, description="Compact machine readable patterns.")
        log.info("Writing patterns (importable by other scripts) to %s", folder)
        for pair in self.keys():
            self.to_csv(md, pair, folder, kind, pretty=optim_mem,
                        only_id=True, pattern_map=pattern_map)

        # Save human readable patterns
        if optim_mem:
            log.warning("Since you asked for args.optim_mem,"
                        "I will not export the human_readable file.")
        else:
            folder = "patterns_human_readable"
            md.register_folder(folder, description="Pretty patterns (for manual examination)")
            log.info("Writing pretty patterns (for manual examination) to %s", folder)
            for pair in self.keys():
                self.to_csv(md, pair, folder, kind,
                            pretty=True)
        return md.prefix

    def to_csv(self, md, pair, folder, kind,
               pretty=False, only_id=False, pattern_map=None):
        """Export a Patterns DataFrame to csv."""
        a, b = pair
        filename = md.register_file(f"{kind}_{a}-{b}.csv", folder=folder)
        export_fun = str if pretty else repr
        export = self[pair].copy()
        export.pattern = export.pattern.map(export_fun)
        if only_id:
            # Replace forms by ids
            export[['form_x', 'form_y']] = \
                export[['form_x', 'form_y']].map(lambda x: x.id)

            # Replace patterns by ids
            export.pattern = export.pattern.map(pattern_map)
        export.drop(["lexeme"], axis=1).to_csv(filename, sep=",", index=False)

    def from_file(self, folder, *args, force=False, **kwargs):
        """Read pattern data from a previous export.

        Arguments:
            folder (str): path to the folder

        """
        collection = defaultdict(lambda: defaultdict(str))
        folder = Path(folder)

        # Read patterns map
        patterns_map = pd.read_csv(folder / 'patterns_map.csv', index_col=0).patterns
        # patterns_map = pd.Series(s.index.values, index=s)

        # Parse patterns for each pair of cells
        first = True
        for path in (folder / "patterns").iterdir():
            self.from_csv(path, patterns_map, collection, *args, **kwargs)
            if first:
                n_files = len(list(folder.iterdir()))
                memory_check(list(self.values())[0], n_files, force=force)
                first = False

        # Raise error if wrong parameters.
        # return table

    def from_csv(self, path, patterns_map, collection,
                 paradigms, defective=True, overabundant=True):
        """
        Read a patterns dataframe for a specific pair of cells

        Arguments:
            paradigms (pandas.DataFrame): a paradigms dataframe, with form id's as index.
            defective (bool): whether to consider defective lexemes.
            overabundant (bool): whether to consider overabundance.
            collection (defaultdict): a defaultdict to avoid recomputing
                patterns from strings.
        """

        def read_pattern(string):
            """
            Reads patterns from string representations.
            Arguments:
                string (str): a string representation of a pattern.
            """
            if string and not pd.isnull(string):
                if string in collection[cells]:
                    result = collection[cells][string]
                else:
                    pattern = Pattern._from_str(cells, string)
                    collection[cells][string] = pattern
                    result = pattern
                return result
            if defective:
                return None
            else:
                return np.nan

        table = pd.read_csv(path, sep=",", dtype="str")

        reg = re.compile(r"[^_]+_(.+)-(.+)\.csv")
        cells = tuple(reg.match(path.name).groups())
        for cell in cells:
            if cell not in self.cells:
                self.cells.append(cell)

        # Restore phon_form based on paradigms and form_ids
        table[['lexeme', 'form_x']] = pd.merge(table, paradigms, right_index=True,
                                               left_on='form_x').iloc[:, [-3, -1]]
        table['form_y'] = pd.merge(table, paradigms, right_index=True,
                                   left_on='form_y').iloc[:, -1]

        table.pattern = table.pattern.astype('int').map(patterns_map).apply(read_pattern)

        if not defective:
            table.dropna(axis=0, subset="pattern", inplace=True)

        if not overabundant:
            dupl = table.duplicated('lexeme')
            if dupl.any():
                raise ValueError("Overabundant is unexpected, but some rows are duplicated. "
                                 "Set entropy.legacy=False or recompute patterns with "
                                 f"pats.overabundant=False. Examples:\n{table[dupl].head()}.")

        if (
                defective
                and (paradigms[paradigms.cell.isin(cells)].form == '').any()
                and table.pattern.notna().all()
        ):
            raise ValueError("It looks like you ignored defective rows"
                             "when computing patterns. Set defective=False.")

        self[cells] = table

    def _generate_rules(self, row, pair, collection):
        """
        Generates the patterns for each pair of forms.

        Arguments:
            row (pandas.Series): a dataframe row containing the two forms.
            collection (defaultdict): the patterns collection
        """
        lex, a, b = row.lexeme, row.form_x, row.form_y

        if a and b:
            if a == b:
                new_rule = Pattern(pair, zip(a.tokens, b.tokens), aligned=True)
                new_rule.lexemes = {(lex, a, b)}
                alt = new_rule.to_alt(exhaustive_blanks=False)
                t = _get_pattern_matchtype(new_rule, pair[0], pair[1])
                collection[alt][t].append(new_rule)
            else:
                done = []
                log.debug("All alignments of {}, {}".format(a, b))
                for aligned in alignment.align_auto(a.tokens, b.tokens,
                                                    self.insert_cost, self.sub_cost):
                    log.debug((aligned))
                    new_rule = Pattern(pair, aligned, aligned=True)
                    new_rule.lexemes = {(lex, a, b)}
                    log.debug("pattern: " + str(new_rule))
                    if str(new_rule) not in done:
                        done.append(str(new_rule))
                        if new_rule._gen_alt:
                            alt = new_rule.to_alt(exhaustive_blanks=False, use_gen=True)
                            log.debug("gen alt: " + str(alt))
                        else:
                            alt = new_rule.to_alt(exhaustive_blanks=False)
                        t = _get_pattern_matchtype(new_rule, pair[0], pair[1])
                        collection[alt][t].append(new_rule)

    def find_cellpair_patterns(self, pair, **kwargs):
        """
        Finds patterns for a pair of cells and returns a Dataframe containing the patterns.
        """

        def _score(p):
            """Scores each pattern"""

            def test_all(row, p):
                """Tests each pattern against each form"""
                correct = 0
                lex, a, b = row.lexeme, row.form_x, row.form_y
                if a and b:
                    if (lex, a, b) in p.lexemes:
                        correct += 1
                    else:
                        B = p.apply(a, pair, raiseOnFail=False)
                        A = p.apply(b, pair[::-1], raiseOnFail=False)

                        if (B == b) and (A == a):
                            correct += 1
                            p.lexemes.add((lex, a, b))
                return correct

            counts = df.apply(test_all, axis=1, p=p)
            return counts.sum()

        def _attribute_best_pattern(sorted_collection, lex, a, b):
            for p in sorted_collection:
                if (lex, a, b) in p.lexemes:
                    return p

        df = self[pair]  # Retrieve the empty df from inner dict
        collection = defaultdict(lambda: defaultdict(list))
        c1, c2 = pair

        # Identify pairwise alternations
        df.apply(self._generate_rules, axis=1, pair=pair, collection=collection)

        sorted_collection = []

        for alt in collection:
            log.debug("\n\n####Considering alt:" + str(alt))
            # Attempt to generalize
            types = list(collection[alt])
            pats = []
            log.debug("1.Generalizing in each type")
            for alt_type in collection[alt]:
                log.debug("\tlooking at :" + str(collection[alt][alt_type]))
                if _compatible_context_type(alt_type):
                    collection[alt][alt_type] = [generalize_patterns(collection[alt][alt_type])]
                    # print("\t\tcontexts compatible",collection[alt][j])
                else:
                    collection[alt][alt_type] = incremental_generalize_patterns(*collection[alt][alt_type])
                    # print("\t\tcontexts not compatibles:",collection[alt][j])
                pats.extend(collection[alt][alt_type])
            log.debug("2.Generalizing across types")
            if _compatible_context_type(*types):
                log.debug("\tlooking at compatible:" + str(pats))
                collection[alt] = [generalize_patterns(pats)]
            else:
                log.debug("\tlooking at incompatible:" + str(pats))
                collection[alt] = incremental_generalize_patterns(*pats)
            log.debug("Result:" + str(collection[alt]))
            # Score
            for p in collection[alt]:
                p.score = _score(p)
                sorted_collection.append(p)

        # Sort by score and choose best patterns
        sorted_collection = sorted(sorted_collection, key=lambda x: x.score, reverse=True)

        def _best_pattern(row):
            lex, a, b = row.lexeme, row.form_x, row.form_y
            if a != '' and b != '':
                return _attribute_best_pattern(sorted_collection, lex, a, b)
            return None

        df['pattern'] = df.apply(_best_pattern, axis=1)
        if self.optim_mem:
            df.pattern = df.pattern.apply(repr)
        return (pair, df)

    def unmerge_columns(self, paradigms):
        """
        Recreates merged columnss

        Arguments:
            paradigms (Paradigms): a Paradigms object with already unmerged columns.
        """

        # Variables to store asynchronous changes.
        mapping = {}
        identity = set()

        # List the values that we will need to add.
        for x, y in dict.keys(self):
            dedup = False
            if "#" in x:
                col_x = x.split('#')
                identity.add(frozenset(sorted(col_x)))
                dedup = True
            else:
                col_x = [x]
            if "#" in y:
                col_y = y.split('#')
                identity.add(frozenset(sorted(col_y)))
                dedup = True
            else:
                col_y = [y]
            if dedup:
                mapping[(x, y)] = list(product(col_x, col_y))

        # We create the new pairs and delete the old ones.
        for old, news in mapping.items():
            # old_loc = (df.cell_x == old[0]) & (df.cell_y == old[1])
            for new in news:
                self[new] = paradigms.get_empty_pattern_df(*new)
                self[new]['pattern'] = self[old].pattern.values
            del self[old]

        # We create identity patterns
        for sets in identity:
            for pair in combinations(sets, 2):
                self[pair] = paradigms.get_empty_pattern_df(*pair)
                defective = (self[pair].form_x == "") | (self[pair].form_y == "")
                self[pair].loc[~defective, 'pattern'] = Pattern.new_identity(pair)
                self[pair].loc[defective, 'pattern'] = None

    def find_cellpair_applicable(self, pair):
        """ Find applicable patterns for a single cell pair.

        Args:
            pair (tuple of str): pair of cells

        Returns:
            Dataframe of applicable patterns.
        """
        def applicable(form):
            """ Return a tuple of all applicable patterns for a given form"""
            return tuple((p for p in available_patterns if p.applicable(form, cell_x)))

        df = self[pair]
        available_patterns = [p for p in self[pair]['pattern'].unique() if p is not None]
        cell_x = pair[0]
        has_pat = ~df['pattern'].isnull()
        applicables = df.loc[has_pat, "form_x"].apply(applicable)
        applicables.name = "applicable"
        return (pair, applicables)

    def find_applicable(self, cpus=1, **kwargs):
        """Find all applicable rules for each form.

        We name sets of applicable rules *classes*. *Classes* are oriented:
        we produce two separate columns (a, b) and (b, a)
        for each pair of columns (a, b) in the paradigm..

        Returns:
            :class:`pandas:pandas.DataFrame`:
                associating a lemma (index)
                and an ordered pair of paradigm cells (columns)
                to a tuple representing a class of applicable patterns.
        """
        log.info("Looking for classes of applicable patterns")

        # Adding oriented patterns
        col_rename = {'form_x': 'form_y', 'form_y': 'form_x'}
        to_add = {}
        for key, value in self.items():
            to_add[key[::-1]] = self[key].rename(columns=col_rename)
        self.update(to_add)

        log.info("total cpus: " + str(cpus))

        with Pool(cpus) as pool:  # Create a multiprocessing Pool
            for pair, applicables in tqdm(pool.imap_unordered(self.find_cellpair_applicable, self), total=len(self)):
                df = self[pair]
                # We're trying to avoid any pandas internal issues in merging the df/series
                # First, we create a new column in the df...
                df.loc[:, "applicable"] = None
                # Then we assign the returned series, which is already called "applicable":
                df.loc[applicables.index, "applicable"] = applicables

    def incidence_table(self, microclasses):
        """ Create a Context from a dataframe of properties.

        Arguments:
                microclasses (iterable): microclass exemplars

        Returns:
            pd.DataFrame: a wide dataframe representing an incidence matrix
        """
        incidence_table = defaultdict(dict)
        for pair in self:
            df = self[pair]
            # Limit to microclasses
            df = df[df.lexeme.isin(microclasses)]
            header = "⇌".join(pair)
            for i, row in df.iterrows():
                pair_has_pattern = f"{header}=<{row.pattern}>"
                incidence_table[pair_has_pattern][row.lexeme] = "X"
        return pd.DataFrame(incidence_table).fillna("")
