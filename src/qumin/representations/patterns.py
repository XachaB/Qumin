# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""author: Sacha Beniamine.

This module addresses the modeling of inflectional alternation patterns."""

# Our modules
from . import alignment
from .segments import Inventory, Form
from .contexts import Context
from .quantity import one, optional, some, kleenestar
from .generalize import generalize_patterns, incremental_generalize_patterns

# External tools
from os.path import commonprefix
from itertools import groupby, zip_longest, combinations, product
from collections import defaultdict
from copy import deepcopy
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import logging

log = logging.getLogger()

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


def _replace_alternation(m, r):
    """ Replace all matches in m using co-indexed functions in r."""

    def iter_replacements(m, r):
        g = m.groups("")
        for i in range(len(g)):
            yield r[i](g[i].strip())

    return "".join(iter_replacements(m, r))


def are_all_identical(iterable):
    """Test whether all elements in the iterable are identical."""
    return iterable and len(set(iterable)) == 1


class NotApplicable(Exception):
    """Raised when a :class:`Pattern` can't be applied to a form."""
    pass


class Pattern(object):
    r"""Represent an alternation pattern and its context.

    The pattern can be defined over an arbitrary number of forms.
    If there are only two forms, a :class:`BinaryPattern` will be created.

    cells (tuple): Cell labels.

    Attributes:
        alternation (dict[str, list[tuple]]):
            Maps the cell's names to a list of tuples of alternating material.

        context (tuple of str):
            Sequence of (str, Quantifier) pairs or "{}" (stands for alternating material.)

        score (float):
            A score used to choose among patterns.

    Example:
        >>> cells = ("prs.1.sg", "prs.1.pl","prs.2.pl")
        >>> forms = ("amEn", "amənõ", "amənE")
        >>> p = patterns.Pattern(cells, forms, aligned=False)
        >>> p
        E_ ⇌ ə_ɔ̃ ⇌ ə_E / am_n_ <0>
    """

    def __new__(cls, *args, **kwargs):
        if len(args[0]) == 2:
            return super(Pattern, cls).__new__(BinaryPattern)
        else:
            return super(Pattern, cls).__new__(cls)

    def __init__(self, cells, forms, aligned=False, **kwargs):
        """Constructor for Pattern.

        Arguments:
            cells (Iterable): Cells labels (str), in the same order.
            forms (Iterable): Forms (str) to be segmented.
            aligned (bool): whether forms are already aligned. Otherwise, left alignment will be performed.
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

    def __deepcopy__(self, memo):
        cls = self.__class__
        copy = cls.__new__(cls, self.cells)
        copy.context = deepcopy(self.context)
        copy.alternation = deepcopy(self.alternation)
        copy.cells = self.cells
        copy.score = self.score
        copy._repr = self._repr
        copy._feat_str = self._feat_str
        copy._gen_alt = self._gen_alt
        return copy

    @classmethod
    def new_identity(cls, cells):
        """Create a new identity pattern for a given set of cells.
        """
        p = cls(cells, [""] * len(cells), aligned=True)
        p.context = Context([(Inventory._max, kleenestar)])
        p._repr = p._make_str_(features=False)
        p._feat_str = p._make_str_(features=True)
        return p

    @classmethod
    def _from_str(cls, cells, string):
        """Parse a repr str to a pattern.
        >>> _ ⇌ E / abEs_ <0.5>
        Note: Phonemes in context classes are now separated by ","


        """
        quantities = {"": one, "?": optional, "+": some, "*": kleenestar}

        simple_segs = sorted((s for s in Inventory._classes if Inventory.is_leaf(s)),
                             key=len, reverse=True)

        seg = r"(?:{})".format("|".join(simple_segs))
        classes = r"[\[{{](?:{sounds}|\-|,)+[}}\]]".format(sounds="|".join(simple_segs))

        def is_class(s):
            return s is not None and (len(s) > 3 or "-" in s or "," in s) and \
                ((s[0], s[-1]) == ("[", "]") or (s[0], s[-1]) == ("{", "}"))

        def get_class(s):
            separator = "," if "," in s else "-"
            segments = s[1:-1].split(separator)
            return frozenset(segments)

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
                elif (len(s) > 2 or "-" in s or "," in s) and \
                        ((s[0], s[-1]) == ("[", "]") or (s[0], s[-1]) == ("{", "}")):
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
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        """Return a repr string, for ex: _ ⇌ E / abEs_ <0.5>."""
        try:
            return '{content} <{score}>'.format(content=self._repr, score=self.score)
        except AttributeError:
            self._repr = self._make_str_(features=False)
            return '{content} <{score}>'.format(content=self._repr, score=self.score)

    def __str__(self):
        try:
            return self._feat_str
        except AttributeError:
            self._feat_str = self._make_str_(features=True)
            return self._feat_str

    def is_identity(self):
        return all(self.alternation[x] == [()] for x in self.cells)

    def _make_str_(self, features=True, reverse=False):
        """Return a string verbosely representing the segmentation.

        """
        alternation = list(self._iter_alt(features=features))
        if reverse:
            alternation = " ⇌ ".join("_".join(alt) for alt in alternation[::-1])
        else:
            alternation = " ⇌ ".join("_".join(alt) for alt in alternation)

        context = self.context.to_str(mode=int(features) + 1)

        return alternation + " / " + context

    def alternation_list(self, exhaustive_blanks=True, use_gen=False, filler="_"):
        """Return a list of the alternating material, where the context is replaced by a filler.

        Arguments:
            exhaustive_blanks (bool): Whether initial and final contexts should be marked by a filler.
            use_gen (bool): Whether the alternation should be the generalized one.
            filler (str): Alternative filler used to join alternation members.

        Returns:
            a list of str of alternating material, where the context is replaced by a filler.
        """

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

        result = [add_ellipsis(alt, initial, final) for alt in self._iter_alt()]

        if use_gen and self._gen_alt:
            self.alternation = tmp_alt
            self._repr = self._make_str_(features=False)
            self._feat_str = self._make_str_(features=True)

        return result

    def to_alt(self, exhaustive_blanks=True, use_gen=False, **kwargs):
        """Join the alternating material obtained with alternation_list() in a str."""
        return " ⇌ ".join(self.alternation_list(exhaustive_blanks=exhaustive_blanks, use_gen=use_gen, **kwargs))

    def _iter_alt(self, *kwargs):
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


class BinaryPattern(Pattern):
    r"""Represent the alternation pattern between two forms.

    A BinaryPattern is a `Patterns.Pattern` over just two forms.
    Applying the pattern to one of the original forms yields the second one.

    As an example, we will use the following alternation
    in a present verb of french:

    ========================== ========================== ==========================
    cells                      Forms                      Transcription
    ========================== ========================== ==========================
    prs.1.sg ⇌ prs.2.pl        j'amène ⇌ vous amenez      amEn ⇌ amənE
    ========================== ========================== ==========================

    Example:
        >>> cells = ("prs.1.sg", "prs.2.pl")
        >>> forms = ("amEn", "amənE")
        >>> p = Pattern(cells, forms, aligned=False)
        >>> type(p)
        representations.patterns.BinaryPattern
        >>> p
        E_ ⇌ ə_E / am_n_ <0>
        >>> p.apply("amEn",cells)
        'amənE'
    """

    def __lt__(self, other):
        """Sort on lexicographic order.

        There is no reason to sort patterns,
        but Pandas wants to do it from time to time,
        this is only implemented to avoid Pandas complaining.
        """
        return str(self) < str(other)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._find_generalized_alt()

    def __deepcopy__(self, memo):
        copy = super().__deepcopy__(self)
        copy._gen_alt = deepcopy(self._gen_alt)
        return copy

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
        """
        c1, c2 = self.cells

        def make_transform_reg(sounds):
            sounds = sorted(sounds)
            return "(?:" + "|".join(x + " " for x in sounds) + ")"

        def make_transform_repl(a, b):
            return lambda x: Inventory.get_from_transform(x, (a, b)) + " "

        def make_sub_repl(chars):
            return lambda x: chars

        def identity(x):
            return x + " "

        def iter_alternation(alt):
            for is_transform, group in groupby(alt, lambda x: not Inventory.is_leaf(x)):
                if is_transform:
                    for x in group:
                        yield is_transform, x
                else:
                    yield is_transform, "".join(Inventory.regex(x) if x else "" for x in group)

        # Build alternation as list of zipped segments / transformations
        alternances = []
        for left, right in zip(self.alternation[c1], self.alternation[c2]):
            alternances.append(
                list(zip_longest(iter_alternation(left), iter_alternation(right), fillvalue=(False, ""))))

        regex = {c1: "", c2: ""}
        repl = {c1: [], c2: []}
        i = 0

        for group in self.context:
            c = group.to_str(mode=0).format("")
            regex[c1] += c
            regex[c2] += c
            repl[c1].append(identity)
            repl[c2].append(identity)

            if group.blank:
                # alternation
                # We build one regex group for each continuous sequence of segments and each transformation
                for (is_transform, chars_1), (is_transform, chars_2) in alternances[i]:
                    # Replacements
                    if is_transform:
                        # Transformation replacement make_transform_repl with two segments in argument
                        repl[c1].append(make_transform_repl(chars_2, chars_1))
                        repl[c2].append(make_transform_repl(chars_1, chars_2))

                        # Regex matches these segments as one group
                        regex[c1] += "({})".format(make_transform_reg(chars_1))
                        regex[c2] += "({})".format(make_transform_reg(chars_2))
                    else:
                        # Substitution replacement make_subl_repl with target segment as argument
                        repl[c1].append(make_sub_repl(chars_1))
                        repl[c2].append(make_sub_repl(chars_2))

                        # Regex matches these segments as one group
                        regex[c1] += "({})".format(chars_1)
                        regex[c2] += "({})".format(chars_2)
                i += 1

        self._saved_regex = {c: re.compile("^" + regex[c] + "$") for c in regex}
        self._saved_repl = repl

    def _find_generalized_alt(self):
        """See if the alternation can be expressed in a more general way using features."""
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
        string, nb_subs = reg.subn(lambda x: _replace_alternation(x, self._repl[to_cell]), form)
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

    def _iter_alt(self, features=True):
        """Generator of formatted alternating material for each cell."""

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
        yield c1_alt
        yield c2_alt


class PatternCollection(tuple):
    """Represent a set of patterns."""

    def __init__(self, items):
        self.collection = tuple(sorted(set(items)))

    def __str__(self):
        return ";".join(str(p) for p in self.collection)

    def __repr__(self):
        return ";".join(repr(p) for p in self.collection)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def __lt__(self, other):
        return self.collection < other.collection


def find_patterns(paradigms, method, **kwargs):
    r"""Find Patterns in a DataFrame according to any general method.

    Methods can be:
        - suffix (align left),
        - prefix (align right),
        - baseline (see Albright & Hayes 2002)
        - levenshtein (dynamic alignment using levenshtein scores)
        - similarity (dynamic alignment using segment similarity scores)

    Arguments:
        paradigms (:class:`pandas:pandas.DataFrame`):
            paradigms (columns are cells, index are lemmas).
        method (str): "suffix", "prefix", "baseline", "levenshtein" or "similarity"

    Returns:
        (tuple):
            **patterns,pattern_dict**. Patterns is the created
            :class:`pandas:pandas.DataFrame`,
            pat_dict is a dict mapping a column name to a list of patterns.
    """
    if method in ["levenshtein", "similarity"]:
        return _with_dynamic_alignment(paradigms, scoring_method=method, **kwargs)
    elif method in ["suffix", "prefix", "baseline"]:
        return _with_deterministic_alignment(paradigms, method=method, **kwargs)
    else:
        raise NotImplementedError("Alignment method {} is not implemented, "
                                  "choose from suffix, prefix or baseline.".format(method))


def _with_deterministic_alignment(paradigms, method="suffix", disable_tqdm=False, **kwargs):
    r"""Find Patterns in a DataFrame according to a deterministic alignment method.

    Methods can be suffix (align left), prefix (align right) or baseline (see Albright & Hayes 2002).

    Arguments:
        paradigms (:class:`pandas:pandas.DataFrame`):
            paradigms (columns are cells, index are lemmas).
        method (str): "suffix", "prefix" or "baseline"
        disable_tqdm (bool): if true, do not show progressbar

    Returns:
        (tuple):
            **patterns,pattern_dict**. Patterns is the created
            :class:`pandas:pandas.DataFrame`,
            pat_dict is a dict mapping a column name to a list of patterns.
    """
    if method == "suffix":
        align_func = alignment.align_left
    elif method == "prefix":
        align_func = alignment.align_right
    elif method == "baseline":
        align_func = alignment.align_baseline
    else:
        raise NotImplementedError("Alignment method {} is not implemented."
                                  "Call find_patterns(paradigms, method) rather than this function.".format(method))

    # Variable for fast access
    cols = paradigms.columns

    # Create tuples pairs

    pairs = list(combinations(cols, 2))

    patterns = pd.DataFrame(index=paradigms.index,
                            columns=pairs)

    pat_dict = {}

    def generate_rules(row, cells, collection):
        if row.iloc[0] == row.iloc[1]:  # If the lists are identical, do not compute product
            pairs = [(x, x) for x in row.iloc[0]]
        else:
            pairs = product(*row)

        for a, b in pairs:
            if a and b:
                aligned = align_func(a.tokens, b.tokens)
                new_rule = Pattern(cells, aligned, aligned=True)
                new_rule.lexemes.add(row.name)
                alt = new_rule.to_alt(exhaustive_blanks=False)
                collection[alt].append(new_rule)

    def find_patterns_in_col(column, pat_dict):
        pattern_collection = defaultdict(list)
        a, b = column.name

        paradigms[[a, b]].apply(generate_rules, axis=1, args=((a, b), pattern_collection))
        results = {}
        pat_dict[(a, b)] = []

        for alt in pattern_collection:
            pattern = generalize_patterns(pattern_collection[alt])
            pat_dict[(a, b)].append(pattern)
            for name in pattern.lexemes:
                results[name] = pattern

        return pd.Series(results)

    tqdm.pandas(leave=False, disable=disable_tqdm)
    patterns = patterns.progress_apply(find_patterns_in_col, axis=0, args=(pat_dict,))

    return patterns, pat_dict


def _with_dynamic_alignment(paradigms, scoring_method="levenshtein", optim_mem=False,
                            disable_tqdm=False, **kwargs):
    """Find Patterns in a DataFrame with automatic alignment.

    Patterns are chosen according to their coverage and accuracy among competing patterns,
    and they are merged as much as possible. Their alternation can be generalized.

    Arguments:
        paradigms: a DataFrame.
        scoring_method (str): method for scoring best pairwise alignments. Can be "levenshtein" or "similarity".
        disable_tqdm (bool): if true, do not show progressbar

    Returns:
        a tuple of the DataFrame and a dict of pairs of cells
        to the list of unique patterns used for that pair of cells.
    """

    if scoring_method == "levenshtein":
        insert_cost = alignment.levenshtein_ins_cost
        sub_cost = alignment.levenshtein_sub_cost
    elif scoring_method == "similarity":
        Inventory.init_dissimilarity_matrix(**kwargs)
        insert_cost = Inventory.insert_cost
        sub_cost = Inventory.sub_cost
    else:
        raise NotImplementedError("Alignment method {} is not implemented."
                                  "Call find_patterns(paradigms, method) "
                                  "rather than this function.".format(scoring_method))

    cells = paradigms.data.cell.unique()
    pairs = list(combinations(cells, 2))

    paradigms_dic = {}
    for pair in pairs:
        paradigms_dic[pair] = paradigms.get_empty_pattern_df(*pair)

    pat_dict = {}

    def generate_rules(row, collection):
        """
        Generates the patterns for each pair of forms.

        Arguments:
            row (pandas.Series): a dataframe row containing the two forms.
            collection (defaultdict): the patterns collection
        """
        cells = [row.cell_x, row.cell_y]
        lex, a, b = row.lexeme, row.form_x, row.form_y

        if a and b:
            if a == b:
                new_rule = Pattern(cells, zip(a.tokens, b.tokens), aligned=True)
                new_rule.lexemes = {(lex, a, b)}
                alt = new_rule.to_alt(exhaustive_blanks=False)
                t = _get_pattern_matchtype(new_rule, cells[0], cells[1])
                collection[alt][t].append(new_rule)
            else:
                done = []
                log.debug("All alignments of {}, {}".format(a, b))
                for aligned in alignment.align_auto(a.tokens, b.tokens, insert_cost, sub_cost):
                    log.debug((aligned))
                    new_rule = Pattern(cells, aligned, aligned=True)
                    new_rule.lexemes = {(lex, a, b)}
                    log.debug("pattern: " + str(new_rule))
                    if str(new_rule) not in done:
                        done.append(str(new_rule))
                        if new_rule._gen_alt:
                            alt = new_rule.to_alt(exhaustive_blanks=False, use_gen=True)
                            log.debug("gen alt: " + str(alt))
                        else:
                            alt = new_rule.to_alt(exhaustive_blanks=False)
                        t = _get_pattern_matchtype(new_rule, cells[0], cells[1])
                        collection[alt][t].append(new_rule)

    def attribute_best_pattern(sorted_collection, lex, a, b):
        for p in sorted_collection:
            if (lex, a, b) in p.lexemes:
                return p

    def _score(p, df):
        """Scores each pattern"""
        def test_all(row, p):
            """Tests each pattern against each form"""
            correct = 0
            cells = [row.cell_x, row.cell_y]
            lex, a, b = row.lexeme, row.form_x, row.form_y
            if a and b:
                if (lex, a, b) in p.lexemes:
                    correct += 1
                else:
                    B = p.apply(a, cells, raiseOnFail=False)
                    A = p.apply(b, cells[::-1], raiseOnFail=False)
                    correct_one = (p.apply(a, cells, raiseOnFail=False) == b) and (
                            p.apply(b, cells[::-1], raiseOnFail=False) == a)
                    if correct_one:
                        correct += 1
                        p.lexemes.add((lex, a, b))
            return correct
        counts = df.apply(test_all, axis=1, p=p)
        return counts.sum()

    def find_patterns_in_col(df, pat_dict):
        collection = defaultdict(lambda: defaultdict(list))
        c1 = df.cell_x.iloc[0]
        c2 = df.cell_y.iloc[0]

        # Identify pairwise alternations
        df.apply(generate_rules, axis=1, collection=collection)

        sorted_collection = []

        for alt in collection:
            log.debug("\n\n####Considering alt:" + str(alt))
            # Attempt to generalize
            types = list(collection[alt])
            pats = []
            log.debug("1.Generalizing in each type")
            for j in collection[alt]:
                log.debug("\tlooking at :" + str(collection[alt][j]))
                if _compatible_context_type(j):
                    collection[alt][j] = [generalize_patterns(collection[alt][j])]
                    # print("\t\tcontexts compatible",collection[alt][j])
                else:
                    collection[alt][j] = incremental_generalize_patterns(*collection[alt][j])
                    # print("\t\tcontexts not compatibles:",collection[alt][j])
                pats.extend(collection[alt][j])
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
                p.score = _score(p, df)
                sorted_collection.append(p)

        # Sort by score and choose best patterns
        sorted_collection = sorted(sorted_collection, key=lambda x: x.score, reverse=True)

        def _best_pattern(row):
            lex, a, b = row.lexeme, row.form_x, row.form_y
            if a != '' and b != '':
                return attribute_best_pattern(sorted_collection, lex, a, b)
            return None

        df['pattern'] = df.apply(_best_pattern, axis=1)
        if optim_mem:
            df.pattern = df.pattern.apply(repr)
        else:
            pat_dict[(c1, c2)] = df['pattern'].unique()
        return df

    tqdm.pandas(leave=False, disable=disable_tqdm)
    paradigms_dic = {cells: find_patterns_in_col(df, pat_dict)
                     # .set_index(['lexeme', 'form_a', 'form_b']).pattern
                     for cells, df in tqdm(paradigms_dic.items())}
    return paradigms_dic, pat_dict


def find_applicable(pat_table, pat_dict, disable_tqdm=False, **kwargs):
    """Find all applicable rules for each form.

    We name sets of applicable rules *classes*. *Classes* are oriented:
    we produce two separate columns (a, b) and (b, a)
    for each pair of columns (a, b) in the paradigm.

    Arguments:
        pat_table (:class:`pandas:pandas.DataFrame`):
            patterns table, containing the forms and the cells.
        pat_dict  (dict): a dict mapping a column name to a list of patterns.
        disable_tqdm (bool): if true, do not show progressbar

    Returns:
        :class:`pandas:pandas.DataFrame`:
            associating a lemma (index)
            and an ordered pair of paradigm cells (columns)
            to a tuple representing a class of applicable patterns.
    """

    def _iter_applicable_patterns(row):
        pair = (row.cell_x, row.cell_y)
        if pair not in pat_dict:
            pair = (row.cell_y, row.cell_x)
        for pattern in pat_dict[pair]:
            if pattern.applicable(row.form_x, row.cell_x):
                yield pattern

    def applicable(*args):
        """Returns all applicable patterns to a single row"""
        return tuple(_iter_applicable_patterns(*args))

    # Making the table bidirectionnal
    col_rename = {'cell_x': 'cell_y', 'cell_y': 'cell_x',
                  'form_x': 'form_y', 'form_y': 'form_x'}
    pat_table = pd.concat([pat_table, pat_table.rename(columns=col_rename)])

    # Computing classes
    tqdm.pandas(leave=False, disable=disable_tqdm)
    has_pat = pat_table.pattern.notna()
    pat_table.loc[has_pat, "applicable"] = pat_table.loc[has_pat, :]\
        .progress_apply(applicable, axis=1)

    return pat_table

def find_alternations(paradigms, method, **kwargs):
    """Find local alternations in a Dataframe of paradigms.

    For each pair of form in the paradigm, keep only the alternating material (words are left-aligned).
    Return the resulting DataFrame.

    Arguments:
        paradigms (pandas.DataFrame):
            a dataframe containing inflectional paradigms.
            Columns are cells, and rows are lemmas.
        method (str): "local" uses pairs of forms, "global" uses entire paradigms.

    Returns:
        pandas.DataFrame:
            a dataframe with the same indexes as `paradigms`
            and as many columns as possible combinations of columns
            in `paradigms`, filled with segmented patterns.
    """
    if method == "local":
        return _local_alternations(paradigms, **kwargs)
    elif method == "global":
        return _global_alternations(paradigms, **kwargs)


def _local_alternations(paradigms, disable_tqdm=False, **kwargs):
    # Variable for fast access
    cols = paradigms.columns

    def segment(forms, cells):
        if not forms[0] or not forms[1]:
            return None
        segmented = Pattern(cells, forms)
        return segmented.to_alt()

    def segment_columns(col):
        a, b = col.name
        return paradigms[[a, b]].apply(segment, args=([a, b],), axis=1)

    # Create tuples pairs
    pairs = list(combinations(cols, 2))
    patterns = pd.DataFrame(index=paradigms.index, columns=pairs)
    tqdm.pandas(leave=False, disable=disable_tqdm)
    patterns = patterns.progress_apply(segment_columns, axis=0)

    return patterns


def _global_alternations(paradigms, **kwargs):
    """Find global alternations in a paradigm.

    Return a DataFrame of alternations where we remove in each cell the material
    common to the whole row, wherever it is in the left aligned words.

    Arguments:
        paradigms (pandas.DataFrame):
            a dataframe containing inflectional paradigms.
            Columns are cells, and rows are lemmas.

    Returns:
        pandas.DataFrame:
            a dataframe of the same shape, columns and indexes as `paradigms`,
            filled with segmented patterns.

    Example:
        >>> df = pd.DataFrame([["amEn", "amEn", "amEn", "amənõ", "amənE", "amEn"]],
             columns=["prs.1.sg",  "prs.2.sg", "prs.3.sg", "prs.1.pl", "prs.2.pl","prs.3.pl"],
             index=["amener"])
        >>> df
               prs.1.sg prs.2.sg prs.3.sg prs.1.pl prs.2.pl prs.3.pl
        amener     amEn     amEn     amEn    amənõ    amənE     amEn
        >>> find_global_alternations(df)
               prs.1.sg prs.2.sg prs.3.sg prs.1.pl prs.2.pl prs.3.pl
        amener      _E_      _E_      _E_    _ə_ɔ̃     _ə_E      _E_

    """

    def row_as_list(cells, row):
        """Make a list of forms from a paradigm row."""
        for cell, forms in zip(cells, row):
            for form in forms.split(";"):
                if pd.notnull(forms) and forms != "":
                    yield cell, form

    def segment(forms, cells):
        newcells, formlist = zip(*row_as_list(cells, forms))
        segmented = Pattern(newcells, formlist)
        forms = defaultdict(list)
        for cell, ending in zip(newcells, segmented.alternation_list()):
            forms[cell].append(ending)

        for cell in cells:
            if cell in forms:
                forms[cell] = ";".join(forms[cell])
            else:
                forms[cell] = "#DEF#"
        return pd.Series(forms)[cells]

    df = paradigms.apply(segment, axis=1, args=(list(paradigms.columns),))
    return df


def find_endings(paradigms, *args, disable_tqdm=False, **kwargs):
    """Find suffixes in a paradigm.

    Return a DataFrame of endings where we remove in each row
    the common prefix to all the row's cells.

    Arguments:
        paradigms (pandas.DataFrame): a dataframe containing inflectional paradigms.
            Columns are cells, and rows are lemmas.
        disable_tqdm (bool): if true, do not show progressbar

    Returns:
        pandas.DataFrame: a dataframe of the same shape filled with segmented endings.

    Example:
        >>> df = pd.DataFrame([["amEn", "amEn", "amEn", "amənõ", "amənE", "amEn"]],
             columns=["prs.1.sg",  "prs.2.sg", "prs.3.sg", "prs.1.pl", "prs.2.pl","prs.3.pl"],
             index=["amener"])
        >>> df
               prs.1.sg prs.2.sg prs.3.sg prs.1.pl prs.2.pl prs.3.pl
        amener     amEn     amEn     amEn    amənõ    amənE     amEn
        >>> find_endings(df)
               prs.1.sg prs.2.sg prs.3.sg prs.1.pl prs.2.pl prs.3.pl
        amener       En       En       En      ənõ      ənE       En

    """

    def row_as_list(row):
        """Make a list of forms from a paradigm row."""
        return [form for forms in row for form in forms.split(";") if (pd.notnull(forms) and forms != "")]

    def segment(forms, l=0):
        """Segment all semicolon separated forms at the l character"""
        return ";".join(form[l:] if form else "#DEF#" for form in forms.split(";"))

    def row_ending(row):
        """Remove the common prefix in all strings of a row."""
        l = len(commonprefix(row_as_list(row)))
        return row.apply(segment, args=(l,))

    tqdm.pandas(leave=False, disable=disable_tqdm)
    return paradigms.progress_apply(row_ending, axis=1)


def make_pairs(paradigms):
    """Join columns with " ⇌ " by combination.

    The output has one column for each pairs on the paradigm's columns.
    """

    def pair_columns(column, paradigms):
        cell1, cell2 = column.name
        return paradigms[cell1] + " ⇌ " + paradigms[cell2]

    # Pairing cells
    pairs = list(combinations(paradigms.columns, 2))
    paired = pd.DataFrame(columns=pairs, index=paradigms.index)
    return paired.apply(pair_columns, args=(paradigms,))


def to_csv(dataframe, filename, pretty=False, only_id=False):
    """Export a Patterns DataFrame to csv."""
    export_fun = str if pretty else repr
    export = dataframe.copy()
    export.pattern = export.pattern.map(export_fun)
    if only_id:
        export[['form_x', 'form_y']] = \
            export[['form_x', 'form_y']].map(lambda x: x.id)
    export.drop(["lexeme", "cell_x", "cell_y"], axis=1).to_csv(filename, sep=",", index=False)


def from_csv(filename, paradigms, defective=True, overabundant=True):
    """Read a Patterns Dataframe from a csv.

    Arguments:
        filename (str): path to the file
        paradigms (pandas.DataFrame): a paradigms dataframe, with form id's as index.
        defective (bool): whether to consider defective lexemes.
        overabundant (bool): whether to consider overabundance.
    """

    # TODO: remove ?
    # def format_header(item):
    #     splitted = item.strip("'() ").split(",")
    #     return splitted[0].strip("' "), splitted[1].strip("' ")

    def read_pattern(row, collection):
        """
        Reads patterns from string representations.
        Arguments:
            row (pandas.Series): a pattern file row, containing
                a string representation of a pattern and cell names.
            collection (defaultdict): a defaultdict to avoid recomputing
                patterns from strings.
        """
        cells = (row.cell_x, row.cell_y)
        string = row.pattern
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

    collection = defaultdict(lambda: defaultdict(str))
    table = pd.read_csv(filename, sep=",", header=0, dtype="str")

    is_alt_str = table.pattern.map(lambda x: type(x) is str and "/" not in x).all()
    if is_alt_str:
        log.warning("These are not patterns but alternation strings")
        return table, {}

    # Restore phon_form based on paradigms and form_ids
    table[['lexeme', 'cell_x', 'form_x']] = pd.merge(table, paradigms, right_index=True,
                                                     left_on='form_x').iloc[:, -3:]
    table[['cell_y', 'form_y']] = pd.merge(table, paradigms, right_index=True,
                                           left_on='form_y').iloc[:, -2:]

    # The paradigms table already dropped unnecessary cells.
    # After the merge, NaN cells can be dropped also.
    table.drop(table[table.cell_x.isna() | table.cell_y.isna()].index,
               inplace=True)

    table['pattern'] = table.apply(read_pattern, collection=collection, axis=1)

    if not defective:
        table.dropna(axis=0, subset="pattern", inplace=True)

    if overabundant:
        raise NotImplementedError
    collection = {column: list(collection[column].values()) for column in collection}
    return table, collection


def unmerge_columns(df, paradigms):
    """
    Recreates merged columns in a memory efficient ways
    (that is, keeping category detype for cells and lexemes).

    Arguments:
        df (pandas.Dataframe): a pandas DataFrame containing patterns for each pair of cells.
        paradigms (Paradigms): a Paradigms object with already unmerged columns.
    """

    # Variables to store asynchronous changes.
    mapping = {}
    identity = set()
    new_cat = {"x": set(), "y": set()}

    # List the values that we will need to add.
    # TODO simplify this
    for x, y in df[['cell_x', 'cell_y']].drop_duplicates().itertuples(index=False):
        dedup = False
        if "#" in x:
            col_x = x.split('#')
            new_cat['x'].update(col_x)
            identity.add(frozenset(sorted(col_x)))
            dedup = True
        else:
            col_x = [x]
        if "#" in y:
            col_y = y.split('#')
            new_cat['y'].update(col_y)
            identity.add(frozenset(sorted(col_y)))
            dedup = True
        else:
            col_y = [y]
        if dedup:
            mapping[(x, y)] = list(product(col_x, col_y))

    # We remove unused categories and add the new ones.
    for n in new_cat.keys():
        s = df['cell_' + n]\
            .cat.remove_unused_categories()\
            .cat.add_categories(new_cat[n])
        df['cell_' + n] = s

    to_add = []
    # We create the new pairs and delete the old ones.
    for old, news in mapping.items():
        old_loc = (df.cell_x == old[0]) & (df.cell_y == old[1])
        for new in news:
            new_df = paradigms.get_empty_pattern_df(*new)
            new_df['pattern'] = df.loc[old_loc, "pattern"].values
            to_add.append(new_df)
        df.drop(df[old_loc].index, axis=0, inplace=True)

    # We create identity patterns
    for sets in identity:
        for pair in combinations(sets, 2):
            new_df = paradigms.get_empty_pattern_df(*pair)
            defective = (new_df.form_x == "") | (new_df.form_y == "")
            new_df.loc[~defective, 'pattern'] = Pattern.new_identity(pair)
            new_df.loc[defective, 'pattern'] = None
            to_add.append(new_df)

    # We resolve all concatenations
    # First set categories in a memory efficient way
    cat = paradigms.data.cell.cat.categories
    df.cell_y = df.cell_y.cat.set_categories(cat)
    df.cell_x = df.cell_x.cat.set_categories(cat)
    df = pd.concat([df] + to_add).reset_index(drop=True)

    return df
