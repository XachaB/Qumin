# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""author: Sacha Beniamine.

This module addresses the modeling of inflectional alternation patterns."""

from os.path import commonprefix
from itertools import combinations, product
from . import alignment
from .segments import Inventory, Form
from .contexts import Context
from .quantity import one, optional, some, kleenestar
from .generalize import generalize_patterns, incremental_generalize_patterns
from itertools import groupby, zip_longest
from collections import defaultdict
from copy import deepcopy
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import logging

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
        >>> Inventory.initialize("tests/data/frenchipa.csv")
        >>> cells = ("prs.1.sg", "prs.1.pl","prs.2.pl")
        >>> forms = (Form("amEn"), Form("amənɔ̃"), Form("amənE"))
        >>> p = Pattern(cells, forms, aligned=False)
        >>> p
        E_ ⇌ Ø_ɔ̃ ⇌ Ø_E / am_n_ <0>
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

        Str patterns look like:
            _ ⇌ E / abEs_ <0.5>

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
        >>> Inventory.initialize("tests/data/frenchipa.csv")
        >>> cells = ("prs.1.sg", "prs.2.pl")
        >>> forms = (Form("amEn"), Form("amənE"))
        >>> p = Pattern(cells, forms, aligned=False)
        >>> type(p)
        <class 'qumin.representations.patterns.BinaryPattern'>
        >>> p
        E_ ⇌ Ø_E / am_n_ <0>
        >>> p.apply(Form("amEn"), cells)
        Form(a m Ø n E )
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


def find_patterns(paradigms, method="levenshtein", optim_mem=False, disable_tqdm=False, **kwargs):
    """Find Patterns in a DataFrame.

    Methods can be:
        - levenshtein (dynamic alignment using levenshtein scores)
        - similarity (dynamic alignment using segment similarity scores)

    Patterns are chosen according to their coverage and accuracy among competing patterns,
    and they are merged as much as possible. Their alternation can be generalized.

    Arguments:
        paradigms (:class:`pandas:pandas.DataFrame`): paradigms (columns are cells, index are lemmas).
        method (str): method for scoring best pairwise alignments. Can be "levenshtein" or "similarity".
        disable_tqdm (bool): if true, do not show progressbar

    Returns:
        (tuple):
            **patterns,pattern_dict**. Patterns is the created
            :class:`pandas:pandas.DataFrame`,
            pat_dict is a dict mapping a column name to a list of patterns.
    """

    if method == "levenshtein":
        insert_cost = alignment.levenshtein_ins_cost
        sub_cost = alignment.levenshtein_sub_cost
    elif method == "similarity":
        Inventory.init_dissimilarity_matrix(**kwargs)
        insert_cost = Inventory.insert_cost
        sub_cost = Inventory.sub_cost
    else:
        raise NotImplementedError("Alignment method {} is not implemented."
                                  "Call find_patterns(paradigms, method) "
                                  "rather than this function.".format(method))

    cols = paradigms.columns
    pairs = list(combinations(cols, 2))
    patterns_df = pd.DataFrame(index=paradigms.index,
                               columns=pairs)

    pat_dict = {}

    def generate_rules(row, cells, collection):
        if row.iloc[0] == row.iloc[1]:  # If the lists are identical, do not compute product
            pairs = [(x, x) for x in row.iloc[0]]
        else:
            pairs = product(*row)

        for a, b in pairs:
            if a and b:
                if a == b:
                    new_rule = Pattern(cells, zip(a.tokens, b.tokens), aligned=True)
                    new_rule.lexemes = {(row.name, a, b)}
                    alt = new_rule.to_alt(exhaustive_blanks=False)
                    t = _get_pattern_matchtype(new_rule, cells[0], cells[1])
                    collection[alt][t].append(new_rule)
                else:
                    done = []
                    log.debug("All alignments of {}, {}".format(a, b))
                    for aligned in alignment.align_auto(a.tokens, b.tokens, insert_cost, sub_cost):
                        log.debug((aligned))
                        new_rule = Pattern(cells, aligned, aligned=True)
                        new_rule.lexemes = {(row.name, a, b)}
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

    def attribute_best_pattern(sorted_collection, l, a, b):
        for p in sorted_collection:
            if (l, a, b) in p.lexemes:
                return p

    def _score(p, cells, forms, index):
        def test_all(row, p, cells):
            correct = 0
            l, a_list, b_list = row
            if a_list == b_list:  # If the lists are identical, do not compute product
                pairs = [(x, x) for x in b_list]
            else:
                pairs = product(a_list, b_list)
            for a, b in pairs:
                if a and b:
                    if (l, a, b) in p.lexemes:
                        correct += 1
                    else:
                        B = p.apply(a, cells, raiseOnFail=False)
                        A = p.apply(b, cells[::-1], raiseOnFail=False)
                        correct_one = (p.apply(a, cells, raiseOnFail=False) == b) and (
                                p.apply(b, cells[::-1], raiseOnFail=False) == a)
                        if correct_one:
                            correct += 1
                            p.lexemes.add((l, a, b))
            return correct

        counts = np.apply_along_axis(test_all, 1, forms, p, cells)
        return counts.sum(axis=0)

    def find_patterns_in_col(column, pat_dict):
        collection = defaultdict(lambda: defaultdict(list))
        c1, c2 = column.name
        col = paradigms[[c1, c2]]
        index = col.index
        forms = col.reset_index().values

        col.apply(generate_rules, axis=1, args=((c1, c2), collection))

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
                p.score = _score(p, (c1, c2), forms, index)
                sorted_collection.append(p)

        # Sort by score and choose best patterns
        sorted_collection = sorted(sorted_collection, key=lambda x: x.score, reverse=True)
        best = defaultdict(set)

        nullset = {''}

        for l in index:
            row = col.at[l, c1], col.at[l, c2]

            if row != (nullset, nullset):
                if row[0] == row[1]:  # If the lists are identical, do not compute product
                    pairs = [(x, x) for x in row[0]]
                else:
                    pairs = product(*row)

                for a, b in pairs:
                    best[l].add(attribute_best_pattern(sorted_collection, l, a, b))
            else:
                best[l].add(None)

        if optim_mem:
            result = [repr(PatternCollection(list(best[l]))) for l in index]
        else:
            result = [PatternCollection(list(best[l])) for l in index]
            pat_dict[(c1, c2)] = list(set.union(*[set(coll.collection) for coll in result]))

        return result

    tqdm.pandas(leave=False, disable=disable_tqdm)
    patterns_df = patterns_df.progress_apply(find_patterns_in_col, axis=0, args=(pat_dict,))

    return patterns_df, pat_dict


def find_applicable(paradigms, pat_dict, disable_tqdm=False, **kwargs):
    """Find all applicable rules for each form.

    We name sets of applicable rules *classes*. *Classes* are oriented:
    we produce two separate columns (a, b) and (b, a)
    for each pair of columns (a, b) in the paradigm.

    Arguments:
        paradigms (:class:`pandas:pandas.DataFrame`):
            paradigms (columns are cells, index are lemmas).
        pat_dict  (dict): a dict mapping a column name to a list of patterns.
        disable_tqdm (bool): if true, do not show progressbar

    Returns:
        :class:`pandas:pandas.DataFrame`:
            associating a lemma (index)
            and an ordered pair of paradigm cells (columns)
            to a tuple representing a class of applicable patterns.
    """

    def _iter_applicable_patterns(form, local_patterns, cell):
        known_regexes = set()
        if type(form) is tuple:  # if overabundant
            form = form[0]  # from tuple to Form
        for pattern in local_patterns:
            regex = pattern._regex[cell]
            if regex in known_regexes:
                yield pattern

            elif pattern.applicable(form, cell):
                known_regexes.add(regex)
                yield pattern

    def applicable(*args):
        return tuple(_iter_applicable_patterns(*args))

    # The result has (a, b) and (b, a) columns
    # for each (a, b) column of patterns
    # -> pairs are ordered, patterns.columns weren't
    pairs = [y for x in pat_dict for y in (x, x[::-1])]

    # Initialisation
    classes = pd.DataFrame(index=paradigms.index,
                           columns=pairs)

    for a, b in tqdm(pat_dict, leave=False, disable=disable_tqdm):
        local_patterns = pat_dict[(a, b)]

        # Iterate on paradigms' rows of corresponding columns to fill with the
        # result
        classes[(a, b)] = paradigms[a].apply(applicable,
                                             args=(local_patterns, a))
        classes[(b, a)] = paradigms[b].apply(applicable,
                                             args=(local_patterns, b))

    return classes


def to_csv(dataframe, filename, pretty=False):
    """Export a Patterns DataFrame to csv."""
    export_fun = str if pretty else repr
    dataframe.map(export_fun).to_csv(filename, sep=",")


def from_csv(filename, defective=True, overabundant=True):
    """Read a Patterns Dataframe from a csv"""

    def format_header(item):
        splitted = item.strip("'() ").split(",")
        return splitted[0].strip("' "), splitted[1].strip("' ")

    def read_pattern(raw_value, names, collection):
        result = []
        if raw_value and not pd.isnull(raw_value):
            for string in raw_value.split(";"):
                if string in collection[names]:
                    result.append(collection[names][string])
                else:
                    pattern = Pattern._from_str(names, string)
                    collection[names][string] = pattern
                    result.append(pattern)
            if overabundant:
                return PatternCollection(result)
            else:
                return PatternCollection([result[0]])
        if defective:
            return PatternCollection([None])
        else:
            return np.nan

    def str_to_pat_column(col, collection):
        names = col.name
        collection[names] = {}
        return col.apply(read_pattern, args=(names, collection))

    collection = {}
    table = pd.read_csv(filename, sep=",", header=0, index_col=0)
    table.columns = [format_header(item) for item in table.columns]
    table = table.map(lambda x: None if x == "None" else x)

    is_alt_str = table.map(lambda x: type(x) is str and "/" not in x).all().all()
    if is_alt_str:
        log.warning("These are not patterns but alternation strings")
        return table, {}
    pat_table = table.apply(str_to_pat_column, args=(collection,))

    if not defective:
        pat_table.dropna(axis=0, inplace=True)

    collection = {column: list(collection[column].values()) for column in collection}

    return pat_table, collection
