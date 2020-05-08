# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""author: Sacha Beniamine.

This module addresses the modeling of inflectional alternation patterns."""

from os.path import commonprefix
from itertools import combinations, product
from representations import alignment, normalize_dataframe
from representations.segments import Segment, _CharClass, restore, restore_string, restore_segment_shortest
from representations.contexts import Context
from representations.quantity import one, optional, some, kleenestar
from representations.generalize import generalize_patterns, incremental_generalize_patterns
from itertools import groupby, zip_longest
from collections import defaultdict
from copy import deepcopy
import numpy as np
import pandas as pd
import re
from tqdm import tqdm

ORTHO = False
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
            yield r[i](g[i])

    return "".join(iter_replacements(m, r))


def are_all_identical(iterable):
    """Test whether all elements in the iterable are identical."""
    return iterable and len(set(iterable)) == 1


class NotApplicable(Exception):
    """Raised when a :class:`patterns.Pattern` can't be applied to a form."""
    pass


class Pattern(object):
    r"""Represent an alternation pattern and its context.

    The pattern can be defined over an arbitrary number of forms.
    If there are only two forms, a :class:`patterns.BinaryPattern` will be created.

    cells (tuple): Cell labels.

    alternation (dict of str: list of tuple):
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
            cells (iterable): Cells labels (str), in the same order.
            forms (iterable): Forms (str) to be segmented.
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
            alignment_of_forms = list(alignment.align_left(*list(forms), fillvalue=""))

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
        p = cls(cells, [""] * len(cells))
        p.context = [(Segment._max, kleenestar)]
        p._repr = p._make_str_(features=False)
        p._feat_str = p._make_str_(features=True)
        return p

    @classmethod
    def _from_str(cls, cells, string):
        """Parse a repr str to a pattern.
        >>> _ ⇌ E / abEs_ <0.5>
        Note: Contexts strings are now separated by "-" (because phonemes can be)
        """
        quantities = {"": one, "?": optional, "+": some, "*": kleenestar}

        def parse_alternation(string, cells):
            simple_segs = "".join(Segment._simple_segments)
            seg = r"[{}]".format(simple_segs)
            classes = r"\[[{}\-]+\]".format(simple_segs)
            regex = r"({seg}|{classes})".format(seg=seg, classes=classes)
            for c, alt in zip(cells, string.split(" ⇌ ")):
                alts = []
                for segs in alt.split("_"):
                    alts.append([])
                    for s in re.findall(regex, segs):
                        if (len(s) > 2 or "-" in s) and (s[0], s[-1]) == ("[", "]"):
                            if "-" in s:
                                raw_segments = s[1:-1].split("-")
                                aliases = [Segment.get(s).alias for s in raw_segments]
                                s = _CharClass("".join(aliases))
                            else:
                                # Legacy parser
                                s = _CharClass(s[1:-1])
                        alts[-1].append(s)
                yield c, [tuple(x) for x in alts]

        def parse_context(string):
            simple_segs = "".join(Segment._simple_segments)
            seg = r"[{}]".format(simple_segs)
            classes = r"\[[{}\-]+\]".format(simple_segs)
            regex = r"({seg}|{classes}|_)([+*?]?)".format(seg=seg, classes=classes)
            for s, q in re.findall(regex, string):
                if (s, q) == ("_", ""):
                    yield "{}"
                elif (len(s) > 2 or "-" in s) and (s[0], s[-1]) == ("[", "]"):
                    if "-" in s:
                        raw_segments = s[1:-1].split("-")
                        aliases = [Segment.get(s).alias for s in raw_segments]
                        s = _CharClass("".join(aliases))
                        yield s, quantities[q]
                    else:
                        # Legacy parser
                        yield _CharClass(s[1:-1]), quantities[q]
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
        new.alternation = dict(parse_alternation(alt_str, cells))
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

    def _iter_alt(self, features=True):
        """Generator of formatted alternating material for each cell."""

        def format_segment_ipa(chars):
            chars_restored = "-".join([restore(c) for c in sorted(chars)])
            if len(chars) > 1:
                chars_restored = "[{}]".format(chars_restored)
            return chars_restored

        def format_segment_shortest_notnode(chars):
            feats = set.intersection(*[Segment.get(x).features for x in chars])
            return min([format_segment_ipa(chars), "[{}]".format(" ".join(sorted(feats)))], key=len)

        if ORTHO:
            format_segment = str
        elif features:
            format_segment = format_segment_shortest_notnode
        else:
            format_segment = format_segment_ipa

        for cell in self.cells:
            formatted = []
            for segs in self.alternation[cell]:
                temp = ""
                for seg in segs:
                    if seg == "":
                        temp += seg
                    else:
                        temp += format_segment(seg)
                formatted.append(temp)
            yield formatted

    def _init_from_alignment(self, alignment):
        alternation = []
        context = []
        comparables = iter(alignment)
        elements = next(comparables, None)
        while elements is not None:
            if elements and are_all_identical(elements):
                context.append(elements[0])
                elements = next(comparables, None)
                while elements and are_all_identical(elements):
                    context.append(elements[0])
                    elements = next(comparables, None)
            if elements and not are_all_identical(elements):
                altbuffer = elements
                context.append("{}")
                elements = next(comparables, None)
                while elements and not are_all_identical(elements):
                    altbuffer = tuple(b + elt for b, elt in zip(altbuffer, elements))
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

        def make_transform_repl(a, b):
            return lambda x: Segment.get_from_transform(x, (a, b))

        def make_sub_repl(chars):
            return lambda x: chars

        def identity(x):
            return x

        def iter_alternation(alt):
            for is_transform, group in groupby(alt, lambda x: type(x) is _CharClass):
                if is_transform:
                    for x in group:
                        yield is_transform, x
                else:
                    yield is_transform, "".join(group)

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
                #  alternation
                #  We build one regex group for each continuous sequence of segments and each transformation
                for (is_transform, chars_1), (is_transform, chars_2) in alternances[i]:
                    # Replacements
                    if is_transform:
                        #  Transformation replacement make_transform_repl with two segments in argument
                        repl[c1].append(make_transform_repl(chars_2, chars_1))
                        repl[c2].append(make_transform_repl(chars_1, chars_2))
                    else:
                        #  Substitution replacement make_subl_repl with target segment as argument
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
                A, B = Segment.transformation(a, b)
                if a != "" and b != "" and (len(A) > 1 or len(B) > 1):
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
        string, nb_subs = re.subn(self._regex[from_cell], lambda x: _replace_alternation(x, self._repl[to_cell]), form)
        if nb_subs == 0 and (not self.applicable(form, from_cell)):
            if raiseOnFail:
                raise NotApplicable("The context {} from the pattern {} and cells {} -> {}"
                                    "doesn't match the form \"{}\""
                                    "".format(restore_string(self._regex[from_cell].pattern), self, from_cell, to_cell,
                                              form))
            else:
                return None
        return string

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
        maxi_seg = Segment._max
        return all([x in [(maxi_seg, kleenestar), "{}"] for x in self.context])


class PatternCollection(object):
    """Represent a set of patterns."""

    def __init__(self, collection):
        self.collection = tuple(sorted(set(collection)))

    def __str__(self):
        return ";".join(str(p) for p in self.collection)

    def _reverse_str(self):
        return ";".join(p._make_str_(features=True, reverse=True) for p in self.collection)

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
    paradigms_sets = paradigms.applymap(lambda x: set(x.split(";")))

    def generate_rules(row, cells, collection):
        if row[0] == row[1]:  # If the lists are identical, do not compute product
            pairs = [(x, x) for x in row[0]]
        else:
            pairs = product(*row)

        for a, b in pairs:
            if a and b:
                aligned = align_func(a, b)
                new_rule = Pattern(cells, aligned, aligned=True)
                new_rule.lexemes.add(row.name)
                alt = new_rule.to_alt(exhaustive_blanks=False)
                collection[alt].append(new_rule)

    def find_patterns_in_col(column, pat_dict):
        pattern_collection = defaultdict(list)
        a, b = column.name

        paradigms_sets[[a, b]].apply(generate_rules, axis=1, args=((a, b), pattern_collection))
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


def _with_dynamic_alignment(paradigms, scoring_method="levenshtein", optim_mem=False, disable_tqdm=False, **kwargs):
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
        Segment.init_dissimilarity_matrix(**kwargs)
        insert_cost = Segment.insert_cost
        sub_cost = Segment.sub_cost
    else:
        raise NotImplementedError("Alignment method {} is not implemented."
                                  "Call find_patterns(paradigms, method) "
                                  "rather than this function.".format(scoring_method))

    cols = paradigms.columns
    pairs = list(combinations(cols, 2))
    patterns_df = pd.DataFrame(index=paradigms.index,
                               columns=pairs)

    pat_dict = {}
    paradigms_sets = paradigms.applymap(lambda x: set(x.split(";")))

    def generate_rules(row, cells, collection):
        if row[0] == row[1]:  # If the lists are identical, do not compute product
            pairs = [(x, x) for x in row[0]]
        else:
            pairs = product(*row)

        # print("row: ",row)
        for a, b in pairs:
            if a and b:
                if a == b:
                    new_rule = Pattern(cells, zip(a, b), aligned=True)
                    new_rule.lexemes = {(row.name, a, b)}
                    alt = new_rule.to_alt(exhaustive_blanks=False)
                    t = _get_pattern_matchtype(new_rule, cells[0], cells[1])
                    collection[alt][t].append(new_rule)
                else:
                    done = []
                    # print("All alignments of {}, {}".format(a,b))
                    for aligned in alignment.align_auto(a, b, insert_cost, sub_cost):
                        # print((aligned))
                        new_rule = Pattern(cells, aligned, aligned=True)
                        new_rule.lexemes = {(row.name, a, b)}
                        if str(new_rule) not in done:
                            done.append(str(new_rule))
                            if new_rule._gen_alt:
                                alt = new_rule.to_alt(exhaustive_blanks=False, use_gen=True)
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
        col = paradigms_sets[[c1, c2]]
        index = col.index
        forms = col.reset_index().values

        col.apply(generate_rules, axis=1, args=((c1, c2), collection))

        sorted_collection = []

        for alt in collection:
            # print("\n\n####Considering alt:",alt)
            #  Attempt to generalize
            types = list(collection[alt])
            pats = []
            # print("1.Generalizing in each type")
            for j in collection[alt]:
                # print("\tlooking at :",collection[alt][j])
                if _compatible_context_type(j):
                    collection[alt][j] = [generalize_patterns(collection[alt][j])]
                    # print("\t\tcontexts compatible",collection[alt][j])
                else:
                    collection[alt][j] = incremental_generalize_patterns(*collection[alt][j])
                    # print("\t\tcontexts not compatibles:",collection[alt][j])
                pats.extend(collection[alt][j])
            # print("2.Generalizing across types")
            if _compatible_context_type(*types):
                # print("\tlooking at compatible:",pats)
                collection[alt] = [generalize_patterns(pats)]
            else:
                # print("\tlooking at incompatible:",pats)
                collection[alt] = incremental_generalize_patterns(*pats)
            # print("Result:",collection[alt])
            #  Score
            for p in collection[alt]:
                p.score = _score(p, (c1, c2), forms, index)
                sorted_collection.append(p)

        #  Sort by score and choose best patterns
        sorted_collection = sorted(sorted_collection, key=lambda x: x.score, reverse=True)
        best = defaultdict(set)

        nullset = {''}

        for l in index:
            row = col.at[l, c1], col.at[l, c2]

            if row != (nullset, nullset):
                if row[0] == row[1]:  #  If the lists are identical, do not compute product
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
        (:class:`pandas:pandas.DataFrame`):
            associating a lemma (index)
            and an ordered pair of paradigm cells (columns)
            to a tuple representing a class of applicable patterns.
    """

    def _iter_applicable_patterns(form, local_patterns, cell):
        known_regexes = set()
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
    print("Making pairs from", paradigms)

    def pair_columns(column, paradigms):
        cell1, cell2 = column.name
        return paradigms[cell1] + " ⇌ " + paradigms[cell2]

    # Pairing cells
    pairs = list(combinations(paradigms.columns, 2))
    paired = pd.DataFrame(columns=pairs, index=paradigms.index)
    return paired.apply(pair_columns, args=(paradigms,))


def to_csv(dataframe, filename, pretty=False):
    """Export a Patterns DataFrame to csv."""
    export_fun = str if pretty else repr
    dataframe.applymap(export_fun).to_csv(filename, sep=",")


def from_csv(filename, defective=True, overabundant=True):
    """Read a Patterns Dataframe from a csv"""

    def format_header(item):
        splitted = item.strip("'() ").split(",")
        return splitted[0].strip("' "), splitted[1].strip("' ")

    def read_pattern(raw_value, names, collection):
        result = []
        if raw_value:
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
    table = table.applymap(lambda x: None if x == "None" else x)
    table = normalize_dataframe(table,
                                Segment._aliases,
                                Segment._normalization,
                                verbose=False)

    is_alt_str = table.applymap(lambda x: x and "/" not in x).all().all()
    if is_alt_str:
        print("Warning: These are not patterns but alternation strings")
        return table, {}
    pat_table = table.apply(str_to_pat_column, args=(collection,))

    if not defective:
        pat_table.dropna(axis=0, inplace=True)

    collection = {column: list(collection[column].values()) for column in collection}

    return pat_table, collection
