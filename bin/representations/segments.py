# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Sacha Beniamine.

This module addresses the modelisation of phonological segments.
"""

import pandas as pd
from collections import defaultdict
from lattice.lattice import ICLattice

import functools
import unicodedata
import numpy as np
from itertools import combinations
import re
from utils import snif_separator


class Form(object):
    def __init__(self, string):
        if Segment._legal_str.fullmatch(string) is None:
            raise ValueError("Unknown sound in: " + repr(string))
        tokens = Segment._segmenter.findall(string)
        self.tokens = [Segment._normalization.get(c, c) for c in tokens]
        self.str = " ".join(self.tokens)+" "

    def __str__(self): return self.str
    def __len__(self): return len(self.tokens)
    def __iter__(self): yield from self.tokens
    def __getitem__(self, item): return self.tokens[item]

class Segment(object):
    """The `Segments.Segment` class holds the definition of a single segment.
*
    Attributes:
        name (str or _CharClass): Name of the segment.
        features (frozenset of tuples):
            The tuples are of the form `(attribute, value)`
            with a positive value, used for set operations.

            TODO: rewrite docstr
    A `_CharClass` is a `str` with sorted chars and brackets.

    Charclasses have a: attr:`Segments._CharClass.REGEX` constant field.
    This class is used for segments names. This way we get:

    * An immutable class.
    * Enforced sorted characters.
    * Identical to a corresponding string.
    * Which __str__() returns a ready-to-use regex character class string.
    * Iterators go through the content (without the brakets)

    Attributes:
        REGEX (str): constant. The string wrapped in "[]"


    We still use get for:

    - pretty, shorthand, regex
    - s1 < s2
    - is s simple
    - is the segment known
    """
    _pool = {}
    _simple_segments = []
    _lattice = None
    _score_matrix = {}
    _gap_score = None
    _normalization = {}
    _segmenter = None
    _legal_str = None
    _max = None

    def __new__(cls, classes, extent, intent, shorthand=None):
        s = " ".join(sorted(extent))
        obj = cls._pool.get(s, None)
        if obj is None:
            obj = object.__new__(cls)
            cls._pool[s] = obj
        return obj

    def __init__(self, classes, extent, intent, shorthand=None):
        """Constructor for Segments."""
        self.charset = frozenset(extent)
        ordered = sorted(self.charset)
        joined = "|".join(ordered)
        if len(self.charset) == 1:
            self.REGEX = joined+" "
            self.pretty = joined
        else:
            # The non capturing group of each segment
            self.REGEX = "(?:" + "|".join(x+" " for x in ordered) + ")"
            self.pretty = "[" + "-".join(ordered)+"]"
        self.classes = set(classes)
        self.features = set(intent)
        self.ipa = " ".join(ordered)
        self.shorthand = shorthand or "[{}]".format(" ".join(self.features))
        self.shortest = min((self.pretty, self.shorthand), key=len)

    def __iter__(self):
        yield from self.charset

    def __str__(self):
        return self.REGEX

    def __repr__(self):
        r"""Return the representation of one segment.

        Example:

            >>> a = [+syl, +rel.ret., -haut, +arr, -cons, +son, +vois,\
            ...      -rond, +cont, +bas, -nas, -ant]

        """
        return "{} = {}".format(self.ipa, self.shorthand)

    def __lt__(self, other):
        """ Checks if self is a descendant of other.

        X is a descendant of Y if Y is in X's ancestor list.
        """
        return other.ipa in self.classes

    def __le__(self, other):
        return (self.ipa == other.ipa) or (self < other)

    def __len__(self):
        return len(self.charset)

    def similarity(self, other):
        """Compute phonological similarity  (Frisch, 2004)

        Measure from "Similarity avoidance and the OCP" , Frisch, S. A.; Pierrehumbert, J. B. & Broe,
        M. B. *Natural Language \& Linguistic Theory*, Springer, 2004, 22, 179-228, p. 198.

        We compute similarity by comparing the number of shared and unshared natural classes
        of two consonants, using the equation in (7). This equation is a direct extension
        of the Pierrehumbert (1993) feature similarity metric to the case of natural classes.

        (7) :math:`Similarity = \\frac{\\text{Shared natural classes}}{\\text{Shared natural classes } + \\text{Non-shared natural classes}}`
        """
        if self == other: return 1
        return len(self.classes & other.classes) / len(self.classes | other.classes)


    @classmethod
    def initialize(cls, filename, sep=None):
        print("Reading table")
        table = pd.read_table(filename, header=0, dtype=str,
                              index_col=False, sep=sep or snif_separator(filename),
                              encoding="utf-8")
        shorten_feature_names(table)
        table["Seg."] = table["Seg."].astype(str)
        na_vals = {c: "-1" for c in table.columns}
        na_vals["Seg."] = ""
        na_vals["UNICODE"] = ""
        na_vals["ALIAS"] = ""
        na_vals["value"] = ""
        table = table.fillna(na_vals)

        # Checking segments names legality
        for seg in table["Seg."]:
            if seg == "":
                raise ValueError("One of your segments doesn't have a name !")
            if seg.strip("#") == "":
                raise ValueError("The symbol \"#\" is reserved and can only "
                                 "be used in a shorthand name (#V# for a vowel, etc)")

        # Legacy columns
        if "value" in table.columns:
            table.drop("value", axis=1, inplace=True)
        if "UNICODE" in table.columns:
            table.drop("UNICODE", axis=1, inplace=True)
        if "ALIAS" in table.columns:
            table.drop("ALIAS", axis=1, inplace=True)

        # Separate shorthand table
        shorthand_selection = table["Seg."].str.match("^#.+#$")
        shorthands = None
        if shorthand_selection.any():
            shorthands = table[shorthand_selection]
            table = table[~shorthand_selection]
            shorthands.set_index("Seg.", inplace=True)
            shorthands = shorthands.applymap(str)  # Why is this necessary ?
        table.set_index("Seg.", inplace=True)

        print("Normalizing identical segments")
        attributes = list(table.columns)
        cls._normalization = normalize(table, attributes)
        table.set_index("Normalized", inplace=True)
        table.drop_duplicates(inplace=True)

        # if verbose:
        #     print("Normalization map: ",
        #           {chr(x): normalization[x] for x in normalization})
        # TODO: replace w logging

        def feature_formatter(columns):
            signs = ["-", "+"] + [str(x) for x in range(2, 11)]
            for c in columns:
                key, val = c.split("=")
                yield signs[int(float(val))] + key.replace(" ", "_")

        leaves = {t: [] for t in table.index}
        table = table.applymap(lambda x: str(x))

        lattice = ICLattice(table, leaves,
                            na_value="-1",
                            col_formatter=feature_formatter,
                            verbose=False)

        cls._lattice = lattice.lattice

        if shorthands is not None:
            shorthand_lattice = ICLattice(shorthands, {t: [] for t in shorthands.index},
                                          na_value="-1",
                                          col_formatter=feature_formatter,
                                          verbose=False)

            shorthands = {cls._lattice[i].intent: e[0].strip("#") for e, i in
                          shorthand_lattice.lattice if
                          e and len(e) == 1}

        for extent, intent in cls._lattice:

            if extent:
                # Define the shortest expression of this segment if possible
                shorthand = shorthands.get(intent, None)
                if len(intent) == 0:
                    shorthand = "X"
                elif shorthand is None and len(extent) > 1:
                    lower = set().union(
                        *(set(x.intent) for x in cls._lattice[extent].upper_neighbors))
                    minimal = set(intent) - lower
                    if minimal:
                        shorthand = "[{}]".format(" ".join(minimal))

                ancestors = lattice.ancestors(extent)
                classes = sorted(
                    ["|".join(sorted(ancestor.extent)) for ancestor in ancestors]
                    + [extent], key=len)

                segment = Segment(classes, extent, intent, shorthand=shorthand)
                if len(extent) == 1:
                    cls._simple_segments.append(segment.ipa)
                cls._lattice[extent].ipa = segment.ipa
            else:
                cls._lattice[extent].ipa = "X"

        not_actual_leaves = []
        for leaf in leaves:
            lattice_node = "|".join(sorted(cls._lattice[(leaf,)].extent))
            if leaf != lattice_node:
                not_actual_leaves.append((leaf, lattice_node))

        if not_actual_leaves:
            alert = ""
            for leaf, lattice_node in not_actual_leaves:
                other = "".join((set(lattice_node) - {leaf}))
                alert += "\n\t" + leaf + " is the same node as " + cls.get(
                    lattice_node).ipa
                alert += "\n\t\t" + repr(cls.get(lattice_node))
                for o in other:
                    alert += "\n\t\t" + repr(cls.get(o))

            raise Exception(
                "Warning, some of the segments aren't actual leaves :" + alert)

        cls.max = max(cls._pool, key=len)

        all_sounds = sorted(cls._simple_segments + list(cls._normalization) + [";"],
                            key=len, reverse=True)
        cls._segmenter = re.compile("(" + "|".join(all_sounds) + ")")
        cls._legal_str = re.compile("(" + "|".join(all_sounds) + ")+")


    @classmethod
    def is_simple_sound(cls, sound):
        return sound == "" or (sound in cls._simple_segments)

    @classmethod
    def init_dissimilarity_matrix(cls, gap_prop=0.24, **kwargs):
        """Compute score matrix with dissimilarity scores."""
        costs = []
        for a, b in combinations(cls._simple_segments, 2):
            seg_a = cls._pool[a]
            seg_b = cls._pool[b]
            cost = 1 - seg_a.similarity(seg_b)
            cls._score_matrix[(a, b)] = cls._score_matrix[(b, a)] = cost
            costs.append(cost)

        cls._gap_score = np.quantile(np.array(costs), 0.5) * gap_prop
        for a in cls._simple_segments:
            cls._score_matrix[(a, a)] = 0

    @classmethod
    def insert_cost(cls, *_):
        return cls._gap_score

    @classmethod
    def sub_cost(self, a, b):
        return self._score_matrix[(a, b)]

    @classmethod
    def intersect(cls, *args):
        """Intersect some segments from their names.
        This is the "meet" operation on the lattice nodes, and returns the lowest common ancestor.

        Returns:
            a str representing the segment which classes are the intersection of the input.
        """
        segs = {s for segment in args for s in segment.split(" ")} # TODO: Having to re-split on spaces is bad
        return cls._lattice[segs].ipa

    @classmethod
    def get(cls, descriptor):
        """Get a Segment from an alias."""
        try:
            # Simple case: the descriptor is a  known alias
            return cls._pool[descriptor]
        except:
            try:
                # Alternate case: use lattice to recover segment
                v = (descriptor,) if type(descriptor) is str else descriptor
                return cls._pool[cls._lattice[v].ipa]
            except:
                raise KeyError("The segment {} isn't known".format(descriptor))

    @classmethod
    def transformation(cls, a, b):
        """Find a transformation between aliases a and b.

        The transformation is a pair of two maximal sets of segments related by a bijective phonological function.

        This function takes a pair of strings representing segments. It calculates the function which relates
        these two segments. It then finds the two maximal sets of segments related by this function.

        Example:
            In French, t -> s can be expressed by a phonological function
            which changes [-cont] and [-rel. ret] to [+cont] and [+rel. ret]

            These other segments are related by the same change:
            d -> z
            b -> v
            p -> f

            >>> a,b = Segment.transformation("t","s")
            >>> print(a,b)
            [bdpt] [fsvz]

        Arguments:
            a,b (str): Segment aliases.

        Returns:
            two charclasses.

        """

        def select_if_reciprocal(cls, segs, left, right):
            tmp = []
            for x in segs.charset:
                try:
                    y = cls.get((cls.get(x).features - left) | right)
                    if y and len(y) == 1:
                        x_back = cls.get((y.features - right) | left)
                        if x == x_back.ipa:
                            tmp.append(x)
                except:
                    pass
            return cls.get(tmp).ipa #TODO: attention a ne plus concatener les segments, mais faire des listes

        a_f = cls.get(a).features
        b_f = cls.get(b).features
        left = a_f - b_f
        right = b_f - a_f
        A, B = cls.get(left), cls.get(right)
        A = select_if_reciprocal(cls, A, left, right)
        B = select_if_reciprocal(cls, B, right, left)
        return A, B

    @classmethod
    def get_transform_features(cls, left, right):
        """ Get the features corresponding to a transformation.

        Arguments:
            left (tuple): string of segment aliases
            right (tuple): string of segment aliases

        Example:
            >>> inventory.get_from_transform("bd", "pt")
            {'+vois'}, {'-vois'}
        """
        # TODO: having to resplit is bad

        t1 = cls.get(left.split(" ")).features
        t2 = cls.get(right.split(" ")).features
        f1 = t1 - t2
        f2 = t2 - t1
        return f1, f2

    @classmethod
    def get_from_transform(cls, a, transform):
        """ Get a segment from another according to a transformation tuple.

        In the following example, the segments have been initialized with French segment definitions.

        Arguments:
            a (str): Segment alias
            transform (tuple): Couple of two strings of segments aliases.

        Example:
            >>> inventory.get_from_transform("d",("bdpt", "fsvz"))
            'z'
        """
        a = cls.get(a).features
        f1, f2 = cls.get_transform_features(*transform)
        return cls.get((a - f1) | f2).ipa

    @classmethod
    def show_pool(cls, only_single=False):
        """Return a string description of the whole segment pool."""
        if not only_single:
            return "\n".join(
                [repr(cls._pool[seg]) for seg in sorted(cls._pool, key=lambda x: len(x))])
        else:
            return "\n".join(
                [repr(cls._pool[seg]) for seg in sorted(cls._pool, key=lambda x: len(x))
                 if len(seg) == 1])


def normalize(ipa, features):
    """Assign a normalized segment to groups of segments with identical rows.

    This function takes a segments table
    and adds **in place** a "Normalized" column.
    This column contains a common value
    for each segment with identical boolean values.
    The function also returns a translation table
    mapping indexes to normalized segments.

    Note: the index are expected to be one char length.

    ============ ============ ==============
    Index        ..features..  Normalized
    ============ ============ ==============
    ɛ               [...]       E
    e               [...]       E
    ============ ============ ==============

    Arguments:
        ipa (:class:`pandas:pandas.DataFrame`):
            Dataframe of segments. Columns are features,
            UNICODE code point representation and segment names,
            indexes are segments.
        features (list): Feature columns' names.

    Returns:
        norm_map (dict):
            translation table from the segment's nameto its normalized name.
    """

    def find_indentical_rows(segment, table):
        seg_features = table.loc[segment, :]
        try:
            same_features_as_seg = (table == seg_features).all(axis=1)
        except ValueError:
            if seg_features.shape[0] > 1:
                raise ValueError(
                    "You have more than one segment definition for {}\n{}".format(segment,
                                                                                  seg_features))
        return same_features_as_seg

    ipa["Normalized"] = ""

    for segment in ipa[features].drop_duplicates().index:
        same_features_as_seg = find_indentical_rows(segment, ipa[features])

        if (ipa.loc[same_features_as_seg, "Normalized"] == "").all():
            ipa.loc[same_features_as_seg, "Normalized"] = segment

    norm_map = {seg: norm for seg, norm in zip(ipa.index, ipa["Normalized"]) if
                seg != norm}
    return norm_map


def shorten_feature_names(table):
    if "Seg." in list(table.iloc[0]):
        # Use shortened names if they exist
        table.columns = table.iloc[0]
        table.drop(0, axis=0, inplace=True)
        return table

    short_features_names = []
    for name in table.columns:
        if name in ["Seg.", "UNICODE", "ALIAS", "value"] or len(name) <= 3:
            short_features_names.append(name)
        else:
            names = [name[:i] for i in range(3, len(name) + 1)]
            while names and names[0] in short_features_names:
                names.pop(0)
            short_features_names.append(names[0])
    table.columns = short_features_names
