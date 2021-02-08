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
        if Inventory._legal_str.fullmatch(string) is None:
            raise ValueError("Unknown sound in: " + repr(string))
        tokens = Inventory._segmenter.findall(string)
        self.tokens = [Inventory._normalization.get(c, c) for c in tokens]
        self.str = " ".join(self.tokens)+" "

    def __str__(self): return self.str
    def __len__(self): return len(self.tokens)
    def __iter__(self): yield from self.tokens
    def __getitem__(self, item): return self.tokens[item]

class Inventory(object):
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
    _lattice = None
    _score_matrix = {}
    _gap_score = None
    _normalization = {}
    _segmenter = None
    _legal_str = None
    _max = None
    _regexes_end = {}
    _regexes = {}
    _pretty_str = {}
    _features = {}
    _features_str = {}
    _classes = {}

    @classmethod
    def _add_segment(cls, classes, extent, intent, shorthand=None):
        """Constructor for Segments."""
        id = frozenset(extent) if len(extent) > 1 else extent[0]
        ordered = sorted(extent)
        joined = "|".join(ordered)
        if len(extent) == 1:
            cls._regexes[id] = id+" "
            cls._regexes_end[id] = id
            cls._pretty_str[id] = id
        else:
            # The non capturing group of each segment
            cls._regexes[id] = "(?:" + "|".join(x+" " for x in ordered) + ")"
            cls._regexes_end[id] = "(?:" + "|".join(x for x in ordered) + ")"
            cls._pretty_str[id] = "{" + ",".join(ordered)+"}" #TODO: change to "{a,b,c}"
        cls._classes[id] = set(classes)
        cls._features[id] = set(intent)
        cls._features_str[id] = shorthand or "[{}]".format(" ".join(sorted(intent)))

    @classmethod
    def regex(cls, sound, end=False):
        if end:
            return cls._regexes_end[sound]
        return cls._regexes[sound]

    @classmethod
    def pretty_str(cls, sound, **kwargs):
        return cls._pretty_str[sound]

    @classmethod
    def features(cls, sound, **kwargs):
        return cls._features[sound]

    @classmethod
    def features_str(cls, sound, **kwargs):
        return cls._features_str[sound]

    @classmethod
    def shortest(cls, sound, **kwargs):
        return min((cls.pretty_str(sound), cls.features_str(sound)), key=len)

    @classmethod
    def is_leaf(cls, sound):
        return (type(sound) is str)

    @classmethod
    def infos(cls, sound):
        return cls.pretty_str(sound) +" = "+cls._features_str[sound]

    @classmethod
    def inf(cls, a, b):
        """ Checks if a is a descendant of b.

        a < b ff b has children and either a is a string
        which is part of b, or a is a subset of b.
        """
        return (not cls.is_leaf(b)) and ((a in b) or (not cls.is_leaf(a) and a < b))

    @classmethod
    def similarity(cls, a, b):
        """Compute phonological similarity  (Frisch, 2004)

        Measure from "Similarity avoidance and the OCP" , Frisch, S. A.; Pierrehumbert, J. B. & Broe,
        M. B. *Natural Language \& Linguistic Theory*, Springer, 2004, 22, 179-228, p. 198.

        We compute similarity by comparing the number of shared and unshared natural classes
        of two consonants, using the equation in (7). This equation is a direct extension
        of the Pierrehumbert (1993) feature similarity metric to the case of natural classes.

        (7) :math:`Similarity = \\frac{\\text{Shared natural classes}}{\\text{Shared natural classes } + \\text{Non-shared natural classes}}`
        """
        if a == b: return 1
        ca = cls._classes[a]
        cb = cls._classes[b]
        return len(ca & cb) / len(ca | cb)

    ####
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

                cls._add_segment(classes, extent, intent, shorthand=shorthand)

        not_actual_leaves = []
        for leaf in leaves:
            lattice_node = frozenset(cls._lattice[(leaf,)].extent)
            if len(lattice_node) > 1:
                not_actual_leaves.append((leaf, lattice_node))

        if not_actual_leaves:
            alert = ""
            for leaf, lattice_node in not_actual_leaves:
                other = set(lattice_node) - {leaf}
                alert += "\n\t" + leaf + " is the same node as " + str(lattice_node)
                alert += "\n\t\t" + cls.infos(lattice_node)
                for o in other:
                    alert += "\n\t\t" +  cls.infos(o)

            raise Exception("Warning, some segments are " # TODO: change doc!!!
                            "ancestors of other segments:" + alert)

        cls.max = max(cls._classes, key=len)

        simple_sounds = [s for s in cls._classes if cls.is_leaf(s)]
        all_sounds = sorted(simple_sounds + list(cls._normalization),
                            key=len, reverse=True)
        cls._segmenter = re.compile("(" + "|".join(all_sounds) + ")")
        cls._legal_str = re.compile("(" + "|".join(all_sounds) + ")+")

    @classmethod
    def init_dissimilarity_matrix(cls, gap_prop=0.24, **kwargs):
        """Compute score matrix with dissimilarity scores."""
        # TODO: should this be delegated to morphalign ?
        # TODO: should this code all on integers ?
        costs = []
        simple_sounds = [s for s in cls._classes if cls.is_leaf(s)]
        for a, b in combinations(simple_sounds, 2):
            cost = 1 - cls.similarity(a, b)
            cls._score_matrix[(a, b)] = cls._score_matrix[(b, a)] = cost
            costs.append(cost)

        cls._gap_score = np.quantile(np.array(costs), 0.5) * gap_prop # TODO: Gap score might need to be different...
        for a in simple_sounds:
            cls._score_matrix[(a, a)] = 0

    @classmethod
    def insert_cost(cls, *_):
        return cls._gap_score

    @classmethod
    def sub_cost(self, a, b):
        return self._score_matrix[(a, b)]

    @classmethod
    def get(cls, descriptor):
        """Get a sound using the lattice."""
        try:
            s = cls._lattice[descriptor].extent
            if len(s) == 1:
                return s[0]
            return frozenset(s)
        except:
            print(descriptor)
            print(cls._lattice[descriptor])
            raise

    @classmethod
    def meet(cls, *args):
        """Intersect some segments from their names.
        This is the "meet" operation on the lattice nodes, and returns the lowest common ancestor.

        Returns:
            lowest common ancestor ID
        """
        segments = set()
        for segment in args:
            if cls.is_leaf(segment):
                segments.add(segment)
            else:
                segments |= segment
        return cls.get(segments)

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

            >>> a,b = Inventory.transformation("t","s")
            >>> print(a,b)
            [bdpt] [fsvz]

        Arguments:
            a,b (str): Segment aliases.

        Returns:
            two charclasses.

        """

        def select_if_reciprocal(cls, segs, left, right):
            tmp = []
            if cls.is_leaf(segs):
                segs = {segs}
            for x in segs:
                try:
                    y = cls.get((cls.features(x) - left) | right)
                    if y and len(y) == 1: # TODO: check if this is still correct
                        x_back = cls.get((y.features - right) | left)
                        if x == x_back.ipa:
                            tmp.append(x)
                except:
                    pass
            return frozenset(tmp) # TODO: warning: this need not be a lattice node

        left, right = cls.get_transform_features(a, b)
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

        t1 = cls.features(left)
        t2 = cls.features(right)
        f1 = t1 - t2
        f2 = t2 - t1
        return f1, f2

    @classmethod
    def get_from_transform(cls, a, transform):
        """ Get a segment from another according to a transformation tuple.

        In the following example, the segments have been initialized with French segment definitions.

        Arguments:
            a (str): Segment alias
            transform (tuple): Couple of two segment IDs

        Example:
            >>> segments.Inventory.get_from_transform("d",
            ...                                     (frozenset({"b","d","p","t"}),
            ...                                     frozenset({"f","s","v","z"})))
            'z'
        """
        a = cls.features(a)
        f1, f2 = cls.get_transform_features(*transform)
        return cls.get((a - f1) | f2)

    @classmethod
    def show_pool(cls):
        """Return a string description of the whole segment pool."""
        return "\n".join([cls.infos(seg)
                          for seg in sorted(cls._classes, key=lambda x: len(x))])


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
