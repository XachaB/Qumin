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


class _CharClass(str):
    """A `_CharClass` is a `str` with sorted chars and brackets.

    Charclasses have a: attr:`Segments._CharClass.REGEX` constant field.
    This class is used for segments names. This way we get:

    * An immutable class.
    * Enforced sorted characters.
    * Identical to a corresponding string.
    * Which __str__() returns a ready-to-use regex character class string.
    * Iterators go through the content (without the brakets)

    Attributes:
        REGEX (str): constant. The string wrapped in "[]"
    """

    def __new__(cls, content):
        """Using new, not init, because _CharClass, as str, is immutable.

        This ensures sorted string.
        """
        content = "".join(sorted(content))
        self = str.__new__(cls, content)
        self.REGEX = "[{!s}]".format(content)
        return self

    def __str__(self):
        """Return the string surrounded by '[]', regex char class style."""
        return self.REGEX


class Segment(object):
    """The `Segments.Segment` class holds the definition of a single segment.

    This is a lightweight class.

    Attributes:
        name (str or _CharClass): Name of the segment.
        features (frozenset of tuples):
            The tuples are of the form `(attribute, value)`
            with a positive value, used for set operations.
    """
    _pool = {}
    _simple_segments = []
    _normalization = {}
    _aliases = {}
    _lattice = None
    _score_matrix = {}
    _gap_score = None

    def __new__(cls, classes, features, alias, chars, shorthand=None):
        obj = cls._pool.get(alias, None)
        if not obj:
            obj = object.__new__(cls)
            cls._pool[alias] = obj
        return obj

    def __init__(self, classes, features, alias, chars, shorthand=None):
        """Constructor for Segments."""
        self.ipa = chars
        self.alias = alias
        self.classes = set(classes)
        self.features = features
        if shorthand:
            self.shorthand = shorthand
        else:
            self.shorthand = "[{}]".format(" ".join(self.features))

    def __lt__(self, other):
        """ Checks if self is a descendant of other.

        X is a descendant of Y if Y is in X's ancestor list.
        """
        return other.alias in self.classes

    def __le__(self, other):
        return (self.alias == other.alias) or (self < other)

    @classmethod
    def init_dissimilarity_matrix(cls, gap_prop=0.49, **kwargs):
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
    def sub_cost(cls, a, b):
        return cls._score_matrix[(a, b)]

    @classmethod
    def set_max(cls):
        """Set a variable to the top of the natural classes lattice."""
        cls._max = max(cls._pool, key=len)

    @classmethod
    def _reinitialize(cls):
        cls._pool.clear()
        del cls._simple_segments[:]
        cls._normalization.clear()
        cls._aliases.clear()
        cls._score_matrix.clear()
        cls._lattice = None
        cls._gap_score = None

    @classmethod
    def intersect(cls, *args):
        """Intersect some segments from their names/aliases.
        This is the "meet" operation on the lattice nodes, and returns the lowest common ancestor.

        Returns:
            a str or _CharClass representing the segment which classes are the intersection of the input.
        """
        # print("  intersecting :",args," to :",functools.reduce(lambda x, y: x & y, (cls.get(x) for x in args if x)).alias)
        return cls.lattice["".join(args)].alias

    @classmethod
    def get(cls, descriptor):
        """Get a Segment from an alias."""
        try:
            #  Simple case: the descriptor is a  known alias
            return cls._pool[descriptor]
        except:
            try:
                #  Alternate case: use lattice to recover segment
                return cls._pool[cls.lattice[descriptor].alias]
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
            tmp = ""
            for x in segs.alias:
                try:
                    y = cls.get((cls.get(x).features - left) | right)
                    if y:
                        x_back = cls.get((y.features - right) | left)
                        # print("x :",x,"y:",y,"x_back",x_back.alias)
                        if x == x_back.alias:
                            tmp += x
                except:
                    pass
            return _CharClass(tmp)

        a = cls.get(a).features
        b = cls.get(b).features
        left = a - b
        right = b - a
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
            >>> Segment.get_from_transform("bd", "pt")
            {'+vois'}, {'-vois'}
        """

        t1 = set.intersection(*[cls.get(x).features for x in left])
        t2 = set.intersection(*[cls.get(x).features for x in right])
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
            >>> Segment.get_from_transform("d",("bdpt", "fsvz"))
            'z'
        """
        a = cls.get(a).features
        f1, f2 = cls.get_transform_features(*transform)
        return cls.get((a - f1) | f2).alias

    @classmethod
    def show_pool(cls, only_single=False):
        """Return a string description of the whole segment pool."""
        if not only_single:
            return "\n".join([repr(cls._pool[seg]) for seg in sorted(cls._pool, key=lambda x: len(x))])
        else:
            return "\n".join([repr(cls._pool[seg]) for seg in sorted(cls._pool, key=lambda x: len(x)) if len(seg) == 1])

    @functools.lru_cache(maxsize=None)
    def similarity(self, other):
        """Compute phonological similarity  (Frisch, 2004)

        The function is memoized. Measure from "Similarity avoidance and the OCP" , Frisch, S. A.; Pierrehumbert, J. B. & Broe,
        M. B. *Natural Language \& Linguistic Theory*, Springer, 2004, 22, 179-228, p. 198.


            We compute similarity by comparing the number of shared and unshared natural classes
            of two consonants, using the equation in (7). This equation is a direct extension
            of the Pierrehumbert (1993) feature similarity metric to the case of natural classes.

            (7) :math:`Similarity = \\frac{\\text{Shared natural classes}}{\\text{Shared natural classes } + \\text{Non-shared natural classes}}`


        """
        if self == other:
            return 1
        return len(self.classes & other.classes) / len(self.classes | other.classes)

    def __repr__(self):
        r"""Return the representation of one segment.

        Example:

            >>> a = [+syl, +rel.ret., -haut, +arr, -cons, +son, +vois,\
            ...      -rond, +cont, +bas, -nas, -ant]

        """
        if str(self.ipa) != str(self.alias):
            return "{} ({}) = {}".format(self.ipa, self.alias, self.shorthand)
        return "{} = {}".format(self.alias, self.shorthand)

    def __str__(self):
        """Return the segment's name's str.

        Example:

            "a" # Simple segment
            "[iEøâô]". # Complex segment
        """
        return str(self.alias)


def make_aliases(ipa):
    """Associate one symbol to segments that take two characters. Return restoration map.

    This function takes a segments table and changes the entries of the "Segs." column
    with a unique character for each multi-chars cell in the Segs.
    A dict is returned that allows for original segment name restoration.

    ============= ==============
    Input Segs.    Output Segs.
    ============= ==============
    ɑ̃                â
    a                a
    ============= ==============

    The table can have an optional UNICODE column.
    It will be dropped at the end of the process.

    Arguments:
        ipa (:class:`pandas:pandas.DataFrame`): Dataframe of segments.
            Columns are features and indexes are segments. A UNICODE col can specify alt chars.
    Returns:
        alias_map (dict):
            maps from the simplified name to the original segments name.
    """
    reserved = ".^$*+?{}[]/|()<>_ ⇌,;"  # these characters shouldn't be names for segments.

    def alias(row, all_segments, dict_confusables, alias_map):
        """alias for a segment's name.

        Arguments:
            row (:class:`pandas.core.series.Series`):
                Serie of two elements: a segment name and an UNICODE value.
            all_segments:
                set of all existing segments names.
            alias_map (dict):
                maps from the simplified name to the original segments name.
            dict_confusables (dict):
                dictionnary providing mapping to similar characters.
        """
        segment, alias, code = row
        if segment in reserved:
            raise ValueError(
                "The characters " + reserved + " are reserved. Please choose other representations for the segments. Occured at: " + str(
                    segment) + " " + str(code))
        if len(segment) == 1:
            return segment
        else:
            alt = []
            if alias and not pd.isna(alias):
                alt = [alias]
            if code and str(code).isdigit():
                i = int(code)
                if 0 < i < 1114111:
                    alt.append(chr(int(code)))

            #  Compose segments that are composable, normalize otherwise
            normalized = unicodedata.normalize("NFKC", segment)
            l = len(normalized)
            s = normalized[0]
            if l == 1:
                alt.append(normalized)
            elif normalized[0] in "ˈˌˑ˘":
                s = normalized[1]

            # Ressembling segment
            alt.extend(dict_confusables[segment.lower()])

            if l < len(segment):
                alt.extend(list(normalized))

            # Segment ressembling  one similar to the first char
            alt.extend(dict_confusables[s.lower()])

            # Just the first char, lower or upper
            alt.extend([s.lower(), s.upper()])

            # Numeric fallback
            alt.extend(str(i) for i in range(9))
            for seg in alt:
                if seg not in all_segments and seg not in reserved:
                    all_segments.add(seg)
                    alias_map[seg] = segment
                    # print("I chose ",seg," as alias for ",segment)
                    return seg
            raise ValueError(f"I can not guess a good one-char alias for {segment}, please use an ALIAS column to provide one.")

    from representations import confusables
    from os.path import dirname

    dict_confusables = confusables.parse(dirname(__file__) + "/confusables.txt")
    alias_map = {}
    all_segments = set(ipa["Seg."])
    if "UNICODE" not in ipa.columns:
        ipa["UNICODE"] = ""
    if "ALIAS" not in ipa.columns:
        ipa["ALIAS"] = ""

    ipa["Seg."] = ipa[["Seg.", "ALIAS", "UNICODE"]].apply(alias,
                                                          args=(all_segments, dict_confusables, alias_map),
                                                          axis=1)

    ipa.drop("UNICODE", axis=1, inplace=True)
    ipa.drop("ALIAS", axis=1, inplace=True)

    return alias_map


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
                raise ValueError("You have more than one segment definition for {}\n{}".format(segment, seg_features))
        return same_features_as_seg

    ipa["Normalized"] = ""

    for segment in ipa[features].drop_duplicates().index:
        same_features_as_seg = find_indentical_rows(segment, ipa[features])

        if (ipa.loc[same_features_as_seg, "Normalized"] == "").all():
            ipa.loc[same_features_as_seg, "Normalized"] = segment

    norm_map = str.maketrans({seg: norm
                              for seg, norm
                              in zip(ipa.index, ipa["Normalized"])
                              if seg != norm})
    return norm_map


def restore(char):
    """Restore the original string from an alias."""
    if char in Segment._pool:
        return Segment.get(char).ipa
    else:
        return str(char)


def restore_string(string):
    """Restore the original string from a string of aliases."""
    return "".join(restore(char) for char in string)


def restore_segment_shortest(segment):
    """Restore segment to the shortest of either the original character or its feature list."""
    if segment:
        return min([restore(segment), Segment.get(segment).shorthand], key=len)
    else:
        return segment

def _merge_duplicate_cols(dataframe):
    """Merge any duplicate columns in the dataframe."""

    def complementaire(values):
        a, b = values
        return (a == b == -1) or (a == (not b))

    agenda = set([c for c in dataframe.columns if c not in ["Seg.", "UNICODE", "ALIAS"]])
    changed = defaultdict(set)
    while agenda:
        col = agenda.pop()
        agenda -= {col}
        for col2 in agenda:
            identical = (dataframe[col] == dataframe[col2]).all()
            compl = dataframe[[col, col2]].apply(complementaire, axis=1).all()
            if identical or compl:
                changed[col].add(col2)
                dataframe.drop(col2, axis=1, inplace=True)

        agenda -= changed[col]

    for col in dataframe.columns:
        if changed[col]:
            print("Merged: ", col, changed[col])


def initialize(filename, sep="\t", verbose=False):
    Segment._reinitialize()
    print("Reading table")
    with open(filename, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if "Seg.," in first_line or '"Seg.",' in first_line:
            sep = ","
        elif "Seg.\t" in first_line or '"Seg."\t' in first_line:
            sep = "\t"
    table = pd.read_table(filename, header=0,
                          index_col=False, sep=sep, encoding="utf-8")

    if "Seg." in list(table.iloc[0]):
        table.columns = table.iloc[0]
        table.drop(0, axis=0, inplace=True)
    else:
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

    na_vals = {c:-1 for c in table.columns}
    na_vals["Seg."] = ""
    na_vals["UNICODE"] = ""
    na_vals["ALIAS"] = ""
    na_vals["value"] = ""
    table = table.fillna(na_vals)
    
    #  Checking segments names legality
    for col in table["Seg."]:
        if col == "":
            raise ValueError("One of your segments doesn't have a name !")
        if col.strip("#") == "":
            raise ValueError(
                "The symbol \"#\" is reserved and can only be used in a shorthand name (#V# for a vowel, etc)")

    if "value" in table.columns:
        table.drop("value", axis=1, inplace=True)

    _merge_duplicate_cols(table)

    shorthand_selection = (table["Seg."].str.endswith("#") & table["Seg."].str.startswith("#"))

    shorthands = None
    if shorthand_selection.any():
        shorthands = table[shorthand_selection]
        table = table[~shorthand_selection]
        shorthands.set_index("Seg.", inplace=True)
        if "UNICODE" in shorthands.columns:
            shorthands.drop("UNICODE", axis=1, inplace=True)
        if "ALIAS" in shorthands.columns:
            shorthands.drop("ALIAS", axis=1, inplace=True)
        shorthands = shorthands.applymap(str)

    print("Aliasing multi-chars segments")
    aliases = make_aliases(table)
    Segment._aliases.update({aliases[x]: x for x in aliases})  #  Need the opposite mapping to make_aliases dataframes
    table.set_index("Seg.", inplace=True)
    if verbose:
        print("Aliases map: ", {x: aliases[x] for x in aliases if aliases[x] != x})

    attributes = list(table.columns)

    print("Normalizing identical segments")
    normalization = normalize(table, attributes)
    Segment._normalization.update(normalization)

    if verbose:
        print("Normalization map: ", {chr(x): normalization[x] for x in normalization})
    table.set_index("Normalized", inplace=True)
    table.drop_duplicates(inplace=True)

    def formatter(columns):
        signs = ["-", "+"] + [str(x) for x in range(2, 11)]
        for c in columns:
            key, val = c.split("=")
            yield signs[int(val)] + key.replace(" ", "_")

    leaves = {t: [] for t in table.index}
    table = table.applymap(lambda x: str(x))

    lattice = ICLattice(table, leaves,
                        na_value="-1",
                        col_formatter=formatter,
                        verbose=False)

    Segment.lattice = lattice.lattice

    if shorthands is not None:
        shorthand_lattice = ICLattice(shorthands, {t: [] for t in shorthands.index},
                                      na_value="-1",
                                      col_formatter=formatter,
                                      verbose=False)

        shorthands = {lattice.lattice[i].intent: e[0].strip("#") for e, i in shorthand_lattice.lattice if
                      e and len(e) == 1}
        shorthands[()] = "X"
    else:
        shorthands = {(): "X"}

    for extent, intent in lattice.lattice:

        if extent:
            alias = "".join(sorted(extent))
            shorthand = "[{}]".format(" ".join(intent))
            if intent in shorthands:
                shorthand = shorthands[intent]
            elif len(extent) > 1:
                lower = set().union(*(set(x.intent) for x in lattice.lattice[extent].upper_neighbors))
                minimal = set(intent) - lower
                if minimal:
                    shorthand = "[{}]".format(" ".join(minimal))

            ancestors = lattice.ancestors(extent)
            classes = sorted(["".join(sorted(ancestor.extent)) for ancestor in ancestors] + [extent], key=len)
            features = set(intent)

            if len(alias) == 1:
                ipa = aliases.get(alias, alias)
                Segment._simple_segments.append(alias)
            else:
                alias = _CharClass(alias)
                ipa = "[{}]".format("-".join(aliases.get(a, a) for a in alias))
            # print("Adding : {} ({}) = {}\n\t{}".format(ipa,alias,lattice.vertex[seg_set],classes))
            Segment(classes, features, alias, ipa, shorthand=shorthand)
            lattice.lattice[extent].alias = alias

    not_actual_leaves = []
    for leaf in leaves:
        lattice_node = "".join(sorted(lattice.lattice[leaf].extent))
        if leaf != lattice_node:
            not_actual_leaves.append((leaf, lattice_node))
            # Segment._pool[leaf] = Segment._pool[lattice_node]

    # lattice.to_html("test.html")
    if not_actual_leaves:
        alert = ""
        for leaf, lattice_node in not_actual_leaves:
            other = "".join((set(lattice_node) - {leaf}))
            leaf_alias = aliases.get(leaf, leaf)

            alert += "\n\t" + leaf_alias + " is the same node as " + Segment.get(lattice_node).ipa
            alert += "\n\t\t" + repr(Segment.get(lattice_node))
            for o in other:
                alert += "\n\t\t" + repr(Segment.get(o))

        raise Exception("Warning, some of the segments aren't actual leaves :" + alert)

    Segment.set_max()
