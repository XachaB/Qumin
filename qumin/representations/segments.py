# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Sacha Beniamine.

This module addresses the modelisation of phonological segments.
"""

import pandas as pd
from ..lattice.lattice import table_to_context
import numpy as np
from itertools import combinations
import re
from ..utils import snif_separator
import functools
import logging
log = logging.getLogger()


inventory = None

class Form(str):
    """ A form is a string of sounds, separated by spaces.

    Sounds might be more than one character long.
    Forms are strings, they are segmented at the object creation.
    They have a tokens attribute, which is a tuple of phonemes.
    """
    def __new__(cls, string):
        if Inventory._legal_str.fullmatch(string) is None:
            raise ValueError("Unknown sound in: " + repr(string))
        tokens = Inventory._segmenter.findall(string)
        tokens = tuple(Inventory._normalization.get(c, c) for c in tokens)
        self = str.__new__(cls, " ".join(tokens) + " ")
        self.tokens = tokens
        return self

    @classmethod
    def from_segmented_str(cls, segmented):
        stripped = segmented.strip(" ")
        self = str.__new__(cls, stripped+" ")
        self.tokens = stripped.split()
        return self

    def __repr__(self):
        return "Form("+self+")"

class Inventory(object):
    """The static `segments.Inventory` class describes a sound inventory.

    This class is static, so that all other modules can access its current state,
    without passing an inventory instance everywhere.

    The inventory first needs to be initialized with a distinctive features file.

    Each sound class in the inventory is a concept in a FCA lattice.
    Sound class identifiers are either strings (for phonemes)
    or frozensets (for sound classes). Phonemes are the leaves of the hierarchy.

    Sound classes can be seen as under-determined phonemes, and both phonemes and sound
    classes are handled in the same way. For this reason, we call both "sound".

    Static Attributes:
        _lattice: the FCA lattice underlying the feature space
        _score_matrix (dict): a dictionnary of sound tuples to alignment score
        _gap_score (float): a score for insertions
        _normalization (dict): a dictionnary of sounds to their normalized counterparts
        _segmenter (re.pattern): a compiled regex to segment words into phonemes
        _legal_str (re.pattern): a compiled regex to recognize words made of known phonemes
        _max (frozenset): the identifier of the supremum in the lattice
        _regexes_end (dict): a dictionnary of sound IDs to regex strings (to use at the end of words) -- currently unused
        _regexes (dict): a dictionnary of sound IDs to regex strings
        _pretty_str (dict): a dictionnary of sound IDs to pretty formatted strings
        _features (dict): a dictionnary of sound IDs to set of features
        _features_str (dict): a dictionnary of sound IDs to a string representing features
        _classes (dict): a dictionnary of sound IDs to a list of classes (ancestors)

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
        """ Adds a single lattice concept to the inventory.

        A concept is a sound class. If there is a single sound in the class,
        the node represents that specific phoneme.

        Args:
            classes: list of ancestors for this concept
            extent: list of phonemes in the sound class
            intent: list of features for this concept
            shorthand: short string expressing the value of this node
        """
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
        """ Returns a regex representing a sound.

        Args:
            sound: identifier of a sound
            end: whether this regex is at the end of a word
                TODO: this is intended to later get rid of trailing spaces in Form

        Returns:
            (str): regex string
        """
        if end:
            return cls._regexes_end[sound]
        return cls._regexes[sound]

    @classmethod
    def pretty_str(cls, sound, **kwargs):
        """ Returns a pretty string representing a sound.

        Args:
            sound: identifier of a sound

        Returns:
            (str): pretty string
        """
        return cls._pretty_str[sound]

    @classmethod
    def features(cls, sound, **kwargs):
        """ Returns a set of features representing a sound.

        Args:
            sound: identifier of a sound

        Returns:
            (set): features
        """
        return cls._features[sound]

    @classmethod
    def features_str(cls, sound, **kwargs):
        """ Returns a string which described the features of a sound.

        Args:
            sound: identifier of a sound

        Returns:
            (str): features string
        """
        return cls._features_str[sound]

    @classmethod
    def shortest(cls, sound, **kwargs):
        """ Returns a string which describes the sound in as little characters as possible.

        Args:
            sound: identifier of a sound

        Returns:
            (str): short string
        """
        return min((cls.pretty_str(sound), cls.features_str(sound)), key=len)

    @classmethod
    def is_leaf(cls, sound):
        """ Returns whether this sound is a leaf (a phoneme, rather than a sound class)

        Args:
            sound: identifier of a sound

        Returns:

        """
        return (type(sound) is str)

    @classmethod
    def infos(cls, sound):
        """ String giving all useful information on a sound.

        Args:
            sound: identifier of a sound

        Returns:
            pretty string and features of a sound.

        """
        return cls.pretty_str(sound) +" = "+cls._features_str[sound]

    @classmethod
    def inf(cls, a, b):
        """ Checks if a is a descendant of b.

        a < b iff b has children and either a is a string
        which is part of b, or a is a subset of b.
        """
        return (not cls.is_leaf(b)) and ((a in b) or (not cls.is_leaf(a) and a < b))

    @classmethod
    def similarity(cls, a, b):
        """Computes phonological similarity  (Frisch, 2004)

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

    @classmethod
    def initialize(cls, filename, sep=None):
        """ Initializes the inventory

        Args:
            filename: path to a csv or tsv file with distinctive features
            sep: separator in the file
        """
        # TODO: this is now much slower !
        log.info("Reading table %s",filename)

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

        log.info("Normalizing identical rows")
        attributes = list(table.columns)
        cls._normalization = normalize(table, attributes)
        table.set_index("Normalized", inplace=True)
        table.drop_duplicates(inplace=True)

        log.debug("Normalization map: %s", cls._normalization)

        def feature_formatter(columns):
            signs = ["-", "+"] + [str(x) for x in range(2, 11)]
            for c in columns:
                key, val = c.split("=")
                yield signs[int(float(val))] + key.replace(" ", "_")

        table = table.applymap(lambda x: str(x))

        context = table_to_context(table, na_value="-1", col_formatter=feature_formatter)
        cls._lattice = context.lattice

        if shorthands is not None:
            shorthand_context = table_to_context(shorthands, na_value="-1",
                                                col_formatter=feature_formatter)

            stack = shorthands.index.tolist()
            shorthands = {}

            for e, i in shorthand_context.lattice:
                full_intent = cls._lattice[i].intent
                for sh in e:
                    if sh in stack:
                        shorthand_name = sh.strip("#")
                        if shorthand_name:
                            shorthands[full_intent] = sh.strip("#")
                            stack.remove(sh)
                if not stack:
                    break

        else:
            shorthands = {}

        for extent, intent in cls._lattice:

            if extent:
                # Define the shortest expression of this segment if possible
                shorthand = shorthands.get(intent, None)
                if len(intent) == 0:
                    shorthand = "X"
                elif shorthand is None and len(extent) > 1:
                    minimals = next(cls._lattice[extent].attributes())
                    if minimals:
                        shorthand = "[{}]".format(" ".join(minimals))

                concept = cls._lattice[extent]
                ancestors = ["|".join(sorted(c.extent)) for c in concept.upset()]
                classes = sorted(ancestors, key=len)
                cls._add_segment(classes, extent, intent, shorthand=shorthand)

        not_actual_leaves = []
        for leaf in table.index:
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

        cls._max = max(cls._classes, key=len)

        simple_sounds = [s for s in cls._classes if cls.is_leaf(s)]
        all_sounds = sorted(simple_sounds + list(cls._normalization),
                            key=len, reverse=True)
        cls._segmenter = re.compile("(" + "|".join(all_sounds) + ")")
        cls._legal_str = re.compile("(" + "|".join(all_sounds) + ")+")

    @classmethod
    def init_dissimilarity_matrix(cls, gap_prop=0.5, **kwargs):
        """Computes score matrix with dissimilarity scores."""
        # TODO: should this be delegated to morphalign ?
        # TODO: should this code all on integers ?
        costs = []
        simple_sounds = [s for s in cls._classes if cls.is_leaf(s)]
        for a, b in combinations(simple_sounds, 2):
            cost = 1 - cls.similarity(a, b)
            cls._score_matrix[(a, b)] = cls._score_matrix[(b, a)] = cost
            costs.append(cost)

        cls._gap_score = np.quantile(np.array(costs), 0.5) * gap_prop # TODO: update gap score
        for a in simple_sounds:
            cls._score_matrix[(a, a)] = 0

    @classmethod
    def insert_cost(cls, *_):
        """Returns the constant insertion/deletion cost"""
        return cls._gap_score

    @classmethod
    def sub_cost(self, a, b):
        """ Returns the cost of aligning sounds `a` and `b`

        Args:
            a: sound identifier
            b: sound identifier

        Returns: (float): substitution cost

        """
        return self._score_matrix[(a, b)]

    @classmethod
    @functools.lru_cache(maxsize=None)
    def get(cls, descriptor):
        """ Get a sound using the lattice.

        Args:
            descriptor: iterable of phonemes OR iterable of features

        Returns: (str or frozenset) sound identifier

        """
        try:
            s = cls._lattice[descriptor].extent
            if len(s) == 1:
                return s[0]
            return frozenset(s)
        except KeyError:
            raise ValueError("Unknown sound descriptor: "+repr(descriptor))

    @classmethod
    def meet(cls, *args):
        """Finds the lowest common ancestors of segments from their identifiers.

        Args: several sound identifiers

        Returns:
            lowest common ancestor identifier
        """
        segments = set()
        for segment in args:
            if cls.is_leaf(segment):
                segments.add(segment)
            else:
                segments |= segment
        return cls.get(frozenset(segments))

    @classmethod
    @functools.lru_cache(maxsize=None)
    def transformation(cls, a, b):
        """Find a transformation between a and b.

        The transformation is a pair of two maximal sets of segments related by a bijective phonological function.

        This function takes a pair of sound identifiers and calculates the function which relates
        these two segments. It then finds and returns the two maximal sets of segments related by this function.

        Example:
            In French, t -> s can be expressed by a phonological function
            which changes [-cont] and [-rel. ret] to [+cont] and [+rel. ret]

            These other segments are related by the same change:
            d -> z
            b -> v
            p -> f

            >>> a,b = Inventory.transformation("t","s")
            >>> print(a,b)
            {"b","d","p","t"} {"f","s","v","z"}

        Arguments:
            a,b (str): Segment identifiers.

        Returns:
            (tuple of frozensets): two sets of sounds.
        """

        def select_if_reciprocal(cls, segs, left, right):
            tmp = []
            for x in cls.id_to_frozenset(segs):
                y = cls.get(frozenset((cls.features(x) - left) | right))
                if y and type(y) is str:
                    x_back = cls.get(frozenset((cls.features(y) - right) | left))
                    if x == x_back:
                        tmp.append(x)
            return frozenset(tmp) # TODO: warning: this need not be a lattice node

        left, right = cls.get_transform_features(a, b)
        A, B = cls.get(left), cls.get(right)
        A = select_if_reciprocal(cls, A, left, right)
        B = select_if_reciprocal(cls, B, right, left)
        return A, B

    @classmethod
    def id_to_frozenset(cls, sound_id):
        if cls.is_leaf(sound_id):
            return frozenset({sound_id})
        return sound_id

    @classmethod
    def get_transform_features(cls, left, right):
        """ Get the features corresponding to a transformation.

        Arguments:
            left (frozenset): set of phonemes
            right (frozenset): set of phonemes

        Example:
            >>> inventory.get_from_transform({"b","d"}, {"p","t"})
            frozenset({'+vois'}), frozenset({'-vois'})
        """

        t1 = cls.features(cls.get(cls.id_to_frozenset(left)))
        t2 = cls.features(cls.get(cls.id_to_frozenset(right)))
        f1 = t1 - t2
        f2 = t2 - t1
        return frozenset(f1), frozenset(f2)

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
        return cls.get(frozenset((a - f1) | f2))

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
