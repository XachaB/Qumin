# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Sacha Beniamine.

This module addresses the modelisation of phonological segments.
"""

import pandas as pd
import numpy as np
from itertools import combinations
import functools
import re
from tqdm import tqdm

from ..lattice.lattice import table_to_context
import logging
log = logging.getLogger("Qumin")

inventory = None

_to_short_feature = {'anterior': 'ant', 'approximant': 'appr', 'back': 'back', 'click': 'click', 'consonantal': 'C',
                     'constr gl': 'cgl', 'constricted': 'constr', 'constricted glottis': 'cgl', 'continuant': 'cont',
                     'coronal': 'coro', 'delayed release': 'del.rel', 'distributed': 'dist', 'dorsal': 'dors',
                     'front': 'front', 'high': 'high', 'labial': 'lab', 'laryngeal': 'laryng', 'lateral': 'lat',
                     'long': 'long', 'low': 'low', 'nasal': 'nas', 'pharyngeal': 'phar', 'place': 'place',
                     'preaspirated': 'preasp', 'preglottalized': 'pregl', 'prenasal': 'prenas', 'round': 'round',
                     'sibilant': 'sib', 'sonorant': 'son', 'spread': 'spread', 'spread gl': 'spre.gl',
                     'spread glottis': 'sg', 'strident': 'stri', 'syllabic': 'syll', 'tap': 'tap', 'tense': 'tens',
                     'voice': 'voic', 'mid': 'mid', 'central': 'centr', 'compact': 'compact', 'diffuse': 'diff',
                     'abrupt': 'abrupt', 'checked': 'check', 'grave': 'grave', 'acute': 'acute', 'medial': 'med',
                     'flat': 'flat', 'sharp': 'sharp', 'trill': 'tril', 'labiodental': 'labdent'}
_short_features = [y for _, y in _to_short_feature.items()]


class Form(str):
    """ A form is a string of sounds, separated by spaces.
    If a form is provided as defective, this information is still stored
    as a Form object with empty content. Defectiveness can be tested with:

        Form('').is_defective()
        >>> True

    Sounds might be more than one character long.
    Forms are strings, they are segmented at the object creation.

    Attributes:
        tokens (Tuple): Tuple of phonemes contained in this form. For defective entries,
            tokens are an empty tuple.
        id (str): form_id of the corresponding form according to the Paralex package.
            If unknown, `None` will be assigned.
    """

    def __new__(cls, string, form_id=None):
        tokens = Inventory._segmenter.findall(string)
        tokens = tuple(Inventory._normalization.get(c, c) for c in tokens)
        if string == "":
            self = str.__new__(cls, "")
        else:
            if Inventory._legal_str.fullmatch("".join(tokens)) is None:
                raise ValueError("Unknown sound in: " + repr(string))
            self = str.__new__(cls, " ".join(tokens) + " ")
        self.tokens = tokens
        self.id = form_id
        return self

    @classmethod
    def from_segmented_str(cls, segmented):
        stripped = segmented.strip(" ")
        self = cls.__new__(cls, stripped + " ")
        self.tokens = stripped.split()
        return self

    def is_defective(self):
        return self == ''

    def __repr__(self):
        return f"Form({self if self != '' else '#DEF#'}, id='{self.id}')" if hasattr(self, "id") and self.id else f"Form({self})"

    def __str__(self):
        return "".join([x.strip() for x in self.tokens]) if self else '#DEF#'


class Inventory(object):
    """The static `segments.Inventory` class describes a sound inventory.

    This class is static, so that all other modules can access its current state,
    without passing an inventory instance everywhere.

    The inventory first needs to be initialized with a distinctive features file.

    >>> Inventory.initialize("tests/data/frenchipa.csv")

    Each sound class in the inventory is a concept in a FCA lattice.
    Sound class identifiers are either strings (for phonemes)
    or frozensets (for sound classes). Phonemes are the leaves of the hierarchy.

    Sound classes can be seen as under-determined phonemes, and both phonemes and sound
    classes are handled in the same way. For this reason, we call both "sound".

    Attributes:
        _lattice: the FCA lattice underlying the feature space
        _score_matrix (dict): a dictionnary of sound tuples to alignment score
        _gap_score (float): a score for insertions
        _normalization (dict): a dictionnary of sounds to their normalized counterparts
        _segmenter (re.Pattern): a compiled regex to segment words into phonemes
        _legal_str (re.Pattern): a compiled regex to recognize words made of known phonemes
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
        cls._regexes[id] = "(?:" + "|".join(x + " " for x in ordered) + ")"
        cls._regexes_end[id] = "(?:" + "|".join(ordered) + ")"
        if len(extent) == 1:
            cls._pretty_str[id] = id
        else:
            cls._pretty_str[id] = "{" + ",".join(ordered) + "}"
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
        return  cls._regexes[sound]

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
        return cls.pretty_str(sound) + " = " + cls._features_str[sound]

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
        M. B. *Natural Language & Linguistic Theory*, Springer, 2004, 22, 179-228, p. 198.

        We compute similarity by comparing the number of shared and unshared natural classes
        of two consonants, using the equation in (7). This equation is a direct extension
        of the Pierrehumbert (1993) feature similarity metric to the case of natural classes.

        (7) :math:`Similarity = \\frac{\\text{Shared natural classes}}{\\text{Shared natural classes } + \\text{Non-shared natural classes}}`
        """
        if a == b:
            return 1
        ca = cls._classes[a]
        cb = cls._classes[b]
        return len(ca & cb) / len(ca | cb)

    @classmethod
    def initialize(cls, filename):
        """ Initializes the inventory

        Args:
            filename: path to a csv or tsv file with distinctive features
        """
        # TODO: this is now much slower !
        log.info("Reading table %s", filename)

        table = pd.read_table(filename, header=0, dtype=str,
                              index_col=False, sep=',',
                              encoding="utf-8")

        sound_id = "sound_id"
        if sound_id not in table.columns:
            raise ValueError("Paralex sound tables must have a sound_id column.")

        drop = {"value", "UNICODE", "ALIAS", "Seg.",  # Legacy columns
                "label", "tier", "CLTS_id",  # Unused Paralex columns
                }
        deprecated_cols = table.columns.intersection({"value", "UNICODE", "ALIAS", "Seg."})
        if not deprecated_cols.empty:
            log.warning(f"Usage of columns {' ,'.join(deprecated_cols)} is deprecated. Edit your sounds file !")

        for col in drop:
            if col in table.columns:
                table.drop(col, axis=1, inplace=True)

        shorten_feature_names(table)

        table[sound_id] = table[sound_id].astype(str)
        na_vals = {c: "-1" for c in table.columns}
        na_vals[sound_id] = ""
        table = table.fillna(na_vals)

        # Checking segments names legality
        for seg in table[sound_id]:
            if seg == "":
                raise ValueError("One of your segments doesn't have a name !")
            if seg.strip("#") == "":
                raise ValueError("The symbol \"#\" is reserved and can only "
                                 "be used in a shorthand name (#V# for a vowel, etc)")

        # Separate shorthand table
        shorthand_selection = table[sound_id].str.match("^#.+#$")
        shorthands = None
        if shorthand_selection.any():
            shorthands = table[shorthand_selection]
            table = table[~shorthand_selection]
            shorthands.set_index(sound_id, inplace=True)
            shorthands = shorthands.map(str)  # Why is this necessary ?
        table.set_index(sound_id, inplace=True)

        log.info("Normalizing identical rows")
        attributes = list(table.columns)
        cls._normalization = normalize(table, attributes)
        table.set_index("Normalized", inplace=True)
        table.drop_duplicates(inplace=True)

        log.debug("Normalization map: %s", cls._normalization)

        def feature_formatter(columns):
            signs = ["-", "+"]
            for c in columns:
                key, val = c.split("=")
                i = int(float(val))
                if i < 2:
                    yield signs[int(float(val))] + key.replace(" ", "_")
                else:
                    yield c

        table = table.map(lambda x: str(x))

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

        log.info('Building classes of segments...')
        for extent, intent in tqdm(cls._lattice):

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
                    alert += "\n\t\t" + cls.infos(cls.get(o))
            raise Exception("Warning, some segments are  ancestors of other segments:" + alert)

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

        cls._gap_score = np.quantile(np.array(costs), 0.5) * gap_prop
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
            raise ValueError("Unknown sound descriptor: " + repr(descriptor))

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
            >>> a == frozenset({'d', 't', 'b', 'p'})
            True
            >>> b == frozenset({'s', 'z', 'f', 'v'})
            True

        Arguments:
            a,b (str): Segment identifiers.

        Returns:
            tuple of frozenset: two sets of sounds.
        """

        def select_if_reciprocal(cls, segs, left, right):
            tmp = []
            for x in cls.id_to_frozenset(segs):
                y = cls.get(frozenset((cls.features(x) - left) | right))
                if y and type(y) is str:
                    x_back = cls.get(frozenset((cls.features(y) - right) | left))
                    if x == x_back:
                        tmp.append(x)
            return frozenset(tmp)  # TODO: warning: this need not be a lattice node

        left, right = cls.get_transform_features(a, b)
        A, B = cls.get(left), cls.get(right)
        A = select_if_reciprocal(cls, A, left, right)
        B = select_if_reciprocal(cls, B, right, left)
        return A, B

    @classmethod
    def id_to_frozenset(cls, sound_id):
        if cls.is_leaf(sound_id):
            return frozenset({sound_id})
        return frozenset(sound_id)

    @classmethod
    def get_transform_features(cls, left, right):
        """ Get the features corresponding to a transformation.

        Arguments:
            left (frozenset): set of phonemes
            right (frozenset): set of phonemes

        Example:
            >>> Inventory.get_transform_features({"b","d"}, {"p","t"})
            (frozenset({'+voi'}), frozenset({'-voi'}))
        """

        t1 = cls.features(cls.get(cls.id_to_frozenset(left)))
        t2 = cls.features(cls.get(cls.id_to_frozenset(right)))
        f1 = t1 - t2
        f2 = t2 - t1
        return frozenset(f1), frozenset(f2)

    @classmethod
    def get_from_transform(cls, a, transform):
        """ Get a segment from another according to a transformation tuple.

        Arguments:
            a (str): Segment alias
            transform (tuple): Couple of two segment IDs

        Example:
            >>> Inventory.get_from_transform("d",
            ...                                     (frozenset({"d","t"}),
            ...                                     frozenset({"s","z"})))
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
        dict: translation table from
            the segment's name to its normalized name.
    """

    def find_identical_rows(segment, table):
        seg_features = table.loc[segment, :]
        try:
            return (table == seg_features).all(axis=1)
        except ValueError:
            if seg_features.shape[0] > 1:
                raise ValueError("You have multiple definitions for {}\n{}".format(segment,
                                                                                  seg_features))

    ipa["Normalized"] = ""

    for segment in ipa[features].drop_duplicates().index:
        same_features_as_seg = find_identical_rows(segment, ipa[features])

        if (ipa.loc[same_features_as_seg, "Normalized"] == "").all():
            ipa.loc[same_features_as_seg, "Normalized"] = segment

    norm_map = {seg: norm for seg, norm in zip(ipa.index, ipa["Normalized"]) if
                seg != norm}

    return norm_map


def shorten_feature_names(table):
    headers = list(table.iloc[0])
    if "Seg." in headers or "sound_id" in headers:
        raise ValueError("Using a second row of headers is not supported anymore.")
    short_features_names = []
    for name in table.columns:
        if name == "sound_id" or len(name) <= 3:  # Not a feature name
            short_features_names.append(name)
        else:
            if name in _to_short_feature:  # Check standard names
                short_features_names.append(_to_short_feature[name])
            elif name.lower() in _to_short_feature:  # Uppercase
                short_features_names.append(_to_short_feature[name.lower()].upper())
            else:
                # Make an abbreviation on the fly by shortening the label
                names = [name[:i] for i in range(3, len(name) + 1)]
                reserved_names = _short_features + short_features_names
                while names and names[0] in reserved_names:
                    names.pop(0)
                if len(names) != 0:
                    new_name = names[0]
                else:  # Fallback strategy: append a unique integer
                    key = 1
                    new_name = name[:3] + str(key)
                    while new_name in reserved_names:
                        key += 1
                        new_name = name[:3] + str(key)
                short_features_names.append(new_name)
    table.columns = short_features_names
