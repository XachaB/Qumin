# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Classes and functions to make clustering decisions and build inflection class trees
according to distances.

    Still experimental, and unfinished.

Author: Sacha Beniamine
"""

from itertools import combinations, product
from collections import Counter
import numpy as np
import re
from clustering import Node, descriptionlength
from clustering.clusters import _BUClustersBuilder
import warnings
from tqdm import tqdm


def hamming(x, y, table, *args, **kwargs):
    """Compute hamming distances between x and y in table.

    Arguments:
        x (any iterable): vector.
        y (any iterable): vector.
        table (:class:`pandas:pandas.DataFrame`): a dataframe of patterns.

    Returns:
        (int): the hamming distance between x and y.
    """
    return sum([1 for a, b in zip(table.loc[x, :], table.loc[y, :]) if a != b])


def split_description(descriptions):
    """Split each description of a list on spaces to obtain symbols."""
    return [re.split(" +", description) for description in descriptions]


def DL(messages):
    """Compute the description length of a list of messages encoded separately.

    Arguments:
        messages (list): List of lists of symbols. Symbols are str. They are treated as atomic."""
    DL = 0
    for message in messages:
        freq = Counter(message)
        total = sum(freq.values())
        DL += sum(descriptionlength.weighted_log(freq[x], total) for x in freq)
    return DL


def table_to_descr(table, exemplars, microclasses):
    """Create a list of descriptions from a paradigmatic table.

    Arguments:
        table (:class:`pandas:pandas.DataFrame`): a dataframe of patterns.
        exemplars (iterable of str): The microclasses to include in the description.
        microclasses (dict of str:list): mapping of microclasses exemplars to microclasses inventories.
    """
    lexemes_descr = []
    microcl_descr = []

    for e in exemplars:
        microclass = microclasses[e]
        lexemes_descr.extend(microclass)
        microcl_descr.extend(len(microclass) * [e])

    lines = [" ".join(lexemes_descr), " ".join(microcl_descr)]
    # lines = [" ".join(exemplars)]
    for col in table.columns:
        lines.append(" ".join([str(col)] + [table.at[e, col].replace(" ", "") for e in exemplars]))

    return lines


def compression_distance_atomic(x, y, table, microclasses, *args, **kwargs):
    """Compute the compression distances between microclasses x and y from their exemplars.

    Arguments:
        x (str): A microclass exemplar.
        y (str): A microclass exemplar.
        table (:class:`pandas:pandas.DataFrame`): a dataframe of patterns.
        microclasses (dict of str:list): mapping of microclasses exemplars to microclasses inventories.
    """
    warnings.warn("The compression distance is experimental and development is not active. Use at your own risks !")
    left_descr = table_to_descr(table, [x], microclasses)
    right_descr = table_to_descr(table, [y], microclasses)
    merged_descr = table_to_descr(table, [x, y], microclasses)
    a = DL(split_description(left_descr))
    b = DL(split_description(right_descr))
    merged = DL(split_description(merged_descr))
    return compression_distance(a, b, merged)


def compression_distance(a, b, merged):
    """Compute the compression distances between description lengths.

    Arguments:
        a (float): Description length of a cluster.
        b (float): Description length of a cluster.
        merged (float): Description length of the cluster merging both the clusters from a and b.
    """
    if a <= b:
        return (merged - a) / b
    else:
        return (merged - b) / a


def dist_matrix(table, *args, labels=None, distfun=hamming, half=False, default=np.inf, **kwargs):
    """Output a distance score_matrix between clusters.

    Arguments:
        table (:class:`pandas:pandas.DataFrame`): a dataframe of patterns.
        distfun (fun): distance function.
        labels (iterable): the labels between which to compute distance. Defaults to the table's index.
        half (bool): Wether to fill only a half score_matrix.
        default (float): Default distance.

    Returns:
        distances (dict): the similarity score_matrix.
    """
    if labels is None:
        labels = table.index

    distances = {frozenset([x]): {frozenset([y]): default for y in labels} for x in labels}

    for a, b in tqdm(combinations(labels, 2)):
        d = distfun(a, b, table, *args, **kwargs)
        a = frozenset([a])
        b = frozenset([b])
        distances[a][b] = d
        if not half:
            distances[b][a] = d
    return distances


class _DistanceClustersBuilder(_BUClustersBuilder):
    """Builder for top down hierarchical clusters of inflection classes with distances.

    This is an abstract class.
    This class inherits attributes.

    Attributes:
        attr (str): (class attribute) always have the value "DL", as the nodes of the Inflection class tree have a "DL" attribute.
        paradigms (:class:`pandas:pandas.DataFrame`): a dataframe of patterns.
        distances (dict): The distance score_matrix between clusters.
    """
    attr = "dist"

    def __init__(self, microclasses, paradigms, *args, distfun=hamming, **kwargs):
        """Constructor.

        Arguments:
            microclasses (dict of str:list): mapping of microclasses exemplars to microclasses inventories.
            paradigms (:class:`pandas:pandas.DataFrame`): a dataframe of patterns.
            distfun (func): The distance function to use
            kwargs : keyword arguments to be used as configuration.
        """
        super().__init__(microclasses, **kwargs)
        self.paradigms = paradigms
        self.printv("Computing distances...")
        self.distances = dist_matrix(paradigms, microclasses, *args,
                                     labels=list(microclasses), distfun=distfun)

    def update_distances(self, *args):
        raise NotImplementedError("this is an abstract class. Daughters should implement update_distances")

    def merge(self, a, b):
        """Merge two Clusters, build a Node to represent the result, update the distances.

        Parameters:
            a (frozenset): the label of a cluster to merge.
            b (frozenset): the label of a cluster to merge."""
        new = a | b
        d = self.distances[a][b]

        self.printv("\nMerging ", list(a), list(b), "with d ", d)

        self.update_distances(new)

        # Make tree
        left = self.nodes.pop(a)
        right = self.nodes.pop(b)
        leaves = left.labels + right.labels
        size = left.attributes["size"] + right.attributes["size"]
        color = "r"
        d = self.distances[a][b]
        self.nodes[new] = Node(leaves, size=size, children=[left, right],
                               dist=d, color=color,
                               macroclass=False)

    def find_ordered_merges(self):
        """Find the list of all best merges of two clusters.

        The list is a list of tuples of length 3 containing two frozensets representing the
        labels of the clusters to merge and the distance between the two clusters.
        """
        minimum = np.inf
        ordered_merges = []
        for a, b in combinations(self.nodes, 2):
            d = self.distances[a][b]
            if d < minimum:
                minimum = d
                ordered_merges = [(a, b, d)]
            if d == minimum:
                ordered_merges.append((a, b, d))

        return ordered_merges


class UPGMAClustersBuilder(_DistanceClustersBuilder):
    """Builder for UPGMA hierarchical clusters of inflection classes with hamming distance.

    Attributes:
        microclasses (dict of str:list): Inherited. mapping of microclasses exemplars to microclasses inventories.
        nodes (dict of frozenset :Node): Inherited. Maps frozensets of microclass exemplars to Nodes representing clusters.
        preferences (dict): Inherited. Configuration parameters.
        attr (str): Inherited. always have the value "DL", as the nodes of the Inflection class tree have a "DL" attribute.
        microclasses (dict of str:list): Inherited. mapping of microclasses exemplars to microclasses inventories.
        nodes (dict of frozenset :Node): Inherited. Maps frozensets of microclass exemplars to Nodes representing clusters.
        preferences (dict): Inherited. Configuration parameters.
        paradigms (:class:`pandas:pandas.DataFrame`): Inherited. a dataframe of patterns.
        distances (dict): Inherited. The distance score_matrix between clusters.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn("The UPGMA clustering is experimental and development is not active. Use at your own risks !")
        super().__init__(*args, distfun=hamming, **kwargs)

    def update_distances(self, new):
        """UPGMA update for distances.

        Arguments:
            new (frozenset) : Frozenset of microclass exemplar representing the new cluster.
        """
        self.distances[new] = {}
        for cluster in self.nodes:
            pairs = product(cluster, new)
            sumdist = sum(self.distances[frozenset([a])][frozenset([b])] for a, b in pairs)
            pairslen = (len(new) * len(cluster))
            d = sumdist / pairslen
            self.distances[new][cluster] = self.distances[cluster][new] = d


class CompressionDistClustersBuilder(_DistanceClustersBuilder):
    """Builder for bottom up hierarchical clusters of inflection classes with compression distance.

    This class inherits attributes.

    Attributes:
        microclasses (dict of str:list): Inherited. mapping of microclasses exemplars to microclasses inventories.
        nodes (dict of frozenset :Node): Inherited. Maps frozensets of microclass exemplars to Nodes representing clusters.
        preferences (dict): Inherited. Configuration parameters.
        attr (str): Inherited. always have the value "DL", as the nodes of the Inflection class tree have a "DL" attribute.
        microclasses (dict of str:list): Inherited. mapping of microclasses exemplars to microclasses inventories.
        nodes (dict of frozenset :Node): Inherited. Maps frozensets of microclass exemplars to Nodes representing clusters.
        preferences (dict): Inherited. Configuration parameters.
        paradigms (:class:`pandas:pandas.DataFrame`): Inherited. a dataframe of patterns.
        distances (dict): Inherited. The distance score_matrix between clusters.
        DL_dict (dict of frozenset:float): Maps each cluster to its description length.
        min_DL (float): the lowest description length for the whole system yet encountered.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The Compression Distance clustering is experimental and development is not active. Use at your own risks !")
        super().__init__(*args, distfun=compression_distance_atomic, **kwargs)
        self.DL_dict = {}
        for microclass in list(self.microclasses):
            descr = table_to_descr(self.paradigms, [microclass], self.microclasses)
            self.DL_dict[frozenset([microclass])] = DL(split_description(descr))
        self.min_DL = sum(self.DL_dict[m] for m in self.nodes)

    def update_distances(self, new):
        """Update for compression distances.

        Arguments:
            new (frozenset) : Frozenset of microclass exemplar representing the new cluster.
        """
        table = self.paradigms
        microclasses = self.microclasses
        left = DL(split_description(table_to_descr(table, new, microclasses)))
        self.DL_dict[new] = left
        self.distances[new] = {}
        for cluster in self.nodes:
            right = self.DL_dict[cluster]
            merged = DL(split_description(table_to_descr(table, new | cluster, microclasses)))
            d = compression_distance(left, right, merged)
            self.distances[new][cluster] = self.distances[cluster][new] = d

    def merge(self, a, b):
        """Merge two Clusters, build a new Node, update the distances, track system DL.

        Parameters:
            a (frozenset): the label of a cluster to merge.
            b (frozenset): the label of a cluster to merge."""
        new = a | b
        super().merge(a, b)
        DL = sum(self.DL_dict[m] for m in self.nodes)
        self.printv("DL of whole system :", DL)
        if DL <= self.min_DL:
            self.min_DL = DL
            self.nodes[new].attributes["color"] = "c"
            self.nodes[new].attributes["macroclass"] = True
        else:
            self.printv("DL is not improving anymore ({})".format(DL))
