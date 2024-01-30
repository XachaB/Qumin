# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Classes to make clustering decisions and build inflection class trees according to description length.

Author: Sacha Beniamine
"""
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
from . import Node
from tqdm import tqdm
import logging

log = logging.getLogger()


class Cluster(object):
    """A single cluster in MDL clustering.

    A Cluster is iterable. Itering on a cluster is itering on its patterns.
    Cluster can be merged or separated by adding or substracting them.

    Attributes:
        patterns (:class:`defaultdict`): For each pair of cell in the paradigms under consideration,
         it holds a counter of the number of microclass using each pattern in this cluster and pair of cells.::

                { str: Counter({Pattern: int }) }
                pairs of cells -> pattern -> number of microclasses using this pattern for this cell

         Note that the Counter's length is written on a .length attribute, to avoid calling len() repeatedly.

        labels (set): the set of all exemplars representing the microclasses in this cluster.
        size (int): The size of this cluster. Depending on external parameters,
            this can be the number of microclasses or the number of lexemes belonging to the cluster.
        totalsize (int): The size of the whole system of clusters, either number of microclasses in the system, or number of lexemes in the system.
        R : The cost in bits to disambiguate for each pair of cells which pattern is to be used with which microclass.
        C : The contribution of this cluster to the cost of mapping from microclasses to clusters.
        """

    def __init__(self, *args):
        """Initialize single cluster.

        Arguments:
            args (str): Names (exemplar) of each microclass belonging to the cluster.
        """
        # cell : Counter(patterns)
        self.patterns = defaultdict(Counter)
        self.labels = set(args)
        self.size = self.R = self.C = self.totalsize = 0

    def init_from_paradigm(self, class_size, paradigms, size):
        """Populate fields according to a paradigm column.

        This assumes an initialization with only one microclass.

        Arguments:
            class_size (int): the size of the microclass
            paradigms (:class:`pandas:pandas.DataFrame`): a dataframe of patterns.
            size (int): total size
        """
        self.size = class_size
        self.totalsize = size
        self.R = 0
        self.C = weighted_log(self.size, self.totalsize)

        for cell in paradigms.index:
            pattern = paradigms[cell]
            self[cell][pattern] = self.size
            self[cell].length = 1
            self.R += sum(weighted_log(self[cell][p], self.size) for p in self[cell])

    def __copy(self):
        new = Cluster()
        new.totalsize = self.totalsize
        new += self
        return new

    def __update_attributes(self, other, update_action):
        self.size = update_action(self.size, other.size)
        self.C = weighted_log(self.size, self.totalsize)
        self.R = 0
        for cell in set(self).union(other):
            self[cell] = update_action(self[cell], other[cell])
            self[cell].length = len(self[cell])
            self.R += sum(weighted_log(self[cell][p], self.size) for p in self[cell])

    def __str__(self):
        template = "<Cluster {} size={}; C={}; R={}; Pattern={}>"
        return template.format(self.labels, self.size, self.C, self.R, self.patterns)

    def __iter__(self):
        return iter(self.patterns)

    def __getitem__(self, key):
        return self.patterns[key]

    def __setitem__(self, key, item):
        self.patterns[key] = item

    def __radd__(self, other):
        if other == 0:
            return self + Cluster()
        else:
            return self + other

    def __add__(self, other):
        new = self.__copy()
        new += other
        return new

    def __sub__(self, other):
        new = self.__copy()
        new -= other
        return new

    def __iadd__(self, other):
        self.labels = self.labels | other.labels
        self.__update_attributes(other, lambda a, b: a + b)
        return self

    def __isub__(self, other):
        self.labels = self.labels - other.labels
        self.__update_attributes(other, lambda a, b: a - b)
        return self


class BUDLClustersBuilder(object):
    """Builder for hierarchical clusters of inflection classes with description length based decisions.

    This class holds two representations of the clusters it builds. On one hand, the class
    Cluster represents the informations needed to compute the description length of a cluster.
    On the other hand, the class Node represents the inflection classes being built.
    A Node can have children and a parent, a Cluster can be splitted or merged.

    This class inherits attributes.

    Attributes:microclasses (dict of str:list): Inherited. mapping of microclasses exemplars to microclasses inventories.
        nodes (dict of frozenset :Node): Inherited. Maps frozensets of microclass exemplars to Nodes representing clusters.
        preferences (dict): Inherited. Configuration parameters.
        attr (str): (class attribute) always have the value "DL", as the nodes of the Inflection class tree have a "DL" attribute.
        DL (float): A description length DL, with DL(system) = DL(M) + DL(C) + DL(P) + DL(R)
        M (float): DL(M), the cost in bits to express the mapping between lexemes and microclasses.
        C (float): DL(C), the cost in bits to express the mapping between microclasses and clusters.
        P (float):  DL(P), the cost in bits to express the relation between clusters and patterns.
        R (float):  DL(R), the cost in bits to disambiguiate which pattern to use in each cluster for each microclasses.
        clusters (dict of frozenset: :class:`Cluster`): Clusters, indexed by a frozenset of microclass examplars.
        patterns (dict of str : Counter): A dict of pairs of cells to a count of patterns
         to the number of clusters presenting this pattern for this cell.::

                { str: Counter({Pattern: int }) }
                pairs of cells -> pattern -> number of clusters with this pattern for this cell

         Note that the Counter's length is written on a .length attribute, to avoid calling len() repeatedly.
         Remark that the count is not the same as in the class Cluster.
        size (int): The size of the whole system in microclasses.
    """

    attr = "DL"

    def __init__(self, microclasses, paradigms, **kwargs):
        """Constructor.

        Arguments:
            microclasses (dict of str:list): mapping of microclasses exemplars to microclasses inventories.
            paradigms (:class:`pandas:pandas.DataFrame`): a dataframe of patterns.
            kwargs : keyword arguments to be used as configuration.
        """
        self.preferences = kwargs
        self.microclasses = microclasses
        self.nodes = {
            frozenset([m]): Node([m], size=len(self.microclasses[m]), macroclass=False)
            for m in
            self.microclasses}

        self.P = self.M = self.C = self.R = self.DL = 0
        self.initialize_clusters(paradigms)
        self.initialize_patterns()
        self.compute_DL(M=True)
        current_partition = " - ".join(", ".join(c) for c in self.clusters)
        log.debug("\t".join(["Partition", "M", "C", "P", "R", "DL"]))
        log.debug(" ".join([current_partition, ":\t", "\t".join(
            (str(self.M), str(self.C), str(self.P), str(self.R), str(self.DL)))]))

    def initialize_clusters(self, paradigms):
        self.clusters = {}
        classes_size = {m: 1 for m in self.microclasses}
        self.size = sum(classes_size.values())

        for microclass in self.microclasses:
            patterns = paradigms.loc[microclass, :]
            cluster = Cluster(microclass)
            cluster.init_from_paradigm(classes_size[microclass], patterns, self.size)
            self.clusters[frozenset([microclass])] = cluster

    def initialize_patterns(self):
        self.patterns = defaultdict(Counter)
        for cell in next(iter(self.clusters.values())):
            for label in self.nodes:
                self.patterns[cell] += Counter(list(self.clusters[label][cell]))
            self.patterns[cell].length = sum(self.patterns[cell].values())

    def compute_DL(self, M=False):
        values = [len(self.microclasses[m]) for m in self.microclasses]
        if M:
            total = sum(values)
            self.M = sum(weighted_log(val, total) for val in values)

        self.size = len(self.microclasses)
        for cell in self.patterns:

            # This is P_p
            for pattern in self.patterns[cell]:
                self.P += weighted_log(self.patterns[cell][pattern],
                                       self.patterns[cell].length)

            # This is P_c
            cluster_patterns = [len(self.clusters[cluster][cell]) for cluster in
                                self.nodes]
            total = sum(cluster_patterns)
            self.P += sum(weighted_log(a, total) for a in cluster_patterns)

        for label in self.nodes:
            self.C += self.clusters[label].C
            self.R += self.clusters[label].R

        self.DL = (self.M + self.C + self.P + self.R)

    def _simulate_merge(self, a, b):
        """Simulate merging two clusters, return parameters for the DL.

        Parameters:
            a (frozenset): the label of a cluster to merge.
            b (frozenset): the label of a cluster to merge."""
        g1 = self.clusters[a]
        g2 = self.clusters[b]
        new = g1 + g2
        C = self.C - g1.C - g2.C + new.C
        P = 0
        patterns = defaultdict(Counter)

        for cell in g1:
            # This is P_p
            patterns[cell] = self.patterns[cell] + \
                             Counter(list(new[cell])) - \
                             Counter(list(g1[cell])) - \
                             Counter(list(g2[cell]))

            patterns[cell].length = self.patterns[cell].length + new[cell].length - g1[
                cell].length - g2[cell].length

            for pattern in patterns[cell]:
                P += weighted_log(patterns[cell][pattern], patterns[cell].length)

            # This is P_c
            cluster_patterns = [new[cell].length]
            for cluster in self.nodes:
                if cluster not in [a, b]:
                    cluster_patterns.append(self.clusters[cluster][cell].length)
            total = sum(cluster_patterns)
            P += sum(weighted_log(a, total) for a in cluster_patterns)

        R = self.R - g1.R - g2.R + new.R

        return R, C, P, patterns, new

    def merge(self, a, b):
        """Merge two Clusters, build a Node to represent the result, update the DL.

        Parameters:
            a (str): the label of a cluster to merge.
            b (str): the label of a cluster to merge."""
        labels = a | b
        self.R, self.C, self.P, self.patterns, self.clusters[
            labels] = self._simulate_merge(a, b)
        # del self.clusters[b]
        # del self.clusters[a]

        prev_DL = self.DL
        self.DL = (self.R + self.C + self.P + self.M)

        left = self.nodes.pop(a)
        right = self.nodes.pop(b)
        leaves = list(labels)
        size = left.attributes["size"] + right.attributes["size"]
        color = "c"
        if self.DL >= prev_DL:
            log.info("DL stopped improving: prev = {}, current best = {}".format(prev_DL,
                                                                                 self.DL))
            color = "r"

        self.nodes[labels] = Node(leaves, size=size, children=[left, right],
                                  DL=self.DL, color=color, macroclass=color != "r")

        log.debug("Merging %s and %s with DL %s", ", ".join(a), ", ".join(b), self.DL)

        current_partition = " - ".join(
            [", ".join(self.nodes[c].labels) for c in self.nodes])
        log.debug(" ".join(
            [current_partition, ":\t", "\t".join(
                (str(self.M), str(self.C), str(self.P), str(self.R), str(self.DL)))]))

    def find_ordered_merges(self):
        """Find the list of all best merges of two clusters.

        The list is a list of tuples of length 3 containing two frozensets representing the
        labels of the clusters to merge and the description length of the resulting system.
        """
        best_merges = []
        best = np.inf
        pairs = combinations(sorted(self.nodes), 2)
        tot = (len(self.nodes) * (len(self.nodes) - 1)) // 2

        for g1, g2 in tqdm(pairs, leave=False, total=tot):
            R, C, P, *_ = self._simulate_merge(g1, g2)
            DL = self.M + R + C + P
            if DL < best:
                best_merges = [(g1, g2, DL)]
                best = DL
            elif DL == best:
                best_merges.append((g1, g2, DL))

        if len(best_merges) > 1:
            choices = ", ".join(
                ["({}, {})".format("-".join(a), "-".join(b)) for a, b, _ in best_merges])
            log.warning("There were {} equivalent choices: %s"
                        .format(len(best_merges)), choices)

        return best_merges

    def rootnode(self):
        """Return the root of the Inflection Class tree, if it exists."""
        assert len(self.nodes) == 1
        return next(iter(self.nodes.values()))


def weighted_log(symbol_count, message_length):
    r"""Compute :math:`-log_{2}(symbol_count/message_length) * message_length`.

    This corresponds to the product inside the sum
    of the description length formula
    when probabilities are estimated on frequencies.

    Arguments:
        symbol_count (int): a count of symbols.
        message_length (int): the size of the message.

    Returns:
        (float): the weighted log
    """
    try:
        if symbol_count == 0:
            return 0
        return symbol_count * -np.log2(symbol_count / message_length)
    except ZeroDivisionError:
        return 0
