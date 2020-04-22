# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Classes to make clustering decisions and build inflection class trees according to description length.

Author: Sacha Beniamine
"""
import numpy as np
from utils import ProgressBar
from collections import defaultdict, Counter
from itertools import combinations, chain
from clustering import Node
from clustering.clusters import _ClustersBuilder, _BUClustersBuilder


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
            class_size (int): the size of the microclass"""
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


class _DLClustersBuilder(_ClustersBuilder):
    """Builder for hierarchical clusters of inflection classes with description length based decisions.

    This is an abstract class.

    This class holds two representations of the clusters it builds. On one hand, the class
    Cluster represents the informations needed to compute the description length of a cluster.
    On the other hand, the class Node represents the inflection classes being built.
    A Node can have children and a parent, a Cluster can be splitted or merged.

    This class inherits attributes.

    Attributes:
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
        super().__init__(microclasses, **kwargs)
        self.P = self.M = self.C = self.R = self.DL = 0
        self.initialize_clusters(paradigms)
        self.initialize_patterns()
        self.compute_DL(M=True)
        current_partition = " - ".join(", ".join(c) for c in self.clusters)
        self.log("\t".join(["Partition", "M", "C", "P", "R", "DL"]) + "\n", name="clusters")
        self.log(" ".join(
            [current_partition, ":\t", "\t".join((str(self.M), str(self.C), str(self.P), str(self.R), str(self.DL))),
             "\n"]), name="clusters")

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
                self.P += weighted_log(self.patterns[cell][pattern], self.patterns[cell].length)

            # This is P_c
            cluster_patterns = [len(self.clusters[cluster][cell]) for cluster in self.nodes]
            total = sum(cluster_patterns)
            self.P += sum(weighted_log(a, total) for a in cluster_patterns)

        for label in self.nodes:
            self.C += self.clusters[label].C
            self.R += self.clusters[label].R

        self.DL = (self.M + self.C + self.P + self.R)


class TDDLClustersBuilder(_DLClustersBuilder):
    """Top down builder for hierarchical clusters of inflection classes with description length based decisions.

    This class holds two representations of the clusters it builds. On one hand, the class
    Cluster represents the informations needed to compute the description length of a cluster.
    On the other hand, the class Node represents the inflection classes being built.
    A Node can have children and a parent, a Cluster can be splitted or merged.

    This class inherits attributes.

    Attributes:
        microclasses (dict of str:list): Inherited. mapping of microclasses exemplars to microclasses inventories.
        nodes (dict of frozenset :Node): Inherited. Maps frozensets of microclass exemplars to Nodes representing clusters.
        preferences (dict): Inherited. Configuration parameters.
        attr (str): Inherited. (class attribute) always have the value "DL", as the nodes of the Inflection class tree have a "DL" attribute.
        DL (float): Inherited. A description length DL, with DL(system) = DL(M) + DL(C) + DL(P) + DL(R)
        M (float): Inherited. DL(M), the cost in bits to express the mapping between lexemes and microclasses.
        C (float): Inherited. DL(C), the cost in bits to express the mapping between microclasses and clusters.
        P (float): Inherited.  DL(P), the cost in bits to express the relation between clusters and patterns.
        R (float): Inherited.  DL(R), the cost in bits to disambiguiate which pattern to use in each cluster for each microclasses.
        clusters (dict of frozenset: :class:`Cluster`): Inherited. Clusters, indexed by a frozenset of microclass examplars.
        patterns (dict of str : Counter): Inherited. A dict of pairs of cells to a count of patterns
         to the number of clusters presenting this pattern for this cell.::

                { str: Counter({Pattern: int }) }
                pairs of cells -> pattern -> number of clusters with this pattern for this cell

         Note that the Counter's length is written on a .length attribute, to avoid calling len() repeatedly.
         Remark that the count is not the same as in the class Cluster.
        size (int): Inherited. The size of the whole system in microclasses.
        minDL (float): The minimum description length yet encountered.
        history (dict of frozenset:tuples): dict associating partitions with (M, C, P, R, DL) tuples.
        left (Cluster): left and right are temporary clusters used to divide a current cluster in two.
        right (Cluster): see left.
        to_split (Node): the node that we are currently trying to split.
    """

    def __init__(self, microclasses, paradigms, **kwargs):
        """Constructor.

        Arguments:
            microclasses (dict of str:list): mapping of microclasses exemplars to microclasses inventories.
            paradigms (:class:`pandas:pandas.DataFrame`): a dataframe of patterns.
            kwargs : keyword arguments to be used as configuration.
        """
        super().__init__(microclasses, paradigms, **kwargs)
        self.minDL = self.DL
        self.printv("Initial cluster has DL : ", self.DL)
        self.nodes[frozenset(list(self.microclasses))].attributes["DL"] = self.DL

    def initialize_nodes(self):
        """Initialize nodes with only one root node which children are all microclasses."""
        root = Node(list(self.microclasses),
                    children=[Node([m],
                                   size=len(self.microclasses[m]),
                                   macroclass=False,
                                   color="c") for m in self.microclasses],
                    size=sum(len(self.microclasses[m]) for m in self.microclasses),
                    color="r",
                    macroclass=False)
        self.nodes = {frozenset(self.microclasses): root}

    def initialize_clusters(self, paradigms):
        """Initialize clusters with one cluster per microclass plus one for the whole.

        Arguments:
            paradigms (:class:`pandas:pandas.DataFrame`): a dataframe of patterns.
        """
        super().initialize_clusters(paradigms)
        # self.atomic_clusters = self.clusters.copy()
        root = frozenset(self.microclasses)
        self.clusters[root] = sum(self.clusters[cluster] for cluster in self.clusters)

    def _partial_DL(self):
        """Compute the description length of the two clusters right and left."""
        C = self.right.C + self.left.C
        R = self.right.R + self.left.R
        P = 0
        patterns = defaultdict(Counter)

        for cell in self.right:
            #  This is P_p
            patterns[cell] = Counter(list(self.right[cell])) + Counter(list(self.left[cell]))
            patterns[cell].length = sum(patterns[cell].values())
            for pattern in patterns[cell]:
                P += weighted_log(patterns[cell][pattern], patterns[cell].length)

            # This is P_c
            right_patterns = self.right[cell].length
            left_patterns = self.left[cell].length
            total_patterns = right_patterns + left_patterns
            P += weighted_log(right_patterns, total_patterns) + \
                 weighted_log(left_patterns, total_patterns)

        return R, C, P, patterns

    def _shift_auxiliary(self, label):
        """Shift one microclass from left to right or vice-versa

        Parameters:
            label (str): the label of the microclass to shift."""
        cluster_to_shift = self.clusters[frozenset([label])]
        if label in self.right.labels:
            self.right -= cluster_to_shift
            self.left += cluster_to_shift
        elif label in self.left.labels:
            self.left -= cluster_to_shift
            self.right += cluster_to_shift

    def _simulate_shift(self, label):
        """Simulate shifting one microclass, return parameters for the DL.

        Parameters:
            label (str): the label of the microclass to shift."""
        self._shift_auxiliary(label)
        R, C, P, patterns = self._partial_DL()
        self._shift_auxiliary(label)
        return R, C, P, patterns

    def shift(self, label):
        """Shift one microclass rom left to right or vice-versa

        Parameters:
            label (str): the label of the microclass to shift.
        """
        self._shift_auxiliary(label)

    def split_leaves(self):
        """Split a cluster by replacing it with the two clusters left and right.

        Recompute the description length when left and right are separated.
        Build two nodes corresponding to left and right, children of to_split.
        """
        leaves = self.to_split.children

        if len(self.left.labels) > 0 and len(self.right.labels) > 0:
            left_leaves = []
            right_leaves = []
            left_labels = self.left.labels
            right_labels = self.right.labels

            for leaf in leaves:
                if leaf.labels[0] in self.left.labels:
                    left_leaves.append(leaf)
                else:
                    right_leaves.append(leaf)

            # del self.clusters[frozenset(self.to_split.labels)]
            self.right.totalsize = self.left.totalsize = self.size
            self.right.C = weighted_log(self.right.size, self.size)
            self.left.C = weighted_log(self.left.size, self.size)
            self.clusters[frozenset(right_labels)] = self.right
            self.clusters[frozenset(left_labels)] = self.left

            self.compute_DL()
            current_partition = " - ".join(", ".join(c) for c in self.nodes)
            self.log(" ".join([current_partition, ":\t",
                               "\t".join((str(self.M), str(self.C), str(self.P), str(self.R), str(self.DL))), "\n"]),
                     name="clusters")

            color = "r"
            if self.DL >= self.minDL:
                color = "c"
            else:
                self.minDL = self.DL
            kwargs = {"macroclass": False, "DL": self.DL, "color": color}
            if len(left_leaves) > 1:
                left = Node(left_labels, size=sum(leaf.attributes["size"] for leaf in left_leaves),
                            children=left_leaves, **kwargs)
            else:
                left = left_leaves[0]
                left.attributes["DL"] = self.DL
            if len(right_leaves) > 1:
                right = Node(right_labels, size=sum(leaf.attributes["size"] for leaf in right_leaves),
                             children=right_leaves, **kwargs)
            else:
                right = right_leaves[0]
                right.attributes["DL"] = self.DL

            self.printv("Splitted:", ", ".join(right.labels), "\n\t", ", ".join(left.labels))
            self.to_split.children = [left, right]

    def initialize_subpartition(self, node):
        """Initialize left and right as a subpartition of a node we want to split.

        Arguments:
            node (Node): The node to be splitted.
        """
        self.to_split = node

        self.left = sum(self.clusters[frozenset([leaf])] for leaf in node.labels)
        self.left.C = 0
        self.right = Cluster()
        self.left.totalsize = self.right.totalsize = self.left.size

        current_partition = frozenset(list(self.nodes))
        self.history = {current_partition: (self.M, self.C, self.P, self.R, self.DL)}

        microclasses = [len(self.microclasses[m]) for m in self.left.labels | self.right.labels]
        total = sum(microclasses)
        M = sum(weighted_log(val, total) for val in microclasses)
        R, C, P, *_ = self._partial_DL()
        DL = M + C + P + R
        leftlabels = frozenset(self.left.labels)
        rightlabels = frozenset(self.right.labels)
        self.history[frozenset([leftlabels, rightlabels])] = (M, C, P, R, DL)

    def find_ordered_shifts(self):
        """Find the list of all best shifts of a microclass between right and left.

        The list is a list of tuples of length 2 containing the label of a node to shift
        and the description length of the node to be splitted if we perform the shift.
        """
        best_shifts = []

        leftlabels = frozenset(self.left.labels)
        rightlabels = frozenset(self.right.labels)
        M, C, P, R, best = self.history[frozenset([leftlabels, rightlabels])]

        for leaf in sorted(chain(self.right.labels, self.left.labels)):
            new = frozenset([leftlabels - {leaf}, rightlabels | {leaf}])

            if new not in self.history:
                R, C, P, *_ = self._simulate_shift(leaf)
                DL = M + R + C + P
                self.history[new] = (M, C, P, R, DL)
                if DL < best:
                    best_shifts = [(leaf, DL)]
                    best = DL
                elif DL == best:
                    best_shifts.append((leaf, DL))
                str_partition = " - ".join(", ".join(p) for p in new)
                self.log(" ".join([str_partition, ":\t", "\t".join((str(M), str(C), str(P), str(R), str(DL))), "\n"]),
                         name="clusters")

        return best_shifts


class BUDLClustersBuilder(_DLClustersBuilder, _BUClustersBuilder):
    """Bottom up Builder for hierarchical clusters of inflection classes with description length based decisions.

    This class holds two representations of the clusters it builds. On one hand, the class
    Cluster represents the informations needed to compute the description length of a cluster.
    On the other hand, the class Node represents the inflection classes being built.
    A Node can have children and a parent, a Cluster can be splitted or merged.

    This class inherits attributes.

    Attributes:
        microclasses (dict of str:list): Inherited. mapping of microclasses exemplars to microclasses inventories.
        nodes (dict of frozenset :Node): Inherited. Maps frozensets of microclass exemplars to Nodes representing clusters.
        preferences (dict): Inherited. Configuration parameters.
        attr (str): Inherited. (class attribute) always have the value "DL", as the nodes of the Inflection class tree have a "DL" attribute.
        DL (float): Inherited. A description length DL, with DL(system) = DL(M) + DL(C) + DL(P) + DL(R)
        M (float): Inherited. DL(M), the cost in bits to express the mapping between lexemes and microclasses.
        C (float): Inherited. DL(C), the cost in bits to express the mapping between microclasses and clusters.
        P (float): Inherited.  DL(P), the cost in bits to express the relation between clusters and patterns.
        R (float): Inherited.  DL(R), the cost in bits to disambiguiate which pattern to use in each cluster for each microclasses.
        clusters (dict of frozenset: :class:`Cluster`): Inherited. Clusters, indexed by a frozenset of microclass examplars.
        patterns (dict of str : Counter): Inherited. A dict of pairs of cells to a count of patterns
         to the number of clusters presenting this pattern for this cell.::

                { str: Counter({Pattern: int }) }
                pairs of cells -> pattern -> number of clusters with this pattern for this cell

         Note that the Counter's length is written on a .length attribute, to avoid calling len() repeatedly.
         Remark that the count is not the same as in the class Cluster.
        size (int): Inherited. The size of the whole system in microclasses.
    """

    def _simulate_merge(self, a, b):
        """Simulate merging two clusters, return parameters for the DL.

        Parameters:
            a (str): the label of a cluster to merge.
            b (str): the label of a cluster to merge."""
        g1 = self.clusters[a]
        g2 = self.clusters[b]
        new = g1 + g2
        C = self.C - g1.C - g2.C + new.C
        P = 0
        patterns = defaultdict(Counter)

        for cell in g1:
            # This is P_p
            patterns[cell] = self.patterns[cell] + Counter(list(new[cell])) - \
                             Counter(list(g1[cell])) - Counter(list(g2[cell]))

            patterns[cell].length = self.patterns[cell].length + new[cell].length - \
                                    g1[cell].length - g2[cell].length

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
        self.R, self.C, self.P, self.patterns, self.clusters[labels] = self._simulate_merge(a, b)
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
            self.printv("\nDL stopped improving: prev = {}, current best = {}".format(prev_DL, self.DL))
            color = "r"

        self.nodes[labels] = Node(leaves, size=size, children=[left, right],
                                  DL=self.DL, color=color, macroclass=color != "r")

        self.printv("\nMerging ", ", ".join(a), " and ", ", ".join(b), "with DL ", self.DL)

        current_partition = " - ".join([", ".join(self.nodes[c].labels) for c in self.nodes])
        self.log(" ".join(
            [current_partition, ":\t", "\t".join((str(self.M), str(self.C), str(self.P), str(self.R), str(self.DL))),
             "\n"]), name="clusters")

    def find_ordered_merges(self):
        """Find the list of all best merges of two clusters.

        The list is a list of tuples of length 3 containing two frozensets representing the
        labels of the clusters to merge and the description length of the resulting system.
        """
        best_merges = []
        best = np.inf
        pairs = list(combinations(sorted(self.nodes), 2))
        progress = ProgressBar(len(pairs))

        for g1, g2 in pairs:
            R, C, P, *_ = self._simulate_merge(g1, g2)
            DL = self.M + R + C + P
            if DL < best:
                best_merges = [(g1, g2, DL)]
                best = DL
            elif DL == best:
                best_merges.append((g1, g2, DL))
            progress.increment()

        if len(best_merges) > 1:
            choices = ", ".join(["({}, {})".format("-".join(a), "-".join(b)) for a, b, _ in best_merges])
            self.printv("\nWarning, {} equivalent choices: ".format(len(best_merges)), choices)

        return best_merges


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
