# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Base classes to make clustering decisions and build inflection class trees.

Author: Sacha Beniamine
"""
from clustering import Node


def _do_nothing(*args, **kwargs):
    """Place holder function for disabled verbosity"""
    pass


class _ClustersBuilder(object):
    """Builder for Hierarchical clustering of inflectiona realizations.

    This is an abstract class.

    Attributes:
        microclasses (dict of str:list): mapping of microclasses exemplars to microclasses inventories.
        nodes (dict of frozenset :Node): Maps frozensets of microclass exemplars to Nodes representing clusters.
        preferences (dict): Configuration parameters.
    """

    def __init__(self, microclasses, *args, **kwargs):
        self.preferences = kwargs
        self.microclasses = microclasses
        self.nodes = {frozenset([m]): Node([m], size=len(self.microclasses[m]), macroclass=False) for m in
                      self.microclasses}

        if "verbose" not in kwargs or not kwargs["verbose"]:
            self.printv = _do_nothing
        if "debug" in kwargs and kwargs["debug"] and kwargs["prefix"]:
            self.preferences["filename"] = self.preferences["prefix"] + "_{}.log"
            print("Writing logs to : ", self.preferences["filename"].format("<...>"))
        else:
            self.log = _do_nothing

    def rootnode(self):
        """Return the root of the Inflection Class tree, if it exists."""
        assert len(self.nodes) == 1
        return next(iter(self.nodes.values()))

    def log(self, *args, name="clusters", **kwargs):
        filename = self.preferences["filename"].format(name)
        with open(filename, "a", encoding="utf-8") as flow:
            flow.write(*args, **kwargs)

    def printv(self, *args, **kwargs):
        print(*args, **kwargs)


class _BUClustersBuilder(_ClustersBuilder):
    """Builder for Hierarchical clusters of inflection classes in bottom-up algorithms.

    This is an abstract class.
    """

    def find_ordered_merges(self):
        """Find the list of all best possible merges."""
        raise NotImplementedError("this is an abstract class. Daughters should implement find_ordered_merges")

    def merge(self, a, b):
        """Merge two clusterBuilder into one."""
        raise NotImplementedError("this is an abstract class. Daughters should implement merge")


class BUComparisonClustersBuilder(_BUClustersBuilder):
    """Comparison between measures for hierarchical clustering bottom-up clustering of Inflection classes.

    This class takes two _BUClustersBuilder classes, a DecisionMaker and an Annotator.
    The DecisionMaker is used to find the ordered merges.
    When merging, the merge is performed on both classes,
    and the Annotator's values (description length or distances)
    are used to annotate the trees of the DecisionMaker.

    Attributes:
        microclasses (dict of str:list): Inherited. mapping of microclasses exemplars to microclasses inventories.
        nodes (dict of frozenset :Node): Inherited. Maps frozensets of microclass exemplars to Nodes representing clusters.
        preferences (dict): Inherited. Configuration parameters.
        DecisionMaker (:class:clustering.clusters._BUClustersBuilder): A class to use for finding ordered merges.
        Annotator (:class:clustering.clusters._BUClustersBuilder): A class to use for annotating the DecisionMaker.
    """

    def __init__(self, *args, DecisionMaker=None, Annotator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.DecisionMaker = DecisionMaker(*args, **kwargs)
        self.Annotator = Annotator(*args, **kwargs)
        self.nodes = self.DecisionMaker.nodes

    def find_ordered_merges(self):
        """Find the list of all best possible merges."""
        return self.DecisionMaker.find_ordered_merges()

    def merge(self, a, b):
        """Merge two clusters into one."""
        self.DecisionMaker.merge(a, b)
        self.Annotator.merge(a, b)

        # Now annotate the new node from decision maker with info from the annotator
        new = a | b
        annotation = self.Annotator.attr
        value = self.Annotator.nodes[new].attributes[annotation]
        self.DecisionMaker.nodes[new].attributes[annotation] = value

    def rootnode(self):
        """Return the root of the Inflection Class tree, if it exists."""
        return self.DecisionMaker.rootnode()
