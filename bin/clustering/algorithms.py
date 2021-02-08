# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Algorithms for inflection classes clustering.

Author: Sacha Beniamine
"""
import numpy as np
from clustering import find_microclasses
import logging
log = logging.getLogger(__name__)

def choose(iterable):
    """Choose a random element in an iterable of iterable."""
    i = np.random.choice(len(iterable), 1)
    return iterable[int(i)]


def log_classes(classes, prefix, suffix):
    filename = prefix + "_" + suffix + ".txt"
    log.info("Found %s %s", len(classes), suffix)
    log.info("Printing log to %s", filename)
    with open(filename, "w", encoding="utf-8") as flow:
        for m in sorted(classes, key=lambda x: len(classes[x])):
            flow.write("\n\n{} ({}) \n\t".format(m,
                                                 len(classes[m]))
                       + ", ".join(classes[m]))


def hierarchical_clustering(patterns, Clusters, **kwargs):
    """Perform hierarchical clustering on patterns according to a clustering algorithm and a measure.

    This function ::
        Finds microclasses.
        Performs the clustering,
        Finds the macroclasses (and exports them),
        Returns the inflection class tree.

    The clustering algorithm is the following::

        Begin with one cluster per microclasses.
        While there is more than one cluster :
            Find the best possible merge of two clusters, among all possible pairs.
            Perform this merge

    Scoring, finding the best merges, merging nodes depends on the Clusters class.

    Arguments:
        patterns (:class:`pandas:pandas.DataFrame`): a dataframe of strings representing alternation patterns.
        Clusters : a cluster class to use in clustering.
        clustering_algorithm (func): a clustering algorithm.
        kwargs: any keywords arguments to pass to Clusters. Some keywords are mandatory :
          "prefix" should be the log file prefix, "patterns" should be a function for pattern finding
    """

    # Clustering
    microclasses = find_microclasses(patterns)

    clusters = Clusters(microclasses, paradigms=patterns, **kwargs)
    while len(clusters.nodes) > 1:
        log.info("N = %s", len(clusters.nodes))
        possible_merges = clusters.find_ordered_merges()
        a, b, score = choose(possible_merges)
        clusters.merge(a, b)
    node = clusters.rootnode()

    # Export macroclasses
    macroclasses = node.macroclasses()
    if macroclasses:
        log_classes(macroclasses, kwargs["prefix"], "macroclasses")
    else:
        log.warning("No macroclasses could be found "
                    " this is not necessarily a bug, but it is surprising !")

    return node
