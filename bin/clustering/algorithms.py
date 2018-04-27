# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Algorithms for inflection classes clustering.

Author: Sacha Beniamine
"""
import numpy as np
from clustering import find_microclasses
import warnings

def choose(iterable):
    """Choose a random element in an iterable of iterable."""
    i = np.random.choice(len(iterable), 1)
    return iterable[int(i)]


# def randomised(clustering_algorithm, patterns, microclasses, Clusters, n=10, **kwargs):
#     """Randomise a clustering algorithm by running it n times.

#     Arguments:
#         clustering_algorithm (func): a clustering algorithm.
#         patterns (:class:`pandas:pandas.DataFrame`): a dataframe of patterns.
#         microclasses (dict of str:list): mapping of microclasses exemplars to microclasses inventories.
#         Clusters : a cluster class to use in clustering.
#         n : the number of repeated experiments to run.
#         kwargs: any keywords arguments to pass to Clusters.
#     """
#     print("Randomised Top down clustering with {} iterations".format(n))

#     solutions = []
#     for i in range(n):
#         tree = clustering_algorithm(patterns, microclasses, Clusters, **kwargs)
#         DL = find_min_attribute(tree, "DL")
#         print("Found solution with DL :", DL)
#         solutions.append((DL, tree))

#     solutions.sort(key=lambda x: x[0])
#     best = solutions[0]
#     worst = solutions[-1]

#     print(" {} Solutions from {} to {}".format(len(solutions), best[0], worst[0]))

#     return best[1]


def top_down_clustering(patterns, microclasses, Clusters, **kwargs):
    """Cluster microclasses in a top-down recursive fashion.

    The algorithm is the following::

        Begin with one unique cluster containing all microclasses, and one empty cluster.
        While we are seeing an improvement:
            Find the best possible shift of a microclass from one cluster to another.
            Perform this shift.
        Build a binary node with the two clusters.
        Recursively apply the same algorithm to each.

    The algorithm stops when it reaches leaves, or when no shift improves the score.

    Scoring, finding the best shits, updating the nodes depends on the Clusters class.

    Arguments:
        patterns (:class:`pandas:pandas.DataFrame`): a dataframe of patterns.
        microclasses (dict of str:list): mapping of microclasses exemplars to microclasses inventories.
        Clusters : a cluster class to use in clustering.
        kwargs: any keywords arguments to pass to Clusters.
    """
    warnings.warn("Top down clustering is experimental and development is not active. Use at your own risks !")
    print("Top down clustering")
    clusters = Clusters(microclasses, paradigms=patterns, **kwargs)

    stack = [clusters.rootnode()]
    while stack:
        node = stack.pop(-1)
        clusters.initialize_subpartition(node)

        possible_shifts = clusters.find_ordered_shifts()
        if possible_shifts:
            while possible_shifts:
                leaf, DL = choose(possible_shifts)
                clusters.shift(leaf)
                possible_shifts = clusters.find_ordered_shifts()
            clusters.split_leaves()
            stack.extend(node.children)

    return clusters.rootnode()


def bottom_up_clustering(patterns, microclasses, Clusters, **kwargs):
    """Cluster microclasses in a top-down recursive fashion.

    The algorithm is the following::

        Begin with one cluster per microclasses.
        While there is more than one cluster :
            Find the best possible merge of two clusters, among all possible pairs.
            Perform this merge

    Scoring, finding the best merges, merging nodes depends on the Clusters class.

    Arguments:
        patterns (:class:`pandas:pandas.DataFrame`): a dataframe of patterns.
        microclasses (dict of str:list): mapping of microclasses exemplars to microclasses inventories.
        Clusters : a cluster class to use in clustering.
        kwargs: any keywords arguments to pass to Clusters.
    """
    print(kwargs)
    clusters = Clusters(microclasses, paradigms=patterns, **kwargs)
    print("Bottom up clustering")
    while len(clusters.nodes) > 1:
        print("N =", len(clusters.nodes))
        possible_merges = clusters.find_ordered_merges()
        a, b, score = choose(possible_merges)
        clusters.merge(a, b)
    return clusters.rootnode()

def log_classes(classes, prefix, suffix):
    filename = prefix + "_"+ suffix + ".txt"
    print("\nFound ", len(classes), suffix, ".\nPrinting log to ", filename)
    with open(filename, "w", encoding="utf-8") as flow:
        for m in sorted(classes, key=lambda m: len(classes[m])):
            flow.write("\n\n{} ({}) \n\t".format(m, len(classes[m])) + ", ".join(classes[m]))


def hierarchical_clustering(patterns, Clusters, clustering_algorithm=bottom_up_clustering,  **kwargs):
    """Perform hierarchical clustering on patterns according to a clustering algorithm and a measure.

    This function ::
        Finds microclasses.
        Performs the clustering,
        Finds the macroclasses (and exports them),
        Returns the inflection class tree.

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
    node = clustering_algorithm(patterns, microclasses, Clusters, **kwargs)

    # Export macroclasses
    macroclasses = node.macroclasses()
    if macroclasses:
        log_classes(macroclasses,kwargs["prefix"],"macroclasses")
    else:
        print("No macroclasses could be found "
              "(this is normal for Top Down and distances based clustering,"
              " but an edge case for Bottom up clustering with description length)")

    return node
