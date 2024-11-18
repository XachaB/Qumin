# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Cluster lemmas according to their paradigms.

Author: Sacha Beniamine.

"""
try:
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_LOADED = True
except ImportError:
    MATPLOTLIB_LOADED = False
    matplotlib = None
    plt = None

from .representations import segments, patterns
from .clustering import algorithms, descriptionlength, find_min_attribute
import pandas as pd
import logging

log = logging.getLogger()


def macroclasses_command(cfg, md):
    r"""Cluster lexemes in macroclasses according to alternation patterns."""
    # Loading files and paths
    data_file_path = cfg.patterns

    # Initializing segments

    if cfg.pats.ortho:
        pat_table = pd.read_csv(data_file_path, index_col=0)
    else:
        sounds_file_name = md.get_table_path("sounds")
        segments.Inventory.initialize(sounds_file_name)
        pat_table, pat_dic = patterns.from_csv(data_file_path, defective=False, overabundant=False)
        pat_table = pat_table.map(str)

    preferences = {"md": md}

    node = algorithms.hierarchical_clustering(pat_table, descriptionlength.BUDLClustersBuilder, **preferences)

    DL = "Min :" + str(find_min_attribute(node, "DL"))
    experiment_id = " ".join(["Bottom-up DL clustering on ", DL])

    computation = "macroclasses"
    # Saving png figure
    if MATPLOTLIB_LOADED:
        fig = plt.figure(figsize=(10, 20))
        figname = md.register_file("figure.png",
                                   {"computation": computation,
                                    "content": "figure"})
        log.info("Drawing figure to: {}".format(figname))
        node.draw(horizontal=True,
                  square=True,
                  layout="qumin",
                  leavesfunc=lambda x: x.labels[0] + " (" + str(x.attributes["size"]) + ")",
                  nodefunc=lambda x: "{0:.3f}".format(x.attributes["DL"]),
                  keep_above_macroclass=True)

        fig.suptitle(experiment_id)
        fig.savefig(figname,
                    bbox_inches='tight', pad_inches=.5)

    # Saving text tree
    treename = md.register_file("tree.txt",
                                {"computation": computation,
                                 "content": "tree"})
    log.info("Printing tree to: {}".format(treename))
    flow = open(treename, "w", encoding="utf8")
    flow.write(node.tree_string())
    flow.write("\n" + experiment_id)
    flow.close()
