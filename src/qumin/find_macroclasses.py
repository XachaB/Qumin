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

import logging

from .clustering import algorithms, descriptionlength, find_min_attribute
from .representations import segments
from .representations.paradigms import Paradigms
from .representations.patterns import ParadigmPatterns

log = logging.getLogger()


def macroclasses_command(cfg, md):
    r"""Cluster lexemes in macroclasses according to alternation patterns."""
    # Loading files and paths
    patterns_folder_path = cfg.patterns
    defective = cfg.pats.defective
    overabundant = cfg.pats.overabundant

    # Initializing segments
    sounds_file_name = md.get_table_path("sounds")
    segments.Inventory.initialize(sounds_file_name)

    # Loading paradigms
    paradigms = Paradigms(md.dataset, defective=defective, overabundant=overabundant,
                          merge_cols=cfg.entropy.merged,
                          segcheck=True,
                          cells=cfg.cells,
                          pos=cfg.pos,
                          force=cfg.force,
                          sample_lexemes=cfg.sample_lexemes,
                          sample_cells=cfg.sample_cells,
                          sample_kws=dict(force_random=cfg.force_random,
                                          seed=cfg.seed),
                          )

    # Loading Patterns
    patterns = ParadigmPatterns()
    patterns.from_file(patterns_folder_path,
                       paradigms.data,
                       defective=defective,
                       overabundant=cfg.pats.overabundant.keep,
                       force=cfg.force,
                       )

    for pair in patterns:
        patterns[pair].loc[:, "pattern"] = patterns[pair].loc[:, "pattern"].map(str)

    preferences = {"md": md}

    node = algorithms.hierarchical_clustering(patterns, paradigms, descriptionlength.BUDLClustersBuilder, **preferences)

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
