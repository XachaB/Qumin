# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Show microclass similarity

Author: Sacha Beniamine.
"""
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
import logging

# Our scripts
from .clustering import find_microclasses
from .representations import segments
from .representations.paradigms import Paradigms
from .representations.patterns import ParadigmPatterns

log = logging.getLogger()


def microclass_heatmap(distances, md, labels=None, cmap_name="BuPu", exhaustive_labels=False):
    """Make a heatmap of microclasses distances"""
    index = list(distances.index)
    tick_value = "auto"
    if exhaustive_labels:
        tick_value = 1

    if labels is not None:
        cats = sorted(set(labels.tolist()))
        labels = [labels[l] for l in distances.index]
        class_pal = sns.cubehelix_palette(len(cats), start=.5, rot=-.75)
        to_color = dict(zip(cats, class_pal))
        class_labels = pd.Series([to_color[l] for l in labels], index=index)
        sns.clustermap(distances, method="average", xticklabels=tick_value, yticklabels=tick_value, linewidths=0,
                       cmap=plt.get_cmap(cmap_name), rasterized=True, row_colors=class_labels, col_colors=class_labels)
        patches = []
        for color, cat in zip(class_pal, cats):
            patches.append(mpatches.Patch(color=color, label=cat))

        plt.legend(handles=patches, bbox_to_anchor=(6, 1))
    else:
        sns.clustermap(distances, method="average", xticklabels=tick_value, yticklabels=tick_value,
                       linewidths=0, cmap=plt.get_cmap(cmap_name), rasterized=True)

    name = md.get_path('vis/microclassHeatmap.pdf')
    log.info("Saving file to: %s", name)
    plt.savefig(name, bbox_inches='tight', pad_inches=0, transparent=True)
    md.register_file("vis/microclassHeatmap.pdf", description="Microclass heatmap")


def distance_matrix(patterns, microclasses, **kwargs):
    """Returns a similarity matrix from a pattern dataframe and microclasses"""

    # Long to wide for just the exemplars
    exemplars = list(microclasses)
    feature_space = pd.DataFrame(index=exemplars, columns=list(patterns))
    for pair in patterns:
        feature_space.loc[exemplars, pair] = patterns[pair].applymap(repr)

    dists = squareform(pdist(feature_space, metric=lambda x, y: sum((a != b) for a, b in zip(x, y))))
    distances = pd.DataFrame(dists, columns=exemplars, index=exemplars)
    for x in exemplars:
        distances.at[x, x] = 0
    distances.columns = [x.split()[0] for x in exemplars]
    distances.index = list(distances.columns)
    return distances


def heatmap_command(cfg, md, patterns_md):
    r"""Draw a clustermap of microclass similarities using seaborn.

    Arguments:
        cfg (omegaconf.dictconfig.DictConfig): Configuration for this run.
        md (qumin.utils.Metadata): Metadata handler for this run.
        patterns_md (qumin.utils.Metadata): Metadata handler for the patterns run.
    """

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    log.info("Reading files")

    categories = None
    if cfg.heatmap.label:
        categories = pd.read_csv(md.get_table_path("lexemes"), index_col=0)[cfg.heatmap.label]

    # Loading files and paths
    defective = cfg.pats.defective
    overabundant = cfg.pats.overabundant

    # Initializing segments
    sounds_file_name = md.get_table_path("sounds")
    segments.Inventory.initialize(sounds_file_name)

    # Loading paradigms
    paradigms = Paradigms(md.paralex, defective=defective, overabundant=overabundant,
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
    patterns.from_file(patterns_md,
                       paradigms.data,
                       defective=defective,
                       overabundant=False,
                       force=cfg.force,
                       )

    log.info("Looking for microclasses")
    microclasses = find_microclasses(paradigms, patterns)

    log.info("Computing distances")
    distances = distance_matrix(patterns, microclasses)

    log.info("Drawing")
    microclass_heatmap(distances, md, labels=categories,
                       cmap_name=cfg.heatmap.cmap,
                       exhaustive_labels=cfg.heatmap.exhaustive_labels)
