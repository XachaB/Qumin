# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Show microclass similarity

Author: Sacha Beniamine.
"""
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from .clustering import find_microclasses
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
import logging
import hydra
from .utils import Metadata

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
    name = md.register_file("microclassHeatmap.pdf",
                            {"computation": "microclass_heatmap",
                             "content": "figure"})
    log.info("Saving file to: " + name)
    plt.savefig(name, bbox_inches='tight', pad_inches=0, transparent=True)


def distance_matrix(pat_table, microclasses, **kwargs):
    """Returns a similarity matrix from a pattern dataframe and microclasses"""
    poplist = list(microclasses)
    dists = squareform(pdist(pat_table.loc[poplist, :], metric=lambda x, y: sum((a != b) for a, b in zip(x, y))))
    distances = pd.DataFrame(dists, columns=poplist, index=poplist)
    for x in poplist:
        distances.at[x, x] = 0
    distances.columns = [x.split()[0] for x in poplist]
    distances.index = list(distances.columns)
    return distances


def heatmap_command(cfg, md):
    r"""Draw a clustermap of microclass similarities using seaborn.
    """
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    log.info(cfg)
    log.info("Reading files")

    categories = None
    if cfg.heatmap.label:
        categories = pd.read_csv(md.get_table_path("lexemes"), index_col=0)[cfg.heatmap.label]
    pat_table = pd.read_csv(cfg.patterns, index_col=0)
    log.info("Looking for microclasses")
    microclasses = find_microclasses(pat_table)
    log.info("Computing distances")
    distances = distance_matrix(pat_table, microclasses)
    log.info("Drawing")
    microclass_heatmap(distances, md, labels=categories,
                       cmap_name=cfg.heatmap.cmap,
                       exhaustive_labels=cfg.heatmap.exhaustive_labels)
