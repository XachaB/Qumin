# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Show microclass similarity

Author: Sacha Beniamine.
"""
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from clustering import find_microclasses
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches


def microclass_heatmap(distances, path, labels=None, cmap_name="BuPu", exhaustive_labels=False):
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

    name = path + "_microclassHeatmap.pdf"
    print("Saving file to: ", name)
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


def main(args):
    r"""Draw a clustermap of microclass similarities using seaborn.

    For a detailed explanation, see the html doc.::
      ____
     / __ \                    /)
    | |  | | _   _  _ __ ___   _  _ __
    | |  | || | | || '_ ` _ \ | || '_ \
    | |__| || |_| || | | | | || || | | |
     \___\_\ \__,_||_| |_| |_||_||_| |_|
      Quantitative modeling of inflection

    """
    print("Reading files")
    categories = None
    if args.labels:
        categories = pd.read_csv(args.labels, index_col=0, squeeze=True)
    pat_table = pd.read_csv(args.patterns, index_col=0)
    print("Looking for microclasses")
    microclasses = find_microclasses(pat_table)
    print("Computing distances")
    distances = distance_matrix(pat_table, microclasses)
    print("Drawing")
    microclass_heatmap(distances, args.output, labels=categories,
                       cmap_name=args.cmap,
                       exhaustive_labels=args.exhaustive_labels)


if __name__ == '__main__':
    import argparse

    usage = main.__doc__

    parser = argparse.ArgumentParser(description=usage, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("patterns",
                        help="patterns file, full path"
                             " (csv separated by ‘, ’)",
                        type=str)

    parser.add_argument("-l", "--labels",
                        help="csv files with class membership to compare"
                             " (csv separated by ‘, ’)",
                        type=str,
                        default=None)

    parser.add_argument("-c", "--cmap",
                        help="cmap name",
                        type=str,
                        default="BuPu")

    parser.add_argument("-e", "--exhaustive_labels",
                        help="by default, seaborn shows only some labels on the heatmap for readability."
                             " This forces seaborn to print all labels.",
                        action="store_true")

    parser.add_argument("output",
                        help="output_path",
                        type=str)

    args = parser.parse_args()

    print(args)
    main(args)
