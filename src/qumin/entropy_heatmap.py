# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Show entropy results as heatmap

Author: Jules Bouton.
"""
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import pandas as pd
import numpy as np
import seaborn as sns
import logging
import os
import argparse
from .utils import ArgumentDefaultsRawTextHelpFormatter, Metadata

log = logging.getLogger()


def get_features_order(features_file, results, sort_order=False):
    """Returns an ordered list of the cells from a Paralex compliant
    cell file."""
    log.info("Reading features")
    features = pd.read_csv(features_file, index_col=0)
    df = results.reset_index()
    cells = set(df['pred'].to_list() + df['out'].to_list())
    df_c = pd.DataFrame(index=list(cells))
    for c in cells:
        feat_order = {}

        for f in c.split('.'):
            feat_order[features.loc[f, 'feature']] = features.loc[f, 'canonical_order']
        for f, v in feat_order.items():
            df_c.loc[c, f] = v

    if not sort_order:
        sort_order = list(df_c.columns)
    return df_c.sort_values(by=sort_order, axis=0).index.to_list()


def entropy_heatmap(results, md, cmap_name="vlag",
                    feat_order=None, short_name=False, annot=False,
                    beta=False):
    """Make a FacetGrid heatmap of all metrics.

    Arguments:
        results (:class:`pandas:pandas.DataFrame`):
            a results DataFrame as produced by calc_paradigm_entropy.
        md: MetaData handler
        feat_order (List[str]): an ordered list of each cell name.
            Used to sort the labels.
        cmap_name (str): name of the cmap to use.
        short_name (bool): whether to use short cell names or not.
        annot (bool): whether to add an annotation overlay.

    """

    log.info("Drawing")
    df = results['metrics'].stack()
    df = df.reset_index().rename({0: 'values', 'params': 'beta', 'name': 'metric'}, axis=1)
    df = df[df['metric'] != 'effectifs']
    if beta:
        df = df[df['beta'].isin(beta)]

    df.replace(['entropies', 'accuracies'],
               ['Utterance probability', 'Successful utterance probability'],
               inplace=True)

    def draw_heatmap(*args, **kwargs):
        df = kwargs.pop('data')
        annot = kwargs.pop('annot')

        # For entropies, we want a reversed colormap.
        if 'Utterance probability' in list(df['metric']):
            reverse = ''
        else:
            reverse = '_r'

        df.index.name = 'predictor'
        df.columns.name = 'target'
        df = df.pivot(index=args[0], columns=args[1], values=args[2])

        if feat_order:
            df = df[feat_order]
            df = df.reindex(feat_order)
        else:
            df = df.reindex(list(df.columns))

        df = df.replace([np.nan], 0)

        if short_name:
            def shorten(x):
                return ".".join([f[0].capitalize() for f in x.split('.')])

            df.index = pd.Index(df.index.to_series().map(lambda i: shorten(i)))
            df.columns = pd.Index(df.columns.to_series().map(lambda i: shorten(i)))

        # Additional options for rendering.
        # Annotations.
        if annot:
            annot = df.copy().round(2)

        # Mask (diagonal)
        df_m = pd.DataFrame(columns=df.columns, index=df.index)
        for col in df.columns:
            df_m.loc[df_m.index == col, [col]] = True

        # Drawing each individual heatmap
        sns.heatmap(df,
                    annot=annot,
                    mask=df_m,
                    cmap=plt.get_cmap(cmap_name+reverse),
                    fmt=".2f",
                    linewidths=2,
                    **kwargs)

    # Plotting the heatmap
    cg = sns.FacetGrid(df, col='metric', row='beta', height=4, margin_titles=True)
    cg.map_dataframe(draw_heatmap, 'pred', 'out', 'values', annot=annot, square=True, cbar=False)

    # Setting labels
    cg.tick_params(axis='x', labelbottom=False, labeltop=True,
                   bottom=False, top=True,
                   labelrotation=90)
    cg.set_ylabels('Predictor')
    cg.set_xlabels('Target')
    cg.set_titles(row_template='{row_var} is {row_name}', col_template='{col_name}')
    cg.tight_layout()

    # We add a custom global colorbar
    # The last value is the width
    cbar_ax = cg.fig.add_axes([0.09, -0.06, 0.84, 0.04])
    cbar = cg.fig.colorbar(cm.ScalarMappable(norm=None, cmap=plt.get_cmap(cmap_name)),
                           cax=cbar_ax,#cg.fig.axes,
                           drawedges=False,
                           orientation='horizontal'
                           )
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Easy', 'Balanced', 'Hard'])
    cbar.outline.set_visible(False)

    name = md.register_file("entropyHeatmap.png",
                            {"computation": "entropy_heatmap",
                             "content": "figure"})
    log.info("Saving file to: " + name)
    cg.savefig(name, pad_inches=0)


def main(args):
    r"""Draw a heatmap of results similarities using seaborn.

    For a detailed explanation, see the html doc.::

          ____
         / __ \                    /)
        | |  | | _   _  _ __ ___   _  _ __
        | |  | || | | || '_ ` _ \ | || '_ \
        | |__| || |_| || | | | | || || | | |
         \___\_\ \__,_||_| |_| |_||_||_| |_|
          Quantitative modeling of inflection

    """
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    log.info(args)
    log.info("Reading files")

    md = Metadata(args, __file__)

    results = pd.read_csv(args.results, sep="\t", index_col=[0, 1, 2], header=[0, 1])
    feat_order = get_features_order(args.features, results, args.order)

    entropy_heatmap(results, md,
                    cmap_name=args.cmap,
                    feat_order=feat_order,
                    short_name=args.dense,
                    beta=args.beta)
    md.save_metadata()


def heatmap_command():
    parser = argparse.ArgumentParser(description=main.__doc__,
                                     formatter_class=ArgumentDefaultsRawTextHelpFormatter)

    parser.add_argument("results",
                        help="results file, full path (csv or tsv)",
                        type=str)

    parser.add_argument("features",
                        help="features file, full path (csv or tsv)",
                        type=str)

    parser.add_argument("--order",
                        help="Priority list for sorting features",
                        nargs='+',
                        type=str,
                        default=False)

    parser.add_argument("-c", "--cmap",
                        help="cmap name",
                        type=str,
                        default="vlag")

    parser.add_argument("-e", "--exhaustive_labels",
                        help="by default, seaborn shows only some labels on the heatmap for readability."
                             " This forces seaborn to print all labels.",
                        action="store_true")

    parser.add_argument("-d", "--dense",
                        help="Will use initials instead of full labels",
                        action="store_true", default=False)

    options = parser.add_argument_group('Options')

    options.add_argument("-v", "--verbose",
                         help="Activate debug logs.",
                         action="store_true", default=False)

    options.add_argument("-f", "--folder",
                         help="Output folder name",
                         type=str, default=os.getcwd())

    options.add_argument("--beta",
                         help="Compute only for specific values of beta",
                         nargs="+",
                         type=int, default=False)

    args = parser.parse_args()

    main(args)


if __name__ == '__main__':
    heatmap_command()
