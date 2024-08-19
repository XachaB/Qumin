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

log = logging.getLogger()


def get_features_order(features_file, results, sort_order=False):
    """Returns an ordered list of the cells from a Paralex compliant
    cell file."""
    if features_file:
        log.info("Reading features")
        features = pd.read_csv(features_file, index_col=0)

        df = results.reset_index()
        cells = set(df.predictor.to_list() + df.predicted.to_list())
        df_c = pd.DataFrame(index=list(cells))
        for c in cells:
            feat_order = {}

            for f in c.split('.'):
                feat_order[features.loc[f, 'feature']] = features.loc[f, 'canonical_order']
            for f, v in feat_order.items():
                df_c.loc[c, f] = v

        if not sort_order:
            sort_order = list(df_c.columns)
        return df_c.sort_values(by=[x for x in sort_order], axis=0).index.to_list()
    else:
        if sort_order:
            return sort_order
        else:
            raise ValueError("""If no features are passed,
    the --order argument should contain an ordered list of cells.""")


def entropy_heatmap(results, md, cmap_name=False,
                    feat_order=None, dense=False, annotate=False,
                    parameter=False):
    """Make a FacetGrid heatmap of all metrics.

    Arguments:
        results (:class:`pandas:pandas.DataFrame`):
            a results DataFrame as produced by calc_paradigm_entropy.
        md (qumin.utils.Metadata): MetaData handler to get access to file location.
        cmap_name (str): name of the cmap to use. Defaults to the following cubehelix
            map, `sns.cubehelix_palette(start=2, rot=0.5, dark=0, light=1, as_cmap=True)`.
        feat_order (List[str]): an ordered list of each cell name.
            Used to sort the labels.
        dense (bool): whether to use short cell names or not.
        annotate (bool): whether to add an annotation overlay.
    """

    if not cmap_name:
        cmap = sns.cubehelix_palette(start=2, rot=0.5, dark=0, light=1, as_cmap=True)
    else:
        cmap = plt.get_cmap(cmap_name)

    log.info("Drawing...")

    df = results[['measure', 'value', 'n_pairs']
                 ].set_index('measure', append=True
                             ).stack().reset_index()
    df.rename(columns={"level_3": "type", 0: "value"}, inplace=True)

    # Compute a suitable size for the heatmaps
    height = 4 + round(len(df['predictor'].unique())/5)

    def _draw_heatmap(*args, **kwargs):
        """ Draws a heatmap in a FacetGrid with custom parameters
        """

        df = kwargs.pop('data')
        annot = kwargs.pop('annotate')

        # For n_pairs, we want a specific set of parameters.
        if 'n_pairs' in list(df['type']):
            hm_cmap = cmap
            annot = True
            fmt = ".0f"
        else:
            hm_cmap = cmap
            fmt = ".2f"

        df.index.name = 'predictor'
        df.columns.name = 'predicted'
        df = df.pivot(index=args[0], columns=args[1], values=args[2])

        if feat_order:
            df = df[feat_order]
            df = df.reindex(feat_order)
        else:
            df = df.reindex(list(df.columns))

        df = df.replace([np.nan], 0)

        # Additional options for rendering.
        # Extra short labels
        if dense:
            def shorten(x):
                return ".".join([f[0].capitalize() for f in x.split('.')])

            df.index = pd.Index(df.index.to_series().map(shorten))
            df.columns = pd.Index(df.columns.to_series().map(shorten))

        # Annotations on the heatmap
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
                    cmap=hm_cmap,
                    fmt=fmt,
                    linewidths=1,
                    **kwargs)

    # Plotting the heatmap
    cg = sns.FacetGrid(df, row='measure', col='type', height=height, margin_titles=True)
    cg.set_titles(row_template='{row_name}', col_template='{col_name}')

    cg.map_dataframe(_draw_heatmap, 'predictor', 'predicted', 'value',
                     annotate=annotate, square=True, cbar=False)

    # Setting labels
    rotate = 0 if dense else 90

    cg.tick_params(axis='x', labelbottom=False, labeltop=True,
                   bottom=False, top=True,
                   labelrotation=rotate)
    cg.tick_params(axis='y',
                   labelrotation=0)

    cg.set_ylabels('Predictor')
    cg.set_xlabels('Predicted')

    # We add a custom global colorbar
    # The last value is the width
    cbar_ax = cg.fig.add_axes([0.09, -0.04, 0.84, 0.04])
    cbar = cg.fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap),
                           cax=cbar_ax,
                           drawedges=False,
                           orientation='horizontal'
                           )
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Balanced', 'High'])
    cbar.outline.set_visible(False)

    name = md.register_file("entropyHeatmap.png",
                            {"computation": "entropy_heatmap",
                             "content": "figure"})

    log.info("Writing heatmap to: " + name)
    cg.tight_layout()
    cg.savefig(name, pad_inches=0.1)


def ent_heatmap_command(cfg, md):
    r"""Draw a heatmap of results similarities using seaborn.
    """
    log.info("Drawing a heatmap of the results...")
    results = pd.read_csv(cfg.ent_hm.results, index_col=[0, 1])
    features_file_name = md.get_table_path("features-values")
    feat_order = get_features_order(features_file_name, results, cfg.ent_hm.order)

    entropy_heatmap(results, md,
                    cmap_name=cfg.ent_hm.cmap,
                    feat_order=feat_order,
                    dense=cfg.ent_hm.dense,
                    annotate=cfg.ent_hm.annotate)
