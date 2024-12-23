# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Show entropy results as heatmap

Author: Jules Bouton.
"""
from matplotlib import pyplot as plt
from frictionless.exception import FrictionlessException
import pandas as pd
import numpy as np
import seaborn as sns
import logging

# Prevent matplotlib font manager from spamming the log
logging.getLogger('matplotlib.font_manager').disabled = True

log = logging.getLogger()


def get_features_order(features_file, results, sort_order=False):
    """Returns an ordered list of the cells from a Paralex compliant
    cell file."""
    if features_file:
        log.info("Reading features")
        features = pd.read_csv(features_file, index_col=0)

        df = results.reset_index()
        cells = sorted(set(df.predictor.to_list() + df.predicted.to_list()))

        # Handle multiple predictor format ('cellA&cellB')
        cells = sorted(set(sum([x.split('&') for x in cells], [])))

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
            log.warning("""No cells order provided. Falling back to alphabetical order.""")
            df = results.reset_index()
            return sorted(list(df.predictor.unique()))


def entropy_heatmap(results, md, cmap_name=False,
                    feat_order=None, dense=False, annotate=False,
                    parameter=False):
    """Make a FacetGrid heatmap of all metrics.

    Arguments:
        parameter: ## What is this ? ##
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

    df = results[['measure', 'value', 'n_pairs', 'n_preds']
                 ].set_index(['measure', 'n_preds'], append=True
                             ).stack().reset_index()

    df.rename(columns={"level_4": "type", 0: "value"}, inplace=True)
    df.loc[:, 'type'] = df.type.replace({'value': 'Result', 'n_pairs': 'Number of pairs'})
    df.loc[:, 'measure'] = df.measure.replace({'cond_entropy': 'Conditional entropy'})
    if len(df.n_preds.unique()) > 1:
        df.measure += df.n_preds.apply(lambda x: f" (n={x})")

    # Compute a suitable size for the heatmaps
    height = 4 + round(len(df['predictor'].unique())/4)

    def _draw_heatmap(*args, **kwargs):
        """ Draws a heatmap in a FacetGrid with custom parameters
        """

        df = kwargs.pop('data')
        annot = kwargs.pop('annotate')

        types = df["type"].unique()
        df.index.name = 'predictor'
        df.columns.name = 'predicted'
        df = df.pivot(index=args[0], columns=args[1], values=args[2])

        # For n_pairs, we want a specific set of parameters.
        if 'Number of pairs' in types:
            hm_cmap = "gray_r"
            annot = df.shape[0] < 10
            fmt = ".0f"
        else:
            hm_cmap = cmap
            fmt = ".2f"

        if feat_order:

            # Sorting for multiple predictors.
            def sorting(x):
                return x.apply(lambda cell: feat_order.index(cell))

            sort = df.index.to_frame().predictor.str.split('&', expand=True)
            sort = sort.sort_values(by=list(sort.columns), key=sorting, axis=0)
            df = df.reindex(sort.index)
            df = df[feat_order]

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
            tmp = df_m.index.to_frame().predictor.str.split('&').apply(lambda x: col in x)
            df_m.loc[tmp, [col]] = True

        # Drawing each individual heatmap
        sns.heatmap(df,
                    annot=annot,
                    mask=df_m,
                    cmap=hm_cmap,
                    fmt=fmt,
                    linewidths=1,
                    **kwargs)

    # Plotting the heatmap
    cg = sns.FacetGrid(df, row='measure', col='type', height=height, margin_titles=True,
                       sharex=False, sharey=False)
    cg.set_titles(row_template='{row_name}', col_template='{col_name}')

    cg.map_dataframe(_draw_heatmap, 'predictor', 'predicted', 'value',
                     annotate=annotate, square=True, cbar=True,
                     cbar_kws=dict(location='bottom',
                                   shrink=0.6,
                                   pad=0.075))  # Spacing between colorbar and hm

    # Setting labels
    rotate = 0 if dense else 90

    cg.tick_params(axis='x', labelbottom=False, labeltop=True,
                   bottom=False, top=True,
                   labelrotation=rotate)
    cg.tick_params(axis='y',
                   labelrotation=0)

    # Override general tick settings.
    for row in cg.axes:
        for hm in row:
            cb = hm.collections[-1].colorbar
            cb.outline.set_visible(True)
            cb.outline.set_linewidth(0.5)
            cb.ax.tick_params(labelbottom=True, labeltop=False,
                              bottom=True, top=False,
                              labelrotation=0)

    cg.set_axis_labels(x_var="Predicted", y_var="Predictor")
    cg.fig.suptitle(f"Measured on the {md.datasets[0].name} dataset, version {md.datasets[0].version}")

    cg.tight_layout()

    name = md.register_file("entropyHeatmap.png",
                            {"computation": "entropy_heatmap",
                             "content": "figure"})

    log.info("Writing heatmap to: " + name)

    cg.savefig(name, pad_inches=0.1)


def ent_heatmap_command(cfg, md):
    r"""Draw a heatmap of results similarities using seaborn.
    """
    log.info("Drawing a heatmap of the results...")
    results = pd.read_csv(cfg.entropy.importFile, index_col=[0, 1])
    try:
        features_file_name = md.get_table_path("features-values")
    except FrictionlessException:
        features_file_name = None
        log.warning("Your package doesn't contain any features-values file. You should provide an ordered list of cells in command line.")

    feat_order = get_features_order(features_file_name, results, cfg.heatmap.order)

    entropy_heatmap(results, md,
                    cmap_name=cfg.heatmap.cmap,
                    feat_order=feat_order,
                    dense=cfg.heatmap.dense,
                    annotate=cfg.heatmap.annotate)
