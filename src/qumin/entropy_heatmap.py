# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Show entropy results as heatmap

Author: Jules Bouton.
"""
import logging

import numpy as np
import pandas as pd
import seaborn as sns
from frictionless.exception import FrictionlessException
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt

# Prevent matplotlib font manager from spamming the log
logging.getLogger('matplotlib.font_manager').disabled = True
log = logging.getLogger("Qumin")


def get_zones(df, threshold=0):
    """ Cluster cells into 'zones' of inter-predictibility.

    By default, two cells are interpredictible if the entropy in both direction is 0.
    Threshold can be used (eg. set to 0.005) to get zones of very good predictibility instead.

    Args:
        df: results table with conditional entropies
        threshold: below this threshold, consider the zones interpredictible.

    Returns:
        a dictionary of cells to zone indexes (clusters).
    """
    is_cond_ent = df["measure"] == "cond_entropy"
    is_one_pred = df["n_preds"] == 1
    df = df[is_cond_ent & is_one_pred].pivot_table(index="predictor",
                                                   columns="predicted", values="value")
    clusters = {x: None for x in df.index}
    n_clusters = 1
    for x in clusters:
        row = df.loc[x, :]
        mates_row = (row.fillna(False) <= threshold)
        col = df.loc[:, x]
        mates_col = (col.fillna(False) <= threshold)
        mates = row[mates_row & mates_col].index
        if clusters[x] is None:
            for m in mates:
                if clusters[m] is not None:
                    clusters[x] = clusters[m]
                    break
            if clusters[x] is None:
                clusters[x] = n_clusters
                n_clusters += 1
        for m in mates:
            if clusters[m] is None:
                clusters[m] = clusters[x]
    return clusters


def zones_heatmap(results, md, features, cell_order=None, cols=None):
    """ Produces a heatmap of zones of interpredictibility clusters.

    Args:
        results: qumin entropy result file
        md: frictionless metadata for the paralex dataset
        features: feature table from the dataset
        cell_order: list of cells sorted in a predefined way
        cols: list of feature dimensions to use as columns (the rest will be cells)

    Returns:
        Clusters of zones with total predictibility.
    """

    def partial_cell(c, decomposed, dims):
        partials = [v for v in decomposed.loc[c, dims] if not pd.isna(v)]
        return ".".join(partials)

    def zone_table_one(clusters, ax):
        table = pd.DataFrame(clusters.items(), columns=["cell", "zones"])
        if cols:
            decomposed = decompose(features, list(table["cell"]))
            table["cols"] = table["cell"].apply(lambda c: partial_cell(c, decomposed, cols))
            rows = [f for f in decomposed.columns if f not in cols]
            table["rows"] = table["cell"].apply(lambda c: partial_cell(c, decomposed, rows))
        else:
            table["rows"] = table["cell"]
            table["cols"] = ""

        table = table.sort_values("cell", key=lambda s: s.apply(cell_order.index))
        maxi = table["zones"].max()
        table = table.pivot_table(index="rows", columns="cols", values="zones", sort=False)
        palette = sns.color_palette(n_colors=maxi)
        g = sns.heatmap(table, cmap=palette, square=True, xticklabels=True, yticklabels=True,
                        cbar=False, annot=True, linewidths=0, ax=ax)
        g.set_title(f"Entropy threshold: {t} ({maxi} zones)")
        g.set_facecolor('white')

    dataset = results["dataset"].iloc[0]

    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    for ax, t in zip(axes.flat, [0.005, 0]):
        clusters = get_zones(results, threshold=t)
        zone_table_one(clusters, ax)
    fig.suptitle(f"Zones of inter-predictibility for {dataset}")
    plt.tight_layout()

    name = md.register_file("zonesTable.png",
                            {"computation": "zone_table",
                             "content": "figure"})

    log.info("Writing zones table to: " + name)
    plt.savefig(name, pad_inches=0.1)
    return clusters  # this is the last computed: with 0 entropy threshold


def decompose(features, cells):
    """ Decompose a set of cells to separate each of their feature-value by feature.

    Args:
        features: features-values file from paralex
        cells: list of cells

    Returns:
        A dataframe with the cells as indexes, the features (dimensions) as columns,
            giving for their intersection the corresponding value.
            NaNs are present if a cell is not specified for a feature.
    """
    df_c = pd.DataFrame(index=list(cells))
    for c in cells:
        for v in c.split('.'):
            f = features.loc[v, 'feature']
            df_c.loc[c, f] = v
    return df_c


def get_features_order(features, results, sort_order=False):
    """Returns an ordered list of the cells from a Paralex compliant
    cell file."""
    if features is not None:

        df = results.reset_index()
        cells = sorted(set(df.predictor.to_list() + df.predicted.to_list()))

        # Handle multiple predictor format ('cellA&cellB')
        cells = sorted(set(sum([x.split('&') for x in cells], [])))

        df_c = decompose(features, list(cells))
        df_c = df_c.map(lambda f: features.loc[f, 'canonical_order'] if not pd.isna(f) else f)
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


def _draw_heatmap(*args, cmap=None, cmap_freqs=None, cell_freqs=None,
                  feat_order=None, dense=False, **kwargs):
    """
    Draws a heatmap in a FacetGrid with custom parameters

    Arguments:
        cmap: Colormap used for the metrics heatmap.
        cmap_freqs: Colormap used for the frequencies scale.
        feat_order (List[str]): an ordered list of each cell name.
            Used to sort the labels.
    """
    df = kwargs.pop('data')
    annot = kwargs.pop('annotate')
    types = df["type"].unique()
    df = df.pivot(index=args[0], columns=args[1], values=args[2])
    df.index.name = 'predictor'
    df.columns.name = 'predicted'

    # For n_pairs, we want a specific set of parameters.
    if 'Number of pairs' in types:
        hm_cmap = "gray_r"
        annot = df.shape[0] < 10
        fmt = ".0f"
        cell_colors = False
    else:
        hm_cmap = cmap
        fmt = ".2f"
        cell_colors = cell_freqs is not None

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

    freqs_mask = pd.DataFrame(False, columns=df.columns, index=df.index)

    if cell_colors:
        row_freqs = cell_freqs.loc[df.index]
        col_freqs = pd.DataFrame(cell_freqs.loc[df.columns]).T
        col_freqs.insert(0, "frequency", 0)
        df = pd.concat([row_freqs, df], axis=1)
        df = pd.concat([col_freqs, df], axis=0)
        df.index.name = 'predictor'
        df.columns.name = 'predicted'
        freqs_mask = pd.DataFrame(False, columns=df.columns, index=df.index)
        freqs_mask.iloc[:, 0] = True
        freqs_mask.iloc[0, :] = True

    # Annotations on the heatmap
    if annot:
        annot = df.copy().round(2)

    # Mask (diagonal)
    diag_mask = pd.DataFrame(False, columns=df.columns, index=df.index)
    for col in df.columns:
        tmp = diag_mask.index.to_frame().predictor.str.split('&').apply(lambda x: col in x)
        diag_mask.loc[tmp, [col]] = True

    # Drawing each individual heatmap
    ax = sns.heatmap(df,
                     annot=annot,
                     mask=diag_mask | freqs_mask,
                     cmap=hm_cmap,
                     fmt=fmt,
                     linewidths=1,
                     vmin=0,
                     cbar=True,
                     cbar_kws=dict(location='bottom',
                                   shrink=0.6,
                                   pad=0.075),  # Spacing between colorbar and hm
                     **kwargs)

    ax.tick_params(axis='x', labelbottom=False, labeltop=True,
                   bottom=False, top=False,
                   labelrotation=0 if dense else 90)

    if cell_colors:
        # Plotting first rows & columns (frequency) in a different cmap
        ax = sns.heatmap(df,
                         annot=annot,
                         mask=diag_mask | ~freqs_mask,
                         cmap=cmap_freqs,
                         fmt=fmt,
                         vmin=0,
                         vmax=cell_freqs.max(),
                         linewidths=1,
                         **kwargs)
        ax.tick_params(axis='x', labelbottom=False, labeltop=True,
                       bottom=False, top=False,
                       labelrotation=0 if dense else 90)


def entropy_heatmap(results, md, cmap_name=False, freq_margins=True,
                    dense=False, annotate=False,
                    n_pairs=False, debug=False, filename="entropyHeatmap.png", **kwargs):
    """Make a FacetGrid heatmap of all metrics

    Arguments:
        results (:class:`pandas:pandas.DataFrame`):
            a results DataFrame as produced by calc_paradigm_entropy.
        md (qumin.utils.Metadata): MetaData handler to get access to file location.
        cmap_name (str): name of the cmap to use. Defaults to the following cubehelix
            map, `sns.cubehelix_palette(start=2, rot=0.5, dark=0, light=1, as_cmap=True)`.
        dense (bool): whether to use short cell names or not.
        annotate (bool): whether to add an annotation overlay.
        filename (str): filename to save the heatmap.
        freq_margins (bool):  whether to add cell frequency margins to dataframe.
        n_pairs (bool): whether to display a heatmap of the number of pairs.
        debug (bool): whether to display a heatmap with debug results.
        **kwargs: Optional arguments are passed to `_draw_heatmap()`
    """

    if not cmap_name:
        cmap = sns.cubehelix_palette(start=2, rot=0.5, dark=0, light=1, as_cmap=True)
    else:
        cmap = plt.get_cmap(cmap_name)

    df = results[['measure', 'value', 'n_pairs', 'n_preds']
                 ].set_index(['measure', 'n_preds'], append=True
                             ).stack().reset_index()

    df.rename(columns={"level_4": "type", 0: "value"}, inplace=True)

    cell_freqs = None
    cmap_freqs = None
    if freq_margins and (results.probability_source == "tokens").all():
        cell_freqs = results.reset_index('predicted').pred_probability.drop_duplicates()
        cell_freqs.name = "frequency"
        cmap_freqs = sns.color_palette("mako_r", as_cmap=True)

    # Clean debug info
    df['debug'] = df.measure.str.endswith('_debug')
    df.loc[df.debug, 'measure'] = df.loc[df.debug].measure.str.replace('_debug', '')

    # Rename measures
    names = {'cond_entropy': 'Conditional entropy'}
    df.loc[:, 'measure'] = df.measure.map(names)

    # Drop unnecessary rows
    if not n_pairs:
        df.drop(df[df.type == "n_pairs"].index, axis=0, inplace=True)
    if not debug:
        df.drop(df.loc[df.debug].index, axis=0, inplace=True)

    # Decide layout
    if not n_pairs and not debug:
        orient_param = dict(col="measure")
    elif not n_pairs:
        orient_param = dict(col="measure", row="debug")
        df.debug = df.debug.replace({True: "debug value", False: "normal value"})
    else:
        orient_param = dict(row="measure", col="type")
        df.loc[df.debug, "measure"] = df.loc[df.debug, "measure"] + " (debug)"

    # Rename metric types
    df.loc[:, 'type'] = df.type.replace({'value': 'Result', 'n_pairs': 'Number of pairs'})

    if len(df.n_preds.unique()) > 1:
        df.measure += df.n_preds.apply(lambda x: f" (n={x})")

    # Compute a suitable size for the heatmaps
    height = 4 + round(len(df['predictor'].unique()) / 4)

    # Plotting the heatmap
    cg = sns.FacetGrid(df, **orient_param,
                       height=height, margin_titles=True,
                       sharex=False, sharey=False)
    cg.set_titles(row_template='{row_name}', col_template='{col_name}')

    cg.map_dataframe(_draw_heatmap, 'predictor', 'predicted', 'value', 'reverse',
                     annotate=annotate, cmap=cmap,
                     cmap_freqs=cmap_freqs, cell_freqs=cell_freqs,
                     dense=dense, square=True,
                     **kwargs)

    cg.tick_params(axis='y', labelrotation=0)

    cg.set_axis_labels(x_var="Predicted", y_var="Predictor")
    cg.fig.suptitle(f"Measured on the {md.dataset.name} dataset, version {md.dataset.version}")

    cg.tight_layout()

    name = md.register_file(filename,
                            {"computation": "entropy_heatmap",
                             "content": "figure"})

    log.info("Writing heatmap to: " + name)
    cg.savefig(name, pad_inches=0.1)


def ent_heatmap_command(cfg, md):
    r"""Draw a heatmap of results similarities and a plot of zones.
    """
    verbose = HydraConfig.get().verbose is not False
    results = pd.read_csv(cfg.entropy.importFile, index_col=[0, 1])

    if not verbose:  # Remove debug results
        is_debug = results["measure"].str.endswith("_debug")
        results = results[~is_debug]
    try:
        features_file_name = md.get_table_path("features-values")
    except FrictionlessException:
        features_file_name = None
        log.warning("Your package doesn't contain any features-values file. "
                    "You should provide an ordered list of cells in command line.")

    features = None
    if features_file_name:
        log.info("Reading features")
        features = pd.read_csv(features_file_name, index_col=0)
    feat_order = get_features_order(features, results, cfg.heatmap.order)

    log.info("Drawing a heatmap of the results...")
    entropy_heatmap(results, md,
                    cmap_name=cfg.heatmap.cmap,
                    feat_order=feat_order,
                    dense=cfg.heatmap.dense,
                    annotate=cfg.heatmap.annotate,
                    n_pairs=cfg.heatmap.display.n_pairs,
                    debug=cfg.heatmap.display.debug,
                    freq_margins=cfg.heatmap.display.freq_margins)

    log.info("Drawing zones of interpredictibility...")
    clusters = zones_heatmap(results, md, features, cell_order=feat_order, cols=cfg.heatmap.cols)

    cl_found = set()
    distillation = []
    for c in feat_order:
        if clusters[c] not in cl_found:
            cl_found.add(clusters[c])
            distillation.append(c)

    if len(distillation) > 1:
        log.info("Drawing a heatmap of a distillation of the results...")
        distil_index = results.index.isin(distillation, level=0)
        distil_col = results.index.isin(distillation, level=1)
        result_subset = results.loc[distil_index & distil_col]
        entropy_heatmap(result_subset, md,
                        cmap_name=cfg.heatmap.cmap,
                        feat_order=distillation,
                        dense=cfg.heatmap.dense,
                        annotate=cfg.heatmap.annotate,
                        n_pairs=cfg.heatmap.display.n_pairs,
                        debug=cfg.heatmap.display.debug,
                        freq_margins=cfg.heatmap.display.freq_margins,
                        filename="entropyHeatmap_distillation.png")
