#!usr/bin/python3
# -*- coding: utf-8 -*-
"""author: Sacha Beniamine.

Compute conditional entropies in inflectional patterns.
"""

import logging

from hydra.core.hydra_config import HydraConfig

from .entropy.distribution import PatternDistribution, SplitPatternDistribution
from .representations import segments, patterns, create_paradigms, create_features
from .representations.frequencies import Frequencies

log = logging.getLogger()


def H_command(cfg, md):
    r"""Compute entropies of flexional paradigms' distributions."""
    verbose = HydraConfig.get().verbose is not False
    md.bipartite = False
    if type(cfg.data) is not str or type(cfg.patterns) is not str:
        assert len(cfg.data) == len(
            cfg.patterns) == 2, "You must pass either a single dataset and patterns file, or a list of two of each (coindexed)."
        md.bipartite = True

    Frequencies.initialize(md.datasets[0], real=True)

    patterns_file_path = cfg.patterns if md.bipartite else [cfg.patterns]
    sounds_file_name = md.get_table_path("sounds")

    preds = [cfg.entropy.n] if type(cfg.entropy.n) is int else sorted(cfg.entropy.n)
    onePred = preds[0] == 1
    if onePred:
        preds.pop(0)

    cells = cfg.cells
    if cells and len(cells) == 1:
        raise ValueError("You can't provide only one cell.")

    segments.Inventory.initialize(sounds_file_name)

    # Inflectional paradigms: columns are cells, rows are lexemes.
    paradigms = create_paradigms(md.datasets[0], defective=True, overabundant=False,
                                 merge_cols=cfg.entropy.merged,
                                 segcheck=True, cells=cells,
                                 sample=cfg.sample,
                                 most_freq=cfg.most_freq)
    pat_table, pat_dic = patterns.from_csv(patterns_file_path[0], defective=True,
                                           overabundant=False)

    if pat_table.shape[0] < paradigms.shape[0]:
        log.info(
            "It looks like you ignored defective rows when computing patterns. I'll drop all defectives.")
        paradigms = paradigms[(paradigms != "").all(axis=1)]

    if verbose and len(pat_table.columns) > 10:
        log.warning("Using verbose mode is strongly "
                    "discouraged on large (>10 cells) datasets."
                    "You should probably stop this process now.")

    if cfg.entropy.features is not None:
        features = create_features(md, cfg.entropy.features)
    else:
        features = None

    if md.bipartite:
        names = [p.name for _, p in md.datasets]
        paradigms2 = create_paradigms(md.datasets[0], defective=True,
                                      overabundant=False,
                                      merge_cols=cfg.entropy.merged, segcheck=True,
                                      cells=cells)
        paradigms2 = paradigms2.loc[paradigms.index, :]
        pat_table2, pat_dic2 = patterns.from_csv(patterns_file_path[1], defective=True,
                                                 overabundant=False)

        log.info("Looking for classes of applicable patterns")
        classes = patterns.find_applicable(paradigms, pat_dic)
        classes2 = patterns.find_applicable(paradigms2, pat_dic2)
        log.debug("Classes:")
        log.debug(classes)
        log.debug(classes2)

        distrib = SplitPatternDistribution([paradigms, paradigms2],
                                           [pat_table, pat_table2],
                                           [classes, classes2],
                                           names,
                                           features=features)

        distrib.mutual_information()
        mean1 = distrib.distribs[0].measures.get_mean()
        mean2 = distrib.distribs[1].measures.get_mean()
        mean3 = distrib.measures.get_mean(measure="mutual_information")
        mean4 = distrib.measures.get_mean(measure="normalized_mutual_information")
        log.debug("Mean remaining H(c1 -> c2) for %s = %s", names[0], mean1)
        log.debug("Mean remaining H(c1 -> c2) for %s = %s", names[1], mean2)
        log.debug("Mean I(%s,%s) = %s", *names, mean3)
        log.debug("Mean NMI(%s,%s) = %s", *names, mean4)

    else:
        log.info("Looking for classes of applicable patterns")
        classes = patterns.find_applicable(paradigms, pat_dic)
        log.debug("Classes:")
        log.debug(classes)
        distrib = PatternDistribution(paradigms,
                                      pat_table,
                                      classes,
                                      "&".join([p.name for p in md.datasets]),
                                      features=features)

    if onePred:
        if not md.bipartite:  # Already computed in bipartite systems :)
            distrib.one_pred_entropy()
        mean = distrib.measures.get_mean(weighting=cfg.entropy.weighting)
        log.info("Mean H(c1 -> c2) = %s ", mean)
        if verbose:
            distrib.one_pred_distrib_log()

    if preds:
        if cfg.entropy.importFile:
            distrib.measures.import_file(cfg.entropy.importFile)

        for n in preds:
            distrib.n_preds_entropy_matrix(n)
            mean = distrib.measures.get_mean(n=n, weighting=cfg.entropy.weighting)
            log.info(f"Mean H(c1, ..., c{n} -> c) = {mean}")

            if verbose:
                distrib.n_preds_distrib_log(n)

    ent_file = md.register_file('entropies.csv',
                                {'computation': 'entropies',
                                 'content': 'results'})
    log.info("Writing to: {}".format(ent_file))
    distrib.measures.export_file(ent_file, weighting=cfg.entropy.weighting)

    return ent_file
