#!usr/bin/python3
# -*- coding: utf-8 -*-
"""author: Sacha Beniamine.

Compute conditional entropies in inflectional patterns.
"""

import logging

from hydra.core.hydra_config import HydraConfig

from .entropy.distribution import PatternDistribution, SplitPatternDistribution
from .representations import segments, patterns, create_paradigms, create_features
from .utils import check_pattern_cells

log = logging.getLogger()


def H_command(cfg, md):
    r"""Compute entropies of flexional paradigms' distributions."""
    verbose = HydraConfig.get().verbose is not False
    md.bipartite = False
    if type(cfg.data) is not str or type(cfg.patterns) is not str:
        assert len(cfg.data) == len(
            cfg.patterns) == 2, "You must pass either a single dataset and patterns file, or a list of two of each (coindexed)."
        md.bipartite = True

    patterns_file_path = cfg.patterns if md.bipartite else [cfg.patterns]
    sounds_file_name = md.get_table_path("sounds")
    real_frequencies = cfg.entropy.real_frequencies
    frequencies_file_path = md.get_table_path("frequencies") if real_frequencies is not None else None

    preds = [cfg.entropy.n] if type(cfg.entropy.n) is int else sorted(cfg.entropy.n)

    overabundant = cfg.entropy.overabundant
    token = cfg.entropy.token
    debug = cfg.entropy.debug  # remove debug option later

    onePred = preds[0] == 1
    if onePred:
        preds.pop(0)

    cells = cfg.cells
    if cells and len(cells) == 1:
        raise ValueError("You can't provide only one cell.")
    segments.Inventory.initialize(sounds_file_name)

    if real_frequencies is None and token:
        log.warning('Frequency computation required but no frequencies were provided.')
        log.warning('Falling back to type frequencies.')
        token = False

    # Inflectional paradigms: columns are cells, rows are lexemes.
    paradigms = create_paradigms(md.datasets[0], defective=True,
                                 overabundant=overabundant,
                                 merge_cols=cfg.entropy.merged,
                                 segcheck=True, cells=cells,
                                 sample=cfg.sample,
                                 most_freq=cfg.most_freq)
    pat_table, pat_dic = patterns.from_csv(patterns_file_path[0], defective=True,
                                           overabundant=overabundant)

    # Keep only useful columns
    if cells:
        pat_dic, pat_table = check_pattern_cells(pat_dic, pat_table, cells)

    if pat_table.shape[0] < paradigms.shape[0]:
        log.info(
            "It looks like you ignored defective rows when computing patterns. I'll drop all defectives.")
        paradigms = paradigms[(paradigms != "").all(axis=1)]

    if len(cfg.entropy.beta) > 1 and debug:
        raise NotImplementedError("Using debug mode is not possible "
                                  "with multiple values of beta.")

    if debug and len(pat_table.columns) > 10:
        log.warning("Using debug mode is strongly "
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
        mean1 = distrib.distribs[0].get_results().loc[:, "value"].mean()
        mean2 = distrib.distribs[1].get_results().loc[:, "value"].mean()
        mean3 = distrib.get_results(measure="mutual_information").loc[:, "value"].mean()
        mean4 = distrib.get_results(measure="normalized_mutual_information").loc[:, "value"].mean()
        log.debug("Mean remaining H(c1 -> c2) for %s = %s", names[0], mean1)
        log.debug("Mean remaining H(c1 -> c2) for %s = %s", names[1], mean2)
        log.debug("Mean I(%s,%s) = %s", *names, mean3)
        log.debug("Mean NMI(%s,%s) = %s", *names, mean4)

    else:
        log.info("Looking for classes of applicable patterns")

        if overabundant:
            classes = patterns.find_applicable_OA(paradigms, pat_dic)
        else:
            classes = patterns.find_applicable(paradigms, pat_dic)
        log.debug("Classes:")
        log.debug(classes)
        distrib = PatternDistribution(paradigms,
                                      pat_table,
                                      classes,
                                      "&".join([p.name for p in md.datasets]),
                                      md,
                                      real_frequencies,
                                      overabundant=overabundant,
                                      features=features)

    if onePred:
        if not md.bipartite:  # Already computed in bipartite systems :)
            if overabundant:
                distrib.one_pred_entropy_OA(cfg.entropy)
            else:
                distrib.one_pred_entropy()

        results = distrib.get_results(measure=['cond_entropy', 'accuracy'])
        mean = results.groupby(by="measure")["value"].mean()
        log.info("Mean H(c1 -> c2) and mean P(success) are \n%s ", mean.to_markdown())
        if verbose:
            if overabundant:
                distrib.one_pred_entropy_OA(cfg.entropy, debug=True)
            else:
                distrib.one_pred_distrib_log()

    if preds:
        if cfg.entropy.importFile:
            distrib.import_file(cfg.entropy.importFile)

        for n in preds:
            distrib.n_preds_entropy_matrix(n)
            mean = distrib.get_results(n=n).loc[:, "value"].mean()
            log.info(f"Mean H(c1, ..., c{n} -> c) = {mean}")

            if verbose:
                distrib.n_preds_distrib_log(n)

    ent_file = md.register_file('entropies.csv',
                                {'computation': 'entropies',
                                 'content': 'results'})
    log.info("Writing to: {}".format(ent_file))
    distrib.export_file(ent_file)
