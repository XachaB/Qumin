#!usr/bin/python3
# -*- coding: utf-8 -*-
"""author: Sacha Beniamine.

Compute conditional entropies in inflectional patterns.
"""

import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from .entropy.distribution import PatternDistribution, SplitPatternDistribution
# Our libraries
from .representations import segments, patterns, create_paradigms, create_features
from .utils import Metadata

log = logging.getLogger()

@hydra.main(version_base=None, config_path="config", config_name="entropy")
def H_command(cfg):
    r"""Compute entropies of flexional paradigms' distributions."""
    verbose = HydraConfig.get().verbose is not False
    md = Metadata(cfg, __file__)
    md.bipartite = False
    if type(cfg.data) is not str or type(cfg.patterns) is not str:
        assert len(cfg.data) == len(
            cfg.patterns) == 2, "You must pass either a single dataset and patterns file, or a list of two of each (coindexed)."
        md.bipartite = True

    patterns_file_path = cfg.patterns if md.bipartite else [cfg.patterns]
    sounds_file_name = md.get_table_path("sounds")
    paradigms_file_path = md.get_table_path("forms")

    preds = [cfg.entropy.n] if type(cfg.entropy.n) is int else sorted(cfg.entropy.n)
    onePred = preds[0] == 1
    if onePred:
        preds.pop(0)

    cells = cfg.cells
    if cells and len(cells) == 1:
        raise ValueError("You can't provide only one cell.")

    log.info(cfg)

    segments.Inventory.initialize(sounds_file_name)

    # Inflectional paradigms: columns are cells, rows are lexemes.
    paradigms = create_paradigms(paradigms_file_path, defective=True, overabundant=False,
                                 merge_cols=cfg.entropy.merged,
                                 segcheck=True, cells=cells)
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
        paradigms_file_path_2 = md.get_table_path("forms", num=1)
        paradigms2 = create_paradigms(paradigms_file_path_2, defective=True,
                                      overabundant=False,
                                      merge_cols=cfg.entropy.merged, segcheck=True,
                                      cells=cells)
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
        if cfg.entropy.comp:
            computation = 'bipartiteEntropies'
            ent_file1 = md.register_file('bipartite1.csv',
                                         {'computation': computation,
                                          'source': names[0],
                                          'content': 'entropies'})
            ent_file2 = md.register_file('bipartite2.csv',
                                         {'computation': computation,
                                          'source': names[1],
                                          'content': 'entropies'})
            I = md.register_file('I.csv', {'computation': computation,
                                           'source': names,
                                           'content': 'I'})
            NMI = md.register_file('NMI.csv',
                                   {'computation': computation,
                                    'source': names,
                                    'content': 'NMI'})

            distrib.distribs[0].one_pred_entropy()
            entropies1 = distrib.distribs[0].entropies[1]
            distrib.distribs[1].one_pred_entropy()
            entropies2 = distrib.distribs[1].entropies[1]
            mutual = distrib.mutual_information()
            normmutual = distrib.mutual_information(normalize=True)

            log.info("Writing to:" + "\n\t".join([ent_file1, ent_file2, I, NMI]))
            entropies1.to_csv(ent_file1, sep="\t")
            entropies2.to_csv(ent_file2, sep="\t")
            mutual.to_csv(I, sep="\t")
            normmutual.to_csv(NMI, sep="\t")
            if verbose:
                # mean on df's index, then on Series' values.
                mean1 = entropies1.mean().mean()
                mean2 = entropies2.mean().mean()
                mean3 = mutual.mean().mean()
                mean4 = normmutual.mean().mean()
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
                                      features=features)

    if onePred:
        distrib.one_pred_entropy()
        mean = distrib.data.loc[(distrib.data.loc[:, "n_preds"] == 1) &
                                (distrib.data.loc[:, "measure"] == "cond_entropy")
                                , "value"].mean()
        log.info("Mean H(c1 -> c2) = %s ", mean)
        if verbose:
            distrib.one_pred_distrib_log()

    if preds:
        if cfg.entropy.importFile:
            distrib.import_file(cfg.entropy.importFile)

        for n in preds:
            distrib.n_preds_entropy_matrix(n)
            n_entropies = distrib.data.loc[(distrib.data["n_preds"] == n) &
                                (distrib.data.loc[:, "measure"] == "cond_entropy"), "value"]
            mean = n_entropies.mean()
            log.info(f"Mean H(c1, ..., c{n} -> c) = {mean}")

            if verbose:
                distrib.n_preds_distrib_log(n)

    ent_file = md.register_file('entropies.csv',
                                {'computation': 'entropies',
                                 'content': 'results'})
    log.info("Writing to: {}".format(ent_file))
    distrib.export_file(ent_file)

    md.save_metadata()
