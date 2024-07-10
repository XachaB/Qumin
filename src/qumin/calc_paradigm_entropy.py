#!usr/bin/python3
# -*- coding: utf-8 -*-
"""author: Sacha Beniamine.

Compute conditional entropies in inflectional patterns.
"""

import logging

import hydra

from .entropy.distribution import PatternDistribution, SplitPatternDistribution
# Our libraries
from .representations import segments, patterns, create_paradigms, create_features
from .utils import Metadata


@hydra.main(version_base=None, config_path="config", config_name="entropy")
def H_command(cfg):
    r"""Compute entropies of flexional paradigms' distributions."""

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

    # Define logging levels (different depending on verbosity)
    if cfg.verbose or cfg.debug:
        logfile_name = md.register_file('debug.log', {'content': 'log'})
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG,
                            filename=logfile_name, filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    log = logging.getLogger()
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

    if cfg.debug and len(pat_table.columns) > 10:
        log.warning("Using debug mode is strongly "
                    "discouraged on large (>10 cells) datasets."
                    "You should probably stop this process now.")
    sanity_check = cfg.debug and len(pat_table.columns) < 10

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

        distrib = SplitPatternDistribution([paradigms, paradigms2],
                                           [pat_table, pat_table2],
                                           [pat_dic, pat_dic2],
                                           names,  # TODO: replace this
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

            distrib.distribs[0].entropy_matrix()
            entropies1 = distrib.distribs[0].entropies[1]
            distrib.distribs[1].entropy_matrix()
            entropies2 = distrib.distribs[1].entropies[1]
            mutual = distrib.mutual_information()
            normmutual = distrib.mutual_information(normalize=True)

            log.info("Writing to:" + "\n\t".join([ent_file1, ent_file2, I, NMI]))
            entropies1.to_csv(ent_file1, sep="\t")
            entropies2.to_csv(ent_file2, sep="\t")
            mutual.to_csv(I, sep="\t")
            normmutual.to_csv(NMI, sep="\t")
            if cfg.debug:
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
        distrib = PatternDistribution(paradigms,
                                      pat_table,
                                      pat_dic,
                                      features=features)

    if onePred:
        computation = 'onePredEntropies'
        ent_file = md.register_file('entropies.csv',
                                    {'computation': computation,
                                     'content': 'entropies'})
        effectifs_file = md.register_file('effectifs.csv',
                                          {'computation': computation,
                                           'content': 'effectifs'})

        distrib.entropy_matrix()
        entropies = distrib.entropies[1]
        effectifs = distrib.effectifs[1]

        if cfg.entropy.stacked:
            entropies = entropies.stack()
            entropies.index = [' -> '.join(index)
                               for index in entropies.index.values]
        log.info("Writing to: {}\n\tand {}".format(ent_file, effectifs_file))
        entropies.to_csv(ent_file, sep="\t")
        effectifs.to_csv(effectifs_file, sep="\t")
        # mean on df's index, then on Series' values.
        mean = entropies.mean().mean()
        log.info("Mean H(c1 -> c2) = %s ", mean)

        if cfg.debug:
            check = distrib.one_pred_distrib_log(sanity_check=sanity_check)

            if sanity_check:
                check_file = md.register_file('entropies_slow_method.csv',
                                              {'computation': computation,
                                               'content': 'entropies_slow_method'})

                log.info("Writing slowly computed entropies to: %s", check_file)

                check.to_csv(check_file, sep="\t")

    if preds:

        if cfg.entropy.importFile:
            distrib.read_entropy_from_file(cfg.entropy.importFile)

        for n in preds:
            computation = 'nPredsEntropies'
            n_ent_file = md.register_file('npreds{}_entropies.csv'.format(n),
                                          {'computation': computation,
                                           'content': 'entropies',
                                           'n': n})
            effectifs_file = md.register_file('npreds{}_effectifs.csv'.format(n),
                                              {'computation': computation,
                                               'content': 'effectifs',
                                               'n': n})

            distrib.n_preds_entropy_matrix(n)
            n_entropies = distrib.entropies[n]
            effectifs = distrib.effectifs[n]
            log.info("\nWriting to: {}\n\tand {}".format(n_ent_file, effectifs_file))
            if cfg.entropy.stacked:
                n_entropies = n_entropies.stack()
                n_entropies.index = [' -> '.join(index)
                                     for index in n_entropies.index.values]
            n_entropies.to_csv(n_ent_file, sep="\t")
            effectifs.to_csv(effectifs_file, sep="\t")
            mean = n_entropies.mean().mean()
            log.info("Mean H(c1, ..., c%s-> c) = %s", n, mean)

            if cfg.debug:
                n_check = distrib.n_preds_distrib_log(n, sanity_check=sanity_check)

                if sanity_check:
                    n_check_file = md.register_file('npreds{}_entropies_slow.csv',
                                                    {'computation': computation,
                                                     'content': 'entropies_slow_method',
                                                     'n': n})

                    log.info("Writing slowly computed"
                             " entropies to: {}".format(n_check_file))
                    n_check.to_csv(n_check_file, sep="\t")

            if onePred and cfg.debug:
                distrib.value_check(n)

    if cfg.debug:
        log.info("Wrote log to: {}".format(logfile_name))

    md.save_metadata()
