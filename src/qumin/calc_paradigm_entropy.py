#!usr/bin/python3
# -*- coding: utf-8 -*-
"""author: Sacha Beniamine.

Compute conditional entropies in inflectional patterns.
"""

import logging

from hydra.core.hydra_config import HydraConfig

from .entropy.distribution import PatternDistribution
from .representations import segments, create_features
from .representations.patterns import ParadigmPatterns
from .representations.paradigms import Paradigms
from .utils import get_cells

log = logging.getLogger()


def H_command(cfg, md):
    r"""Compute entropies of flexional paradigms' distributions."""
    verbose = HydraConfig.get().verbose is not False
    patterns_folder_path = cfg.patterns
    sounds_file_name = md.get_table_path("sounds")

    preds = [cfg.entropy.n] if type(cfg.entropy.n) is int else sorted(cfg.entropy.n)
    onePred = preds[0] == 1
    if onePred:
        preds.pop(0)

    cells = get_cells(cfg.cells, cfg.pos, md.dataset)

    # Initialize segment inventory for phonological computations
    segments.Inventory.initialize(sounds_file_name)

    # Inflectional paradigms: rows are forms, with lexeme and cell..
    paradigms = Paradigms(md.dataset, defective=cfg.defective, overabundant=False,
                          merge_cols=cfg.entropy.merged,
                          segcheck=True, cells=cells, pos=cfg.pos,
                          sample=cfg.sample,
                          most_freq=cfg.most_freq,
                          force=cfg.force,
                          )

    patterns = ParadigmPatterns()
    patterns.from_file(patterns_folder_path,
                       paradigms.data,
                       defective=cfg.defective,
                       overabundant=False,
                       force=cfg.force,
                       )

    if verbose and len(patterns.keys()) > 45:
        log.warning("Using verbose mode is strongly "
                    "discouraged on large (>10 cells) datasets."
                    "You should probably stop this process now.")

    if cfg.entropy.features is not None:
        features = create_features(md, cfg.entropy.features)
    else:
        features = None

    patterns.find_applicable()
    patterns.info()

    distrib = PatternDistribution(patterns,
                                  md.dataset.name,
                                  features=features)

    if onePred:
        distrib.one_pred_entropy()
        mean = distrib.get_results().loc[:, "value"].mean()
        log.info("Mean H(c1 -> c2) = %s ", mean)
        if verbose:
            distrib.one_pred_entropy(debug=verbose)

    if preds:
        if cfg.entropy.importFile:
            distrib.import_file(cfg.entropy.importFile)

        for n in preds:
            distrib.n_preds_entropy_matrix(n, paradigms)
            mean = distrib.get_results(n=n).loc[:, "value"].mean()
            log.info(f"Mean H(c1, ..., c{n} -> c) = {mean}")

            if verbose:
                distrib.n_preds_distrib_log(n)

    ent_file = md.register_file('entropies.csv',
                                {'computation': 'entropies',
                                 'content': 'results'})

    log.info("Writing to: {}".format(ent_file))
    distrib.export_file(ent_file)

    return ent_file
