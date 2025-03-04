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

log = logging.getLogger()


def H_command(cfg, md):
    r"""Compute entropies of flexional paradigms' distributions."""
    verbose = HydraConfig.get().verbose is not False
    patterns_folder_path = cfg.patterns
    sounds_file_name = md.get_table_path("sounds")
    defective = cfg.pats.defective

    assert not ((cfg.entropy.token_freq.patterns or cfg.entropy.token_freq.patterns)
                and cfg.entropy.legacy), \
        "Token frequencies can't be used in entropy computations with legacy=True."

    preds = [cfg.entropy.n] if type(cfg.entropy.n) is int else sorted(cfg.entropy.n)
    assert (preds[-1] == 1 or cfg.entropy.legacy), \
        "N predictor computations are only available with legacy=True"
    onePred = preds[0] == 1
    if onePred:
        preds.pop(0)

    # Initialize segment inventory for phonological computations
    segments.Inventory.initialize(sounds_file_name)

    # Inflectional paradigms: rows are forms, with lexeme and cell..
    paradigms = Paradigms(md.dataset,
                          defective=defective,
                          overabundant=False,
                          merge_cols=cfg.entropy.merged,
                          segcheck=True,
                          cells=cfg.cells,
                          pos=cfg.pos,
                          force=cfg.force,
                          sample_lexemes=cfg.sample_lexemes,
                          sample_cells=cfg.sample_cells,
                          sample_kws=dict(force_random=cfg.force_random,
                                          seed=cfg.seed),
                          )
    patterns = ParadigmPatterns()
    patterns.from_file(patterns_folder_path,
                       paradigms.data,
                       cells=cfg.cells,
                       defective=defective,
                       overabundant=False,
                       force=cfg.force,
                       )

    if verbose and len(patterns.cells) > 10:
        log.warning("Using verbose mode is strongly "
                    "discouraged on large (>10 cells) datasets."
                    "You should probably stop this process now.")

    if cfg.entropy.features is not None:
        features = create_features(md, cfg.entropy.features)
    else:
        features = None

    patterns.find_applicable(cpus=cfg.cpus)
    patterns.info()

    distrib = PatternDistribution(patterns,
                                  md.dataset.name,
                                  paradigms.frequencies,
                                  features=features)

    if onePred:
        if verbose:
            distrib.one_pred_entropy(debug=verbose,
                                     legacy=cfg.entropy.legacy,
                                     token_patterns=cfg.entropy.token_freq.patterns,
                                     token_predictors=cfg.entropy.token_freq.predictors,
                                     )
        distrib.one_pred_entropy(legacy=cfg.entropy.legacy,
                                 token_patterns=cfg.entropy.token_freq.patterns,
                                 token_predictors=cfg.entropy.token_freq.predictors,
                                 )
        mean = distrib.get_results().loc[:, "value"].mean()
        log.info("Mean H(c1 -> c2) = %s ", mean)

    if preds:
        if cfg.entropy.importFile:
            distrib.import_file(cfg.entropy.importFile)
        for n in preds:
            if verbose:
                distrib.n_preds_entropy(n, paradigms, debug=verbose)
            distrib.n_preds_entropy(n, paradigms)
            mean = distrib.get_results(n=n).loc[:, "value"].mean()
            log.info(f"Mean H(c1, ..., c{n} -> c) = {mean}")

    ent_file = md.register_file('entropies.csv',
                                {'computation': 'entropies',
                                 'content': 'results'})

    log.info("Writing to: {}".format(ent_file))
    distrib.export_file(ent_file)

    return ent_file
