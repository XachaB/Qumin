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
    defective = cfg.pats.defective

    assert not ((cfg.entropy.tokens.patterns or cfg.entropy.tokens.patterns) and cfg.entropy.legacy), \
        "Tokens can't be used in entropy computations with legacy=True."

    preds = [cfg.entropy.n] if type(cfg.entropy.n) is int else sorted(cfg.entropy.n)
    assert (preds[-1] == 1 or cfg.entropy.legacy), \
        "N predictor computations are only available with legacy=True"
    onePred = preds[0] == 1
    if onePred:
        preds.pop(0)

    cells = get_cells(cfg.cells, cfg.pos, md.dataset)

    # Initialize segment inventory for phonological computations
    segments.Inventory.initialize(sounds_file_name)

    # Inflectional paradigms: rows are forms, with lexeme and cell..
    paradigms = Paradigms(md.dataset,
                          defective=defective,
                          overabundant=False,
                          merge_cols=cfg.entropy.merged,
                          segcheck=True,
                          cells=cells,
                          pos=cfg.pos,
                          force=cfg.force,
                          sample=cfg.sample,
                          sample_kws=dict(force_random=cfg.force_random,
                                          seed=cfg.seed),
                          )
    patterns = ParadigmPatterns()
    patterns.from_file(patterns_folder_path,
                       paradigms.data,
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
            distrib.one_pred_metrics(debug=verbose,
                                     legacy=cfg.entropy.legacy,
                                     token_patterns=cfg.entropy.tokens.patterns,
                                     token_predictors=cfg.entropy.tokens.predictors,
                                     )
        distrib.one_pred_metrics(legacy=cfg.entropy.legacy,
                                 token_patterns=cfg.entropy.tokens.patterns,
                                 token_predictors=cfg.entropy.tokens.predictors,
                                 )
        means = distrib.get_results().groupby('measure').value.mean()
        log.info("Mean metrics:\n " + means.to_markdown())

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
