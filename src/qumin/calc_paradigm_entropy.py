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
    use_extra = cfg.entropy.use_extra
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
    paradigms = Paradigms(md.dataset,
                          defective=cfg.pats.defective,
                          overabundant=cfg.pats.overabundant,
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
                       defective=cfg.pats.defective,
                       overabundant=cfg.pats.overabundant,
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
                                  md.dataset,
                                  features=features)

    if onePred:
        if verbose:
            distrib.one_pred_entropy(use_extra=use_extra,
                                     debug=verbose,
                                     **cfg.entropy.extra)
        distrib.one_pred_entropy(use_extra=use_extra,
                                 **cfg.entropy.extra)
        measures = ['cond_entropy', 'accuracy'] if use_extra else ['cond_entropy']
        mean = distrib.get_results(measure=measures)\
            .loc[:, ["value", "measure"]].groupby('measure').mean()
        if verbose:
            distrib.sanity_check()
        log.info(f"Mean metrics:\n{mean.to_markdown()}")
    if preds:
        if use_extra:
            raise NotImplementedError
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
