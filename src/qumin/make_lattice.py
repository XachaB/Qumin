# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Cluster lemmas according to their paradigms.

Author: Sacha Beniamine.
"""

import logging

import pandas as pd

from .clustering import find_microclasses
from .lattice.lattice import ICLattice
from .representations import segments
from .representations.paradigms import Paradigms
from .representations.patterns import ParadigmPatterns
from .utils import get_cells

log = logging.getLogger()


def lattice_command(cfg, md):
    r"""Infer Inflection classes as a lattice from alternation patterns."""
    # Loading files and paths
    patterns_folder_path = cfg.patterns
    defective = cfg.pats.defective
    overabundant = cfg.pats.overabundant
    cells = get_cells(cfg.cells, cfg.pos, md.dataset)

    # Initializing segments
    sounds_file_name = md.get_table_path("sounds")
    segments.Inventory.initialize(sounds_file_name)

    # Loading paradigms
    paradigms = Paradigms(md.dataset,
                          defective=defective,
                          overabundant=overabundant,
                          merge_cols=cfg.entropy.merged,
                          segcheck=True,
                          cells=cells,
                          pos=cfg.pos,
                          force=cfg.force,
                          sample=cfg.sample,
                          sample_kws=dict(force_random=cfg.force_random,
                                          seed=cfg.seed),
                          )

    # Loading Patterns
    patterns = ParadigmPatterns()
    patterns.from_file(patterns_folder_path,
                       paradigms.data,
                       defective=defective,
                       overabundant=overabundant,
                       force=cfg.force,
                       )

    for pair in patterns:
        patterns[pair].loc[:, "pattern"] = patterns[pair].loc[:, "pattern"].map(str)

    microclasses = find_microclasses(paradigms, patterns)

    # Builde a wide df of patterns


    log.info("Building the lattice...")
    lattice = ICLattice(patterns, microclasses,
                        aoc=cfg.lattice.aoc,
                        keep_names=(not cfg.lattice.shorten))

    if cfg.lattice.stat:
        statname = md.register_file('stats.txt', {"computation": cfg.action,
                                                  "content": "stats"})
        with open(statname, "w", encoding="utf-8") as flow:
            flow.write(lattice.stats().to_frame().T.to_latex())
            log.info(lattice.stats().to_frame().T.to_latex())

    if cfg.lattice.png:
        lattpng = md.register_file('lattice.png', {'computation': cfg.action,
                                                   'content': 'figure'})
        lattice.draw(lattpng, figsize=(20, 10), title=None, point=True)

    if cfg.lattice.pdf:
        lattpdf = md.register_file('lattice.pdf', {'computation': cfg.action,
                                                   'content': 'figure'})
        lattice.draw(lattpdf, figsize=(20, 10), title=None, point=True)

    if cfg.lattice.html:
        latthtml = md.register_file('lattice.html', {'computation': cfg.action,
                                                     'content': 'figure'})
        log.info("Exporting to html: " + latthtml)
        lattice.to_html(latthtml)

    if cfg.lattice.ctxt:
        lattcxt = md.register_file('lattice.cxt', {'computation': cfg.action,
                                                   'content': 'figure'})
        log.info(" ".join(["Exporting context to file:", lattcxt]))
        lattice.context.tofile(lattcxt, frmat='cxt')

    log.info("Here is the first level of the hierarchy:")
    log.info("Root:")
    obj, common = lattice.nodes.attributes["objects"], lattice.nodes.attributes["common"]
    if obj or common:
        log.info("\tdefines: " + str(obj) + str(common))
    for child in lattice.nodes.children:
        extent, common = child.labels, child.attributes["common"]
        log.info(" ".join(["\n\textent:", str(extent), "\n\tdefines:", str(common), ">"]))
