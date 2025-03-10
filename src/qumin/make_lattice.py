# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Cluster lemmas according to their paradigms.

Author: Sacha Beniamine.
"""

import logging

from .clustering import find_microclasses
from .lattice.lattice import ICLattice
from .representations import segments
from .representations.paradigms import Paradigms
from .representations.patterns import ParadigmPatterns

log = logging.getLogger()


def lattice_command(cfg, md):
    r"""Infer Inflection classes as a lattice from alternation patterns."""
    # Loading files and paths
    patterns_folder_path = cfg.patterns
    defective = cfg.pats.defective
    overabundant = cfg.pats.overabundant

    # Initializing segments
    sounds_file_name = md.get_table_path("sounds")
    segments.Inventory.initialize(sounds_file_name)

    # Loading paradigms
    paradigms = Paradigms(md.paralex,
                          defective=defective,
                          overabundant=overabundant,
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
    log.info("Building the lattice...")
    incidence_table = patterns.incidence_table(microclasses)
    lattice = ICLattice(incidence_table, microclasses,
                        aoc=cfg.lattice.aoc,
                        keep_names=(not cfg.lattice.shorten))

    if cfg.lattice.stat:
        statname = md.get_path('lattice/stats.txt')
        with open(statname, "w", encoding="utf-8") as flow:
            flow.write(lattice.stats().to_frame().T.to_latex())
            log.info(lattice.stats().to_frame().T.to_latex())
        md.register_file('lattice/stats.txt',  description="Lattice statistics")

    if cfg.lattice.png:
        lattpng = md.get_path('lattice/lattice.png')
        lattice.draw(lattpng, figsize=(20, 10), title=None, point=True)
        md.register_file('lattice/lattice.png',  description="Lattice PNG figure")

    if cfg.lattice.pdf:
        lattpdf = md.get_path('lattice/lattice.pdf')
        lattice.draw(lattpdf, figsize=(20, 10), title=None, point=True)
        md.register_file('lattice/lattice.pdf',  description="Lattice PDF figure")

    if cfg.lattice.html:
        latthtml = md.get_path('lattice/lattice.html')
        log.info("Exporting to html: " + latthtml)
        lattice.to_html(latthtml)
        md.register_file('lattice/lattice.html',  description="Lattice HTML figure")

    if cfg.lattice.ctxt:
        lattcxt = md.get_path('lattice/lattice.cxt')
        log.info(" ".join(["Exporting context to file:", lattcxt]))
        lattice.context.tofile(lattcxt, frmat='cxt')
        md.register_file('lattice/lattice.cxt',  description="Lattice CXT figure")

    log.info("Here is the first level of the hierarchy:")
    log.info("Root:")
    obj, common = lattice.nodes.attributes["objects"], lattice.nodes.attributes["common"]
    if obj or common:
        log.info("\tdefines: " + str(obj) + str(common))
    for child in lattice.nodes.children:
        extent, common = child.labels, child.attributes["common"]
        log.info(" ".join(["\n\textent:", str(extent), "\n\tdefines:", str(common), ">"]))
