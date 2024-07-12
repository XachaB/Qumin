# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Cluster lemmas according to their paradigms.

Author: Sacha Beniamine.
"""

import logging

import pandas as pd
import hydra

from .clustering import find_microclasses
from .lattice.lattice import ICLattice
from .representations import segments, patterns
from .utils import Metadata

log = logging.getLogger()
@hydra.main(version_base=None, config_path="config", config_name="lattice")
def lattice_command(cfg):
    r"""Infer Inflection classes as a lattice from alternation patterns."""
    log.info(cfg)

    md = Metadata(cfg, __file__)
    md.bipartite = type(cfg.patterns) is not str

    # Loading files and paths
    patterns_file_path = cfg.patterns if md.bipartite else [cfg.patterns]
    comp = None

    if cfg.pats.ortho:
        log.info("Reading patterns...")
        pat_table = pd.read_csv(patterns_file_path[0], index_col=0)
        collections = False
    else:
        # Initializing segments
        log.info("Initializing segments...")
        sounds_file_name = md.get_table_path("sounds")
        segments.Inventory.initialize(sounds_file_name)

        log.info("Reading patterns...")
        pat_table, _ = patterns.from_csv(patterns_file_path[0])
        collections = True
        if md.bipartite:
            comp = "<comp>"
            try:
                pat_table2, _ = patterns.from_csv(patterns_file_path[1])
                pat_table2.columns = [(comp + c1, c2) for (c1, c2) in pat_table2.columns]
            except:
                pat_table2 = pd.read_csv(patterns_file_path[1], index_col=0).fillna("")
                pat_table2.columns = [comp + c for c in pat_table2.columns]
            pat_table = pat_table.join(pat_table2)

    microclasses = find_microclasses(pat_table.map(str))

    log.info("Building the lattice...")
    lattice = ICLattice(pat_table.loc[list(microclasses), :], microclasses,
                        overabundant=collections,
                        comp_prefix=comp,
                        aoc=cfg.lattice.aoc,
                        keep_names=(not cfg.lattice.shorten))

    if cfg.export.stat:
        statname = md.register_file('stats.txt', {"computation": cfg.scriptname,
                                                  "content": "stats"})
        with open(statname, "w", encoding="utf-8") as flow:
            flow.write(lattice.stats().to_frame().T.to_latex())
            log.info(lattice.stats().to_frame().T.to_latex())

    if cfg.export.png:
        lattpng = md.register_file('lattice.png', {'computation': cfg.scriptname,
                                                   'content': 'figure'})
        lattice.draw(lattpng, figsize=(20, 10), title=None, point=True)

    if cfg.export.pdf:
        lattpdf = md.register_file('lattice.pdf', {'computation': cfg.scriptname,
                                                   'content': 'figure'})
        lattice.draw(lattpdf, figsize=(20, 10), title=None, point=True)

    if cfg.export.html:
        latthtml = md.register_file('lattice.html', {'computation': cfg.scriptname,
                                                     'content': 'figure'})
        log.info("Exporting to html: " + latthtml)
        lattice.to_html(latthtml)

    if cfg.export.ctxt:
        lattcxt = md.register_file('lattice.cxt', {'computation': cfg.scriptname,
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

    md.save_metadata()
