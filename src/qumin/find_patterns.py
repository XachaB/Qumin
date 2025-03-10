# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Cluster lemmas according to their paradigms.

Author: Sacha Beniamine.
"""
from os import cpu_count
import logging

from .clustering import find_microclasses
from .representations import segments
from .representations.paradigms import Paradigms
from .representations.patterns import ParadigmPatterns

log = logging.getLogger()


def pat_command(cfg, md):
    r"""Find pairwise alternation patterns from paradigms."""
    # Loading files and paths
    kind = cfg.pats.kind
    defective = cfg.pats.defective
    overabundant = cfg.pats.overabundant
    segcheck = True

    # Initializing segments
    sounds_file_name = md.get_table_path("sounds")
    segments.Inventory.initialize(sounds_file_name)

    merge_cols = True

    paradigms = Paradigms(md.paralex,
                          defective=defective,
                          overabundant=overabundant,
                          merge_cols=merge_cols,
                          segcheck=segcheck,
                          cells=cfg.cells,
                          pos=cfg.pos,
                          force=cfg.force,
                          sample_lexemes=cfg.sample_lexemes,
                          sample_cells=cfg.sample_cells,
                          sample_kws=dict(force_random=cfg.force_random,
                                          seed=cfg.seed),
                          )

    patterns = ParadigmPatterns()

    patterns.find_patterns(paradigms,
                           method=kind,
                           optim_mem=cfg.pats.optim_mem,
                           gap_prop=cfg.pats.gap_proportion,
                           cpus=cfg.cpus or min(1, cpu_count() - 2))

    if merge_cols and not cfg.pats.merged:  # Re-build duplicate columns
        paradigms.unmerge_columns()
        patterns.unmerge_columns(paradigms)

    empty = [((df.form_x != '') & (df.form_y != '') & (df.pattern.isnull()))
             for df in patterns.values()]

    if any([x.any() for x in empty]):
        log.warning("Some words don't have any patterns "
                    "-- This means something went wrong."
                    "Please report this as a bug !")

    microclasses = find_microclasses(paradigms, patterns)
    filename = md.get_path("microclasses.txt")
    log.info("Found %s microclasses. Printing microclasses to %s", len(microclasses), filename)
    with open(filename, "w", encoding="utf-8") as flow:
        for m in sorted(microclasses, key=lambda m: len(microclasses[m])):
            flow.write("\n\n{} ({}) \n\t".format(m, len(microclasses[m])) + ", ".join(microclasses[m]))
    md.register_file("microclasses.txt", description="Microclass computation")

    patterns.export(md, kind, optim_mem=cfg.pats.optim_mem)
