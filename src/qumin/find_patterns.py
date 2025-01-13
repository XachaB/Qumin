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
from .utils import get_cells

log = logging.getLogger()


def pat_command(cfg, md):
    r"""Find pairwise alternation patterns from paradigms."""
    # Loading files and paths
    kind = cfg.pats.kind
    defective = cfg.pats.defective
    overabundant = cfg.pats.overabundant
    cells = get_cells(cfg.cells, cfg.pos, md.dataset)
    segcheck = True

    # Initializing segments
    sounds_file_name = md.get_table_path("sounds")
    segments.Inventory.initialize(sounds_file_name)

    merge_cols = True

    paradigms = Paradigms(md.dataset,
                          defective=defective,
                          overabundant=overabundant,
                          merge_cols=merge_cols,
                          segcheck=segcheck,
                          cells=cells,
                          pos=cfg.pos,
                          sample=cfg.sample,
                          most_freq=cfg.most_freq,
                          force=cfg.force,
                          )

    log.info("Looking for patterns...")
    patterns = ParadigmPatterns()

    patterns.find_patterns(paradigms,
                           method=kind,
                           optim_mem=cfg.pats.optim_mem,
                           gap_prop=cfg.pats.gap_proportion,
                           cpus=cfg.cpus or min(1, cpu_count() - 2))

    # Concatenate the patterns as a dict. Cell names are turned into columns.
    # patterns_df = pd.concat([df for df in patterns_dfs.values()]).reset_index(drop=True)

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
    filename = md.register_file("microclasses.txt",
                                {'computation': cfg.pats.kind, 'content': 'microclasses'})
    log.info("Found %s microclasses. Printing microclasses to %s", len(microclasses), filename)
    with open(filename, "w", encoding="utf-8") as flow:
        for m in sorted(microclasses, key=lambda m: len(microclasses[m])):
            flow.write("\n\n{} ({}) \n\t".format(m, len(microclasses[m])) + ", ".join(microclasses[m]))

    return patterns.export(md, kind, optim_mem=cfg.pats.optim_mem)
