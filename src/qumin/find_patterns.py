# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Cluster lemmas according to their paradigms.

Author: Sacha Beniamine.
"""
import logging
import pandas as pd

from .clustering import find_microclasses
from .representations import patterns, segments
from .representations.paradigms import Paradigms
from .utils import get_cells

log = logging.getLogger()


def pat_command(cfg, md):
    r"""Find pairwise alternation patterns from paradigms."""
    # Loading files and paths
    kind = cfg.pats.kind
    defective = cfg.defective
    overabundant = cfg.overabundant
    cells = get_cells(cfg.cells, cfg.pos, md.datasets)
    segcheck = True

    # Initializing segments
    sounds_file_name = md.get_table_path("sounds")
    segments.Inventory.initialize(sounds_file_name)

    merge_cols = True

    paradigms = Paradigms(md.datasets, defective=defective,
                          overabundant=overabundant, merge_cols=merge_cols,
                                 segcheck=segcheck, cells=cells, pos=cfg.pos,
                          sample=cfg.sample,
                          most_freq=cfg.most_freq,
                          )

    log.info("Looking for patterns...")
    patterns_dfs, dic = patterns.find_patterns(paradigms, kind, optim_mem=cfg.pats.optim_mem,
                                               gap_prop=cfg.pats.gap_proportion)


    # Concatenate the patterns as a dict. Cell names are turned into columns.
    patterns_df = pd.concat([df for df in patterns_dfs.values()]).reset_index(drop=True)

    if merge_cols and not cfg.pats.merged:  # Re-build duplicate columns
        paradigms.unmerge_columns()
        patterns_df = patterns.unmerge_columns(patterns_df, paradigms)

    empty = (patterns_df.form_x != '') & (patterns_df.form_y != '') & (patterns_df.pattern.isnull())

    if empty.any():
        log.warning("Some words don't have any patterns "
                    "-- This means something went wrong."
                    "Please report this as a bug !")
        log.warning(patterns_df[empty])

    microclasses = find_microclasses(patterns_df.map(str))
    filename = md.register_file("microclasses.txt",
                                {'computation': cfg.pats.kind, 'content': 'microclasses'})
    log.info("Found %s microclasses. Printing microclasses to %s", len(microclasses), filename)
    with open(filename, "w", encoding="utf-8") as flow:
        for m in sorted(microclasses, key=lambda m: len(microclasses[m])):
            flow.write("\n\n{} ({}) \n\t".format(m, len(microclasses[m])) + ", ".join(microclasses[m]))

    patfilename = md.register_file(kind + ".csv",
                                   {'computation': cfg.pats.kind, 'content': 'patterns'})
    log.info("Writing patterns (importable by other scripts) to %s", patfilename)
    if cfg.pats.optim_mem:
        patterns.to_csv(patterns_df, patfilename, pretty=True, only_id=True)  # uses str because optim_mem already used repr
        log.warning("Since you asked for args.optim_mem, I will not export the human_readable file.")
    else:
        patterns.to_csv(patterns_df, patfilename, pretty=False, only_id=True)  # uses repr
        pathumanfilename = md.register_file("human_readable_" + kind + ".csv",
                                            {'computation': cfg.pats.kind, 'content': 'patterns_human'})
        log.info("Writing pretty patterns (for manual examination) to %s", pathumanfilename)
        patterns.to_csv(patterns_df, pathumanfilename, pretty=True)  # uses str

    return patfilename
