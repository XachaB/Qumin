# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Cluster lemmas according to their paradigms.

Author: Sacha Beniamine.
"""
import logging
from itertools import combinations

from .clustering import find_microclasses
from .representations import patterns, segments, create_paradigms

log = logging.getLogger()


def pat_command(cfg, md):
    r"""Find pairwise alternation patterns from paradigms."""
    # Loading files and paths
    kind = cfg.pats.kind
    defective = cfg.pats.defective
    overabundant = cfg.pats.overabundant
    cells = cfg.cells
    if cells and len(cells) == 1:
        raise ValueError("You can't provide only one cell.")

    is_of_pattern_type = kind.startswith("patterns")
    segcheck = True

    # Initializing segments
    if not cfg.pats.ortho:
        sounds_file_name = md.get_table_path("sounds")
        segments.Inventory.initialize(sounds_file_name)
    elif is_of_pattern_type:
        raise ValueError("You can't find patterns on orthographic material, only alternations or endings.")
    else:
        segcheck = False

    method = {'globalAlt': 'global',
              'localAlt': 'local',
              'patternsLevenshtein': 'levenshtein',
              'patternsPhonsim': 'similarity',
              'patternsSuffix': 'suffix',
              'patternsPrefix': 'prefix',
              'patternsBaseline': 'baseline'}

    merge_cols = False
    if is_of_pattern_type:
        merge_cols = True

    paradigms = create_paradigms(md.datasets[0], defective=defective,
                                 overabundant=overabundant, merge_cols=merge_cols,
                                 segcheck=segcheck, cells=cells,
                                 sample=cfg.sample,
                                 most_freq=cfg.most_freq
                                 )

    log.info("Looking for patterns...")
    if kind.startswith("endings"):
        patterns_df = patterns.find_endings(paradigms)
        if kind.endswith("Pairs"):
            patterns_df = patterns.make_pairs(patterns_df)
            log.info(patterns_df)
    elif is_of_pattern_type:
        patterns_df, dic = patterns.find_patterns(paradigms, method[kind], optim_mem=cfg.pats.optim_mem,
                                                  gap_prop=cfg.pats.gap_proportion)
    else:
        patterns_df = patterns.find_alternations(paradigms, method[kind])

    if merge_cols and not cfg.pats.merged:  # Re-build duplicate columns
        for a, b in patterns_df.columns:
            if "#" in a:
                cols = a.split("#")
                for c in cols:
                    patterns_df[(c, b)] = patterns_df[(a, b)]
                patterns_df.drop((a, b), axis=1, inplace=True)
                for x, y in combinations(cols, 2):
                    patterns_df[(x, y)] = patterns.Pattern.new_identity((x, y))

        for a, b in patterns_df.columns:
            if "#" in b:
                cols = b.split("#")
                for c in cols:
                    patterns_df[(a, c)] = patterns_df[(a, b)]
                patterns_df.drop((a, b), axis=1, inplace=True)
                for x, y in combinations(cols, 2):
                    patterns_df[(x, y)] = patterns.Pattern.new_identity((x, y))

    if patterns_df.isnull().values.any():
        log.warning("Some words don't have any patterns "
                    "-- This means something went wrong."
                    "Please report this as a bug !")
        log.warning(patterns_df[patterns_df.isnull().values])

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
    if is_of_pattern_type:
        if cfg.pats.optim_mem:
            patterns.to_csv(patterns_df, patfilename, pretty=True)  # uses str because optim_mem already used repr
            log.warning("Since you asked for args.optim_mem, I will not export the human_readable file ")
        else:
            patterns.to_csv(patterns_df, patfilename, pretty=False)  # uses repr
            pathumanfilename = md.register_file("human_readable_" + kind + ".csv",
                                                {'computation': cfg.pats.kind, 'content': 'patterns_human'})
            log.info("Writing pretty patterns (for manual examination) to %s", pathumanfilename)
            patterns.to_csv(patterns_df, pathumanfilename, pretty=True)  # uses str
    else:
        patterns_df.to_csv(patfilename, sep=",")

    return patfilename
