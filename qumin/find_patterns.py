# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Cluster lemmas according to their paradigms.

Author: Sacha Beniamine.
"""
from .utils import get_version, get_default_parser
from .representations import patterns, segments, create_paradigms
from .clustering import find_microclasses
from itertools import combinations
import logging
import argparse
from pathlib import Path
import time

def main(args):
    r"""Find pairwise alternation patterns from paradigms.

    For a detailed explanation, see the html doc.::
      ____
     / __ \                    /)
    | |  | | _   _  _ __ ___   _  _ __
    | |  | || | | || '_ ` _ \ | || '_ \
    | |__| || |_| || | | | | || || | | |
     \___\_\ \__,_||_| |_| |_||_||_| |_|
      Quantitative modeling of inflection

    """
    if args.verbose:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    log = logging.getLogger()
    log.info(args)
    now = time.strftime("%Hh%M")
    day = time.strftime("%Y%m%d")

    # Loading files and paths
    kind = args.kind
    defective = args.defective
    overabundant = args.overabundant
    features_file_name = args.segments
    data_file_path = args.paradigms
    data_file_name = Path(data_file_path).name.rstrip("_")

    version = get_version()
    # Setting up the output path.
    result_dir = Path(args.folder)
    result_dir.mkdir(exist_ok=True, parents=True)
    result_prefix = "{}/{}_{}_{}_{}_".format(result_dir, data_file_name, version, day, now)

    is_of_pattern_type = kind.startswith("patterns")
    segcheck = True

    # Initializing segments
    if features_file_name != "ORTHO":
        segments.Inventory.initialize(features_file_name)
    elif is_of_pattern_type:
        raise argparse.ArgumentTypeError("You can't find patterns on orthographic material.")
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

    paradigms = create_paradigms(data_file_path, defective=defective, overabundant=overabundant, merge_cols=merge_cols,
                                 segcheck=segcheck, long=args.long,
                                 col_names=args.cols_names)

    log.info("Looking for patterns...")
    if kind.startswith("endings"):
        patterns_df = patterns.find_endings(paradigms)
        if kind.endswith("Pairs"):
            patterns_df = patterns.make_pairs(patterns_df)
            log.info(patterns_df)
    elif is_of_pattern_type:
        patterns_df, dic = patterns.find_patterns(paradigms, method[kind], optim_mem=args.optim_mem,
                                                  gap_prop=args.gap_proportion)
    else:
        patterns_df = patterns.find_alternations(paradigms, method[kind])

    if merge_cols and not args.merge_cols:  # Re-build duplicate columns
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

    microclasses = find_microclasses(patterns_df.applymap(str))
    filename = result_prefix + "_microclasses.txt"
    log.info("Found %s microclasses. Printing microclasses to %s", len(microclasses), filename)
    with open(filename, "w", encoding="utf-8") as flow:
        for m in sorted(microclasses, key=lambda m: len(microclasses[m])):
            flow.write("\n\n{} ({}) \n\t".format(m, len(microclasses[m])) + ", ".join(microclasses[m]))

    patfilename = result_prefix + "_" + kind + ".csv"
    log.info("Writing patterns (importable by other scripts) to %s", patfilename)
    if is_of_pattern_type:
        if args.optim_mem:
            patterns.to_csv(patterns_df, patfilename, pretty=True)  # uses str because optim_mem already used repr
            log.warning("Since you asked for args.optim_mem, I will not export the human_readable file ")
        else:
            patterns.to_csv(patterns_df, patfilename, pretty=False)  # uses repr
            pathumanfilename = result_prefix + "_human_readable_" + kind + ".csv"
            log.info("Writing pretty patterns (for manual examination) to %s", pathumanfilename)
            patterns.to_csv(patterns_df, pathumanfilename, pretty=True)  # uses str
    else:
        patterns_df.to_csv(patfilename, sep=",")

def pat_command():

    parser = get_default_parser(main.__doc__, "./Results/Patterns/", paradigms=True, patterns=False)


    parser.add_argument('-k', '--kind',
                        help="Kind of patterns to infer:"
                             "Patterns with various alignments (patterns_);"
                             " alternations as in Beniamine et al. (2017) (_Alt);"
                             "endings (endings_), ",
                        choices=['endings', 'endingsPairs', 'globalAlt', 'localAlt', 'endingsDisc',
                                 'patternsLevenshtein', 'patternsPhonsim', 'patternsSuffix', 'patternsPrefix',
                                 'patternsBaseline'],
                        default='patternsPhonsim')

    parser.add_argument("-d", "--defective",
                        help="Keep defective entries.",
                        action="store_true", default=False)

    parser.add_argument("-o", "--overabundant",
                        help="Keep overabundant entries.",
                        action="store_true", default=False)

    parser.add_argument("--gap_proportion",
                        help="Proportion of the median similarity cost assigned to the insertion cost.",
                        type=float, default=.5)

    parser.add_argument("--optim_mem",
                        help="Attempt to optimize RAM usage",
                        action="store_true", default=False)

    parser.add_argument("-m", "--merge_cols",
                        help="Whether to merge identical columns before looking for patterns.",
                        action="store_true", default=False)


    args = parser.parse_args()

    main(args)


if __name__ == '__main__':
    pat_command()