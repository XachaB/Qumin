# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Cluster lemmas according to their paradigms.

Author: Sacha Beniamine.
"""
from utils import get_repository_version
from representations import patterns, segments, create_paradigms
from clustering import find_microclasses
from itertools import combinations


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
    from os import path, makedirs
    import time
    now = time.strftime("%Hh%M")
    day = time.strftime("%Y%m%d")

    # Loading files and paths
    kind = args.kind
    defective = args.defective
    overabundant = args.overabundant
    features_file_name = args.segments
    data_file_path = args.paradigms
    data_file_name = path.basename(data_file_path).rstrip("_")

    version = get_repository_version()
    # Setting up the output path.
    result_dir = "../Results/{}/".format(args.folder)
    makedirs(result_dir, exist_ok=True)
    result_prefix = "{}{}_{}_{}_{}_".format(result_dir, data_file_name, version, day, now)

    is_of_pattern_type = kind.startswith("patterns")
    segcheck = True

    # Initializing segments
    if features_file_name != "ORTHO":
        segments.initialize(features_file_name, sep="\t")
    elif is_of_pattern_type:
        raise argparse.ArgumentTypeError("You can't find patterns on orthographic material.")
    else:
        segcheck = False
        patterns.ORTHO = True

    patternfinder = {'endings':patterns.find_endings,
                     'endingsPairs':patterns.find_endings_pairs,
                     'endingsDisc':patterns.find_global_alternations,
                     'globalAlt':patterns.find_global_alternations_pairs,
                     'localAlt':patterns.find_local_alternations,
                     'patternsLevenshtein':patterns.find_levenshtein_patterns,
                     'patternsPhonsim':patterns.find_phonsim_patterns,
                     'patternsSuffix':patterns.find_suffixal_patterns,
                     'patternsPrefix':patterns.find_prefixal_patterns,
                     'patternsBaseline':patterns.find_baseline_patterns,
                      }


    merge_cols = False
    if is_of_pattern_type:
        merge_cols = True

    paradigms = create_paradigms(data_file_path, defective=defective, overabundant=overabundant, merge_cols=merge_cols, segcheck=segcheck)

    print("Looking for patterns...")
    if is_of_pattern_type:
        patterns_df, dic = patternfinder[kind](paradigms, optim_mem=args.optim_mem)
    else:
        patterns_df = patternfinder[kind](paradigms, optim_mem=args.optim_mem)

    if merge_cols and not args.merge_cols: # Re-build duplicate columns
        for a, b in patterns_df.columns:
            if "#" in a:
                cols = a.split("#")
                for c in cols:
                    patterns_df[(c, b)] = patterns_df[(a, b)]
                patterns_df.drop((a,b), axis=1, inplace=True)
                for x, y in combinations(cols, 2):
                    patterns_df[(x, y)] = patterns.max_identity_pattern((x, y))

        for a, b in patterns_df.columns:
            if "#" in b:
                cols = b.split("#")
                for c in cols:
                    patterns_df[(a, c)] = patterns_df[(a,b)]
                patterns_df.drop((a, b), axis=1, inplace=True)
                for x, y in combinations(cols, 2):
                    patterns_df[(x, y)] = patterns.max_identity_pattern((x, y))

    if patterns_df.isnull().values.any():
        print("Warning: error, some patterns are None")
        patterns_df[patterns_df.isnull().values]

    microclasses = find_microclasses(patterns_df.applymap(str))
    filename = result_prefix + "_microclasses.txt"
    print("\nFound ", len(microclasses), " microclasses.\nPrinting microclasses to ", filename)
    with open(filename, "w", encoding="utf-8") as flow:
        for m in sorted(microclasses, key=lambda m: len(microclasses[m])):
            flow.write("\n\n{} ({}) \n\t".format(m, len(microclasses[m])) + ", ".join(microclasses[m]))

    patfilename = result_prefix+"_" + kind +".csv"
    print("Printing patterns (importable by other scripts) to "+patfilename)
    if is_of_pattern_type:
        patterns.to_csv(patterns_df, patfilename, pretty=False)
        pathumanfilename = result_prefix+"_human_readable_" + kind + ".csv"
        print("Printing pretty patterns (for manual examination) to "+pathumanfilename)
        patterns.to_csv(patterns_df, pathumanfilename, pretty=True)
    else:
        patterns_df.to_csv(patfilename, sep=",")


if __name__ == '__main__':

    import argparse
    usage = main.__doc__

    parser = argparse.ArgumentParser(description=usage,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("paradigms",
                        help="paradigm file, full path"
                             " (csv separated by ‘, ’)",
                        type=str)

    parser.add_argument("segments",
                        help="segments file, full path"
                             " (csv separated by '\\t')"
                             " enter ORTHO if using endings on orthographic forms.",
                        type=str)

    parser.add_argument('-k', '--kind',
                        help="Kind of patterns to infer:"
                             "Patterns with various alignments (patterns_);"
                             " alternations as in Beniamine et al. (2017) (_Alt);"
                             "endings (endings_), ",
                        choices=['endings', 'endingsPairs', 'globalAlt', 'localAlt','endingsDisc',
                                 'patternsLevenshtein', 'patternsPhonsim', 'patternsSuffix', 'patternsPrefix',
                                 'patternsBaseline'],
                        default='patternsPhonsim')

    parser.add_argument("-d", "--defective",
                         help="Keep defective entries.",
                         action="store_true", default=False)


    parser.add_argument("-o", "--overabundant",
                         help="Keep overabundant entries.",
                         action="store_true", default=False)

    parser.add_argument("--optim_mem",
                         help="Attempt to optimize RAM usage",
                         action="store_true", default=False)

    parser.add_argument("-m", "--merge_cols",
                         help="Whether to merge identical columns before looking for patterns.",
                         action="store_true", default=False)

    parser.add_argument("-f", "--folder",
                         help="Output folder name",
                         type=str, default="Patterns")

    args = parser.parse_args()

    print(args)
    main(args)
