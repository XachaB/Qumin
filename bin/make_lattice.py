# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Cluster lemmas according to their paradigms.

Author: Sacha Beniamine.
"""

from clustering import find_microclasses
from representations import segments, patterns
from utils import get_repository_version
from lattice.lattice import ICLattice
import pandas as pd


def main(args):
    r""" Infer Inflection classes as a lattice from alternation patterns.
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

    features_file_name = args.segments
    data_file_path = args.patterns
    data_file_name = path.basename(data_file_path)
    version = get_repository_version().rstrip("_")

    # Setting up the output path.
    result_dir = "../Results/{}/{}".format(args.folder, day)
    makedirs(result_dir, exist_ok=True)
    result_prefix = "{}/{}_{}_{}_{}_{}_{}lattice".format(result_dir, data_file_name, version, day, now,
                                                         "aoc" if args.aoc else "full",
                                                         "bipartite_" if args.bipartite else "_")

    if features_file_name != "ORTHO":

        # Initializing segments
        print("Initializing segments...")
        segments.Inventory.initialize(features_file_name)

        print("Reading patterns...")
        pat_table, _ = patterns.from_csv(data_file_path)
        # pat_table = pat_table.applymap(str)
        # pat_table.columns = [x+" ~ "+y for x,y in pat_table.columns]
        collections = True
        comp = None
        if args.bipartite is not None:
            comp = "<comp>"
            try:
                pat_table2, _ = patterns.from_csv(args.bipartite)
                pat_table2.columns = [(comp + c1, c2) for (c1, c2) in pat_table2.columns]
            except:
                pat_table2 = pd.read_csv(args.bipartite, index_col=0).fillna("")
                pat_table2.columns = [comp + c for c in pat_table2.columns]
            pat_table = pat_table.join(pat_table2)
    else:
        print("Reading patterns...")
        pat_table = pd.read_csv(data_file_path, index_col=0)
        collections = False

    microclasses = find_microclasses(pat_table.applymap(str))

    print("Building the lattice...")
    lattice = ICLattice(pat_table.loc[list(microclasses), :], microclasses,
                        collections=collections, comp_prefix=comp, AOC=args.aoc, keep_names=(not args.shorten))

    if args.stat:
        with open(result_prefix + "_stats.txt", "w", encoding="utf-8") as flow:
            print(lattice.stats().to_frame().T.to_latex(), file=flow)
            print(lattice.stats().to_frame().T.to_latex())

    if args.png:
        lattice.draw(result_prefix + ".png", figsize=(20, 10), title=None, point=True)

    if args.pdf:
        lattice.draw(result_prefix + ".pdf", figsize=(20, 10), title=None, point=True)

    if args.html:
        print("Exporting to html:", result_prefix + ".html")
        lattice.to_html(result_prefix + ".html")

    if args.cxt:
        print("Exporting context to file:", result_prefix + ".cxt")
        lattice.context.tofile(result_prefix + ".cxt", frmat='cxt')

    if args.first:
        print("Here is the first level of the hierarchy:")
        print("Root:")
        obj, common = lattice.nodes.attributes["objects"], lattice.nodes.attributes["common"]
        if obj or common:
            print("\tdefines:", obj, common)
        for child in lattice.nodes.children:
            extent, common = child.labels, child.attributes["common"]
            print("extent:", extent, "\n\tdefines:", common, ">")


if __name__ == '__main__':
    import argparse

    usage = main.__doc__

    parser = argparse.ArgumentParser(description=usage,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("patterns",
                        help="patterns file, full path"
                             " (csv separated by ‘, ’)",
                        type=str)

    parser.add_argument("segments",
                        help="segments file, full path"
                             " (csv separated by '\\t')"
                             " Enter ORTHO if orthographic data",
                        type=str)

    parser.add_argument('--shorten',
                        help="Drop redundant columns altogether."
                             " Useful for big contexts,"
                             "but loses information. "
                             "The lattice shape and stats will be the same. "
                             "Avoid using with --html",
                        action="store_true", default=False)

    parser.add_argument('-b', '--bipartite',
                        help="Add a second paradigm dataset, for bipartite systems.",
                        type=str,
                        default=None)

    parser.add_argument("--aoc",
                        help="Only attribute and object concepts",
                        action="store_true", default=False)

    parser.add_argument("--html",
                        help="Export to html",
                        action="store_true", default=False)

    parser.add_argument("--cxt",
                        help="Export as a context",
                        action="store_true", default=False)

    parser.add_argument("--stat",
                        help="Output stats about the lattice",
                        action="store_true", default=False)

    parser.add_argument("--pdf",
                        help="Export as png",
                        action="store_true", default=False)

    parser.add_argument("--png",
                        help="Export as png",
                        action="store_true", default=False)

    parser.add_argument("--first",
                        help="Write first level",
                        action="store_true", default=False)

    options = parser.add_argument_group('Options')

    options.add_argument("-f", "--folder",
                         help="Output folder name",
                         type=str, default="lattice")

    args = parser.parse_args()

    print(args)
    main(args)
