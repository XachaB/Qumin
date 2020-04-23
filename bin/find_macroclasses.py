# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Cluster lemmas according to their paradigms.

Author: Sacha Beniamine.
"""
try:
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_LOADED = True
except ImportError:
    MATPLOTLIB_LOADED = False

from utils import get_repository_version
from representations import segments, patterns
from clustering import algorithms, descriptionlength, find_min_attribute, distances
import pandas as pd


def main(args):
    r"""Cluster lexemes in macroclasses according to alternation patterns.

    We strongly recommend the default setting for the measure (-m) and the algorithm (-a)
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
    import re
    now = time.strftime("%Hh%M")
    day = time.strftime("%Y%m%d")

    # Loading files and paths
    features_file_name = args.segments
    data_file_path = args.patterns
    data_file_name = path.basename(data_file_path).rstrip("_")
    version = get_repository_version()
    print(data_file_name)
    kind = re.match(r".+_(.+)\.csv", data_file_name).groups()[0]

    # Setting up the output path.
    result_dir = "../Results/{}/{}".format(args.folder, day)
    makedirs(result_dir, exist_ok=True)
    result_prefix = "{}/{}_{}_{}_{}_".format(result_dir, data_file_name, version, day, now)

    # Initializing segments

    if features_file_name != "ORTHO":
        segments.initialize(features_file_name, sep="\t")
        pat_table, pat_dic = patterns.from_csv(data_file_path, defective=False, overabundant=False)
        pat_table = pat_table.applymap(str)
    else:
        pat_table = pd.read_csv(data_file_path, index_col=0)

    result_prefix += args.algorithm + "_" + args.measure

    measures = {"BU": {"DL": descriptionlength.BUDLClustersBuilder,
                       "CD": distances.CompressionDistClustersBuilder,
                       "UPGMA": distances.UPGMAClustersBuilder},
                "TD": {"DL": descriptionlength.TDDLClustersBuilder}}

    algorithm_choice = {"BU": algorithms.bottom_up_clustering,
                        "TD": algorithms.top_down_clustering}

    preferences = {"prefix": result_prefix,
                   "clustering_algorithm": algorithm_choice[args.algorithm],
                   "verbose": args.verbose,
                   "debug": args.debug}

    attr = {"DL": "DL", "CD": "dist", "UPGMA": "dist"}

    # if args.randomised:
    #     func = preferences["clustering_algorithm"]
    #     randomised_algo = partial(algorithms.randomised, func, n=args.randomised)
    #     preferences["clustering_algorithm"] = randomised_algo

    node = algorithms.hierarchical_clustering(pat_table, measures[args.algorithm][args.measure], **preferences)

    if args.measure == "DL":
        DL = "Min :" + str(find_min_attribute(node, "DL"))
    else:
        DL = ""

    experiment_id = " ".join([args.algorithm, args.measure, " on ", kind, DL, "(", version, day, now, ")", ])

    # Saving png figure
    if MATPLOTLIB_LOADED:
        fig = plt.figure(figsize=(10, 20))
        figname = result_prefix + "_figure.png"
        print("Drawing figure to: {}".format(figname))
        node.draw(horizontal=True,
                  square=True,
                  leavesfunc=lambda x: x.labels[0] + " (" + str(x.attributes["size"]) + ")",
                  nodefunc=lambda x: "{0:.3f}".format(x.attributes[attr[args.measure]]),
                  keep_above_macroclass=True)

        fig.suptitle(experiment_id)
        fig.savefig(result_prefix + "_figure.png",
                    bbox_inches='tight', pad_inches=.5)

    # Saving text tree
    print("Printing tree to: {}".format(result_prefix + "_tree.txt"))
    string_tree = repr(node)
    flow = open(result_prefix + "_tree.txt", "w", encoding="utf8")
    flow.write(string_tree)
    flow.write("\n" + experiment_id)
    flow.close()


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
                             " enter ORTHO if using endings on orthographic forms.",
                        type=str)

    parser.add_argument('-m', '--measure',
                        help="Cluster with measure: "
                             " Description length (DL)"
                             " Hamming distances (UPGMA, experimental)"
                        # " Compression distances (CD, experimental)"
                             "Defaults to DL. "
                             "UPGMA is experimental and not in development. It is only available for Bottom up algorithm",
                        choices=['UPGMA', 'DL'],  # 'CD'],
                        default='DL')

    parser.add_argument('-a', '--algorithm',
                        help="Cluster with algorithm: "
                             "Top down (TD, experimental) "
                             "Bottom up (BU) ",
                        choices=['TD', 'BU'],
                        default='BU')

    options = parser.add_argument_group('Options')

    options.add_argument("-v", "--verbose",
                         help="Activate verbosity.",
                         action="store_true", default=False)

    options.add_argument("-d", "--debug",
                         help="Activate debug logs.",
                         action="store_true", default=False)

    options.add_argument("-f", "--folder",
                         help="Output folder name",
                         type=str, default="Clustering")

    # options.add_argument("-r", "--randomised",
    #                      help="Run N times and keep the best result.",
    #                      type=int, default=None)

    args = parser.parse_args()

    print(args)
    main(args)
