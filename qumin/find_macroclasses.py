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

from .utils import get_repository_version, get_default_parser
from .representations import segments, patterns
from .clustering import algorithms, descriptionlength, find_min_attribute
import pandas as pd
import logging
from pathlib import Path
import time
import re

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
    if args.verbose:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    log = logging.getLogger()
    log.info(args)

    now = time.strftime("%Hh%M")
    day = time.strftime("%Y%m%d")

    # Loading files and paths
    features_file_name = args.segments
    data_file_path = args.patterns
    data_file_name = Path(data_file_path).name.rstrip("_")
    version = get_repository_version()

    pattern_type_match = re.match(r".+_(.+)\.csv", data_file_name)
    if pattern_type_match is None:
        log.warning("Did you rename the patterns file ? "
                    "As a result, I do not know which type of pattern you used..")
        kind = "unknown"
    else:
        kind = pattern_type_match.groups()[0]

    # Setting up the output path.
    result_dir = Path(args.folder) / day
    result_dir.makedir(exist_ok=True)
    result_prefix = "{}/{}_{}_{}_{}_BU_DL".format(result_dir, data_file_name, version, day, now)

    # Initializing segments

    if features_file_name != "ORTHO":
        segments.Inventory.initialize(features_file_name)
        pat_table, pat_dic = patterns.from_csv(data_file_path, defective=False, overabundant=False)
        pat_table = pat_table.applymap(str)
    else:
        pat_table = pd.read_csv(data_file_path, index_col=0)

    preferences = {"prefix": result_prefix}

    # if args.randomised:
    #     func = preferences["clustering_algorithm"]
    #     randomised_algo = partial(algorithms.randomised, func, n=args.randomised)
    #     preferences["clustering_algorithm"] = randomised_algo

    node = algorithms.hierarchical_clustering(pat_table, descriptionlength.BUDLClustersBuilder, **preferences)

    DL = "Min :" + str(find_min_attribute(node, "DL"))
    experiment_id = " ".join(["Bottom-up DL clustering on ", kind, DL, "(", version, day, now, ")", ])

    # Saving png figure
    if MATPLOTLIB_LOADED:
        fig = plt.figure(figsize=(10, 20))
        figname = result_prefix + "_figure.png"
        log.info("Drawing figure to: {}".format(figname))
        node.draw(horizontal=True,
                  square=True,
                  layout="qumin",
                  leavesfunc=lambda x: x.labels[0] + " (" + str(x.attributes["size"]) + ")",
                  nodefunc=lambda x: "{0:.3f}".format(x.attributes["DL"]),
                  keep_above_macroclass=True)

        fig.suptitle(experiment_id)
        fig.savefig(result_prefix + "_figure.png",
                    bbox_inches='tight', pad_inches=.5)

    # Saving text tree
    log.info("Printing tree to: {}".format(result_prefix + "_tree.txt"))
    flow = open(result_prefix + "_tree.txt", "w", encoding="utf8")
    flow.write(node.tree_string())
    flow.write("\n" + experiment_id)
    flow.close()

def macroclasses_command():
    parser = get_default_parser(main.__doc, "Results/Clustering",
                                patterns=True, paradigms=False)
    args = parser.parse_args()
    main(args)

if __name__ == '__main__':
    macroclasses_command()

