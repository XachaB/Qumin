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

from .utils import get_default_parser, Metadata
from .representations import segments, patterns
from .clustering import algorithms, descriptionlength, find_min_attribute
import pandas as pd
import logging
from pathlib import Path
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
    md = Metadata(args, __file__)

    # Loading files and paths
    data_file_path = args.patterns
    data_file_name = Path(data_file_path).name.rstrip("_")

    pattern_type_match = re.match(r".+_(.+)\.csv", data_file_name)
    if pattern_type_match is None:
        log.warning("Did you rename the patterns file ? "
                    "As a result, I do not know which type of pattern you used..")
        kind = "unknown"
    else:
        kind = pattern_type_match.groups()[0]

    # Initializing segments

    if args.ortho:
        pat_table = pd.read_csv(data_file_path, index_col=0)
    else:
        sounds_file_name = md.get_table_path("sounds")
        segments.Inventory.initialize(sounds_file_name)
        pat_table, pat_dic = patterns.from_csv(data_file_path, defective=False, overabundant=False)
        pat_table = pat_table.map(str)

    preferences = {"md": md}

    node = algorithms.hierarchical_clustering(pat_table, descriptionlength.BUDLClustersBuilder, **preferences)

    DL = "Min :" + str(find_min_attribute(node, "DL"))
    experiment_id = " ".join(["Bottom-up DL clustering on ", kind, DL])

    computation = "macroclasses"
    # Saving png figure
    if MATPLOTLIB_LOADED:
        fig = plt.figure(figsize=(10, 20))
        figname = md.register_file("figure.png",
                                   {"computation": computation,
                                    "content": "figure"})
        log.info("Drawing figure to: {}".format(figname))
        node.draw(horizontal=True,
                  square=True,
                  layout="qumin",
                  leavesfunc=lambda x: x.labels[0] + " (" + str(x.attributes["size"]) + ")",
                  nodefunc=lambda x: "{0:.3f}".format(x.attributes["DL"]),
                  keep_above_macroclass=True)

        fig.suptitle(experiment_id)
        fig.savefig(figname,
                    bbox_inches='tight', pad_inches=.5)

    # Saving text tree
    treename = md.register_file("tree.txt",
                                {"computation": computation,
                                 "content": "tree"})
    log.info("Printing tree to: {}".format(treename))
    flow = open(treename, "w", encoding="utf8")
    flow.write(node.tree_string())
    flow.write("\n" + experiment_id)
    flow.close()

    md.save_metadata()


def macroclasses_command():
    parser = get_default_parser(main.__doc__, patterns=True)

    parser.add_argument("--ortho",
                        help="the patterns are orthographic",
                        action="store_true", default=False)

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    macroclasses_command()
