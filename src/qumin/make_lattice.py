# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Cluster lemmas according to their paradigms.

Author: Sacha Beniamine.
"""

from .clustering import find_microclasses
from .representations import segments, patterns
from .utils import get_default_parser, Metadata
from .lattice.lattice import ICLattice

import pandas as pd
import logging


def main(args):
    r"""Infer Inflection classes as a lattice from alternation patterns.

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
    features_file_name = args.segments
    data_file_path = args.patterns

    if features_file_name != "ORTHO":

        # Initializing segments
        log.info("Initializing segments...")
        segments.Inventory.initialize(features_file_name)

        log.info("Reading patterns...")
        pat_table, _ = patterns.from_csv(data_file_path)
        # pat_table = pat_table.map(str)
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
        log.info("Reading patterns...")
        pat_table = pd.read_csv(data_file_path, index_col=0)
        collections = False

    microclasses = find_microclasses(pat_table.map(str))

    log.info("Building the lattice...")
    lattice = ICLattice(pat_table.loc[list(microclasses), :], microclasses,
                        overabundant=collections, comp_prefix=comp, aoc=args.aoc, keep_names=(not args.shorten))
    computation = "lattice"

    if args.stat:
        statname = md.register_file('stats.txt', {"computation": computation,
                                                  "content": "stats"})
        with open(statname, "w", encoding="utf-8") as flow:
            flow.write(lattice.stats().to_frame().T.to_latex())
            log.info(lattice.stats().to_frame().T.to_latex())

    if args.png:
        lattpng = md.register_file('lattice.png', {'computation': computation,
                                                   'content': 'figure'})
        lattice.draw(lattpng, figsize=(20, 10), title=None, point=True)

    if args.pdf:
        lattpdf = md.register_file('lattice.pdf', {'computation': computation,
                                                   'content': 'figure'})
        lattice.draw(lattpdf, figsize=(20, 10), title=None, point=True)

    if args.html:
        latthtml = md.register_file('lattice.html', {'computation': computation,
                                                     'content': 'figure'})
        log.info("Exporting to html: " + latthtml)
        lattice.to_html(latthtml)

    if args.cxt:
        lattcxt = md.register_file('lattice.cxt', {'computation': computation,
                                                   'content': 'figure'})
        log.info(" ".join("Exporting context to file:", lattcxt))
        lattice.context.tofile(lattcxt, frmat='cxt')

    if args.first:
        log.info("Here is the first level of the hierarchy:")
        log.info("Root:")
        obj, common = lattice.nodes.attributes["objects"], lattice.nodes.attributes["common"]
        if obj or common:
            log.info("\tdefines: " + str(obj) + str(common))
        for child in lattice.nodes.children:
            extent, common = child.labels, child.attributes["common"]
            log.info(" ".join("extent:", extent, "\n\tdefines:", common, ">"))

    md.save_metadata()


def lattice_command():

    parser = get_default_parser(main.__doc__,  patterns=True,
                                paradigms=False)

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

    args = parser.parse_args()

    main(args)


if __name__ == '__main__':
    lattice_command()
