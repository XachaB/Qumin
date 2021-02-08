#!usr/bin/python3
# -*- coding: utf-8 -*-
"""author: Sacha Beniamine.

Compute conditional entropies in inflectional patterns.
"""

import argparse
from os import path, makedirs

# Our libraries
from representations import segments, patterns, create_paradigms, create_features
from entropy.distribution import PatternDistribution, SplitPatternDistribution
from utils import get_repository_version
import logging

def main(args):
    r"""Compute entropies of flexional paradigms' distributions.

    For a detailed explanation, see the corresponding ipython Notebook
    and the html doc.::

          ____
         / __ \                    /)
        | |  | | _   _  _ __ ___   _  _ __
        | |  | || | | || '_ ` _ \ | || '_ \
        | |__| || |_| || | | | | || || | | |
         \___\_\ \__,_||_| |_| |_||_||_| |_|
          Quantitative modeling of inflection

    """
    patterns_file_path = args.patterns
    paradigms_file_path = args.paradigms
    data_file_name = path.basename(patterns_file_path).rstrip("_")

    features_file_name = args.segments

    import time

    now = time.strftime("%Hh%M")
    day = time.strftime("%Y%m%d")

    result_dir = "../Results/{}/{}".format(args.folder, day)
    makedirs(result_dir, exist_ok=True)
    version = get_repository_version()
    preds = sorted(args.nPreds)
    onePred = preds[0] == 1
    if onePred:
        preds.pop(0)
    result_prefix = "{}/{}_{}_{}_{}_".format(result_dir, data_file_name, version, day, now)

    # Define logging levels (different depending on verbosity)
    if args.debug:
        logfile_name = result_prefix + ".log"
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG, filename=logfile_name, filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    log = logging.getLogger()
    log.info(args)

    # Initialize the class of segments.
    segments.Inventory.initialize(features_file_name)

    # Inflectional paradigms: columns are cells, rows are lexemes.
    paradigms = create_paradigms(paradigms_file_path, defective=True, overabundant=False, merge_cols=args.cols_merged,
                                 segcheck=True)
    pat_table, pat_dic = patterns.from_csv(patterns_file_path, defective=True, overabundant=False)

    if args.debug and len(pat_table.columns) > 10:
        log.warning("Using debug mode is strongly "
                    "discouraged on large (>10 cells) datasets."
                    "You should probably stop this process now.")
    sanity_check = args.debug and len(pat_table.columns) < 10

    if args.features is not None:
        features = create_features(args.features)
    else:
        features = None

    if args.bipartite:

        result_prefix = "{}/{}_{}_{}_{}_bipartite".format(result_dir, data_file_name, version, day, now)
        paradigms2 = create_paradigms(args.bipartite[1], defective=True, overabundant=False,
                                      merge_cols=args.cols_merged, segcheck=True)
        pat_table2, pat_dic2 = patterns.from_csv(args.bipartite[0], defective=True, overabundant=False)

        distrib = SplitPatternDistribution([paradigms, paradigms2],
                                           [pat_table, pat_table2],
                                           [pat_dic, pat_dic2],
                                           args.names,
                                           features=features)
        if args.comp:
            ent_file1 = "{}onepredEntropies-{}.csv".format(result_prefix, args.names[0])
            ent_file2 = "{}onepredEntropies-{}.csv".format(result_prefix, args.names[1])
            I = "{}EntropiesI-{}{}.csv".format(result_prefix, *args.names)
            NMI = "{}EntropiesNMI-{}{}.csv".format(result_prefix, *args.names)

            entropies1, _ = distrib.distribs[0].entropy_matrix()
            entropies2, _ = distrib.distribs[1].entropy_matrix()
            mutual = distrib.mutual_information()
            normmutual = distrib.mutual_information(normalize=True)

            log.info("Writing to:" + "\n\t".join([ent_file1, ent_file2, I, NMI]))
            entropies1.to_csv(ent_file1, sep="\t")
            entropies2.to_csv(ent_file2, sep="\t")
            mutual.to_csv(I, sep="\t")
            normmutual.to_csv(NMI, sep="\t")
            if args.debug:
                # mean on df's index, then on Series' values.
                mean1 = entropies1.mean().mean()
                mean2 = entropies2.mean().mean()
                mean3 = mutual.mean().mean()
                mean4 = normmutual.mean().mean()
                log.debug("Mean remaining H(c1 -> c2) for %s = %s", args.names[0], mean1)
                log.debug("Mean remaining H(c1 -> c2) for %s = %s", args.names[1], mean2)
                log.debug("Mean I(%s,%s) = %s",*args.names, mean3)
                log.debug("Mean NMI(%s,%s) = %s",*args.names, mean4)

    else:
        distrib = PatternDistribution(paradigms,
                                      pat_table,
                                      pat_dic,
                                      features=features)

    if onePred:
        ent_file = "{}onePredEntropies.csv".format(result_prefix)
        effectifs_file = "{}onePredEntropiesEffectifs.csv".format(result_prefix)
        entropies, effectifs = distrib.entropy_matrix()
        if args.stacked:
            entropies = entropies.stack()
            entropies.index = [' -> '.join(index[::-1])
                               for index in entropies.index.values]
        log.info("Writing to: {}\n\tand {}".format(ent_file, effectifs_file))
        entropies.to_csv(ent_file, sep="\t")
        effectifs.to_csv(effectifs_file, sep="\t")
        # mean on df's index, then on Series' values.
        mean = entropies.mean().mean()
        log.info("Mean H(c1 -> c2) = %s ", mean)
        log.debug("Mean H(c1 -> c2) = %s ", mean)

        if args.debug:
            check = distrib.one_pred_distrib_log(sanity_check=sanity_check)

            if sanity_check:
                scsuffix = "{}onePredEntropies_slow_method.csv"
                check_file = scsuffix.format(result_prefix)

                log.info("Writing slowly computed entropies to: {}".format(check_file))

                check.to_csv(check_file, sep="\t")

    if preds:

        if args.importFile:
            distrib.read_entropy_from_file(args.importFile)

        for n in preds:
            n_ent_file = "{}{}PredsEntropies.csv".format(result_prefix, n)
            effectifs_file = "{}{}PredsEntropiesEffectifs.csv".format(result_prefix, n)
            n_entropies, effectifs = distrib.n_preds_entropy_matrix(n)
            log.info("Writing to: {}\n\tand {}".format(n_ent_file, effectifs_file))
            if args.stacked:
                n_entropies = n_entropies.stack()
                n_entropies.index = [' -> '.join(index[::-1])
                                     for index in n_entropies.index.values]
            n_entropies.to_csv(n_ent_file, sep="\t")
            effectifs.to_csv(effectifs_file, sep="\t")
            mean = n_entropies.mean().mean()
            log.info("Mean H(c1, ..., c%s-> c) = %s",n, mean)
            log.debug("Mean H(c1, ..., c%s -> c) = %s", n, mean)
            if args.debug:
                n_check = distrib.n_preds_distrib_log(n, sanity_check=sanity_check)

                if sanity_check:
                    scsuffix = "{}{}PredsEntropies_slow_method.csv"
                    n_check_file = scsuffix.format(result_prefix, n)
                    log.info("Writing slowly computed"
                          " entropies to: {}".format(n_check_file))
                    n_check.to_csv(n_check_file, sep="\t")

            if onePred and args.debug:
                distrib.value_check(n)

    if args.debug:
        log.info("Wrote log to: {}".format(logfile_name))


if __name__ == '__main__':
    usage = main.__doc__

    parser = argparse.ArgumentParser(description=usage,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("patterns",
                        help="patterns file, full path"
                             " (csv separated by ‘, ’)",
                        type=str)

    parser.add_argument("paradigms",
                        help="paradigms file, full path"
                             " (csv separated by ‘, ’)",
                        type=str)

    parser.add_argument("segments",
                        help="segments file, full path"
                             " (csv separated by '\\t')",
                        type=str, default=None)

    parser.add_argument('-b', '--bipartite',
                        help="Add a second paradigm dataset, for bipartite systems.",
                        nargs=2,
                        type=str,
                        default=None)

    parser.add_argument('--features',
                        help="Feature file. Features will "
                             "be considered known in conditional probabilities:"
                             " P(X~Y|X,f1,f2...)",
                        type=str,
                        default=None)

    parser.add_argument('--names',
                        help="Ordered names of bipartite systems (-b argument is 2nd)",
                        nargs=2,
                        type=str,
                        default=None)

    parser.add_argument("-d", "--debug",
                        help="show debug information."
                             "On small datasets "
                             "this will compute a sanity-check "
                             "and output write probability tables.",
                        action="store_true")

    parser.add_argument("-i", "--importFile", metavar="file",
                        help="Import entropy file with n-1 predictors"
                             " (allows for acceleration "
                             "on nPreds entropy computation).",
                        type=str, default=None)

    parser.add_argument("-m", "--cols_merged",
                        help="Whether identical columns are merged in the input.",
                        action="store_true", default=False)

    actions = parser.add_argument_group('actions')

    actions.add_argument("--comp",
                         help="Thorough comparison for bipartite systems:"
                              "Logs H(c1->c2), H(c1'->c2'),"
                              " I(c1'->c2';c1->c2) and  NMI(c1'->c2';c1->c2)",
                         action="store_true",
                         default=False)

    actions.add_argument("-n", "--nPreds", metavar="N",
                         help="Compute entropy for prediction "
                              "from with n predictors. Enter n",
                         nargs='+', type=int, default=[1])

    options = parser.add_argument_group('Optional outputs for actions')

    options.add_argument("-s", "--stacked",
                         help="Export result as only one column.",
                         action="store_true",
                         default=False)

    options.add_argument("-f", "--folder",
                         help="Output folder name",
                         type=str, default="JointPred")

    args = parser.parse_args()

    main(args)
