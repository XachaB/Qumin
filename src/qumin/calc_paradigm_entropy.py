#!usr/bin/python3
# -*- coding: utf-8 -*-
"""author: Sacha Beniamine.

Compute conditional entropies in inflectional patterns.
"""

import logging
import argparse

# Our libraries
from .representations import segments, patterns, create_paradigms, create_features
from .entropy.distribution import PatternDistribution, SplitPatternDistribution
from .utils import get_default_parser, Metadata


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

    md = Metadata(args, __file__)

    patterns_file_path = args.patterns
    paradigms_file_path = args.paradigms
    features_file_name = args.segments

    preds = sorted(args.nPreds)
    overabundant = args.overabundant

    onePred = preds[0] == 1
    if onePred:
        preds.pop(0)

    cells = args.cells
    if len(cells) == 1:
        raise argparse.ArgumentTypeError("You can't provide only one cell.")

    # Define logging levels (different depending on verbosity)
    if args.verbose or args.debug:
        logfile_name = md.register_file('debug.log', {'content': 'log'})
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG,
                            filename=logfile_name, filemode='w')
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
    paradigms = create_paradigms(paradigms_file_path, defective=True,
                                 overabundant=overabundant,
                                 merge_cols=args.cols_merged,
                                 segcheck=True,
                                 frequencies=True,
                                 col_names=args.cols_names, cells=cells)
    pat_table, pat_dic = patterns.from_csv(patterns_file_path, defective=True,
                                           overabundant=overabundant)

    if pat_table.shape[0] < paradigms.shape[0]:
        log.warning(
            "It looks like you ignored defective rows when computing patterns. I'll drop all defectives.")
        paradigms = paradigms[(paradigms != "").all(axis=1)]

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
        paradigms2 = create_paradigms(args.bipartite[1], defective=True,
                                      overabundant=False,
                                      merge_cols=args.cols_merged, segcheck=True,
                                      col_names=args.cols_names, cells=cells)
        pat_table2, pat_dic2 = patterns.from_csv(args.bipartite[0], defective=True,
                                                 overabundant=False)

        distrib = SplitPatternDistribution([paradigms, paradigms2],
                                           [pat_table, pat_table2],
                                           [pat_dic, pat_dic2],
                                           args.names,
                                           features=features)
        if args.comp:
            computation = 'bipartiteEntropies'
            ent_file1 = md.register_file('bipartite1.csv',
                                         {'computation': computation,
                                          'source': args.name[0],
                                          'content': 'entropies'})
            ent_file2 = md.register_file('bipartite2.csv',
                                         {'computation': computation,
                                          'source': args.name[1],
                                          'content': 'entropies'})
            I = md.register_file('I.csv', {'computation': computation,
                                           'source': args.names,
                                           'content': 'I'})
            NMI = md.register_file('NMI.csv',
                                   {'computation': computation,
                                    'source': args.names,
                                    'content': 'NMI'})

            distrib.distribs[0].entropy_matrix()
            entropies1 = distrib.distribs[0].entropies[1]
            distrib.distribs[1].entropy_matrix()
            entropies2 = distrib.distribs[1].entropies[1]
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
                log.debug("Mean I(%s,%s) = %s", *args.names, mean3)
                log.debug("Mean NMI(%s,%s) = %s", *args.names, mean4)

    else:
        distrib = PatternDistribution(paradigms,
                                      pat_table,
                                      pat_dic,
                                      overabundant=overabundant,
                                      features=features)

    if onePred:
        computation = 'onePredEntropies'

        ent_file = md.register_file('entropies.csv',
                                    {'computation': computation,
                                     'content': 'entropies'})
        acc_file = md.register_file('accuracies.csv',
                                    {'computation': computation,
                                     'content': 'accuracies'})
        effectifs_file = md.register_file('effectifs.csv',
                                          {'computation': computation,
                                           'content': 'effectifs'})
        if overabundant:
            distrib.entropy_matrix_OA(beta=args.beta)
        else:
            distrib.entropy_matrix()

        accuracies = distrib.accuracies[1]
        entropies = distrib.entropies[1]
        effectifs = distrib.effectifs[1]

        if args.stacked:
            entropies = entropies.stack()
            accuracies = accuracies.stack()
            entropies.index = [' -> '.join(index[::-1])
                               for index in entropies.index.values]
            accuracies.index = [' -> '.join(index[::-1])
                                for index in accuracies.index.values]
        log.info("Writing to: \n\t{}\n\t{}\n\t{}".format(ent_file,
                                                         effectifs_file,
                                                         acc_file))
        entropies.to_csv(ent_file, sep="\t")
        effectifs.to_csv(effectifs_file, sep="\t")
        accuracies.to_csv(acc_file, sep="\t")
        # mean on df's index, then on Series' values.
        mean_ent = entropies.mean().mean()
        mean_acc = accuracies.mean().mean()
        log.info("Mean H(c1 -> c2) = %s ", mean_ent)
        log.info("Mean E(c1 -> c2) = %s ", mean_acc)
        log.debug("Mean H(c1 -> c2) = %s ", mean_ent)

        if args.debug:
            if overabundant:
                check = distrib.one_pred_distrib_log_OA(
                    sanity_check=sanity_check)
            else:
                check = distrib.one_pred_distrib_log(
                    sanity_check=sanity_check)

            if sanity_check:
                check_file = md.register_file('entropies_slow_method.csv',
                                              {'computation': computation,
                                               'content': 'entropies_slow_method'})

                log.info("Writing slowly computed entropies to: %s", check_file)

                check.to_csv(check_file, sep="\t")

    if preds:

        if args.importFile:
            distrib.read_entropy_from_file(args.importFile)

        for n in preds:
            computation = 'nPredsEntropies'
            n_ent_file = md.register_file('npreds{}_entropies.csv'.format(n),
                                          {'computation': computation,
                                           'content': 'entropies',
                                           'n': n})
            effectifs_file = md.register_file('npreds{}_effectifs.csv'.format(n),
                                              {'computation': computation,
                                               'content': 'effectifs',
                                               'n': n})

            distrib.n_preds_entropy_matrix(n)
            n_entropies = distrib.entropies[n]
            effectifs = distrib.effectifs[n]
            log.info("\nWriting to: {}\n\tand {}".format(n_ent_file, effectifs_file))
            if args.stacked:
                n_entropies = n_entropies.stack()
                n_entropies.index = [' -> '.join(index[::-1])
                                     for index in n_entropies.index.values]
            n_entropies.to_csv(n_ent_file, sep="\t")
            effectifs.to_csv(effectifs_file, sep="\t")
            mean = n_entropies.mean().mean()
            log.info("Mean H(c1, ..., c%s-> c) = %s", n, mean)
            log.debug("Mean H(c1, ..., c%s -> c) = %s", n, mean)
            if args.debug:
                n_check = distrib.n_preds_distrib_log(n, sanity_check=sanity_check)

                if sanity_check:
                    n_check_file = md.register_file('npreds{}_entropies_slow.csv',
                                                    {'computation': computation,
                                                     'content': 'entropies_slow_method',
                                                     'n': n})

                    log.info("Writing slowly computed"
                             " entropies to: {}".format(n_check_file))
                    n_check.to_csv(n_check_file, sep="\t")

            if onePred and args.debug:
                distrib.value_check(n)

    if args.debug:
        log.info("Wrote log to: {}".format(logfile_name))

    md.save_metadata()


def H_command():
    parser = get_default_parser(main.__doc__, paradigms=True, patterns=True)

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

    parser.add_argument('--weights',
                        help="Weights for overabundant forms",
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

    parser.add_argument("--cells",
                        help="List of cells to use. Defaults to all.",
                        nargs='+', default=None)

    parser.add_argument("-o", "--overabundant",
                        help="Use overabundant entries for computation.",
                        action="store_true", default=False)

    parser.add_argument("--beta",
                        help="Value of beta to use for softmax.",
                        type=int, default=10)

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

    args = parser.parse_args()

    main(args)


if __name__ == '__main__':
    H_command()
