# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Evaluate patterns with prediction task.

Author: Sacha Beniamine.
"""
from .entropy import cond_P, P
import numpy as np
from .representations import segments, create_paradigms, patterns, create_features
import pandas as pd
import argparse
from .utils import get_default_parser, Metadata
from itertools import combinations, chain
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
log = logging.getLogger()
sns.set()


def prepare_arguments(paradigms, iterations, methods, features):
    """Generate argument tuples for each evaluation task.

    The tuples are:


    Arguments:
        paradigms (:class:`pandas.DataFrame`): a dataframe of forms
        iterations (bool): number of train/test splits to evaluate
        methods (list of str): alignment methods to use
        features (DataFrame): features to help prediction

    Yields:
        a tuple of (test_items, train_items, method, features, infos)
        test_items is a DataFrame containing test forms for two paradigm cells
        train_items is a DataFrame containing train forms for the same two paradigm cells
        method is the name of a method for finding patterns
        features is either None or a dataframe of features used to help the prediction (ex:gender, transitivity, ...)
        infos is a dictionnary with more information on the current iteration
    """
    idx = pd.IndexSlice

    def get_set(table, a, b, index_range):
        return table.loc[:, idx[:, [a, b]]].iloc[index_range, :].dropna()

    cells = paradigms.columns.levels[1].tolist()
    shuffled_indexes = np.random.permutation(paradigms.shape[0])
    folds = np.array_split(shuffled_indexes, 10)
    l_folds = len(folds)
    l = iterations if iterations is not None else l_folds

    for i in range(l):
        test_range = folds[i]
        train_range = [idx for j in range(l_folds) for idx in folds[j] if j != i]
        infos = {"fold": i}
        for a, b in combinations(cells, 2):
            train_items = get_set(paradigms, a, b, train_range)
            test_items = get_set(paradigms, a, b, test_range)
            for method in methods:
                yield test_items, train_items, method, features, dict(infos)


def evaluate(task):
    """Learn and predict for a given split and pair of cells, then calculate evaluation metrics.

    Arguments:
        task: a tuple of (test_items, train_items, method, features, infos)
        test_items is a DataFrame containing test forms for two paradigm cells
        train_items is a DataFrame containing train forms for the same two paradigm cells
        method is the name of a method for finding patterns
        features is either None or a dataframe of features used to help the prediction (ex:gender, transitivity, ...)
        row is a dictionnary with more information on the current iteration

    Returns:
        row: The dictionnary passed in argument, with stats for this evaluation.
    """
    test_items, train_items, method, features, row = task  # because multiprocessing does not have "istarmap"
    table_ids = test_items.columns.levels[0].tolist()
    ab_predictions = None
    ba_predictions = None
    counts = 0
    for id in table_ids:
        test = test_items[id]
        train = train_items[id]
        pred_ab, pred_ba, count = predict_two_directions(test, train, method, features=features)
        if ab_predictions is None and ba_predictions is None:
            ab_predictions = pred_ab
            ba_predictions = pred_ba
        else:
            ab_predictions &= pred_ab
            ba_predictions &= pred_ba
        counts += count

    a, b = test_items[id].columns  # any of the previous values for table id would work
    row["count"] = counts
    row["total_train"] = train_items.shape[0]
    row["total_test"] = test_items.shape[0]
    row["method"] = method
    row_ab = dict(row)
    row_ba = dict(row)
    row_ab["correct"] = ab_predictions.sum()
    row_ab["from"] = a
    row_ab["to"] = b
    row_ba["correct"] = ba_predictions.sum()
    row_ba["from"] = b
    row_ba["to"] = a
    return (row_ab, row_ba)


def predict_two_directions(test_items, train_items, method, features=None):
    """Predict forms of cell a from those of cells b and reciprocally.

    Arguments:
        test_items: the test items
        train_items: the train items
        method : the method used to find patterns

    Returns:
        a tuple of three elementss: predicted_correct1, predicted_correct2, count
        predicted_correct1 (Series) indicates test lexemes which were predicted correctly in one direction,
        predicted_correct1 (Series) indicates test lexemes which were predicted correctly in the opposite direction,
        count (int): Is the number of patterns found for this set of train items.
    """

    def predict(row, prediction, cells, repli):
        form, solution, classe = row
        result = "<Error>"
        pat = None
        if classe:
            if classe in prediction:
                pat = prediction[classe]
            else:
                pat = sorted(classe, key=lambda x: repli[x])[-1]

        # form and solution are length-1 tuples with Form objects
        # They are tuples because paradigms can have overabundant forms
        # for eval, we ignore overabundant items
        if pat is not None:
            result = pat.apply(form[0], names=cells, raiseOnFail=False)
        return result == solution[0]

    def prepare_prediction(patrons, classes):
        return cond_P(patrons, classes).groupby(level=0).aggregate(lambda x: x.idxmax()[1]).to_dict()

    def repli_prediction(patrons):
        return dict(P(patrons))

    a, b = test_items.columns

    train_range = train_items.index
    test_range = test_items.index

    A, dic = patterns.find_patterns(train_items, method, disable_tqdm=True)
    try:
        A = A[A.columns[0]].apply(lambda x: x.collection[0])
    except AttributeError:
        A = A[A.columns[0]]

    classes = patterns.find_applicable(train_items.append(test_items), dic, disable_tqdm=True)

    B = classes[(a, b)]

    C = classes[(b, a)]

    if features is not None:
        B = B + features[B.index]
        C = C + features[C.index]

    repli = repli_prediction(A)

    pred = prepare_prediction(A, B[train_range])
    test_set = pd.concat([test_items, B[test_range]], axis=1)
    predicted_correct1 = test_set.apply(predict, args=(pred, (a, b), repli), axis=1)

    pred = prepare_prediction(A, C[train_range])
    test_set = pd.concat([test_items[[b, a]], C[test_range]], axis=1)
    predicted_correct2 = test_set.apply(predict, args=(pred, (b, a), repli), axis=1)

    return predicted_correct1, predicted_correct2, len(dic[(a, b)])


def prepare_data(args):
    """Create a multi-index paradigm table and if given a path, a features table."""
    paradigms = []

    # Read all files
    for file in args.paradigms:
        table = create_paradigms(file, segcheck=True, fillna=False, merge_cols=True,
                                 overabundant=False, defective=True, col_names=args.cols_names)
        paradigms.append(table)

    # Keep only common indexes
    indexes = list(set.intersection(*[set(t.index.tolist()) for t in paradigms]))
    assert len(indexes) > 0, "No common indexes in your tables"
    paradigms = [t.loc[indexes, :] for t in paradigms]

    # Make a multi-index df composed of all tables
    paradigms = pd.concat(paradigms, axis=1, keys=range(len(args.paradigms))).drop_duplicates()

    features = None
    if features is not None:
        features = create_features(args.features)
        features, _ = pd.DataFrame.sum(features.map(lambda x: (str(x),)), axis=1).factorize()
        features = features[indexes].apply(lambda x: (x,))

    if args.randomsample:
        paradigms = paradigms.sample(args.randomsample)

    return paradigms, features


def print_summary(results, general_infos):
    log.info("# Evaluation summary")
    summary = results[["method",
                       "correct",
                       "count",
                       "total_test",
                       "total_train"]].groupby("method").agg({"correct": "sum",
                                                              "total_test": "sum",
                                                              "count": "mean"})
    summary["average accuracy"] = (summary["correct"] / summary["total_test"]).apply(lambda x: "{:.2%}".format(x))
    summary["average count of patterns"] = summary["count"]
    log.info(summary[["average accuracy", "average count of patterns"]].to_markdown())

    for info in sorted(general_infos):
        log.info('{}: {}'.format(info, general_infos[info]))


def to_heatmap(results, cells):
    cells = list(cells)
    for method in results["method"].unique():
        r = results[results["method"] == method]
        per_cell = r[["from", "to", "correct", "total_test"]].groupby(["from", "to"], as_index=False).sum()
        per_cell["avg"] = per_cell["correct"] / per_cell["total_test"]
        matrix = per_cell.pivot_table(index="from", columns="to", values="avg", aggfunc=lambda x: sum(x) / len(x))
        matrix = matrix.loc[cells, cells].fillna(1)
        ax = sns.heatmap(matrix, cmap="Blues", vmin=0, vmax=1)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        fig = ax.get_figure()
        yield method, fig
        plt.close(fig)


def main(args):
    r"""Evaluate pattern's accuracy with 10 folds.

    For a detailed explanation, see the html doc.::
      ____
     / __ \                    /)
    | |  | | _   _  _ __ ___   _  _ __
    | |  | || | | || '_ ` _ \ | || '_ \
    | |__| || |_| || | | | | || || | | |
     \___\_\ \__,_||_| |_| |_||_||_| |_|
      Quantitative modeling of inflection

    """
    log.info(args)
    md = Metadata(args, __file__)
    now = md.day+"_"+md.now
    np.random.seed(0)  # make random generator determinist

    segments.Inventory.initialize(args.segments)
    paradigms, features = prepare_data(args)

    files = [Path(file).stem for file in args.paradigms]

    general_infos = {"Qumin_version": md.version,
                     "lexemes": paradigms.shape[0],
                     "paradigms": ";".join(files),
                     "day_time": now}

    tasks = prepare_arguments(paradigms, args.iterations,
                              args.methods, features)
    if args.workers == 1:
        results = list(chain(*(evaluate(t) for t in tqdm(tasks))))
    else:
        pool = Pool(args.workers)
        results = list(chain(*tqdm(pool.imap_unordered(evaluate, tasks))))
        pool.close()

    results = pd.DataFrame(results)
    for info in general_infos:
        results[info] = general_infos[info]

    computation = 'evalPatterns'
    filename = md.register_file("eval_patterns.csv",
                                {"computation": computation,
                                 "content": "scores",
                                 "source": files})
    results.to_csv(filename)

    print_summary(results, general_infos)
    figs = to_heatmap(results, paradigms.columns.levels[1].tolist())
    for name, fig in figs:
        figname = md.register_file("eval_patterns_{}.png".format(name),
                                   {"computation": computation,
                                    "content": "heatmap",
                                    "name": name,
                                    "source": files})
        fig.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.5)

    md.save_metadata()


def eval_command():

    parser = get_default_parser(main.__doc__, paradigms=True,
                                patterns=False, multipar=True)

    parser.add_argument('-i', '--iterations',
                        help="How many test/train folds to do. Defaults to full cross validation.",
                        type=int,
                        default=None)

    parser.add_argument('-m', '--methods',
                        help="Methods to align forms. Default: compare all.",
                        choices=["suffix", "prefix", "baseline", "levenshtein",
                                 "similarity"],
                        nargs="+",
                        default=["suffix", "prefix", "baseline", "levenshtein",
                                 "similarity"])

    parser.add_argument('--workers',
                        help="number of workers",
                        type=int,
                        default=1)

    parser.add_argument('-r', '--randomsample',
                        help="Mostly for debug",
                        type=int,
                        default=None)

    parser.add_argument('--features',
                        help="Feature file. Features will be considered known when predicting",
                        type=str,
                        default=None)

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    eval_command()
