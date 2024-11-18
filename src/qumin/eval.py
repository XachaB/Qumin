# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Evaluate patterns with prediction task.

Author: Sacha Beniamine.
"""
from .entropy import cond_P, P
import numpy as np
from .representations import segments, patterns, create_features
import pandas as pd
from itertools import combinations, chain
from multiprocessing import Pool
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
import logging

log = logging.getLogger()
sns.set()


def prepare_arguments(paradigms, iterations, methods, features):
    """Generate argument tuples for each evaluation task.

    The tuples are:


    Arguments:
        paradigms (:class:`pandas.DataFrame`): a dataframe of forms
        iterations (bool): number of train/test splits to evaluate
        methods (List[str]): alignment methods to use
        features (:class:`pandas.DataFrame`): features to help prediction

    Yields:
        a tuple of (test_items, train_items, method, features, infos)
            test_items is a DataFrame containing test forms for two paradigm cells
            train_items is a DataFrame containing train forms for the same two paradigm cells
            method is the name of a method for finding patterns
            features is either None or a dataframe of features used to help the prediction (ex:gender, transitivity, ...)
            infos is a dictionnary with more information on the current iteration
    """

    def get_set(table, a, b, index_range):
        return table.loc[:, [a, b]].iloc[index_range, :].dropna()

    cells = paradigms.columns.tolist()
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
        task (tuple): a tuple of (test_items, train_items, method, features, infos)
            test_items is a DataFrame containing test forms for two paradigm cells
            train_items is a DataFrame containing train forms for the same two paradigm cells
            method is the name of a method for finding patterns
            features is either None or a dataframe of features used to help the prediction (ex:gender, transitivity, ...)
            row is a dictionnary with more information on the current iteration

    Returns:
        tuple: The dictionnary passed in argument,
            with stats for this evaluation for both directions.
    """
    test_items, train_items, method, features, row = task  # because multiprocessing does not have "istarmap"
    ab_predictions = None
    ba_predictions = None
    counts = 0
    pred_ab, pred_ba, count = predict_two_directions(test_items, train_items, method, features=features)
    if ab_predictions is None and ba_predictions is None:
        ab_predictions = pred_ab
        ba_predictions = pred_ba
    else:
        ab_predictions &= pred_ab
        ba_predictions &= pred_ba
    counts += count

    a, b = test_items.columns  # any of the previous values for table id would work
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
        tuple: a tuple of three elements: predicted_correct1, predicted_correct2, count
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

    classes = patterns.find_applicable(pd.concat([train_items, test_items]), dic, disable_tqdm=True)

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


def prepare_data(cfg, md):
    """Create a multi-index paradigm table and if given a path, a features table."""
    paradigms = create_paradigms(md.datasets[0],
                                 segcheck=True,
                                 fillna=False,
                                 merge_cols=cfg.pats.merged,
                                 overabundant=cfg.pats.overabundant,
                                 defective=cfg.pats.defective,
                                 sample=cfg.sample,
                                 most_freq=cfg.most_freq
                                 )
    indexes = paradigms.index
    features = None

    if cfg.entropy.features is not None:
        features = create_features(md, cfg.entropy.features)
        features, _ = pd.DataFrame.sum(features.map(lambda x: (str(x),)), axis=1).factorize()
        features = features[indexes].apply(lambda x: (x,))

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


def eval_command(cfg, md):
    r"""Evaluate pattern's accuracy with 10 folds."""
    now = md.day + "_" + md.now
    np.random.seed(0)  # make random generator determinist

    sounds_file_name = md.get_table_path("sounds")
    paradigms_file_path = md.get_table_path("forms")

    segments.Inventory.initialize(sounds_file_name)
    paradigms, features = prepare_data(cfg, md)

    general_infos = {"Qumin_version": md.version,
                     "lexemes": paradigms.shape[0],
                     "paradigms": paradigms_file_path,
                     "day_time": now}

    kind_to_method = {
        'patternsLevenshtein': 'levenshtein',
        'patternsPhonsim': 'similarity',
        'patternsSuffix': 'suffix',
        'patternsPrefix': 'prefix',
        'patternsBaseline': 'baseline'
    }

    methods = [kind_to_method[k] for k in cfg.pats.kind]
    tasks = prepare_arguments(paradigms,
                              cfg.eval.iter,
                              methods,
                              features)
    l = cfg.eval.iter * len(list(combinations(range(paradigms.shape[1]), 2)))

    if cfg.eval.workers == 1:
        results = list(chain(*(evaluate(t) for t in tqdm(tasks, total=l))))
    else:
        pool = Pool(cfg.eval.workers)
        results = list(chain(*tqdm(pool.imap_unordered(evaluate, tasks), total=l)))
        pool.close()

    results = pd.DataFrame(results)
    for info in general_infos:
        results[info] = general_infos[info]

    computation = 'evalPatterns'
    filename = md.register_file("eval_patterns.csv",
                                {"computation": computation,
                                 "content": "scores",
                                 "source": paradigms_file_path})
    results.to_csv(filename)

    print_summary(results, general_infos)
    figs = to_heatmap(results, paradigms.columns.tolist())
    for name, fig in figs:
        figname = md.register_file("eval_patterns_{}.png".format(name),
                                   {"computation": computation,
                                    "content": "heatmap",
                                    "name": name,
                                    "source": paradigms_file_path})
        fig.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.5)
