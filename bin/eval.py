# !usr/bin/python3
# -*- coding: utf-8 -*-
"""Evaluate patterns with prediction task.

Author: Sacha Beniamine.
"""
from entropy import cond_P, P
import numpy as np
from representations import segments, create_paradigms, patterns, create_features
import pandas as pd
import argparse
from utils import get_repository_version
from collections import defaultdict
from itertools import combinations
from os import path
import time

def mean(x):
    '''Compute the mean of values in an iterable.'''
    return sum(x)/len(x)

def k_fold_mean_accuracy(paradigms, l, step, k, train_funcs, foldback, bipartite=False, split=False,log=None,**kwargs):
    '''Compute the mean accuracy and mean number of patterns for all known methods, all pairs of columns, and k folds.

    Arguments:
        paradigms (:class:`pandas.DataFrame`): a dataframe of forms
        l : the length of paradigms
        step: the length of one chunk of data, such that there is k chunks.
        k: the number of folds

    Returns:
        a dict of the shape {method : (mean accuracy, mean number of patterns)}
    '''
    k_shuffle = list(range(l))
    np.random.shuffle(k_shuffle)
    accuracy = defaultdict(int)
    count = defaultdict(int)
    total = 0
    total_count = 0


    if bipartite:
        columns = paradigms[0].columns
        accuracy_func = bipartite_accuracy_twocols
    else:
        columns = paradigms.columns
        accuracy_func = accuracy_twocols

    combinations_cols = list(combinations(columns, 2))

    if split:
        st,s = args.split
        len_comb = len(combinations_cols)
        st = min(st,len_comb)
        if s > st:
            exit()
        step_cols = len_comb//st
        left = step_cols*s
        right = step_cols*(s+1)
        if st == s:
            right = len_comb
        combinations_cols = combinations_cols[left:right]
        print("Nb de paires de cases à traiter:",len(combinations_cols))

    total_it = k*len(train_funcs)*len(combinations_cols)
    done = 0
    bound  = 0
    for i in range(k):
        new_bound = bound + step
        test_range = [k_shuffle[i] for i in range(bound,new_bound)]
        train_range = [k_shuffle[i] for i in range(bound)]
        train_range += [k_shuffle[i] for i in range(new_bound,l)]
        print("ITERATION ",i,":")
        for a,b in combinations_cols:
            if bipartite:
                concatenated = pd.concat([paradigms[0][[a,b]],paradigms[1][[a,b]]])
                test_items = concatenated.iloc[test_range,:].dropna()
                train_items = concatenated.iloc[train_range,:].dropna()
            else:
                test_items = paradigms[[a,b]].iloc[test_range,:].dropna()
                train_items = paradigms[[a,b]].iloc[train_range,:].dropna()
            test_len = len(test_items)
            train_len = len(train_items)
            if test_len > 0 and train_len > 0:
                total += test_len*2
                total_count +=1
                for method in train_funcs:
                    r  = accuracy_func(paradigms, a, b, train_funcs[method], test_items, train_items, foldback, **kwargs)
                    accuracy[method] += r["accuracy"]
                    count[method] += r["count"]
                    done += 1
                    print("\t",a,"<->",b,method,
                         "\tthis accuracy:",round(r["accuracy"]/(test_len*2),4),
                         "\tmean accuracy:",round(accuracy[method]/total,4),
                         "\tmean count:",round(count[method]/total_count,4),
                         "\tdone", (done/total_it)*100 ,"%")
        for method in train_funcs:
            print("END ITER",i,method,
                 "\tmean accuracy:",round(accuracy[method]/total,4),
                 "\tmean count:",round(count[method]/total_count,4),
                 "\tdone",(done/total_it)*100,"%")
        bound = new_bound

    result = {}
    for method in accuracy:
        result[method] = (accuracy[method]/total,count[method]/total_count)
        if log is not None:
            print(method, accuracy[method], total, count[method],total_count,file=log)

    return {method: (accuracy[method]/total,count[method]/total_count) for method in accuracy}

def accuracy_twocols(*args,**kwargs):
    prediction_right, prediction_left, count = prediction_twocols(*args,**kwargs)
    return {"accuracy": sum(prediction_right)+sum(prediction_left), "count":count }

def bipartite_accuracy_twocols(paradigms,*args,**kwargs):
    paradigmsA,paradigmsB = paradigms
    prediction_rightA, prediction_leftA, countA = prediction_twocols(paradigmsA,*args,**kwargs)
    prediction_rightB, prediction_leftB, countB = prediction_twocols(paradigmsB,*args,**kwargs)
    return {"accuracy": sum(prediction_rightA & prediction_rightB)+sum(prediction_leftA & prediction_leftB), "count":countA+countB }

def prediction_twocols(paradigms, a, b, train_func, test_items, train_items, foldback, features=None):
    '''Compute the mean accuracy and mean number of patterns for one method and one pair of columns, with given train and test sets.

    Arguments:
        paradigms (:class:`pandas.DataFrame`): a dataframe of forms
        a,b: the pair of cells
        train_func : the function used to find patterns
        test: the index for the test set
        train: the index for the train set

    Returns:
        a dict of  the form : {"accuracy": [accuracy values], "count": [counts of number of patterns]}
    '''

    def predict(row, prediction, cells, repli):
        form, solution, classe = row
        result = "<Error>"
        pat = None
        if classe:
            if classe in prediction:
                pat = prediction[classe]
            elif repli:
                pat = sorted(classe, key=lambda x: repli[x])[-1]

        if pat is not None:
            result = pat.apply(form,names=cells,raiseOnFail=False)
        return (result == solution)

    def prepare_prediction(patrons,classes):
        return cond_P(patrons,classes).groupby(level=0).aggregate(lambda x: x.idxmax()[1]).to_dict()

    def repli_prediction(patrons):
        return dict(P(patrons))

    train_range = train_items.index
    test_range = test_items.index

    A, dic = train_func(train_items)
    try:
        A = A[A.columns[0]].apply(lambda x:x.collection[0])
    except AttributeError:
        A = A[A.columns[0]]

    classes = patterns.find_applicable(train_items.append(test_items), dic)

    B = classes[(a,b)]
    C = classes[(b,a)]

    if features is not None:
        B = B + features[B.index]
        C = C + features[C.index]

    repli = None
    if foldback:
        repli = repli_prediction(A)

    pred = prepare_prediction(A,B[train_range])
    test_set = pd.concat([test_items,B[test_range]], axis=1)
    predicted_correct1 = test_set.apply(predict,args=(pred,(a,b),repli),axis=1)

    pred = prepare_prediction(A,C[train_range])
    test_set = pd.concat([test_items[[b,a]],C[test_range]], axis=1)
    predicted_correct2 = test_set.apply(predict,args=(pred,(b,a),repli),axis=1)

    return predicted_correct1, predicted_correct2, len(dic[(a,b)])



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
    logfile = None
    try:
        np.random.seed(0) # make random generator determinist

        train_funcs = {"levenshtein": patterns.find_levenshtein_patterns,
                    "phono": patterns.find_phonsim_patterns,
                    "suffixal": patterns.find_suffixal_patterns,
                    "prefixal": patterns.find_prefixal_patterns,
                    "baseline": patterns.find_baseline_patterns }

        train_funcs = {m : train_funcs[m] for m in args.methods}

        segments.initialize(args.segments)
        paradigms = create_paradigms(args.paradigms, segcheck=True, fillna=False, merge_cols=True, overabundant=False, defective=True)

        if args.split:
            now = time.strftime("%Hh%M")
            day = time.strftime("%Y%m%d")
            drop = "dropduplicates" if args.dropduplicates else ""
            filename = path.basename(args.paradigms).rstrip("_")
            bipartite = "combined_"+path.basename(args.bipartite).rstrip("_") if args.bipartite else ""
            infos = "_".join([day,now,filename,bipartite,drop,str(args.split[0]),str(args.split[1])])
            logfile = open("../Results/Evals/"+infos+"_eval.log","w",encoding="utf-8")
            print("method,accuracy,total_accuracy,patterns,total_patterns",file=logfile)


        if args.randomsample:
            paradigms = paradigms.sample(args.randomsample)

        if args.dropduplicates:
            paradigms.drop_duplicates(inplace=True)
            if args.bipartite:
                assert args.dropduplicates is None, "You shouldn't pass dropduplicates with a bipartite system. Aborting."

        if args.bipartite is not None:
            paradigms2 = create_paradigms(args.bipartite, segcheck=True, fillna=False, merge_cols=True, overabundant=False, defective=True)
            index = list(set(paradigms.index) & set(paradigms2.index))
            assert len(index) > 0, "For bipartite systems, you need both file to have at least some common indexes. Aborting."
            paradigms = (paradigms.loc[index,:],paradigms2.loc[index,:])
        else:
            index = list(paradigms.index)

        l = len(index)
        step = max(1, (l//10))
        train = l-step
        test = step
        print("Total lexemes: ",len(index))

        #print("Train on {}, test on {}.".format(train,test))
        if args.features is not None:
            features = create_features(args.features)
            features = features.apply(lambda x: x.name+"="+x.apply(str),axis=0)
            features = pd.DataFrame.sum(features.applymap(lambda x: (str(x),)),axis=1)

        else:
            features = None

        results = k_fold_mean_accuracy(paradigms, l, step, args.kfolds, train_funcs,args.foldback,bipartite=args.bipartite,split=args.split,log=logfile, features=features)

        version = get_repository_version()
        print("\n\n############### Results ###############")
        print("Total lexemes: ",len(index))
        #print("Train on {}, test on {}.".format(train,test))
        print("\nEvaluating patterns strategies for {}".format(args.paradigms))
        if args.bipartite is not None:
            print("\n...combined with {}".format(args.bipartite))
        elif args.dropduplicates:
            print("(duplicate rows were dropped)")
        print("version :", version)
        print("Fold back on P(pattern) ? ",args.foldback)
        print("method","mean accuracy","mean number of patterns",sep="\t")
        for result in results:
            print(result,results[result][0],results[result][1],sep="\t")

    finally:
        if logfile:
            logfile.close()


if __name__ == '__main__':
    usage = main.__doc__

    parser = argparse.ArgumentParser(description=usage,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("paradigms",
                        help="paradigm file, full path"
                             " (csv separated by ‘, ’)",
                        type=str)

    parser.add_argument("segments",
                        help="segments file, full path"
                             " (csv separated by '\\t')",
                        type=str)

    parser.add_argument('-k', '--kfolds',
                        help="Number of folds",
                        type=int,
                        default=10)

    parser.add_argument('-m', '--methods',
                        help="Methods to align forms. Default: compare all.",
                        choices=["levenshtein", "phono", "suffixal", "prefixal", "baseline"],
                        nargs="+",
                        default=["levenshtein", "phono", "suffixal", "prefixal", "baseline"])

    parser.add_argument('-b', '--bipartite',
                        help="Add a second paradigm dataset, for bipartite systems.",
                        type=str,
                        default=None)

    parser.add_argument('-d', '--dropduplicates',
                        help="Drop duplicated lines (even when index differs). Useful when testing one half of a bipartite system.",
                        action="store_true",
                        default=False)

    parser.add_argument('-f', '--foldback',
                        help="Use P(pattern) as a fold back when class isn't known (normal prediction is according to  P(pattern|class)).",
                        action="store_true",
                        default=False)

    parser.add_argument('-s', '--split',
                        help="Runs are split in m processes, run the nth split.",
                        nargs=2,
                        type=int,
                        metavar=("m_splits","nth_split"),
                        default=None)

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
