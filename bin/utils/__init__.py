# -*- coding: utf-8 -*-
# !/usr/bin/python3
import sys
import pandas as pd
from collections import defaultdict

class ProgressBar(object):
    """Homemade progressbar"""
    silent = False

    def __init__(self, iterations):

        self.iterations = int(iterations)
        self.prog_bar = ''
        self.fill_char = '█'
        self.width = 50
        self.value = 0
        self.done = 0
        self.animate()

    def increment(self):
        self.value += 1
        new_pct = (100 * self.value) // self.iterations
        if new_pct > self.done and new_pct <= 100:
            self.done = new_pct
            self.animate()
        if self.value == self.iterations and not ProgressBar.silent:
            print("\n")

    def animate(self):
        if not ProgressBar.silent:
            sys.stdout.write('\r'+ str(self))
            sys.stdout.flush()

    def _text(self):
        template = '{:d}% ({!s} of {!s}) complete'
        return template.format(self.done, self.value, self.iterations)

    def _bar(self):
        num_hashes = self.done * self.width // 100
        return (self.fill_char * num_hashes).ljust(self.width)

    def __str__(self):
        return "▕{}▏{}".format(self._bar(), self._text())


def get_repository_version():
    """Return an ID for the current git or svn revision.

    If the directory isn't under git or svn, the function returns an empty str.

     Returns:
        (str): svn/git version or ''.
     """
    import subprocess
    try:
        kwargs = {"universal_newlines": True}
        try:
            no_svn = subprocess.call(["svn", "info"], **kwargs)
        except FileNotFoundError:
            no_svn = 1
        if no_svn == 0:
            version = subprocess.check_output(["svnversion"], **kwargs)
        else:
            version = subprocess.check_output(["git", "describe"], **kwargs)
        return version.strip("\n ")
    except:
        return ''

def merge_duplicate_columns(df,sep=";",keep_names=True):
    """Merge duplicate columns and return new DataFrame.

    Arguments:
        df (:class:`pandas:pandas.DataFrame`): A dataframe
        sep (str): separator to use when joining columns names.
        keep_names (bool): Whether to keep the names of the original duplicated
            columns by merging them onto the columns we keep.
    """
    names = defaultdict(list)
    prb = ProgressBar(len(df.columns))
    l = len(df.columns)

    for c in df.columns:
        hashable = tuple(df.loc[:,c])
        names[hashable].append(c)
        prb.increment()

    keep = [names[i][0] for i in names]
    new_df = df[keep]
    if keep_names:
        new_df.columns = [";".join(names[i]) for i in names]

    print("Reduced from",l,"to",len(new_df.columns),"columns")
    return new_df
