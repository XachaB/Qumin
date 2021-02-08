# -*- coding: utf-8 -*-
# !/usr/bin/python3
from collections import defaultdict
import subprocess
from tqdm import tqdm
import logging
log = logging.getLogger(__name__)

def snif_separator(filename):
    with open(filename, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if "\t" in first_line:
            return "\t"
        elif "," in first_line:
            return ","
        raise ValueError("File {} should be comma or tab separated".format(filename))

def get_repository_version():
    """Return an ID for the current git or svn revision.

    If the directory isn't under git or svn, the function returns an empty str.

     Returns:
        (str): svn/git version or ''.
     """
    try:
        kwargs = {"universal_newlines": True}
        version = subprocess.check_output(["git", "describe", "--tags"], **kwargs)
        return version.strip("\n ")
    except:
        return ''


def merge_duplicate_columns(df, sep=";", keep_names=True):
    """Merge duplicate columns and return new DataFrame.

    Arguments:
        df (:class:`pandas:pandas.DataFrame`): A dataframe
        sep (str): separator to use when joining columns names.
        keep_names (bool): Whether to keep the names of the original duplicated
            columns by merging them onto the columns we keep.
    """
    names = defaultdict(list)
    l = len(df.columns)

    for c in tqdm(df.columns):
        hashable = tuple(df.loc[:, c])
        names[hashable].append(c)

    keep = [names[i][0] for i in names]
    new_df = df[keep]
    if keep_names:
        new_df.columns = [sep.join(names[i]) for i in names]

    log.info("Reduced from %s to %s columns", l, len(new_df.columns))
    return new_df
