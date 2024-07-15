# -*- coding: utf-8 -*-
# !/usr/bin/python3
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import hydra
from frictionless import Package
from tqdm import tqdm

from .. import __version__

log = logging.getLogger()


class Metadata():
    """Metadata manager for Qumin scripts. Basic usage :

        1. Register Metadata manager;
        2. Before writing any important file, register it with a short name;
        3. Save all metadata in a secure place

    Examples:
        .. code-block:: python

            md = Metadata(args, __file__)
            filename = md.register_file(name, suffix)
            # Now, you can open an IO stream and write to ``filename``.
            md.save_metadata(path)

    Arguments:
        cfg (:class:`pandas:pandas.DataFrame`):
            arguments passed to the script
        filename (str): name of the main script, passing __file__ is fine.

    Attributes:
        now (:class:`time.strftime`)
        day (:class:`time.strftime`)
        version (str) : svn/git version or '' if unknown
        prefix (str) : normalized prefix for the output files
        arguments (dict): all arguments passed to the python script
        output (dict) : all output files produced by the script.
        datasets (list): a list of (directory path, Package) tuples; each representing a dataset.
    """

    def __init__(self, cfg, filename):
        # Basic information
        self.now = time.strftime("%Hh%M")
        self.day = time.strftime("%Y%m%d")
        self.version = get_version()
        self.script = filename
        self.working_dir = os.getcwd()

        # Check directory
        self.prefix = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/"

        # Make it robust to multiple
        if "data" in cfg:
            data = [cfg.data] if type(cfg.data) is str else cfg.data
            self.datasets = [Package(path) for path in data]

        # Additional CLI arguments
        self.arguments = dict(cfg)
        self.output = []

    def get_table_path(self, table_name, num=0):
        dataset = self.datasets[num]
        return Path(dataset.basepath) / dataset.get_resource(table_name).path

    def save_metadata(self):
        """ Save the metadata as a JSON file."""

        def default_serializer(x):
            if type(x) is Package:
                return x.to_dict()
            return str(x)

        path = self.prefix + "metadata.json"
        log.info("Writing metadata to %s", path)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=4, default=default_serializer)

    def register_file(self, suffix, properties=None):
        """ Register a file to save. Returns a normalized name.

        Arguments:
            suffix (str): the suffix to append to the normalized prefix
            properties (dict): optional set of properties to keep along

        Returns:
            (str): the full registered path"""

        # Always check if the folder still exists.
        filename = self.prefix + suffix
        self.output.append({'filename': filename,
                            'properties': properties})
        return filename


def get_version():
    """Return an ID for the current git or svn revision.

    If the directory isn't under git or svn, the function returns an empty str.

    Returns:
        (str): svn/git version or ''.
     """
    return __version__


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
