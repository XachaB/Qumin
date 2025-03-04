# -*- coding: utf-8 -*-
# !/usr/bin/python3
import json
import logging
import os
import time
from pathlib import Path

import hydra
from frictionless import Package

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
        now (str)
        day (str)
        version (str) : svn/git version or '' if unknown
        prefix (str) : normalized prefix for the output files
        arguments (dict): all arguments passed to the python script
        output (dict) : all output files produced by the script.
        dataset (tuple): a (directory path, Package) tuple representing a dataset.
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
            data = cfg.data
            self.dataset = Package(data)

        # Additional CLI arguments
        self.arguments = dict(cfg)
        self.output = []

    def get_table_path(self, table_name):
        dataset = self.dataset
        basepath = Path(dataset.basepath or "./")
        return basepath / dataset.get_resource(table_name).path

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

    def register_file(self, suffix, properties=None, folder=None):
        """ Register a file to save. Returns a normalized name.

        Arguments:
            suffix (str): the suffix to append to the normalized prefix
            properties (dict): optional set of properties to keep along
            folder (str): name of a registered subdirectory where the file should be saved.

        Returns:
            (str): the full registered path"""

        # Always check if the folder still exists.
        if folder:
            filename = Path(self.prefix) / folder / suffix
            folder_mds = [entity for entity in self.output if entity.get('folder') == folder]
            if len(folder_mds) == 0:
                raise ValueError('This folder should first be registered')
            else:
                parent = folder_mds[0]['files']
        else:
            filename = Path(self.prefix) / suffix
            parent = self.output

        parent.append({'filename': str(filename),
                       'properties': properties})
        return str(filename)

    def register_folder(self, name, description=None):
        """ Register a folder of data. Returns a normalized name.

        Arguments:
            name (str): the name of the folder
            description (str): a description of the content of the folder

        Returns:
            (str): the full registered path"""

        # Always check if the folder still exists.
        foldername = Path(self.prefix) / name
        foldername.mkdir(parents=True, exist_ok=True)
        self.output.append({'folder': name,
                            'description': description,
                            'files': []})
        return str(foldername)


def get_version():
    """Return an ID for the current git or svn revision.

    If the directory isn't under git or svn, the function returns an empty str.

    Returns:
        (str): svn/git version or ''.
     """
    return __version__


def memory_check(df, factor, max_gb=2, force=False):
    """
    Checks memory usage for a dataframe and warn if it exceeds a certain limit.

    Arguments:
        df (`pandas.DataFrame`): dataframe to test
        factor (int): multiplication factor for the test.
        max_gb (float): Threshold for memory warning.
        force (bool): whether to allow overpassing the limit. Defaults to False.
    """
    mem = df.memory_usage(deep=True, index=True).sum()/(1024**3) * factor
    if mem > max_gb:
        if not force:
            raise Warning(f'The memory required might exceed {mem} GB of RAM. '
                          'If this is what you want, try again with force=true. '
                          'You could also sample some lexemes or select some cells.')
        else:
            log.warning(f'The required memory might exceed {mem} GB of RAM,'
                        'but you passed force=true.')
