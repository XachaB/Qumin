# -*- coding: utf-8 -*-
# !/usr/bin/python3
import logging
import datetime
from omegaconf import OmegaConf
from pathlib import Path

import hydra
from frictionless import Package, Resource

from .. import __version__

log = logging.getLogger()


class Metadata():
    """Metadata manager for Qumin scripts. Wrapper around the Frictionless Package class.
    Basic usage :

        1. Register Metadata manager;
        2. Get an absolute path to the metadata folder;
        3. Write to that path;
        4. After writing a file, register it and set metadata (description, custom dict);
        3. Export the JSON descriptor.

    Examples:
        .. code-block:: python

            md = Metadata(args, __file__)
            name = 'path/myfile.txt'
            filename = md.get_path(name)
            # Open an IO stream and write to ``filename``.
            md.register_file(name, description="My nice file", custom={"property": "value"})
            md.save_metadata(path)

    Arguments:
        cfg (:class:`pandas:pandas.DataFrame`):
            arguments passed to the script
        path (str): name of the main script, passing __file__ is fine.

    Attributes:
        start (datetime) : timestamp at the beginning of the run.
        prefix (Path) : normalized prefix for the output files
        cfg (OmegaConf): all arguments passed to the python script
        paralex (frictionless.Package): a frictionless Package representing a dataset.
    """

    def __init__(self, path=None, cfg=None, **kwargs):
        self.package = Package(path) if path else Package()
        self.cfg = cfg
        self.prefix = Path(self.package.basepath if path else
                           hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

        if path is None:
            self.start = datetime.datetime.now()
            self.paralex = Package(cfg.data)
            self.package.name = self.start.strftime("qumin_results_%Hh%M_%Y%m%d")
            self.package.title = "Qumin Computation Results"
            self.package.homepage = "https://qumin.readthedocs.io/"
            self.package.description = "This package contains the output of a Qumin run. " \
                                       "It can be imported by other Qumin scripts."
            self.package.created = datetime.datetime.now().isoformat()
            self.package.custom['qumin_version'] = __version__
            if cfg:
                self.package.custom['omega_conf'] = OmegaConf.to_container(cfg)
            self.package.custom['paralex_dataset'] = self.paralex.to_dict()

    def get_table_path(self, table_name):
        """ Return the path to a dataset table """
        dataset = self.paralex
        basepath = Path(dataset.basepath or "./")
        return basepath / dataset.get_resource(table_name).path

    def get_resource_path(self, resource):
        """ Return the full path to a resource """
        return self.prefix / self.package.get_resource(resource).path

    def save_metadata(self):
        """ Save the metadata as a JSON file."""
        end = datetime.datetime.now()
        self.package.custom['duration'] = {"start": str(self.start),
                                           "end": str(end),
                                           "delta": str(end - self.start)
                                           }
        self.package._basepath = str(self.prefix)
        self.package.infer()
        self.package.to_json(self.prefix / 'metadata.json')

    def get_path(self, rel_path):
        """ Return an absolute path to a file and create parent directories. """
        path = Path(self.prefix) / rel_path
        if rel_path[-1] != "/":
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def register_file(self, rel_path, custom=None, **kwargs):
        """ Add a file as a frictionless resource.

        Arguments:
            rel_path (str): the relative path to the file.
            custom (dict): Custom properties to save.
            **kwargs (dict): Optional keyword arguments passed to Resource,
                e.g. `description`.
        """
        res = Resource(path=rel_path, **kwargs)
        if custom is not None:
            res.custom = custom
        self.package.add_resource(res)


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
