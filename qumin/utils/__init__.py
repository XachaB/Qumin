# -*- coding: utf-8 -*-
# !/usr/bin/python3
import argparse
from collections import defaultdict
from tqdm import tqdm
import logging
import time
import json
import os
from pathlib import Path
from .. import __version__
log = logging.getLogger()


class Metadata():
    """Metadata manager for Qumin scripts. Basic usage :
    1) Register Metadata manager;
    2) Before saving a file, register it with a short name;
    3) Save all metadata in a secure place

        md = Metadata(args, __file__)
        filename = md.register_file(name, suffix)
        # Now, you can open an IO stream and write to ``filename``.
        md.save_metadata(path)

    Arguments:
        args (:class:`pandas:pandas.DataFrame`):
            arguments passed to the script
        file (str): name of the main script
        relative (bool) : whether to use absolute or relative paths. Relative refers to working_dir

    Attributes:
        nom (:class:`time.strftime`)
        day (:class:`time.strftime`)
        version (str) : svn/git version or '' if unknown
        result_dir (str) : output directory
        prefix (str) : normalized prefix for the output files
        arguments (dict): all arguments passed to the python script
        output (dict) : all output files produced by the script.

    """

    def __init__(self, args, filename, relative=True):
        # Basic information
        self.now = time.strftime("%Hh%M")
        self.day = time.strftime("%Y%m%d")
        self.version = get_version()
        self.script = filename
        self.working_dir = os.getcwd()

        # Check directory
        self.result_dir_relative = str(Path(args.folder) / self.day)
        self.result_dir_absolute = Path(args.folder).absolute() / self.day
        self.result_dir_absolute.mkdir(exist_ok=True, parents=True)
        self.result_dir_absolute = str(self.result_dir_absolute)

        if relative:
            self.prefix = "{}/{}_{}".format(self.result_dir_relative,
                                            self.day, self.now)
        else:
            self.prefix = "{}/{}_{}".format(self.result_dir_absolute,
                                            self.day, self.now)

        # Additional CLI arguments
        self.arguments = args.__dict__
        self.output = []

    def save_metadata(self, path=None):
        """ Save the metadata as a JSON file.

        Arguments:
            path (str) : path to the metadata file.
            Defaults to prefx_metadata.json"""

        if path is None:
            path = self.prefix + "_metadata.json"
            log.info('No metadata path provided. Writing to %s', path)
        else:
            log.info("Writing metadata to %s", path)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=4)

    def register_file(self, suffix, properties=None):
        """ Register a file to save. Returns a normalized name.

        Arguments :
            suffix (str): the suffix to append to the normalized prefix
            properties (dict): optional set of properties to keep along

        Returns:
            (str) : the full registered path"""

        filename = self.prefix+"_"+suffix
        self.output.append({'filename': filename,
                            'properties': properties})
        return filename


def snif_separator(filename):
    with open(filename, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if "\t" in first_line:
            return "\t"
        elif "," in first_line:
            return ","
        raise ValueError("File {} should be comma or tab separated".format(filename))


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


class ArgumentDefaultsRawTextHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Combines RawTextHelpFormatter & class ArgumentDefaultsHelpFormatter
    """

    def _split_lines(self, text, width):
        return text.splitlines()

    def _get_help_string(self, action):
        help = action.help
        if '%(default)' not in action.help:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    help += ' (default: %(default)s)'
        return help

def get_default_parser(usage, patterns=False, paradigms=True, ):

    parser = argparse.ArgumentParser(description=usage,
                                     formatter_class=ArgumentDefaultsRawTextHelpFormatter)

    if patterns:
        parser.add_argument("patterns",
                            help="patterns file, full path"
                                 " (csv separated by ‘, ’)",
                            type=str)

    if paradigms:
        parser.add_argument("paradigms",
                            help="paradigms file, full path"
                                 " (csv separated by ‘, ’)",
                            type=str)



    parser.add_argument("segments",
                        help="segments file, full path (csv or tsv)",
                        type=str)

    if paradigms:
        parser.add_argument("-c", "--cols_names",
                            help="In long form, specify the name of respectively the lexeme, cell and form columns.",
                            nargs=3, type=str, default=["lexeme", "cell", "phon_form"])

    options = parser.add_argument_group('Options')

    options.add_argument("-v", "--verbose",
                         help="Activate debug logs.",
                         action="store_true", default=False)

    options.add_argument("-f", "--folder",
                        help="Output folder name",
                        type=str, default=os.getcwd())

    return parser
