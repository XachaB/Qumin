# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Sacha Beniamine.

This module is used to get characters similar to other utf8 characters.
"""
from collections import defaultdict


def parse(filename):
    """Parse a file with confusable chars association, return a dict."""
    confusables = defaultdict(set)

    with open(filename, "r", encoding="utf8") as flow:
        for line in flow:
            if line[0] != "#":
                a, b = line.strip().split(";")
                if len(b) == 1:
                    confusables[a].add(b)
                if len(a) == 1:
                    confusables[b].add(a)
    return confusables
