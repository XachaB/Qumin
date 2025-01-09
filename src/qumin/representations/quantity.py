# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""author: Sacha Beniamine.

This module provides Quantity objects to represent quantifiers.
"""

import numpy as np


def quantity_largest(args):
    """Reduce on the "&" operator of quantities.


    Returns a quantity with the minimum left value
    and maximum right value.

    Example:
        >>> quantity_largest([Quantity(0,1),Quantity(1,1),Quantity(1,np.inf)])
        Quantity(0,inf)

    Argument:
        args: an iterable of quantities.
    """
    res = args[0]
    for i in range(1, len(args)):
        res &= args[i]
    return res


def quantity_sum(args):
    """Reduce on the "+" operator of quantities.


    Returns a quantity with the minimum left value and the sum of the right value.

    Example:
        >>> quantity_largest([Quantity(0,1),Quantity(1,1),Quantity(0,0)])
        Quantity(0,1)

    Argument:
        args: an iterable of quantities.
    """
    return sum(args, Quantity(0, 0))


class Quantity(object):
    r""" Represents a quantifier as an interval.

    This is a flyweight class and the presets are :

    ============ ======= ====== ============== ==========================
    description    mini   maxi  regex symbol    variable name
    ============ ======= ====== ============== ==========================
    Match one       1       1                    `quantity.one`
    Optional        0       1        ?           `quantity.optional`
    Some            1      inf       \+          `quantity.some`
    Any             0      inf       \*          `quantity.kleenestar`
    None            0       0
    ============ ======= ====== ============== ==========================

    """
    _pool = {}
    _named = {(0, 0): "<None>", (1, 1): "", (0, 1): "?", (1, np.inf): "+", (0, np.inf): "*"}

    def __new__(cls, mini, maxi):
        mini, maxi = cls._to_licit(mini, maxi)
        obj = cls._pool.get((mini, maxi), None)
        if not obj:
            obj = object.__new__(cls)
            cls._pool[(mini, maxi)] = obj
        return obj

    def __init__(self, mini, maxi):
        """
        Arguments:
            mini (int): the minimum number of elements matched.
            maxi (int): the maximum number of elements matched.
        """
        mini, maxi = Quantity._to_licit(mini, maxi)
        self.value = (mini, maxi)
        if self.value in Quantity._named:
            self._str = Quantity._named[self.value]
        else:
            self._str = "{{" + str(self.value[0]) + "-" + str(self.value[1]) + "}}"

    def __deepcopy__(self, memo):
        return self

    @classmethod
    def _to_licit(cls, mini, maxi):
        mini = 0 if mini < 0 else mini
        maxi = np.inf if maxi > 1 else maxi
        return mini, maxi

    def __str__(self):
        return self._str

    def __repr__(self):
        return "Quantity({},{})".format(*self.value)

    def __lt__(self, other):
        return self != other and self.value[0] >= other.value[0] and self.value[1] <= other.value[1]

    def __and__(self, other):
        return Quantity(min(self.value[0], other.value[0]), max(self.value[1], other.value[1]))

    def __add__(self, other):
        return Quantity(min(self.value[0], other.value[0]), self.value[1] + other.value[1])


one = Quantity(1, 1)
optional = Quantity(0, 1)
some = Quantity(1, np.inf)
kleenestar = Quantity(0, np.inf)
