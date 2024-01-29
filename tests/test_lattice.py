#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

import pandas as pd
import numpy as np
import random
from qumin.lattice import ICLattice, to_dummies_overabundant
from qumin.clustering import Node
from . import TestCaseWithPandas
from itertools import combinations


def parse_lattice(nodes):
    for node, children in nodes:
        if children:
            node.children = [nodes[i][0] for i in children]
    return nodes[-1][0]


class AOCTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.idx = list("123456789")
        cls.cols = list("ABCDEFGHIJKL")

    def conformity_check(self, l):
        # Check that all aoc concepts are in the extent
        aoc_extents = {tuple(sorted(c.extent)) for c in l.lattice if
                       (c == l.lattice.supremum)
                       or c.objects or c.properties}
        kept_extents = {tuple(sorted(n.labels)) for n in l.nodes}
        self.assertEqual(kept_extents, aoc_extents)

        # Check that the range of each node is its extent
        for node in l.nodes:
            extent = set(node.labels)
            descendants = set().union(*[x.labels for x in node])
            self.assertEqual(extent, descendants)

        # Check that node children are not grand-children
        for node in l.nodes:
            children_extents = [set(x.labels) for x in node.children]
            for c1, c2 in combinations(children_extents, 2):
                try:
                    self.assertFalse(c1 < c2)
                    self.assertFalse(c2 < c1)
                except:
                    print("C1, C2", c1, c2)
                    raise
    def test_AOC_simplecase(self):
        """This test checks that non aoc nodes are removed in a minimal case.

        Input: ::

            supremum -> B1 A1
            B1 -> C1 del_a
            A1 -> del_a C2
            C1 -> 3 1
            del -> 1 2
            C2 -> 2 4

        Expected output: ::

            supremum -> B1 A1
            B1 -> C1 2
            A1 -> 1 C2
            C1 -> 3 1
            C2 -> 2 4

        """
        data = pd.DataFrame([[1, 1, 1],
                             [1, 1, 2],
                             [2, 1, 1],
                             [1, 2, 2],
                             ], index=self.idx[:4], columns=self.cols[:3]).map(str)
        l = ICLattice(data, {f: (f,) for f in data.index}, aoc=True)

        self.conformity_check(l)
        expected = parse_lattice([(Node(('1',)), []),  # 0
                                  (Node(('2',)), []),  # 1
                                  (Node(('3',)), []),  # 2
                                  (Node(('4',)), []),  # 3
                                  (Node(('1', '3')), [0, 2]),  # 4
                                  (Node(('2', '4')), [1, 3]),  # 5
                                  (Node(('1', '2', '4')), [5, 0]),  # 6
                                  (Node(('1', '2', '3')), [4, 1]),  # 7
                                  (Node(('1', '2', '3', '4')), [7, 6]),  # 8
                                  ])

        self.assertEqual(l.nodes, expected)

    def test_AOC_complex1(self):
        """This example is interesting because on  the first level,
        three nodes (A1, B1, C2) have overlapping spans, and all
        of their direct children must be deleted:

        Input: ::

            supremum -> A1 B1 C2
            A1 -> del_a del_b
            B1 -> del_b del_c
            C2 -> del_a del_c
            del_a -> 1 3
            del_b -> 3 2
            del_c -> 3 4

        Expected output: ::

            supremum -> A1 B1 C2
            A1 -> 1 3 2
            B1 -> 3 2 4
            C2 -> 1 3 4

        """
        data = pd.DataFrame([[1, 3, 2],
                             [1, 1, 1],
                             [1, 1, 2],
                             [2, 1, 2]
                             ], index=self.idx[:4], columns=self.cols[:3]).map(str)
        l = ICLattice(data, {f: (f,) for f in data.index}, aoc=True)

        self.conformity_check(l)

        expected = parse_lattice([(Node(('1',)), []),
                                  (Node(('2',)), []),
                                  (Node(('3',)), []),
                                  (Node(('4',)), []),
                                  (Node(('1', '2', '3')), [0, 1, 2]),
                                  (Node(('1', '3', '4')), [0, 2, 3]),
                                  (Node(('2', '3', '4')), [1, 2, 3]),
                                  (Node(('1', '2', '3', '4')), [4, 5, 6]),
                                  ])

        self.assertEqual(l.nodes, expected)

    def test_AOC_complex2(self):
        """This example is similar to the preceding one,
        but the structure is deeper, resulting in more edge cases."""
        data = pd.DataFrame([[1, 1, 1, 4, 2, 1],
                             [1, 2, 1, 4, 3, 1],
                             [1, 1, 1, 4, 1, 1],
                             [2, 1, 1, 1, 1, 1],
                             [2, 3, 2, 1, 4, 1],
                             [3, 3, 1, 4, 5, 2]
                             ], index=self.idx[:6], columns=self.cols[:6]).map(str)
        l = ICLattice(data, {f: (f,) for f in data.index}, aoc=True)

        self.conformity_check(l)

        expected = parse_lattice([(Node(('4',)), []),
                                  (Node(('5',)), []),
                                  (Node(('2',)), []),
                                  (Node(('1',)), []),
                                  (Node(('3',)), []),
                                  (Node(('6',)), []),
                                  (Node(('4', '5')), [0, 1]),
                                  (Node(('3', '4')), [4, 0]),
                                  (Node(('5', '6')), [1, 5]),
                                  (Node(('1', '2', '3')), [2, 3, 4]),
                                  (Node(('1', '3', '4')), [7, 3]),
                                  (Node(('1', '2', '3', '6')), [9, 5]),
                                  (Node(('1', '2', '3', '4', '5')), [6, 9, 10]),
                                  (Node(('1', '2', '3', '4', '6')), [11, 10]),
                                  (Node(('1', '2', '3', '4', '5', '6')), [12, 13, 8]),
                                  ])

        self.assertEqual(l.nodes, expected)

    def test_double_or(self):
        """This test is interesting because l3 inherits from two generations of 'or' nodes,
        that is, both its parents and grandparents will be deleted. Errors in the AOC
        algorithm tend to generate results where l3 is missing entirely.
        """
        data = pd.DataFrame([['2', '1', '4', '9', '4', '9'],
                             ['8', '6', '4', '9', '4', '4'],
                             ['8', '3', '1', '9', '4', '8'],
                             ['8', '4', '7', '9', '0', '4'],
                             ['8', '4', '8', '6', '0', '7'],
                             ['6', '9', '3', '7', '4', '6']],
                            index=['l1', 'l2', 'l3', 'l4', 'l5', 'l6'],
                            columns=['A', 'B', 'C', 'D', 'E', 'F'])

        l = ICLattice(data, {f: (f,) for f in data.index}, aoc=True)
        self.conformity_check(l)

    def test_hierarchical_constraint(self):
        data = pd.DataFrame([['1', '2', '2', '1', '1'],
                             ['2', '1', '1', '2', '2'],
                             ['3', '1', '1', '1', '3'],
                             ['1', '1', '1', '1', '4'],
                             ['1', '3', '1', '3', '5']],
                            index=['l1', 'l2', 'l3', 'l4', 'l5'],
                            columns=['A', 'B', 'C', 'D', 'E', ])

        l = ICLattice(data, {f: (f,) for f in data.index}, aoc=True)
        self.conformity_check(l)

    def test_on_random_lattices(self):
        """Generate larger random datasets, and make principled checks."""
        alpha = [chr(i) for i in range(97, 122)]

        def id_str(i, length, prefix):
            id = [prefix, str(i), "-"] + [random.choice(alpha) for _ in range(length - 1)]
            return "".join(id)

        for i in range(10):
            shape = np.random.randint(10, high=100, size=(2,))
            values = np.random.randint(0, high=10, size=shape)
            indexes = [id_str(i, 4, "l") for i in range(shape[0])]
            columns = [id_str(i, 4, "f").upper() for i in range(shape[1])]
            data = pd.DataFrame(values, index=indexes, columns=columns).map(str)
            data = data.drop_duplicates()
            l = ICLattice(data, {f: (f,) for f in indexes}, aoc=True)
            self.conformity_check(l)


class TestFuncs(TestCaseWithPandas):
    def test_to_dummies_overabundant(self):
        df = pd.DataFrame([['e', 'f', 'f', 'd', 'c'],
                           ['d;e', 'd', 'a', 'a', 'a'],
                           ['e', 'e', 'c', 'e;a', 'e'],
                           ['a', 'd', 'f', 'f', 'd;c'],
                           ['e', 'a', 'a', 'b', 'b']],
                          columns=['TB', 'LX', 'AW', 'UA', 'VN'],
                          index=['jgl', 'woq', 'rfo', 'tfp', 'box'])
        dummies = to_dummies_overabundant(df)
        expected = pd.DataFrame([['X', 'X', 'X', 'X', 'X', '', '', '', '',
                                  '', '', '', '', '', '', '', '', '', '', ''],
                                 ['X', '', '', '', '', 'X', 'X', 'X', 'X',
                                  'X', '', '', '', '', '', '', '', '', '', ''],
                                 ['X', '', '', '', '', '', '', '', 'X', '',
                                  'X', 'X', 'X', 'X', '', '', '', '', '', ''],
                                 ['', '', 'X', '', 'X', '', 'X', '', '', '',
                                  '', '', '', '', 'X', 'X', 'X', '', '', ''],
                                 ['X', '', '', '', '', '', '', 'X', '', '',
                                  '', '', '', '', '', '', '', 'X', 'X', 'X']],
                                columns=['TB=e', 'LX=f', 'AW=f', 'UA=d', 'VN=c', 'TB=d',
                                         'LX=d', 'AW=a', 'UA=a', 'VN=a', 'LX=e', 'AW=c',
                                         'UA=e', 'VN=e', 'TB=a', 'UA=f', 'VN=d', 'LX=a',
                                         'UA=b', 'VN=b'],
                                index=['jgl', 'woq', 'rfo', 'tfp', 'box'],
                                )
        self.assertEqual(dummies, expected)
