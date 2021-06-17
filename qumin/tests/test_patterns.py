#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import pandas as pd
from ..representations import patterns, segments
from pathlib import Path
class PatternsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        segs_file = Path(__file__).parent / "data" / "portuguese_phonemes.csv"
        segments.Inventory.initialize(segs_file, sep=",")

    def test_from_str(self):
        strs = ["sˈomuʃ ⇌ ˈɛ / _ <1.0>",
            "u{ĩ,ˈĩ,ˈẽ,ẽ}d ⇌ {o,u,ˈo,ˈu} / " \
             "{a,aj,aw,b,d,e,f,i,iw,j,k,l,m,n,o,oj,p,s,t,u,uj,v,w,z,õ,ĩ,ũ,ɐ,ɐj,ɐ̃,ɐ̃j,ɐ̃w," \
             "ɔ,ə,ɛ,ɡ,ɲ,ɾ,ʀ,ʃ,ʎ,ʒ,ˈa,ˈaj,ˈaw,ˈe,ˈew,ˈi,ˈiw,ˈo,ˈoj,ˈu,ˈuj,ˈõ,ˈõj,ˈĩ,ˈũ," \
             "ˈɐ,ˈɐj,ˈɐ̃,ˈɐ̃j,ˈɐ̃w,ˈɔ,ˈɔj,ˈɛ,ˈẽ,ẽ}*{b,d,k,l,m,n,p,s,t,z,ɡ,ɲ,ɾ,ʀ,ʃ,ʎ,ʒ}_u <55.0>",
                ]

        pretty = ["sˈomuʃ ⇌ ˈɛ / _", "u{ĩ,ˈĩ,ˈẽ,ẽ}d ⇌ {o,u,ˈo,ˈu} / X*[+cons -lab-dent]_u"]
        for pat_str, pretty_result in zip(strs, pretty):
            new = patterns.Pattern._from_str(("a", "b"), pat_str)
            self.assertEqual(repr(new), pat_str)
            self.assertEqual(str(new), pretty_result)


        strs_old_format = [
            "u[ĩ-ˈĩ-ˈẽ-ẽ]d ⇌ [o-u-ˈo-ˈu] / " \
            "[a-aj-aw-b-d-e-f-i-iw-j-k-l-m-n-o-oj-p-s-t-u-uj-v-w-z-õ-ĩ-ũ-ɐ-ɐj-ɐ̃-ɐ̃j-ɐ̃w-" \
            "ɔ-ə-ɛ-ɡ-ɲ-ɾ-ʀ-ʃ-ʎ-ʒ-ˈa-ˈaj-ˈaw-ˈe-ˈew-ˈi-ˈiw-ˈo-ˈoj-ˈu-ˈuj-ˈõ-ˈõj-ˈĩ-ˈũ-" \
            "ˈɐ-ˈɐj-ˈɐ̃-ˈɐ̃j-ˈɐ̃w-ˈɔ-ˈɔj-ˈɛ-ˈẽ-ẽ]*[b-d-k-l-m-n-p-s-t-z-ɡ-ɲ-ɾ-ʀ-ʃ-ʎ-ʒ]_u <55.0>"]
        old_as_str = [strs[1], ]
        for old_str, new_str in zip(strs_old_format, old_as_str):
            new = patterns.Pattern._from_str(("a", "b"), old_str)
            self.assertEqual(repr(new), new_str)
