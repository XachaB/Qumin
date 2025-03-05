#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from pathlib import Path

from qumin.representations import patterns, segments


class PatternsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        segs_file = Path(__file__).parent / "data" / "portuguese_phonemes.csv"
        segments.Inventory.initialize(segs_file)

    def test_identity(self):
        c = ("a", "b")
        p = patterns.Pattern.new_identity(c)
        f = segments.Form.from_segmented_str("a b a b ")
        self.assertTrue(p.applicable(f, "a"))
        self.assertTrue(p.applicable(f, "b"))
        self.assertEqual(p.apply(f, c), f)
        self.assertEqual(p.apply(f, c[::-1]), f)
        self.assertTrue(p.is_identity())

    def test_find_generalized_alt(self):
        forms = (segments.Form.from_segmented_str("b a "),
                 segments.Form.from_segmented_str("b ˈa "))
        aligned = zip(forms[0].tokens, forms[1].tokens)
        c = ("a", "b")
        p = patterns.Pattern(c, aligned, aligned=True)
        expected = {'a': ((frozenset(
            {'ẽ', 'ɔ', 'ɛ', 'aw', 'uj', 'ɐ̃', 'oj', 'iw', 'a', 'i', 'ɐ̃j', 'ɐ̃w', 'ĩ',
             'ũ', 'o', 'õ', 'u', 'ɐj', 'e', 'ɐ', 'aj'}),),),
            'b': ((frozenset(
                {'ˈaw', 'ˈa', 'ˈɐ̃', 'ˈi', 'ˈɐ', 'ˈĩ', 'ˈɔ', 'ˈe', 'ˈɛ', 'ˈuj',
                 'ˈoj', 'ˈiw', 'ˈũ', 'ˈõ', 'ˈo', 'ˈu', 'ˈɐ̃w', 'ˈɐj', 'ˈɐ̃j',
                 'ˈẽ', 'ˈaj'}),),)}
        self.assertDictEqual(p._gen_alt, expected)

    def test_applicable_optional_end(self):
        c = ("A", "B")
        fa = segments.Form.from_segmented_str("s a t")
        p = patterns.Pattern._from_str(c, "t ⇌  / sa_i?<2.0>")
        self.assertTrue(p.applicable(fa, "A"), f"{p} should be applicable to {fa}")

    def test_applicable_optional_end_multiple(self):
        c = ("A", "B")
        fa = segments.Form.from_segmented_str("s a t e")
        p = patterns.Pattern._from_str(c, "te ⇌  / sa_<2.0>")
        p_reverse = patterns.Pattern._from_str(c, " ⇌ te / sa_<2.0>")

        self.assertTrue(p.applicable(fa, "A"), f"{p} should be applicable to {fa}")
        self.assertTrue(p_reverse.applicable(fa, "B"), f"{p_reverse} should be applicable to {fa}")

    def test_applicable(self):
        c = ("a", "b")
        p = patterns.Pattern._from_str(c,
                                       "{a,aj,aw,e,i,iw,o,oj,u,uj,õ,ĩ,ũ,ɐ,ɐj,ɐ̃,ɐ̃j,ɐ̃w"
                                       ",ɔ,ɛ,ẽ}_ˈɐm_ʃ ⇌ {ˈa,ˈaj,ˈaw,ˈe,ˈi,ˈiw,ˈo,ˈoj,"
                                       "ˈu,ˈuj,ˈõ,ˈĩ,ˈũ,ˈɐ,ˈɐj,ˈɐ̃,ˈɐ̃j,ˈɐ̃w,ˈɔ,ˈɛ,ˈẽ}__"
                                       " / {a,aj,aw,b,d,e,f,i,iw,j,k,l,m,n,o,oj,p,s,t"
                                       ",u,uj,v,w,z,õ,ĩ,ũ,ɐ,ɐj,ɐ̃,ɐ̃j,ɐ̃w,ɔ,ə,ɛ,ɡ,ɲ,ɾ,ʀ,ʃ"
                                       ",ʎ,ʒ,ˈa,ˈaj,ˈaw,ˈe,ˈew,ˈi,ˈiw,ˈo,ˈoj,ˈu,ˈuj,ˈõ"
                                       ",ˈõj,ˈĩ,ˈũ,ˈɐ,ˈɐj,ˈɐ̃,ˈɐ̃j,ˈɐ̃w,ˈɔ,ˈɔj,ˈɛ,ˈẽ,ẽ}*_"
                                       "{a,aj,aw,b,d,e,f,i,iw,j,k,l,m,n,o,oj,p,s,t,u,u"
                                       "j,v,w,z,õ,ĩ,ũ,ɐ,ɐj,ɐ̃,ɐ̃j,ɐ̃w,ɔ,ə,ɛ,ɡ,ɲ,ɾ,ʀ,ʃ,ʎ,ʒ,"
                                       "ˈa,ˈaj,ˈaw,ˈe,ˈew,ˈi,ˈiw,ˈo,ˈoj,ˈu,ˈuj,ˈõ,ˈõj,ˈ"
                                       "ĩ,ˈũ,ˈɐ,ˈɐj,ˈɐ̃,ˈɐ̃j,ˈɐ̃w,ˈɔ,ˈɔj,ˈɛ,ˈẽ,ẽ}+_u_ "
                                       "<2154.0>")
        self.assertTrue(p.applicable("ɐ k ˈɐ m u ", c[1]))

    def test_apply(self):
        c = ("a", "b")
        p = patterns.Pattern._from_str(c,
                                       "{a,aj,aw,e,i,iw,o,oj,u,uj,õ,ĩ,ũ,ɐ,ɐj,ɐ̃,ɐ̃j,ɐ̃w"
                                       ",ɔ,ɛ,ẽ}_ˈɐm_ʃ ⇌ {ˈa,ˈaj,ˈaw,ˈe,ˈi,ˈiw,ˈo,ˈoj,"
                                       "ˈu,ˈuj,ˈõ,ˈĩ,ˈũ,ˈɐ,ˈɐj,ˈɐ̃,ˈɐ̃j,ˈɐ̃w,ˈɔ,ˈɛ,ˈẽ}__"
                                       " / {a,aj,aw,b,d,e,f,i,iw,j,k,l,m,n,o,oj,p,s,t"
                                       ",u,uj,v,w,z,õ,ĩ,ũ,ɐ,ɐj,ɐ̃,ɐ̃j,ɐ̃w,ɔ,ə,ɛ,ɡ,ɲ,ɾ,ʀ,ʃ"
                                       ",ʎ,ʒ,ˈa,ˈaj,ˈaw,ˈe,ˈew,ˈi,ˈiw,ˈo,ˈoj,ˈu,ˈuj,ˈõ"
                                       ",ˈõj,ˈĩ,ˈũ,ˈɐ,ˈɐj,ˈɐ̃,ˈɐ̃j,ˈɐ̃w,ˈɔ,ˈɔj,ˈɛ,ˈẽ,ẽ}*_"
                                       "{a,aj,aw,b,d,e,f,i,iw,j,k,l,m,n,o,oj,p,s,t,u,u"
                                       "j,v,w,z,õ,ĩ,ũ,ɐ,ɐj,ɐ̃,ɐ̃j,ɐ̃w,ɔ,ə,ɛ,ɡ,ɲ,ɾ,ʀ,ʃ,ʎ,ʒ,"
                                       "ˈa,ˈaj,ˈaw,ˈe,ˈew,ˈi,ˈiw,ˈo,ˈoj,ˈu,ˈuj,ˈõ,ˈõj,ˈ"
                                       "ĩ,ˈũ,ˈɐ,ˈɐj,ˈɐ̃,ˈɐ̃j,ˈɐ̃w,ˈɔ,ˈɔj,ˈɛ,ˈẽ,ẽ}+_u_ "
                                       "<2154.0>")
        print(p._regex)
        self.assertTrue(p.apply("ɐ k ˈɐ m u ", ("b", "a"),
                                "ɐ k ɐ m ˈɐ m u ʃ "))

    def test_from_str(self):
        self.maxdiff = 2000
        strs = ["sˈomuʃ ⇌ ˈɛ / _ <1.0>",
                "u{ĩ,ˈĩ,ˈẽ,ẽ}d ⇌ {o,u,ˈo,ˈu} / "
                "{a,aj,aw,b,d,e,f,i,iw,j,k,l,m,n,o,oj,p,s,t,u,uj,v,w,z,õ,ĩ,ũ,ɐ,ɐj,ɐ̃,ɐ̃j,ɐ̃w,"
                "ɔ,ə,ɛ,ɡ,ɲ,ɾ,ʀ,ʃ,ʎ,ʒ,ˈa,ˈaj,ˈaw,ˈe,ˈew,ˈi,ˈiw,ˈo,ˈoj,ˈu,ˈuj,ˈõ,ˈõj,ˈĩ,ˈũ,"
                "ˈɐ,ˈɐj,ˈɐ̃,ˈɐ̃j,ˈɐ̃w,ˈɔ,ˈɔj,ˈɛ,ˈẽ,ẽ}*{b,d,k,l,m,n,p,s,t,z,ɡ,ɲ,ɾ,ʀ,ʃ,ʎ,ʒ}_u <55.0>",
                ]

        pretty = ["sˈomuʃ ⇌ ˈɛ / _",
                  "u{ĩ,ˈĩ,ˈẽ,ẽ}d ⇌ {o,u,ˈo,ˈu} / X*[+C -labdent]_u"]
        for pat_str, pretty_result in zip(strs, pretty):
            new = patterns.Pattern._from_str(("a", "b"), pat_str)
            self.assertEqual(repr(new), pat_str)
            self.assertEqual(str(new), pretty_result)

    def test_from_alignment(self):
        c = ("a", "b")
        alignment = [('ɐ', 'ɐ'), ('f', 'f'), ('ə', ''), ('ɾ', ''), ('ˈi', 'ˈi'), ('m', ''), ('', 'ɾ'), ('u', 'u'),
                     ('ʃ', '')]
        p = patterns.Pattern(c, alignment, aligned=True)
        expected = {'a': [('ə', 'ɾ'), ('m',), ('ʃ',)], 'b': [('',), ('ɾ',), ('',)]}
        self.assertDictEqual(p.alternation, expected)

        alignment = [('ɐ', 'ɐ'), ('k', 'k'), ('u', ''), ('', 'ˈu'), ('d', 'd'), ('ˈi', ''), ('m', ''),
                     ('u', 'u'), ('ʃ', '')]
        p = patterns.Pattern(c, alignment, aligned=True)
        expected = {'a': [('u',), ('ˈi', 'm'), ('ʃ',)], 'b': [('ˈu',), ('',), ('',)]}

        self.assertDictEqual(p.alternation, expected)
