Usages
======

Usage of `bin/find_patterns.py`
-----------------------------------------

Find pairwise alternation patterns from paradigms.
This is a preliminary step necessary to obtain patterns used as input in the three scripts below.

**Computing automatically aligned patterns** for paradigm entropy or macroclass::

    bin/$ python3 find_patterns.py <paradigm.csv> <segments.csv>

**Computing automatically aligned patterns** for lattices::

    bin/$ python3 find_patterns.py -d -o -c <paradigm.csv> <segments.csv>

The option -k allows one to choose the algorithm for inferring alternation patterns.

====================== ====================== ==================================================================================
 Option                 Description            Strategy
====================== ====================== ==================================================================================
`endings`              Affixes                 Removes the longest common initial string for each row.
`endingsPairs`         Pairs of affixes        Endings, tabulated as pairs for all combinations of columns.
`endingsDisc`          Discontinuous endings   Removes the longest common substring, left aligned

**....Alt**            **Alternations**       **Alternations have no contextes. These were used for comparing macroclass**
                                              **strategies on French and European Portuguese.**

`globalAlt`            Alternations            As `EndingsDisc`, tabulated as pairs for all combinations of columns.
`localAlt`             Alternations            Inferred from local pairs of cells, left aligned.

**patterns...**        **Binary Patterns**     **All patterns have alternations and generalized contexts. Various alignment**
                                               **strategies are offered for comparison. Arbitrary number of changes supported.**

`patternsLevenshtein`  Patterns                Aligned with simple edit distance.
`patternsPhonsim`      Patterns                Aligned with edit distances based on phonological similarity.
`patternsSuffix`       Patterns                Fixed left alignment, only interesting for suffixal languages.
`patternsPrefix`       Patterns                Fixed right alignment, only interesting for prefixal languages.
`patternsBaseline`     Patterns                Baseline alignment, follows Albright & Hayes 2002.
                                               A single change, with a priority order:
                                               Suffixation > Prefixation > Stem-internal alternation (ablaut/infixation)
====================== ====================== ==================================================================================

Most of these were implemented for comparison purposes. I recommend to use the default `patternsPhonsim` in most cases. To avoid relying on your phonological features files for alignment scores, use `patternsLevenshtein`. Only these two are full patterns with generalization both in the context and alternation.

For lattices, we keep defective and overabundant entries. We do not usually keep them for other applications.
The latest code for entropy can handle defective entries.
The file you should use as input for the below scripts has a name that ends in "_patterns". The "_human_readable_patterns" file is nicer to review but is only meant for human usage.


Usage of `bin/calc_paradigm_entropy.py`
-----------------------------------------

Compute entropies of flexional paradigms' distributions.

**Computing entropies from one cell** ::

    bin/$ python3 calc_paradigm_entropy.py -o <patterns.csv> <paradigm.csv> <segments.csv>


**Computing entropies from one cell, with a split dataset** ::

    bin/$ python3 calc_paradigm_entropy.py -names <data1 name> <data2 name> -b <patterns1.csv> <paradigm1.csv> -o <patterns2.csv> <paradigm2.csv> <segments.csv>

**Computing entropies from two cell** ::

    bin/$ python3 calc_paradigm_entropy.py -n 2 <patterns.csv> <paradigm.csv> <segments.csv>

More complete usage can be obtained by typing ::

    bin/$ python3 calc_paradigm_entropy.py --help

With `--nPreds` and N>2 the computation can get quite long on large datasets.

Usage of `bin/find_macroclasses.py`
-------------------------------------

Cluster lexemes in macroclasses according to alternation patterns.


**Inferring macroclasses** ::

    bin/$ python3 find_macroclasses.py  <patterns.csv> <segments.csv>

More complete usage can be obtained by typing ::

    bin/$ python3 find_macroclasses.py --help

The options "-m UPGMA", "-m CD" and "-m TD" are experimental and will not undergo further development, use at your own risks. The default is to use Description Length (DL) and a bottom-up algorithm (BU).

Usage of `bin/make_lattice.py`
-------------------------------------

Infer Inflection classes as a lattice from alternation patterns.
This will produce a context and an interactive html file.


**Inferring a lattice of inflection classes, with html output** ::

    bin/$ python3 make_lattice.py --html <patterns.csv> <segments.csv>

More complete usage can be obtained by typing ::

    bin/$ python3 make_lattice.py --help
