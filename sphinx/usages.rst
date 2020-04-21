Usages
======


Data shape
----------

Segment file
^^^^^^^^^^^^

* The file is a csv table (separated by tabulation or comma) with segments in rows and features in columns, plus a `Seg.` column with the character as it appears in the paradigm..
* The `Seg.` column is the utf8 representation of the segment.
* The segments symbols, in the `Seg.` column cannot be one of the reserved character : `. ^ $ * + ? { } [ ] / | ( ) < > _  ⇌ , ;`
* The file must have two levels of headers, both beginning by `Seg.`, then the attributes for the first rows, the abbreviated attributes for the second rows.
* The values in the features columns can be `-1` (if the relevant isn't valid for this segment) or a positive integer. We recommand them to be in [-1,1].
* If the file contains a "value" column, it will be ignored. This is used to provide a human-readable description of segments.
* You can also add an ALIAS column to provide custom 1-character aliases for your segments (useful if you have a lot of bigrams or if you want to control the inner representations). Otherwise, the program will try to find automatically a resembling character.

======= ======= =========== =========== ============ ========= =========
 Seg.    ALIAS   sonority    syllabic    consonantic anterior    ...
======= ======= =========== =========== ============ ========= =========
 Seg.               son          syl        cons        ant       ...
 p                   0            0          1          1         ...
 t                   0            0          1          1         ...
 b                   0            0          1          1         ...
 d                   0            0          1          1         ...
 ɛ̃        ẽ          1            1          0          -1        ...
 ɑ̃        ã          1            1          0          -1        ...
 ɔ̃        õ          1            1          0          -1        ...
======= ======= =========== =========== ============ ========= =========

* You can add shorthands. Their names have to start and end by "#".They will be used when pretty printing patterns. Here an example for C (consonants), V (vowels) and G (glides) shorthands:

======= =========== =========== ============ ========= =========
 Seg.    sonority    syllabic    consonantic anterior    ...
======= =========== =========== ============ ========= =========
 Seg.    son          syl        cons            ant      ...
 #C#       0            0          1              -1      ...
 #V#       1            1          0              -1      ...
 #G#       1            0          0              -1      ...
======= =========== =========== ============ ========= =========

The Paradigms file
^^^^^^^^^^^^^^^^^^^
* The file is a comma separated csv table with forms from one lexeme in rows and cells of the paradigm in columns, plus one meta columns for lemmas name.
* The file's headers (first row) are the morphological cell's names. Columns headers shouldn't contain the character "#".
* the first column contains lexeme identifiers. It is usually convenient to use orthographic citation forms for this purpose (e.g. infinitive for verbs).
* If the second column's header is "variants", then the whole column will be ignored. This fits Vlexique's format, where the second column holds orthographic variants.
* All other columns are cells of the paradigm, and contain one or more form per rows and cell.
* If there are several forms, they must be separated by ";". Note that overabundant lines will be dropped for both the macroclasses search and the implicative entropy calculation.
* Missing values can be signaled by the string "#DEF#".Note that lines with missing values will be dropped for the macroclasses search.

============ ============ ================ =============
lexeme       variants        prs.2.sg        prs.3.sg
============ ============ ================ =============
abaisser     abaisser        abɛs            abɛs
abandonner   abandonner      abɑ̃dɔn          abɑ̃dɔn
abasourdir   abasourdir      abazuʁdi        abazuʁdi
abâtardir    abâtardir       abataʁdi        abataʁdi
============ ============ ================ =============


Usage of `bin/find_patterns.py`
-----------------------------------------

Find pairwise alternation patterns from paradigms.
This is a preliminary step necessary to obtain patterns used as input in the three scripts below.

**usage**::

      find_patterns.py [-h]
                       [-k {endings,endingsPairs,globalAlt,localAlt,endingsDisc,patternsLevenshtein,patternsPhonsim,patternsSuffix,patternsPrefix,patternsBaseline}]
                       [-d] [-o] [-m] [-f FOLDER]
                        paradigms segments


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

More complete usage can be obtained by typing ::

    bin/$ python3 find_patterns.py --help

For lattices, we keep defective and overabundant entries. We do not for other applications.
The file you should use as input for the below scripts has a name that ends in "_patterns". The "_human_readable_patterns" file is nicer to review but is only meant for human usage.


Usage of `bin/calc_paradigm_entropy.py`
-----------------------------------------

Compute entropies of flexional paradigms' distributions.

**usage**::

    calc_paradigm_entropy.py [-h] [-b BIPARTITE BIPARTITE]
                                    [--names NAMES NAMES] [-v] [-i file] [-m]
                                    [--comp] [-o] [-n N [N ...]] [-s] [-p]
                                    [-f FOLDER]
                                    patterns paradigms segments

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

**usage** ::

     find_macroclasses.py [-h] [-m {UPGMA,DL}] [-a {TD,BU}] [-v] [-d] [-f FOLDER]
                            patterns segments


**Inferring macroclasses** ::

    bin/$ python3 find_macroclasses.py  <patterns.csv> <segments.csv>

More complete usage can be obtained by typing ::

    bin/$ python3 find_macroclasses.py --help

The options "-m UPGMA", "-m CD" and "-m TD" are experimental and will not undergo further development, use at your own risks. The default is to use Description Length (DL) and a bottom-up algorithm (BU).

Usage of `bin/make_lattice.py`
-------------------------------------

Infer Inflection classes as a lattice from alternation patterns.
This will produce a context and an interactive html file.

**usage** ::

    make_lattice.py [-h] [--shorten] [-b BIPARTITE] [--aoc] [--html]
                       [--cxt] [--stat] [--pdf] [--png] [--first] [-f FOLDER]
                       patterns segments

**Inferring a lattice of inflection classes, with html output** ::

    bin/$ python3 make_lattice.py --html <patterns.csv> <segments.csv>

More complete usage can be obtained by typing ::

    bin/$ python3 make_lattice.py --help
