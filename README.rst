********************************************
Qumin: Quantitative modelling of inflection
********************************************

Qumin (QUantitative Modelling of INflection) is a collection of scripts for the computational modelling of the inflectional morphology of languages. It was developed by me (`Sacha Beniamine <xachab.github.io>`_) for my PhD, which was supervised by `Olivier Bonami <http://www.llf.cnrs.fr/fr/Gens/Bonami>`_ . 

**The documentation has moved to ReadTheDocs** at: https://qumin.readthedocs.io/

This is version 2, which was significatively updated since the below publications.

For more detail, you can refer to my dissertation (in French):

`Sacha Beniamine. Classifications flexionnelles. Étude quantitative des structures de paradigmes. Linguistique. Université Sorbonne Paris Cité - Université Paris Diderot (Paris 7), 2018. Français. <https://tel.archives-ouvertes.fr/tel-01840448>`_


Quick Start
============

Install
--------

First, open the terminal and navigate to the folder where you want the Qumin code. Clone the repository from github: ::

    git clone https://github.com/XachaB/Qumin.git

Install the Qumin package: ::

    cd Qumin
    pip install -e ./


Data
-----

The scripts expect full paradigm data in phonemic transcription, as well as a feature key for the transcription.

To provide a data sample in the correct format, Qumin includes a subset of the French `flexique lexicon <http://www.llf.cnrs.fr/fr/flexique-fr.php>`_, distributed under a `Creative Commons Attribution-NonCommercial-ShareAlike license <http://creativecommons.org/licenses/by-nc-sa/3.0/>`_.

For more data, see the [Paralex datasets](http://www.paralex-standard.org).


Scripts
--------


Patterns
^^^^^^^^^

Alternation patterns serve as a basis for all the other scripts. The algorithm to find the patterns was presented in: Sacha Beniamine. `Un algorithme universel pour l'abstraction automatique d'alternances morphophonologiques
24e Conférence sur le Traitement Automatique des Langues Naturelles <https://halshs.archives-ouvertes.fr/hal-01615899>`_ (TALN), Jun 2017, Orléans, France. 2 (2017), 24e Conférence sur le Traitement Automatique des Langues Naturelles.

**Computing automatically aligned patterns** for  macroclass (ignore defective lexemes and overabundant forms)::

    bin/$ qumin.patterns <paradigm.csv> <segments.csv>

**Computing automatically aligned patterns** for paradigm entropy (keep defective lexemes, but not overabundant forms)::

    bin/$ qumin.patterns -d <paradigm.csv> <segments.csv>

**Computing automatically aligned patterns** for lattices (keep defective lexemes and overabundant forms)::

    bin/$ qumin.patterns -d -o <paradigm.csv> <segments.csv>

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

**Full usage and more details:**::

    bin/$ qumin.patterns --help


Microclasses
^^^^^^^^^^^^^

To visualize the microclasses and their similarities, you can use the new script `microclass_heatmap.py`:

**Computing a microclass heatmap**::

    bin/$ qumin.heatmap <paradigm.csv> <output_path>

**Computing a microclass heatmap, comparing with class labels**::

    bin/$ qumin.heatmap -l  <labels.csv> -- <paradigm.csv> <output_path>

The labels file is a csv file. The first column give lexemes names, the second column provides inflection class labels. This allows to visually compare a manual classification with pattern-based similarity. This script relies heavily on `seaborn's clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`__ function.

**Full usage and more details:**::

    bin/$ qumin.heatmap --help


Paradigm entropy
^^^^^^^^^^^^^^^^^^


This script was used in:

- Bonami, Olivier, and S. Beniamine. "`Joint predictiveness in inflectional paradigms <http://www.llf.cnrs.fr/fr/node/4789>`_." Word Structure 9, no. 2 (2016): 156-182. Some improvements have been implemented since then.


**Computing entropies from one cell** ::

    bin/$ qumin.H -n 1 -- <patterns.csv> <paradigm.csv> <segments.csv>

**Computing entropies from two cells** (you can specify any number of predictors, e.g. `-n 1 2 3` works too) ::

    bin/$ qumin.H -n 2 -- <patterns.csv> <paradigm.csv> <segments.csv>

**Add a file with features to help prediction** (for example gender -- features will be added to the known information when predicting) ::

    bin/$ qumin.H -n 2 --features <features.csv> -- <patterns.csv> <paradigm.csv> <segments.csv>

With `-n` and N>2 the computation can get quite long on large datasets.

**Full usage and more details:**::

    bin/$ qumin.H --help



Macroclass inference
^^^^^^^^^^^^^^^^^^^^^

Our work on automatical inference of macroclasses was published in Beniamine, Sacha, Olivier Bonami, and Benoît Sagot. "`Inferring Inflection Classes with Description Length. <http://jlm.ipipan.waw.pl/index.php/JLM/article/view/184>`_" Journal of Language Modelling (2018).

**Inferring macroclasses** ::

    bin/$ qumin.macroclasses  <patterns.csv> <segments.csv>

**Full usage and more details:**::

    bin/$ qumin.macroclasses --help


Lattices
^^^^^^^^^

This script was used in:

- Beniamine, Sacha. (in press) "`One lexeme, many classes: inflection class systems as lattices <https://xachab.github.io/papers/Beniamine2019.pdf>`_" , In: One-to-Many Relations in Morphology, Syntax and Semantics , Ed. by Berthold Crysmann and Manfred Sailer. Berlin: Language Science Press.

**Inferring a lattice of inflection classes, with html output** ::

    bin/$ qumin.lattice --html <patterns.csv> <segments.csv>

**Full usage and more details:**::

    bin/$ qumin.lattice --help

