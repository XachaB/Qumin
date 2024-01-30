
|tests| |DocStatus|_

.. |tests| image:: https://github.com/xachab/qumin/actions/workflows/python-package.yml/badge.svg

.. |DocStatus| image:: https://readthedocs.org/projects/qumin/badge/?version=dev
.. _DocStatus: https://qumin.readthedocs.io/dev/?badge=latest

Qumin (QUantitative Modelling of INflection) is a package for the computational modelling of the inflectional morphology of languages. It was initially developed for `my PhD dissertation <https://tel.archives-ouvertes.fr/tel-01840448>`_.

**Contributors**: Sacha Beniamine, Jules Bouton.

**Documentation**: https://qumin.readthedocs.io/

**Github**: https://github.com/XachaB/Qumin


This is version 2, which was significantly updated since the publications cited below.

For more detail, you can refer to my dissertation (in French, `Beniamine 2018 <https://tel.archives-ouvertes.fr/tel-01840448>`_). If you use(d) Qumin in a publication, send me an email with the reference at s.<last name>@surrey.ac.uk


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

The package expect full paradigm data in phonemic transcription, as well as a feature key for the transcription.

For compatible data, see the `Paralex datasets <http://www.paralex-standard.org>`_. The sounds files may sometimes require edition, as Qumin as more constraints on sound definitions.


Scripts
--------


Patterns
^^^^^^^^^

Alternation patterns serve as a basis for all the other scripts. An early version of the algorithm to find the patterns was presented in `Beniamine (2017) <https://halshs.archives-ouvertes.fr/hal-01615899>`_. An updated description of the algorithm figures in `Beniamine, Bonami and  Luís (2021) <https://doi.org/10.5565/rev/isogloss.109>`_.

**Computing automatically aligned patterns** for  macroclass (ignore defective lexemes and overabundant forms)::

    bin/$ qumin.patterns <paradigm.csv> <segments.csv>

**Computing automatically aligned patterns** for paradigm entropy (keep defective lexemes, but not overabundant forms)::

    bin/$ qumin.patterns -d <paradigm.csv> <segments.csv>

**Computing automatically aligned patterns** for lattices (keep defective lexemes and overabundant forms)::

    bin/$ qumin.patterns -d -o <paradigm.csv> <segments.csv>

The option -k allows one to choose the algorithm for inferring alternation patterns.

* **Affixal strategies**:
    * ``endings``: Removes the longest common initial string for each row.
    * ``endingsPairs``: Endings, tabulated as pairs for all combinations of columns.
    * ``endingsDisc``: Discontinuous endings. Removes the longest common substring, left aligned
* **Alternations**:  Alternations have no contextes. These were used for comparing macroclass strategies on French and European Portuguese.
    * ``globalAlt``: As ``EndingsDisc``, tabulated as pairs for all combinations of columns.
    * ``localAlt``:  Inferred from pairs of cells, left aligned.
* **Patterns**: Binary alternation Patterns. All patterns have alternations and generalized contexts. Various alignment strategies are offered for comparison. Arbitrary number of changes supported.
    * ``patternsLevenshtein``: Aligned with simple edit distance.
    * ``patternsPhonsim``: Aligned with edit distances based on phonological similarity.
    * ``patternsSuffix``: Fixed left alignment, only interesting for suffixal languages.
    * ``patternsPrefix``: Fixed right alignment, only interesting for prefixal languages.
    * ``patternsBaseline``: Baseline alignment, follows Albright & Hayes 2002. A single change, with a priority order: Suffixation > Prefixation > Stem-internal alternation (ablaut/infixation)

Most of these were implemented for comparison purposes. I recommend to use the default `patternsPhonsim` in most cases. To avoid relying on your phonological features files for alignment scores, use `patternsLevenshtein`. Only these two are full patterns with generalization both in the context and alternation.

For lattices, we keep defective and overabundant entries. We do not usually keep them for other applications.
The latest code for entropy can handle defective entries.
The file you should use as input for the below scripts has a name that ends in "_patterns". The "_human_readable_patterns" file is nicer to review but is only meant for human usage.

**Full usage and more details:**::

    bin/$ qumin.patterns --help


Microclasses
^^^^^^^^^^^^^

To visualize the microclasses and their similarities, one can compute a **microclass heatmap**::

    bin/$ qumin.heatmap <paradigm.csv> <output_path>

**Computing a microclass heatmap, comparing with class labels**::

    bin/$ qumin.heatmap -l  <labels.csv> -- <paradigm.csv> <output_path>

The labels file is a csv file. The first column give lexemes names, the second column provides inflection class labels. This allows to visually compare a manual classification with pattern-based similarity. This command relies heavily on `seaborn's clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`__ function.

**Full usage and more details:**::

    bin/$ qumin.heatmap --help


Paradigm entropy
^^^^^^^^^^^^^^^^^^


This software was used in `Bonami and Beniamine 2016 <http://www.llf.cnrs.fr/fr/node/4789>`_,  `Beniamine, Bonami and Luís (2021) <https://doi.org/10.5565/rev/isogloss.109>`_

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

Our work on automatical inference of macroclasses was published in `Beniamine, Bonami and Sagot (2018) <http://jlm.ipipan.waw.pl/index.php/JLM/article/view/184>`_".

**Inferring macroclasses** ::

    bin/$ qumin.macroclasses  <patterns.csv> <segments.csv>

**Full usage and more details:**::

    bin/$ qumin.macroclasses --help


Lattices
^^^^^^^^^

This software was used in `Beniamine (2021) <https://langsci-press.org/catalog/book/262>`_".

**Inferring a lattice of inflection classes, with html output** ::

    bin/$ qumin.lattice --html <patterns.csv> <segments.csv>

**Full usage and more details:**::

    bin/$ qumin.lattice --help

