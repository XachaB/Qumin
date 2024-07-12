
|tests| |DocStatus|_

.. |tests| image:: https://github.com/xachab/qumin/actions/workflows/python-package.yml/badge.svg

.. |DocStatus| image:: https://readthedocs.org/projects/qumin/badge/?version=dev
.. _DocStatus: https://qumin.readthedocs.io/dev/?badge=latest

Qumin (QUantitative Modelling of INflection) is a package for the computational modelling of the inflectional morphology of languages. It was initially developed for `my PhD dissertation <https://tel.archives-ouvertes.fr/tel-01840448>`_.

**Contributors**: Sacha Beniamine, Jules Bouton.

**Documentation**: https://qumin.readthedocs.io/

**Github**: https://github.com/XachaB/Qumin


This is version 2, which was significantly updated since the publications cited below.

For more detail, you can refer to Sacha's dissertation (in French, `Beniamine 2018 <https://tel.archives-ouvertes.fr/tel-01840448>`_). If you use(d) Qumin in a publication, send me an email with the reference at s.<last name>@surrey.ac.uk


Quick Start
============

Install
--------

First, open the terminal and navigate to the folder where you want the Qumin code. Clone the repository from github: ::

    git clone https://github.com/XachaB/Qumin.git

Install the Qumin package: ::

    cd Qumin
    pip install -e ./

Navigate to your working directory: ::

    cd ~/my/workdir/


Data
-----

The package expects full paradigm data in phonemic transcription, as well as a feature key for the transcription.

For compatible data, see the `Paralex datasets <http://www.paralex-standard.org>`_. The sounds files may sometimes require edition, as Qumin as more constraints on sound definitions.


Scripts
--------

.. note::
    We now rely on `hydra <https://hydra.cc/>`_ to manage CLI interface and configurations. Hydra will create a folder ``outputs/<yyyy-mm-dd>/<hh-mm-ss>/`` containing all results. A subfolder ``outputs/<yyyy-mm-dd>/<hh-mm-ss>/.hydra/`` contains details of the configuration as it was when the script was run. Hydra permits a lot more configuration. For example, any of the following scripts can accept a verbose argument of the form `hydra.verbose=True`.

Patterns
^^^^^^^^^

Alternation patterns serve as a basis for all the other scripts. An early version of the algorithm to find the patterns was presented in `Beniamine (2017) <https://halshs.archives-ouvertes.fr/hal-01615899>`_. An updated description of the algorithm figures in `Beniamine, Bonami and  Luís (2021) <https://doi.org/10.5565/rev/isogloss.109>`_.

**Full usage and more details:**::

    /$ qumin.patterns --help

The patterns script comes with a few presets for typical configurations. For more details on alternate configurations, see `qumin.patterns --help` and the `patterns docs page<patterns>`_

**For macroclasses**: ignore defective lexemes and overabundant forms::

    qumin.patterns data=<dataset.package.json>

**For paradigm entropy:** keep defective lexemes, but not overabundant forms::

    /$ qumin.patterns pats=entropy data=<dataset.package.json>

**For inflection class lattices** keep defective lexemes and overabundant forms::

    /$ qumin.patterns pats=lattice data=<dataset.package.json>

Microclasses
^^^^^^^^^^^^^
**Full usage and more details:**::

    /$ qumin.heatmap --help

To visualize the microclasses and their similarities, one can compute a **microclass heatmap**::

    /$ qumin.heatmap patterns=<path/to/patterns.csv> data=<dataset.package.json>

**Computing a microclass heatmap, comparing with class labels**::

    /$ qumin.heatmap label=inflection_class patterns=<path/to/patterns.csv> data=<dataset.package.json>

The labels is the name of the column in the Paralex `lexemes` table to use as labels. This allows to visually compare a manual classification with pattern-based similarity. This command relies heavily on `seaborn's clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`__ function.

A few more parameters can be changed: ::

    patterns: ???             # Path to patterns
    label: null              # lexeme column to use as label (eg. inflection_class)
    cmap: null               # colormap name
    exhaustive_labels: False # by default, seaborn shows only some labels on
                             # the heatmap for readability.
                             # This forces seaborn to print all labels.


Paradigm entropy
^^^^^^^^^^^^^^^^^^

This software was used in `Bonami and Beniamine 2016 <http://www.llf.cnrs.fr/fr/node/4789>`_,  `Beniamine, Bonami and Luís (2021) <https://doi.org/10.5565/rev/isogloss.109>`_

**Full usage and more details:**::

    /$ qumin.H --help

**Computing entropies from one cell** ::

    /$ qumin.H  patterns=<patterns.csv> data=<dataset.package.json>

**Computing entropies for other number of predictors**::

    /$ qumin.H  n=2 patterns=<patterns.csv> data=<dataset.package.json>
    /$ qumin.H  n="[2,3]" patterns=<patterns.csv> data=<dataset.package.json>

**Take into account some features to help prediction** (for example gender and inflection class -- features will be added to the known information when predicting) ::

    /$ qumin.H  feature=inflection_class patterns=<patterns.csv> data=<dataset.package.json>
    /$ qumin.H  feature="[inflection_class,gender]" patterns=<patterns.csv> data=<dataset.package.json>

The features are names of columns from the Paralex `lexemes` table.
With `-n` and N>2 the computation can get quite long on large datasets.

The config file contains the following keys, which can be set through the command line: ::

    patterns: null        # pre-computed patterns
    entropy:
      n:                  # Compute entropy for prediction from with n predictors.
        - 1
      features: null      # Feature column in the Lexeme table.
                          # Features will be considered known in conditional probabilities: P(X~Y|X,f1,f2...)
      importFile: null    # Import entropy file with n-1 predictors (allows for acceleration on nPreds entropy computation).
      merged: False       # Whether identical columns are merged in the input.
      stacked: False      # whether to stack results in long form

For bipartite systems, it is possible to pass two values to both patterns and data, eg: ::

    /$ qumin.H  patterns="[<patterns1.csv>,<patterns2.csv>]" data="[<dataset1.package.json>,<dataset2.package.json>]"


Macroclass inference
^^^^^^^^^^^^^^^^^^^^^

Our work on automatical inference of macroclasses was published in `Beniamine, Bonami and Sagot (2018) <http://jlm.ipipan.waw.pl/index.php/JLM/article/view/184>`_".

**Inferring macroclasses** ::

    /$ qumin.macroclasses  patterns=<patterns.csv> data=<dataset.package.json>

**Full usage and more details:**::

    /$ qumin.macroclasses --help


Lattices
^^^^^^^^^

This software was used in `Beniamine (2021) <https://langsci-press.org/catalog/book/262>`_".

**Inferring a lattice of inflection classes, with (default) html output** ::

    /$ qumin.lattice patterns=<patterns.csv> data=<dataset.package.json>

**Full usage and more details:**::

    /$ qumin.lattice --help


**Further config options**: ::

    patterns: null        # pre-computed patterns
    pats:
      ortho: False        # whether patterns were orthographic
    lattice:
      shorten: False      # Drop redundant columns altogether.
                          #  Useful for big contexts, but loses information.
                          # The lattice shape and stats will be the same.
                          # Avoid using with --html
      aoc: False          # Only attribute and object concepts

    export:
      html: False         # Export to html
      ctxt: False         # Export as a context
      stat: False         # Output stats about the lattice
      pdf: True           # Export as pdf
      png: False          # Export as png