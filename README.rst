
|PyPi|_ |tests| |DocStatus|_

.. |PyPi| image:: https://img.shields.io/pypi/v/qumin
.. _PyPi: https://pypi.org/project/qumin/

.. |tests| image:: https://github.com/xachab/qumin/actions/workflows/python-package.yml/badge.svg

.. |DocStatus| image:: https://readthedocs.org/projects/qumin/badge/?version=dev
.. _DocStatus: https://qumin.readthedocs.io/dev/?badge=latest

Qumin (QUantitative Modelling of INflection) is a package for the computational modelling of the inflectional morphology of languages. It was initially developed for `Sacha Beniamine's PhD dissertation <https://tel.archives-ouvertes.fr/tel-01840448>`_.

**Contributors**: Sacha Beniamine, Jules Bouton.

**Documentation**: https://qumin.readthedocs.io/

**Github**: https://github.com/XachaB/Qumin


This is version |version|, which was significantly updated since the publications cited below. These updates do not affect results, and focused on bugfixes, command line interface, paralex compatibility, workflow improvement and overall tidyness.

For more detail, you can refer to Sacha's dissertation (in French, `Beniamine 2018 <https://tel.archives-ouvertes.fr/tel-01840448>`_).


Citing
============

If you use Qumin in your research, please cite Sacha's dissertation (`Beniamine 2018 <https://tel.archives-ouvertes.fr/tel-01840448>`_), as well as the relevant paper for the specific actions used (see below). To appear in the publications list, send Sacha an email with the reference of your publication at s.<last name>@surrey.ac.uk

Quick Start
============

Install
--------

Install the Qumin package using pip: ::

    pip install qumin

Data
-----

Qumin works from full paradigm data in phonemic transcription.

The package expects `Paralex datasets <http://www.paralex-standard.org>`_, containing at least a `forms` and a `sounds` table. Note that the sounds files may sometimes require edition, as Qumin imposes more constraints on sound definitions than paralex does.


Scripts
--------

.. note::
    We now rely on `hydra <https://hydra.cc/>`_ to manage CLI interface and configurations. Hydra will create a folder ``outputs/<yyyy-mm-dd>/<hh-mm-ss>/`` containing all results. A subfolder ``outputs/<yyyy-mm-dd>/<hh-mm-ss>/.hydra/`` contains details of the configuration as it was when the script was run. Hydra permits a lot more configuration. For example, any of the following scripts can accept a verbose argument of the form ``hydra.verbose=Qumin``, and the output directory can be customized with ``hydra.run.dir="./path/to/output/dir"``.

**More details on configuration:**::

    /$ qumin --help

Patterns
^^^^^^^^^

Alternation patterns serve as a basis for all the other scripts. An early version of the patterns algorithm is described in `Beniamine (2017) <https://halshs.archives-ouvertes.fr/hal-01615899>`_. An updated description figures in `Beniamine, Bonami and  Luís (2021) <https://doi.org/10.5565/rev/isogloss.109>`_.

The default action for Qumin is to compute patterns only, so these two commands are identical: ::

    /$ qumin data=<dataset.package.json>
    /$ qumin action=patterns data=<dataset.package.json>

By default, Qumin will ignore defective lexemes and overabundant forms.

For paradigm entropy, it is possible to explicitly keep defective lexemes: ::

    /$ qumin pats.defective=True data=<dataset.package.json>

For inflection class lattices, both can be kept: ::

    /$ qumin pats.defective=True pats.overabundant.keep=True data=<dataset.package.json>

Microclasses
^^^^^^^^^^^^^

To visualize the microclasses and their similarities, one can compute a **microclass heatmap**::

    /$ qumin action=heatmap data=<dataset.package.json>

This will compute patterns, then the heatmap. To pass pre-computed patterns, pass the file path: ::

    /$ qumin action=heatmap patterns=<path/to/patterns.csv> data=<dataset.package.json>

It is also possible to pass class labels to facilitate comparisons with another classification: ::

    /$ qumin.heatmap label=inflection_class patterns=<path/to/patterns.csv> data=<dataset.package.json>

The label key is the name of the column in the Paralex `lexemes` table to use as labels.

A few more parameters can be changed: ::

    heatmap:
        cmap: null               # colormap name
        exhaustive_labels: False # by default, seaborn shows only some labels on
                                # the heatmap for readability.
                                # This forces seaborn to print all labels.


Paradigm entropy
^^^^^^^^^^^^^^^^^^

An early version of this software was used in `Bonami and Beniamine 2016 <http://www.llf.cnrs.fr/fr/node/4789>`_, and a more recent one in `Beniamine, Bonami and Luís (2021) <https://doi.org/10.5565/rev/isogloss.109>`_

By default, this will start by computing patterns. To work with pre-computed patterns, pass their path with ``patterns=<path/to/patterns.csv>``.

**Computing entropies from one cell** ::

    /$ qumin action=H data=<dataset.package.json>

**Computing entropies for other number of predictors**::

    /$ qumin action=H  n=2 data=<dataset.package.json>
    /$ qumin action=H  n="[2,3]" data=<dataset.package.json>

.. warning::
    With `n` and N>2 the computation can get quite long on large datasets, and it might be better to run Qumin on a server.

Predicting with known lexeme-wise features (such as gender or inflection class) is also possible. This feature was used in `Pellegrini (2023) <https://doi.org/10.1007/978-3-031-24844-3>`_. To use features, pass the name of any column(s) from the ``lexemes`` table: ::

    /$ qumin.H  feature=inflection_class patterns=<patterns.csv> data=<dataset.package.json>
    /$ qumin.H  feature="[inflection_class,gender]" patterns=<patterns.csv> data=<dataset.package.json>


The config file contains the following keys, which can be set through the command line: ::

    patterns: null        # pre-computed patterns
    entropy:
      n:                  # Compute entropy for prediction from with n predictors.
        - 1
      features: null      # Feature column in the Lexeme table.
                          # Features will be considered known in conditional probabilities: P(X~Y|X,f1,f2...)
      importFile: null    # Import entropy file with n-1 predictors (allows for acceleration on nPreds entropy computation).
      merged: False       # Whether identical columns are merged in the input.

Visualizing results
^^^^^^^^^^^^^^^^^^^

Since Qumin 2.0, results are shipped as long tables. This allows to store several metrics in the same file, with results for several runs. Results file now look like this: ::

    predictor,predicted,measure,value,n_pairs,n_preds,dataset
    <cell1>,<cell2>,cond_entropy,0.39,500,1,<dataset_name>
    <cell1>,<cell2>,cond_entropy,0.35,500,1,<dataset_name>
    <cell1>,<cell2>,cond_entropy,0.2,500,1,<dataset_name>
    <cell1>,<cell2>,cond_entropy,0.43,500,1,<dataset_name>
    <cell1>,<cell2>,cond_entropy,0.6,500,1,<dataset_name>
    <cell1>,<cell2>,cond_entropy,0.1,500,1,<dataset_name>

All results are in the same file, including different number of predictors (indicated in the `n_preds` column), and different measures (indicated in the `measure` column).

To facilitate a quick general glance at the results, we output an entropy heatmap in the wide matrix format. This behaviour can be disabled by passing `entropy.heatmap=False`. It takes advantage of the Paralex `features-values` table to sort the cells in a canonical order on the heatmap. The `heatmap.order` setting is used to specify which feature should have higher priority in the sorting: ::

    /$ qumin action=H data=<dataset.package.json> heatmap.order="[number, case]"

It is also possible to draw an entropy heatmap without running entropy computations: ::

    /$ qumin action=ent_heatmap entropy.importFile=<entropies.csv>

The config file contains the following keys, which can be set through the command line: ::

    heatmap:
      cmap: null               # colormap name
      exhaustive_labels: False # by default, seaborn shows only some labels on
                               # the heatmap for readability.
                               # This forces seaborn to print all labels.
      dense: False             # Use initials instead of full labels (only for entropy heatmap)
      annotate: False          # Display values on the heatmap. (only for entropy heatmap)
      order: False             # Priority list for sorting features (for entropy heatmap)
                               # ex: [number, case]). If no features-values file available,
                               # it should contain an ordered list of the cells to display.
    entropy:
      heatmap: True        # Whether to draw a heatmap.


Macroclass inference
^^^^^^^^^^^^^^^^^^^^^

Our work on automatical inference of macroclasses was published in `Beniamine, Bonami and Sagot (2018) <http://jlm.ipipan.waw.pl/index.php/JLM/article/view/184>`_".

By default, this will start by computing patterns. To work with pre-computed patterns, pass their path with ``patterns=<path/to/patterns.csv>``.

**Inferring macroclasses** ::

    /$ qumin action=macroclasses data=<dataset.package.json>


Lattices
^^^^^^^^^

By default, this will start by computing patterns. To work with pre-computed patterns, pass their path with ``patterns=<path/to/patterns.csv>``.

This software was used in `Beniamine (2021) <https://langsci-press.org/catalog/book/262>`_".

**Inferring a lattice of inflection classes, with (default) html output** ::

    /$ qumin action=lattice pats.defective=True pats.overabundant.keep=True data=<dataset.package.json>


**Further config options**: ::

    lattice:
      shorten: False      # Drop redundant columns altogether.
                          #  Useful for big contexts, but loses information.
                          # The lattice shape and stats will be the same.
                          # Avoid using with --html
      aoc: False          # Only attribute and object concepts
      stat: False         # Output stats about the lattice
      html: False         # Export to html
      ctxt: False         # Export as a context
      pdf: True           # Export as pdf
      png: False          # Export as png

