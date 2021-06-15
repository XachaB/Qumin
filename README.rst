********************************************
Qumin: Quantitative modelling of inflection
********************************************

Qumin (QUantitative Modelling of INflection) is a collection of scripts for the computational modelling of the inflectional morphology of languages. It was developed by me (`Sacha Beniamine <xachab.github.io>`_) for my PhD, which was supervised by `Olivier Bonami <http://www.llf.cnrs.fr/fr/Gens/Bonami>`_ . 

**The documentation has moved to ReadTheDocs** at: https://qumin.readthedocs.io/

For more detail, you can refer to my dissertation (in French):

`Sacha Beniamine. Classifications flexionnelles. Étude quantitative des structures de paradigmes. Linguistique. Université Sorbonne Paris Cité - Université Paris Diderot (Paris 7), 2018. Français. <https://tel.archives-ouvertes.fr/tel-01840448>`_


Quick Start
============

Install
--------

First, open the terminal and navigate to the folder where you want the Qumin code. Clone the repository from github: ::

    git clone https://github.com/XachaB/Qumin.git

Make sure to have all the python dependencies installed. The dependencies are listed in `environment.yml`. A simple solution is to use conda and create a new environment from the `environment.yml` file: ::

    conda env create -f environment.yml

There is now a new conda environment named Qumin. It needs to be activated before using any Qumin script: ::

    conda activate Qumin


Data
-----

The scripts expect full paradigm data in phonemic transcription, as well as a feature key for the transcription.

To provide a data sample in the correct format, Qumin includes a subset of the French `flexique lexicon <http://www.llf.cnrs.fr/fr/flexique-fr.php>`_, distributed under a `Creative Commons Attribution-NonCommercial-ShareAlike license <http://creativecommons.org/licenses/by-nc-sa/3.0/>`_.

For Russian nouns, see the `Inflected lexicon of Russian Nouns in IPA notation <https://zenodo.org/record/3428591>`_.


Scripts
--------


Patterns
^^^^^^^^^

Alternation patterns serve as a basis for all the other scripts. The algorithm to find the patterns was presented in: Sacha Beniamine. `Un algorithme universel pour l'abstraction automatique d'alternances morphophonologiques
24e Conférence sur le Traitement Automatique des Langues Naturelles <https://halshs.archives-ouvertes.fr/hal-01615899>`_ (TALN), Jun 2017, Orléans, France. 2 (2017), 24e Conférence sur le Traitement Automatique des Langues Naturelles.

**Computing automatically aligned patterns** for  macroclass (ignore defective lexemes and overabundant forms)::

    bin/$ python3 find_patterns.py <paradigm.csv> <segments.csv>

**Computing automatically aligned patterns** for paradigm entropy (keep defective lexemes, but not overabundant forms)::

    bin/$ python3 find_patterns.py -d <paradigm.csv> <segments.csv>

**Computing automatically aligned patterns** for lattices (keep defective lexemes and overabundant forms)::

    bin/$ python3 find_patterns.py -d -o <paradigm.csv> <segments.csv>


Microclasses
^^^^^^^^^^^^^

To visualize the microclasses and their similarities, you can use the new script `microclass_heatmap.py`:

**Computing a microclass heatmap**::

    bin/$ python3 microclass_heatmap.py <paradigm.csv> <output_path>

**Computing a microclass heatmap, comparing with class labels**::

    bin/$ python3 microclass_heatmap.py -l  <labels.csv> -- <paradigm.csv> <output_path>

The labels file is a csv file. The first column give lexemes names, the second column provides inflection class labels. This allows to visually compare a manual classification with pattern-based similarity. This script relies heavily on `seaborn's clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`__ function.


Paradigm entropy
^^^^^^^^^^^^^^^^^^


This script was used in:

- Bonami, Olivier, and S. Beniamine. "`Joint predictiveness in inflectional paradigms <http://www.llf.cnrs.fr/fr/node/4789>`_." Word Structure 9, no. 2 (2016): 156-182. Some improvements have been implemented since then.


**Computing entropies from one cell** ::

    bin/$ python3 calc_paradigm_entropy.py -n 1 -- <patterns.csv> <paradigm.csv> <segments.csv>

**Computing entropies from two cells** (you can specify any number of predictors, e.g. `-n 1 2 3` works too) ::

    bin/$ python3 calc_paradigm_entropy.py -n 2 -- <patterns.csv> <paradigm.csv> <segments.csv>

**Add a file with features to help prediction** (for example gender -- features will be added to the known information when predicting) ::

    bin/$ python3 calc_paradigm_entropy.py -n 2 --features <features.csv> -- <patterns.csv> <paradigm.csv> <segments.csv>

Macroclass inference
^^^^^^^^^^^^^^^^^^^^^

Our work on automatical inference of macroclasses was published in Beniamine, Sacha, Olivier Bonami, and Benoît Sagot. "`Inferring Inflection Classes with Description Length. <http://jlm.ipipan.waw.pl/index.php/JLM/article/view/184>`_" Journal of Language Modelling (2018).

**Inferring macroclasses** ::

    bin/$ python3 find_macroclasses.py  <patterns.csv> <segments.csv>

Lattices
^^^^^^^^^

This script was used in:

- Beniamine, Sacha. (in press) "`One lexeme, many classes: inflection class systems as lattices <https://xachab.github.io/papers/Beniamine2019.pdf>`_" , In: One-to-Many Relations in Morphology, Syntax and Semantics , Ed. by Berthold Crysmann and Manfred Sailer. Berlin: Language Science Press.

**Inferring a lattice of inflection classes, with html output** ::

    bin/$ python3 make_lattice.py --html <patterns.csv> <segments.csv>

