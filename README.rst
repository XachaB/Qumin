
Qumin: Quantitative modelling of inflection
===========================================

Qumin (QUantitative Modelling of INflection) is a collection of scripts for the computational modelling of the inflectional morphology of languages. It is developed by `Sacha Beniamine <http://www.llf.cnrs.fr/Gens/Beniamine>`_.
My Phd is supervised by `Olivier Bonami <http://www.llf.cnrs.fr/fr/Gens/Bonami>`_ . The `documentation can be found on the LLF website <http://drehu.linguist.univ-paris-diderot.fr/qumin/>`_.

It includes a subset of the French `flexique lexicon <http://www.llf.cnrs.fr/fr/flexique-fr.php>`_, distributed under a `Creative Commons Attribution-NonCommercial-ShareAlike license <http://creativecommons.org/licenses/by-nc-sa/3.0/>`_.

The python version expected is 3.4 or higher. Current python dependencies are : concepts, numpy, pandas (0.18), scipy, scikit-learn, matplotlib, prettytable.

Patterns
---------

Alternation patterns serve as a basis for all the other scripts. The algorithm to find the patterns was presented in: Sacha Beniamine. `Un algorithme universel pour l'abstraction automatique d'alternances morphophonologiques
24e Conférence sur le Traitement Automatique des Langues Naturelles <https://halshs.archives-ouvertes.fr/hal-01615899>`_ (TALN), Jun 2017, Orléans, France. 2 (2017), 24e Conférence sur le Traitement Automatique des Langues Naturelles.

**Computing automatically aligned patterns** for paradigm entropy or macroclass::

    bin/$ python3 find_patterns.py <paradigm.csv> <segments.csv>

**Computing automatically aligned patterns** for lattices::

    bin/$ python3 find_patterns.py -d -o <paradigm.csv> <segments.csv>

Paradigm entropy
------------------

This script corresponds to the work published in Bonami, Olivier, and S. Beniamine. "`Joint predictiveness in inflectional paradigms <http://www.llf.cnrs.fr/fr/node/4789>`_." Word Structure 9, no. 2 (2016): 156-182.

**Computing entropies from one cell** ::

    bin/$ python3 calc_paradigm_entropy.py -o <patterns.csv> <paradigm.csv> <segments.csv>

**Computing entropies from two cell** ::

    bin/$ python3 calc_paradigm_entropy.py -n 2 <patterns.csv> <paradigm.csv> <segments.csv>

Macroclass inference
----------------------

Our work on automatical inference of macroclasses was published in Beniamine, Sacha, Olivier Bonami, and Benoît Sagot. "`Inferring Inflection Classes with Description Length. <http://jlm.ipipan.waw.pl/index.php/JLM/article/view/184>`_" Journal of Language Modelling (2018).

**Inferring macroclasses** ::

    bin/$ python3 find_macroclasses.py  <patterns.csv> <segments.csv>

Lattices
---------

Our work on inflection class lattices has been presented at the Annual Meeting of the Linguistics Association of Great Britain 2016 (Beniamine & Bonami 2016).

**Inferring a lattice of inflection classes, with html output** ::

    bin/$ python3 make_lattice.py --html <patterns.csv> <segments.csv>
