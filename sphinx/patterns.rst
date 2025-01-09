Alternation patterns
=================================

.. note::
    This script generates a human readable file `human_readable_<kind>.csv` which is only intended for manual inspection, as well as a file `<kind>.csv`, eg `patternsPhonsim.csv`, which is intended to be passed to further Qumin scripts.

The patterns can be further configured, modifying the following keys: ::

    data: null                # path to paralex.package.json paradigms, segments
    cells: null               # List of cells to use (subset)
    pos: null                 # List of parts of speech to use (subset)
    patterns: null            # path to pre-computed patterns. If null, will compute patterns.
    most_freq: null           # (int) restrict to N most frequent items (use the lexeme "frequency" column)
    sample: null              # ( int) A number of lexemes to sample, for debug purposes
    pats:
      kind: patternsPhonsim   # Options are (see docs): patternsLevenshtein, patternsPhonsim
      defective: False        # Whether to keep defective entries
      overabundant: False     # Whether to keep overabundant entries
      gap_proportion: .4      # Proportion of the median score used to set the gap score
      optim_mem: False        # Attempt to use a little bit less memory
      merged: False           # Whether to merge identical columns in the data

values for these keys can be given through the command line, eg::

    /$ qumin verbose=True cells=="[ind.prs.1.sg,ind.fut.1.sg]" pats.defective=True data=<dataset.package.json>


Patterns kinds
~~~~~~~~~~~~~~~

Qumin can compute various kinds of patterns that can be used for entropy calculations. They have alternations and generalized contexts:

* ``patternsLevenshtein``: Aligned with simple edit distance.
* ``patternsPhonsim``: Aligned with edit distances based on phonological similarity.

It is recommended to use the default `patternsPhonsim` in most cases. To avoid relying on your phonological features files for alignment scores, use `patternsLevenshtein`. Only these two are full patterns with generalization both in the context and alternation.

.. warning::
    Additional strategies were implemented for comparison purposes during Sacha Beniamine's PhD. They are now deprecated but are available until Qumin 2.0.1.

    * **Affixal strategies**:
        * ``endings``: Removes the longest common initial string for each row.
        * ``endingsPairs``: Endings, tabulated as pairs for all combinations of columns.
        * ``endingsDisc``: Discontinuous endings. Removes the longest common substring, left aligned
    * **Alternations**:  Alternations have no contextes. These were used for comparing macroclass strategies on French and European Portuguese.
        * ``globalAlt``: As ``EndingsDisc``, tabulated as pairs for all combinations of columns.
        * ``localAlt``:  Inferred from pairs of cells, left aligned.
    * **Patterns**: Binary alternation Patterns. All patterns have alternations and generalized contexts. Various alignment strategies are offered for comparison. Arbitrary number of changes supported.
        * ``patternsSuffix``: Fixed left alignment, only interesting for suffixal languages.
        * ``patternsPrefix``: Fixed right alignment, only interesting for prefixal languages.
        * ``patternsBaseline``: Baseline alignment, follows Albright & Hayes 2002. A single change, with a priority order: Suffixation > Prefixation > Stem-internal alternation (ablaut/infixation)
