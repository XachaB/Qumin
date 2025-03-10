Alternation patterns
=================================

.. note::
    This script generates a human readable file `human_readable_<kind>.csv` which is only intended for manual inspection, as well as a file `<kind>.csv`, eg `phon.csv`, which is intended to be passed to further Qumin scripts.

The patterns can be further configured, modifying the following keys: ::

    data: null                # path to paralex.package.json paradigms, segments
    cells: null               # List of cells to use (subset)
    pos: null                 # List of parts of speech to use (subset)
    patterns: null            # path to pre-computed patterns. If null, will compute patterns.
    sample: null              # (int) A number of lexemes to sample, for debug purposes.
                              # Samples by frequency if possible, otherwise randomly.
    force_random: False       # Whether to force random sampling.
    seed: 1                   # Random seed for reproducible random effects.
    pats:
      kind: phon   # Options are (see docs): phon, edits
      defective: False        # Whether to keep defective entries
      overabundant:
        keep: False           # Whether to keep overabundant entries
        freq: True            # Prioritize by frequency when dropping overabundance. If false: prioritize first in file.
        tags: null            # Tags to prefer when dropping overabundance. Then fallback on freq.
      gap_proportion: .4      # Proportion of the median score used to set the gap score
      optim_mem: False        # Attempt to use a little bit less memory
      merged: False           # Whether to merge identical columns in the data

values for these keys can be given through the command line, eg::

    /$ qumin verbose=True cells=="[ind.prs.1.sg,ind.fut.1.sg]" pats.defective=True data=<dataset.package.json>


Patterns kinds
~~~~~~~~~~~~~~~

Qumin can compute various kinds of patterns that can be used for entropy calculations. They have alternations and generalized contexts:

* ``edits``: Aligned with simple edit distance.
* ``phon``: Aligned with edit distances based on phonological similarity.

It is recommended to use the default `phon` in most cases. To avoid relying on your phonological features files for alignment scores, use `edit`. Only these two are full patterns with generalization both in the context and alternation.

.. warning::
    Additional strategies were implemented for comparison purposes during Sacha Beniamine's PhD. Support for these has been discontinued after Qumin 2.0.1.
