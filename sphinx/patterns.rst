Alternation patterns
====================

This script generates alternation patterns. They can be consumed by further Qumin scripts by passing the path to the ``metadata.json`` file produced by a computation. It also writes human readable patterns in the ``patterns/human_readable`` folder, which are intended for manual inspection.

The patterns can be further configured, modifying the following keys: ::

    cpus: null                  # (int) Number of cpus to use for big computations
                                # (defaults to the number of available cpus - 2).
    pats:
        kind: phon              # (str) Options are (see docs): phon, edits
        defective: False        # (bool) Whether to keep defective entries
        overabundant:
            keep: False         # (bool) Whether to keep overabundant entries
            freq: True          # (bool) Whether to prioritize overabundant forms by frequency (fallback on file order)
            tags: null          # (list) Tags to prefer when dropping overabundance (fallback on freq)
        gap_proportion: .4      # (float) Proportion of the median score used to set the gap score
        optim_mem: False        # (bool) Attempt to use a little bit less memory
        merged: False           # (bool) Whether to merge identical columns in the data


values for these keys can be given through the command line, eg::

    /$ qumin verbose=True cells=="[ind.prs.1.sg,ind.fut.1.sg]" pats.defective=True data=<dataset.package.json>


Patterns kinds
~~~~~~~~~~~~~~

Qumin can compute various kinds of patterns that can be used for entropy calculations. They have alternations and generalized contexts:

* ``edits``: Aligned with simple edit distance.
* ``phon``: Aligned with edit distances based on phonological similarity.

It is recommended to use the default `phon` in most cases. To avoid relying on your phonological features files for alignment scores, use `edit`. Only these two are full patterns with generalization both in the context and alternation.

.. warning::
    Additional strategies were implemented for comparison purposes during Sacha Beniamine's PhD. Support for these has been discontinued after Qumin 2.0.1.
