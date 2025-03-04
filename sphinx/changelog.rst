Changelog
=========

Qumin follows the `semver <https://semver.org/>`_ principles for versioning. This changelog only refers to MAJOR and MINOR versions.

dev version (forthcoming)
~~~~~~~~~~~~~~~~~~~~~~~~~

- Add a `pos` keyword to filter paradigms on POS and improve the behaviour of `cells`.
- Prevent Matplotlib font manager from spamming the log in debug mode.
- Adds a Frequency class able to handle multiple paralex sources for frequencies.
- Switch patterns management to long format everywhere
- Implement parallelisation for finding patterns and finding applicable patterns. See `cpus` config option.
- Change management of sampling: `most_freq=False` replaced by `force_random=True`.
    Addition of a seed option to determinise sampling.  By default, sample by frequency.
    Change `sample` for `sample_lexemes` and add `sample_cells` with the same behaviour for cells.
- Removal:
    - Support for sound table "Seg." column.
    - Bipartite entropy computations
    - All alternation algorithms except `phon` and `edits` (former `patternsPhonsim` and `patternsLevenshtein`).
    
Version 2.0
~~~~~~~~~~~

* Support for the Paralex standard.
* Automatic generation of heatmaps for entropy computations.
* Several bugfixes
* Removal:
    * Support for wide paradigms.

Version 1.1
~~~~~~~~~~~

- Several bugfixes

Version 1.0
~~~~~~~~~~~

- Initial release
