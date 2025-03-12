Changelog
=========

Qumin follows the `semver <https://semver.org/>`_ principles for versioning. This changelog only refers to MAJOR and MINOR versions.

Version 3.0
~~~~~~~~~~~

- Implement parallelisation for finding patterns and finding applicable patterns. See `cpus` config option.
- **[breaking]** Switch patterns management to long format everywhere.
- **[breaking]** Change management of imports/exports. Outputs from Qumin are now shipped as a `Frictionless DataPackage <https://datapackage.org/>`_. To import them, a path to the computation ``metadata.json`` must be provided.
- Addition of frequencies:
    - Read frequencies from as much sources as possible in the Paralex package (Frequencies class).
    - Weight cells based on the predictor-target pair frequency.
- Paralex compliance changes:
    - Usage of the `form_id` introduced in Paralex.
    - Sounds file in conformity with Paralex (`Segs` -> `sound_id`)
- User interface improvements:
    - Change the format of human readable patterns to a more readable markdown export
    - Add keywords ``pos``, `sample_cells`, `sample_lexemes` keyword to filter paradigms.
    - Prevent Matplotlib font manager from spamming the log in debug mode.
    - Improve and extend visualisations of Entropy distributions.
- Sampling management changes:
    - ``most_freq=False`` replaced by ``force_random=True``.
    - Addition of a seed option to determinise sampling.  By default, sample by frequency.
    - Change ``sample`` for ``sample_lexemes`` and add ``sample_cells`` with the same behaviour for cells.
    - Removal of overabundant forms is now by default done by frequency (``pats.overabundant.freq=True``),
    and can optionally prioritize some tags (``pats.overabudant.tags="[standard_form,preferred_form]"``).
- **[breaking]** Removal:
    - Support for sound table "Seg." column.
    - Bipartite entropy computations
    - All alternation algorithms except ``phon`` and ``edits`` (former ``patternsPhonsim`` and ``patternsLevenshtein``).
    - Patterns evaluation with ``action=eval``.
- And some various smaller bug fixes !


Version 2.0
~~~~~~~~~~~

* Support for the Paralex standard.
* Automatic generation of heatmaps for entropy computations.
* Add a ``cells`` keyword to filter paradigms on cells.
* Several bugfixes
* Removal:
    * Support for wide paradigms.

Version 1.1
~~~~~~~~~~~

- Several bugfixes

Version 1.0
~~~~~~~~~~~

- Initial release
