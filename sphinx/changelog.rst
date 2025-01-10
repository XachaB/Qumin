Changelog
=========

Qumin follows the `semver <https://semver.org/>`_ principles for versioning. This changelog only refers to MAJOR and MINOR versions.

dev version (forthcoming)
~~~~~~~~~~~~~~~~~~~~~~~~~

- Add a `pos` keyword to filter paradigms on POS.
- Prevent Matplotlib font manager from spamming the log in debug mode.
- Adds a Frequency class able to handle multiple paralex sources for frequencies.
- Support removed for:
    - "Seg." column in sound tables.
    - All alternation algorithms except `phon` and `edit` (former `patternsPhonsim` and `patternsLevenshtein`).

Version 2.0
~~~~~~~~~~~

* Support for the Paralex standard.
* Automatic generation of heatmaps for entropy computations.
* Several bugfixes
* Deprecations:
    * Wide paradigms not supported anymore.

Version 1.1
~~~~~~~~~~~

- Several bugfixes

Version 1.0
~~~~~~~~~~~

- Initial release
