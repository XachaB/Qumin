Changelog
=========

Qumin follows the `semver <https://semver.org/>`_ principles for versioning. This changelog only refers to MAJOR and MINOR versions.

dev version (forthcoming)
~~~~~~~~~~~~~~~~~~~~~~~~~

- Add a `pos` keyword to filter paradigms on POS.
- Usage of frequencies:
    - Read frequencies from as much sources as possible in the Paralex package (Frequencies class).
    - Weight cells based on the predictor-target pair frequency.
- Prevent Matplotlib font manager from spamming the log in debug mode.
- Deprecations:
    - Sound table "Seg." column not supported anymore.

Version 2.0
~~~~~~~~~~~

* Support for the Paralex standard.
* Automatic generation of heatmaps for entropy computations.
* Add a `cells` keyword to filter paradigms on cells.
* Several bugfixes
* Deprecations:
    * Wide paradigms not supported anymore.

Version 1.1
~~~~~~~~~~~

- Several bugfixes

Version 1.0
~~~~~~~~~~~

- Initial release
