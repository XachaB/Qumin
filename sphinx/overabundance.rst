Overabundant systems
====================

Overabundant paradigms are only handled for some very specific computations.

Patterns
~~~~~~~~

Pattern computations should always run with the overabundance option: ::

    /$ qumin  pats.overabundant=True pats.defective=True

Computations detailed below were only tested with simple one predictor computations. Since the overabundant approach comes along with more complex computations, we did not reimplement some features of Qumin.

Entropies and accuracies
~~~~~~~~~~~~~~~~~~~~~~~~

Given that in overabundant situations, entropy and probability of succes do not always yield similar results, both metrics are computed, which turns out to be significantly slower.

Entropy should be computed with the -o option: ::

    /$ qumin action=H data=<dataset.package.json> patterns=<patterns.csv> entropy.overabundant=True

A lot of additional options are available to improve the results. Most can be combined together and even used for non overabundant datasets. However, you should still use the ``entropy.overabundant`` option since these options are not available in the standard approach.

Frequencies
-----------

The default approach will only use type frequencies. If the Paralex dataset contains frequencies, Qumin is also able to use them for weighting. You can even ask Qumin to only use those frequencies for token-based weighting with the ``entropy.token`` parameter. In other words, you have three options:

* Default : **type** frequencies everywhere. Overabundance ratios are **uniform distributions**.
* ``entropy.real_frequencies=True`` : **type** frequencies for weighting. Overabundance ratios are based on **token frequencies**.
* ``entropy.real_frequencies=True entropy.token=True``: **token** frequencies everywhere.

E.g. ::

    /$ qumin action=H data=<dataset.package.json> patterns=<patterns.csv> entropy.overabundant=True entropy.real_frequencies=True entropy.token=True

Pattern probabilities strategy
------------------------------

By default, patterns are computed on overabundant forms by considering a probability between 0 and 1 for each form. The sum of the probabilities of all forms for a given cell should sum to 1. This means that theoretical forms with lower corpus frequencies will be neglected.

It's possible to consider all possible forms as correct, irrespective of their corpus attestations. For this, use the ``entropy.cat_pattern`` parameter: ::

    /$ qumin action=H data=<dataset.package.json> patterns=<patterns.csv> entropy.cat_pattern=True

Pattern probabilities mapping
-----------------------------

As a default, Qumin will use a normalized distribution. However, interesting results can be achieved with a softmax function. The kind of function to use can be specified with the ``entropy.function`` parameter. There are three possible values:

* ``norm`` (default): the **normalized** probability distribution.
* ``soft``: a **softmax** function.
* ``uni``: a **uniform** distribution (baseline assumption)

E.g. ::

    /$ qumin action=H data=<dataset.package.json> patterns=<patterns.csv> entropy.overabundant=True entropy.function=uni


When using a softmax function, it is recommended to specify an appropriate value of beta (you probably want either 5 (few changes) or 20 (skewed distribution). A list of values can be passed to the ``entropy.beta`` setting: ::

    /$ qumin action=H data=<dataset.package.json> patterns=<patterns.csv> entropy.overabundant=True entropy.function=soft entropy.beta="[1, 2, 5, 10, 20]"

In this case, the debug option is not implemented and should not be used.

Success evaluation
------------------

By default, a pattern is considered correct if the ouput form exists (0 or 1). However, it is possible to leverage this by the probability of the output form (for overabundant forms). For this, use the ``entropy.grad_success`` parameter: ::

    /$ qumin action=H data=<dataset.package.json> patterns=<patterns.csv> entropy.overabundant=True entropy.grad_success=True
