The morphological paradigms file
=================================

This file relates phonological forms to their lexemes and paradigm cells.
There are two possible formats for this file, either `wide or long format <https://en.wikipedia.org/wiki/Wide_and_narrow_data>`_ :

-  In **wide format** (deprecated), there is one row per lexeme, and forms are given in successive columns. The entire paradigms fits on the row.
-  In **long format**, there is one row per inflected form. There are three useful columns: the lexeme identifier, the form itself, and the paradigm cell.

The wide format is kept for backward compatibility, but I recommend using the long format (see `Paralex datasets <https://www.paralex-standard.org>`_ in future work.

The files are expected in ``csv`` format, meaning that the values are written as plain text, in utf-8 format, separated by commas. This format can be read by spreadsheet programs as well as programmatically.

Transcriptions
~~~~~~~~~~~~~~

Qumin assumes transcribed phonemic or phonetic forms (rather than orthographic), where each phoneme is ideally separated by spaces. Using non-segmented strings (with no spaces) is still supported, but deprecated, as it can lead to ambiguities.

Wide format
~~~~~~~~~~~~

.. warning::
    The wide format is deprecated. Please use Paralex-style long-form tables.

Qumin used to accept data in wide format, where:

-  each row represents a lexeme, and each column represents a cell.
-  The first column indicates a unique identifier for each lexeme. It is usually convenient to use orthographic citation forms for this purpose (e.g. infinitive for verbs).
-  If a second "Variant" column (for orthographic variants) is provided, it will be ignored.
-  the very first row indicates the names of the cells as column headers. Columns headers shouldn’t contain the character “#”.

 =========== ========= ========= ========= ========== ========== =========
  lexeme      prs.1sg   prs.2sg   prs.3sg   prs.1pl    prs.2pl    prs.3pl
 =========== ========= ========= ========= ========== ========== =========
  peler        pɛl       pɛl       pɛl       pəlɔ̃       pəle       pɛl
  soudoyer     sudwa     sudwa     sudwa     sudwajɔ̃    sudwaje    sudwa
  inféoder     ɛ̃fEɔd     ɛ̃fEɔd     ɛ̃fEɔd     ɛ̃fEOdɔ̃     ɛ̃fEOde     ɛ̃fEɔd
 =========== ========= ========= ========= ========== ========== =========



Long format
~~~~~~~~~~~~

This is now the preferred format, following the Paralex standard.

- each row represents an inflected form;
- three columns are necessary: lexeme identifier, form and cell. By convention, Qumin expects them to be called "lexeme", "phon_form" and "cell".
- the order of these three columns does not matter;
- All other columns are ignored

Example:

 =========== ========= =========
  lexeme       cell     phon_form
 =========== ========= =========
  peler       prs.1sg     p ɛ l
  peler       prs.2sg     p ɛ l
  peler       prs.3sg     p ɛ l
  peler       prs.1pl     p ə l ɔ̃
  peler       prs.2pl     p ə l e
  peler       prs.3pl     p ɛ l
  peler       ipfv.1sg    p ə l E
  peler       ipfv.2sg    p ə l E
  peler       ipfv.3sg    p ə l E
 =========== ========= =========


Overabundance
~~~~~~~~~~~~~

Inflectional paradigms sometimes have some overabundant forms, where the same lexeme and paradigm cell can be realized in various ways, as in “dreamed” vs “dreamt” for the English past of “to dream”.  Only some scripts can make use of this information, the other scripts will use the first value only.

In long format, overabundant forms give rise to two (or more) distinct rows, keyed by the same lexeme and cell. Example:

 =========== ========= ===========
   lexeme      cell      form
 =========== ========= ===========
   bind        ppart     b aˑɪ n d
   bind        ppart     b aˑʊ n d
   wind(air)   ppart     w aˑʊ n d
   wind(air)   ppart     w aˑɪ n d ɪ d
   weave       ppart     w əˑʊ v n̩
   weave       ppart     w iːv d
   slink       ppart     s l ʌ ŋ k
   slink       ppart     s l æ ŋ k
   slink       ppart     s l ɪ ŋ k t
 =========== ========= ===========

**deprecated:**: In wide format, concurrent forms can be written in the same cell, separated by “;”.


Defectivity
~~~~~~~~~~~

On the contrary, some lexemes might be defective for some cells, and have no values whatsoever for these cells.

In wide format, the most explicit way to indicate these missing values is to write “#DEF#” in the cell. The cell can also be left empty. Note that some scripts ignore all lines with defective values. Example:

 ========= ========== ============
  lexeme    form        cell
 ========= ========== ============
 advenir      #DEF#      prs.1sg
 advenir      #DEF#      prs.2sg
 advenir    a d v j ɛ̃    prs.3sg
 ========= ========== ============

