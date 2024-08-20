The morphological paradigms file
=================================

This file relates phonological forms to their lexemes and paradigm cells.

.. warning::
    The `wide format <https://en.wikipedia.org/wiki/Wide_and_narrow_data>`_, or Plat, is no longer supported by qumin.
    In wide format,  there is one row per lexeme, and forms are given in successive columns. The entire paradigms fits on the row.

The format is described by the Paralex standard (see `Paralex datasets <https://www.paralex-standard.org>`_).
The data files are expected in ``csv`` format, meaning that the values are written as plain text, in utf-8 format, separated by commas. This format can be read by spreadsheet programs as well as programmatically.

Transcriptions
~~~~~~~~~~~~~~

Qumin assumes transcribed phonemic or phonetic forms (rather than orthographic), where each phoneme is ideally separated by spaces. Using non-segmented strings (with no spaces) is still supported, but deprecated, as it can lead to ambiguities.

Long format
~~~~~~~~~~~~

- each row represents an inflected form;
- three columns are necessary for Qumin: `lexeme`, `phon_form` and `cell`.
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

