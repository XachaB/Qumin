




Alternation patterns
====================


We take inflectional behaviour to be represented by alternation
patterns. The biggest issue in finding a system of alternation patterns
from a paradigm lies in the implicit segmentation of forms : qualifying
the change between two forms requires to determine what are the constant
and the variable elements in each form.

We use two classes to represent patterns. The ``patterns.Pattern`` class
models an alternation pattern between a number of forms. Its
``patterns.BinaryPattern`` subclass specializes in binary alternation
patterns.

First, we load the ``Segments``:

.. code:: python 
    
    >>> from representations import patterns, segments, restore, alignment, generalize    
    >>> import pandas as pd    
    >>> import warnings    
    >>> warnings.filterwarnings('ignore')    
    >>> from IPython.display import HTML,Pretty    
    >>> ipy_HTML = HTML    
    >>> HTML = lambda x : ipy_HTML('<center>'+x+'</center>')    
    >>> features_file_name = "../Data/Vlexique/frenchipa.csv"    
    >>> segments.initialize(features_file_name,sep="\t")    
    >>> alignment.PHON_INS_COST = alignment.get_mean_cost()



.. parsed-literal::
    :class: output

    /home/sacha/.local/lib/python3.5/site-packages/matplotlib/\_\_init\_\_.py:913: UserWarning: axes.color\_cycle is deprecated and replaced with axes.prop\_cycle; please use the latter.
      warnings.warn(self.msg\_depr % (key, alt\_key))
    /home/sacha/These/qumin/bin/lattice/lattice.py:14: UserWarning: Warning: mpld3 could not be imported. No html export possible.
      warnings.warn("Warning: mpld3 could not be imported. No html export possible.")


.. parsed-literal::
    :class: output

    Reading table
    Aliasing multi-chars segments
    Normalizing identical segments



From alignment to pattern
-------------------------


We will play around with two present forms of french verbal paradigm
(note that below we note the two-character segment 'ɔ̃ ' as 'õ'):

.. code:: python 
    
    >>> cells  = [ 'prs.1.sg' , 'prs.2.sg' , 'prs.3.sg' , 'prs.1.pl'  , 'prs.2.pl'   , 'prs.3.pl']    
    >>> amener = [  'amEn'    ,   'amEn'   ,   'amEn'   ,    'amØnõ',   'amØnE'    ,    'amEn'    ]    
    >>> valeter= [  'valEt'   ,   'valEt'  ,   'valEt'  ,   'valØtõ',   'valØtE'   ,   'valEt'   ]



To find what is constant or alternating material in a serie of forms, we
first need to align the characters in the form. We provide four
alignment functions, which effects are depicted below on the alternation
amEn ~ amənõ. The function ``align_levenshtein`` and ``align_phono`` can
only align two forms together (they relies on edit distances) and return
a list of best alignments. The others can align an arbitrary number of
forms but return only one alignment.

.. code:: python 
    
    >>> print("align_left when we are looking for a suffix:\n\t",alignment.align_left(amener[0],amener[3]))    
    >>> print("align_right is ideal when we are looking for a prefix:\n\t",alignment.align_right(amener[0],amener[3]))    
    >>> print("align_baseline re-implements the strategy from Albright & Hayes (2002):\n\t ", alignment.align_baseline(amener[0],amener[3]))    
    >>> print("align_phono attempts to find the best alignment based on phonological similarity:\n\t" ,alignment.align_phono(amener[0],amener[3]))    
    >>> print("align_levenshtein attempts to find the best alignment based on levenshtein distances:\n\t" ,alignment.align_levenshtein(amener[0],amener[3]))



.. parsed-literal::
    :class: output

    align\_left when we are looking for a suffix:
    	 [('a', 'a'), ('m', 'm'), ('E', 'Ø'), ('n', 'n'), ('', 'õ')]
    align\_right is ideal when we are looking for a prefix:
    	 [('', 'a'), ('a', 'm'), ('m', 'Ø'), ('E', 'n'), ('n', 'õ')]
    align\_baseline re-implements the strategy from Albright & Hayes (2002):
    	  [('a', 'a'), ('m', 'm'), ('E', 'Ø'), ('n', 'n'), ('', 'õ')]
    align\_phono attempts to find the best alignment based on phonological similarity:
    	 [[('a', 'a'), ('m', 'm'), ('E', 'Ø'), ('n', 'n'), ('', 'õ')]]
    align\_levenshtein attempts to find the best alignment based on levenshtein distances:
    	 [[('a', 'a'), ('m', 'm'), ('E', ''), ('', 'Ø'), ('n', 'n'), ('', 'õ')], [('a', 'a'), ('m', 'm'), ('', 'Ø'), ('E', ''), ('n', 'n'), ('', 'õ')]]



By default, patterns are left aligned, which works for the current data.
They can take any number of forms:

.. code:: python 
    
    >>> p1 = patterns.Pattern(cells,amener)    
    >>> print(p1)    
    >>> p2 = patterns.Pattern(cells,valeter)    
    >>> print(p2)



.. parsed-literal::
    :class: output

    E\_ ⇌ E\_ ⇌ E\_ ⇌ Ø\_ɔ̃ ⇌ Ø\_E ⇌ E\_ / am\_n\_
    E\_ ⇌ E\_ ⇌ E\_ ⇌ Ø\_ɔ̃ ⇌ Ø\_E ⇌ E\_ / val\_t\_



We can also make a pattern from already aligned forms (in this
particular case, the result is the same):

.. code:: python 
    
    >>> p1 = patterns.Pattern(cells,alignment.align_baseline(*amener),aligned=True)    
    >>> print(p1)    
    >>> p2 = patterns.Pattern(cells,alignment.align_baseline(*valeter),aligned=True)    
    >>> print(p2)



.. parsed-literal::
    :class: output

    E\_ ⇌ E\_ ⇌ E\_ ⇌ Ø\_ɔ̃ ⇌ Ø\_E ⇌ E\_ / am\_n\_
    E\_ ⇌ E\_ ⇌ E\_ ⇌ Ø\_ɔ̃ ⇌ Ø\_E ⇌ E\_ / val\_t\_



A ``Pattern`` can also be represented as a list of alternating material,
i.e. forms where the constants have been replaced by "…". This is used
by the `clustering experiments <../doc/Clusteringipynb.html>`__.

.. code:: python 
    
    >>> p2.to_alt()


.. parsed-literal::
    :class: output 

        '…E… ⇌ …E… ⇌ …E… ⇌ …Ø…ɔ̃ ⇌ …Ø…E ⇌ …E…'

Binary alternation patterns
---------------------------


``BinaryPattern``\ s are ``Patterns`` over only two forms. Two
``Pattern``\ s with identical alternation but different contexts can be
merged by generalising the contexts. Applying the ``Pattern`` to one of
its forms produces the second forms.

.. code:: python 
    
    >>> cells = cells[2:4]    
    >>> forms = valeter[2:4]    
    >>> forms2 = amener[2:4]    
    >>> print(cells)    
    >>> print(forms)    
    >>> print(forms2)



.. parsed-literal::
    :class: output

    ['prs.3.sg', 'prs.1.pl']
    ['valEt', 'valØtõ']
    ['amEn', 'amØnõ']



Creating a ``Pattern`` over two forms always returns a
``BinaryPattern``:

.. code:: python 
    
    >>> r1 = patterns.Pattern(cells,forms)    
    >>> print(r1, " is of type ",type(r1))    
    >>> r2 = patterns.Pattern(cells,forms2)    
    >>> print(r2, " is of type ",type(r2))    
    >>> r1.lexemes = ["amener"]    
    >>> r2.lexemes = ["valeter"]



.. parsed-literal::
    :class: output

    E\_ ⇌ Ø\_ɔ̃ / val\_t\_  is of type  <class 'representations.patterns.BinaryPattern'>
    E\_ ⇌ Ø\_ɔ̃ / am\_n\_  is of type  <class 'representations.patterns.BinaryPattern'>



Generalizing contexts
~~~~~~~~~~~~~~~~~~~~~


Two rules with identical alternation can be combined by generalizing
their context. The context will now be encoded using natural classes and
quantifiers.

.. code:: python 
    
    >>> r = generalize.generalize_patterns([r1,r2])    
    >>> print(r)



.. parsed-literal::
    :class: output

    E\_ ⇌ Ø\_ɔ̃ / v?a[lmn]\_[dnt]\_



Given enough different rules with the same pattern, the generalization
becomes adequately generic:

.. code:: python 
    
    >>> formlist = [("pEl","pØlõ"),    
    ...          ("ʒEl","ʒØlõ"),    
    ...          ("pʁOmEn","pʁOmØnõ"),    
    ...          ("sEm","sØmõ"),    
    ...          ("lEv","lØvõ"),    
    ...          ("pEz","pØzõ"),    
    ...          ("sEvʁ","sØvʁõ")]    
    >>> pats = [r1,r2] + [patterns.Pattern(cells,forms) for forms in formlist]    
    >>> print("Merging patterns:")    
    >>> for p in pats:    
    ...     print(p)    
    >>> print()    
    >>> r = generalize.generalize_patterns(pats)    
    ...         
    >>> print("Generalized rule:",r)



.. parsed-literal::
    :class: output

    Merging patterns:
    E\_ ⇌ Ø\_ɔ̃ / val\_t\_
    E\_ ⇌ Ø\_ɔ̃ / am\_n\_
    E\_ ⇌ Ø\_ɔ̃ / p\_l\_
    E\_ ⇌ Ø\_ɔ̃ / ʒ\_l\_
    E\_ ⇌ Ø\_ɔ̃ / pʁOm\_n\_
    E\_ ⇌ Ø\_ɔ̃ / s\_m\_
    E\_ ⇌ Ø\_ɔ̃ / l\_v\_
    E\_ ⇌ Ø\_ɔ̃ / p\_z\_
    E\_ ⇌ Ø\_ɔ̃ / s\_vʁ\_
    
    Generalized rule: E\_ ⇌ Ø\_ɔ̃ / [-nas -haut]*[bdflmnpstvzɲʃʒ]\_[bdflmnpstvzʁ]+\_



The function ``str`` converts the pattern to a format where the context
is expressed in the shortest way possible, whether natural classes of
segments have to be written as features ("[-nas]"), as chars
("[bdflmnpstvzɲʃʒ]") or as a shorthand ("C"). The ``repr`` function
provides a more explicit but longer representation, always using chars.
One can re-create a Pattern object from this string:

.. code:: python 
    
    >>> repr_pat = repr(r)    
    >>> print("repr:",repr_pat)    
    >>> print("str:",r)    
    >>> print("imported back: ",patterns.BinaryPattern._from_str(cells,repr_pat))



.. parsed-literal::
    :class: output

    repr: E\_ ⇌ Ø\_ɔ̃ / [EOabdflpstvzØʁ]*[bdflmnpstvzɲʃʒ]\_[bdflmnpstvzʁ]+\_ <0>
    str: E\_ ⇌ Ø\_ɔ̃ / [-nas -haut]*[bdflmnpstvzɲʃʒ]\_[bdflmnpstvzʁ]+\_
    imported back:  E\_ ⇌ Ø\_ / [-nas -haut]*[bdflmnpstvzɲʃʒ]\_[bdflmnpstvzʁ]+\_



Applying patterns
~~~~~~~~~~~~~~~~~


A pattern can be applied to a string to produce the other cell:

.. code:: python 
    
    >>> forms = ["amEn","mEn","pEl","valEt"]    
    
    >>> for form in forms:    
    ...     computed_form = restore(r.apply(form,names=r.cells))    
    ...     print("Applying pattern '{}' to {} produces {}".format(r,form,computed_form))



.. parsed-literal::
    :class: output

    Applying pattern 'E\_ ⇌ Ø\_ɔ̃ / [-nas -haut]*[bdflmnpstvzɲʃʒ]\_[bdflmnpstvzʁ]+\_' to amEn produces amØnõ
    Applying pattern 'E\_ ⇌ Ø\_ɔ̃ / [-nas -haut]*[bdflmnpstvzɲʃʒ]\_[bdflmnpstvzʁ]+\_' to mEn produces mØnõ
    Applying pattern 'E\_ ⇌ Ø\_ɔ̃ / [-nas -haut]*[bdflmnpstvzɲʃʒ]\_[bdflmnpstvzʁ]+\_' to pEl produces pØlõ
    Applying pattern 'E\_ ⇌ Ø\_ɔ̃ / [-nas -haut]*[bdflmnpstvzɲʃʒ]\_[bdflmnpstvzʁ]+\_' to valEt produces valØtõ



Of course, that is only if the pattern's context matches the form

.. code:: python 
    
    >>> verb = "mange"    
    >>> cell = "prs.3.sg"    
    >>> restore(r.apply("mange",names=r.cells))



::


    ---------------------------------------------------------------------------

    NotApplicable                             Traceback (most recent call last)

    <ipython-input-13-169f29b66ee2> in <module>()
          1 verb = "mange"
          2 cell = "prs.3.sg"
    ----> 3 restore(r.apply("mange",names=r.cells))
    

    /home/sacha/These/qumin/bin/representations/patterns.py in apply(self, form, names)
        575             raise NotApplicable("The context {}"
        576                                 "doesn't match the form \"{}\""
    --> 577                                 "".format(self._regex[from_cell], form))
        578         return string
        579 


    NotApplicable: The context re.compile('([EOabdflpstvzØʁ]*[bdflmnpstvzɲʃʒ])(E)([bdflmnpstvzʁ]+)()')doesn't match the form "mange"



We can check wether a pattern is applicable :

.. code:: python 
    
    >>> verb = "mange"    
    >>> cell = "prs.3.sg"    
    >>> print("Is {} applicable to '{}' from the cell '{}' ? {}".format(r,verb,cell,r.applicable(verb,cell)))    
    >>> verb = "pEl"    
    >>> print("Is {} applicable to '{}' from the cell '{}' ? {}".format(r,verb,cell,r.applicable(verb,cell)))



.. parsed-literal::
    :class: output

    Is E\_ ⇌ Ø\_ɔ̃ / [-nas -haut]*[bdflmnpstvzɲʃʒ]\_[bdflmnpstvzʁ]+\_ applicable to 'mange' from the cell 'prs.3.sg' ? False
    Is E\_ ⇌ Ø\_ɔ̃ / [-nas -haut]*[bdflmnpstvzɲʃʒ]\_[bdflmnpstvzʁ]+\_ applicable to 'pEl' from the cell 'prs.3.sg' ? True



Generalizing an alternation
---------------------------


A pattern's alternation can also be expressed in more general ways :

.. code:: python 
    
    >>> cells = ('prs.1sg', 'pst.1sg')    
    >>> formlist = [("admE","admi"), ("EmØ","Emy"), ("pØ","py"), ("plØ","ply")]    
    
    >>> pats = [patterns.Pattern(cells,forms) for forms in formlist]    
    >>> print("These patterns appear to have varying alternations :")    
    >>> for p in pats:    
    ...     print(p)    
    >>> print("\nBut the change is similar, and we can merge them")    
    >>> r = generalize.generalize_patterns(pats)    
    >>> print(r)



.. parsed-literal::
    :class: output

    These patterns appear to have varying alternations :
    E ⇌ i / adm\_
    Ø ⇌ y / Em\_
    Ø ⇌ y / p\_
    Ø ⇌ y / pl\_
    
    But the change is similar, and we can merge them
    [EOØ] ⇌ [iuy] / [-nas -haut]*[+ant]\_



This relies on a "transformation" function in ``Segment`` which finds
pairs or segment with analogical relations :

.. code:: python 
    
    >>> print(segments.Segment.transformation("E","i"))    
    >>> print(segments.Segment.transformation("Ø","y"))



.. parsed-literal::
    :class: output

    ('EOØ', 'iuy')
    ('EOØ', 'iuy')



Finding all patterns in a paradigm
----------------------------------


The ``pattern`` module provides functions to compute all relevant
patterns in a paradigm:

Four functions share the same interface (see the
`API <../doc/representations.html#representations.patterns.find_auto_patterns>`__).

-  ``patterns.find_levenshtein_patterns`` and
   ``patterns.find_phonsim_patterns``: All best alignments of forms are
   considered, according to either levenshtein distances or phonological
   similarity. The resulting competing patterns are chosen according to
   their coverage and accuracy. Patterns are generalized both in
   contexts and alternations as much as possible.
-  ``patterns.find_suffixal_patterns``,
   ``patterns.find_prefixal_patterns`` and
   ``patterns.find_baseline_patterns``: These functions differ regarding
   their alignment of forms. They generalize the contexts two by two
   incrementally but never the alternation

.. code:: python 
    
    >>> paradigms = pd.DataFrame([["amEn", "amØnõ", "amØnE"],    
    ...                          ["mãʒ","mãʒõ","mãʒE"],    
    ...                          ["ʒEl","ʒØlõ","ʒØlE"],    
    ...                          ["mõtʁ","mõtʁõ","mõtʁE"],    
    ...                          ["fini","finisõ","finisE"],    
    ...                          ["ãsEɲ","ãsEɲõ","ãsEɲE"]],    
    ...              columns=["prs.3.sg", "prs.1.pl", "prs.2.pl"],    
    ...              index=["amener","manger","geler","montrer","finir","enseigner"])    
    
    >>> HTML(paradigms.to_html())


.. raw:: html 

    <center><table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>prs.3.sg</th>
          <th>prs.1.pl</th>
          <th>prs.2.pl</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>amener</th>
          <td>amEn</td>
          <td>amØnõ</td>
          <td>amØnE</td>
        </tr>
        <tr>
          <th>manger</th>
          <td>mãʒ</td>
          <td>mãʒõ</td>
          <td>mãʒE</td>
        </tr>
        <tr>
          <th>geler</th>
          <td>ʒEl</td>
          <td>ʒØlõ</td>
          <td>ʒØlE</td>
        </tr>
        <tr>
          <th>montrer</th>
          <td>mõtʁ</td>
          <td>mõtʁõ</td>
          <td>mõtʁE</td>
        </tr>
        <tr>
          <th>finir</th>
          <td>fini</td>
          <td>finisõ</td>
          <td>finisE</td>
        </tr>
        <tr>
          <th>enseigner</th>
          <td>ãsEɲ</td>
          <td>ãsEɲõ</td>
          <td>ãsEɲE</td>
        </tr>
      </tbody>
    </table></center>
.. code:: python 
    
    >>> p,d = patterns.find_levenshtein_patterns(paradigms)    
    >>> HTML(p.to_html())



.. parsed-literal::
    :class: output

    ▕██████████████████████████████████████████████████▏100% (3 of 3) complete
    

.. raw:: html 

    <center><table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>(prs.3.sg, prs.1.pl)</th>
          <th>(prs.3.sg, prs.2.pl)</th>
          <th>(prs.1.pl, prs.2.pl)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>amener</th>
          <td>E_ ⇌ Ø_ɔ̃ / a?[lmnvzɲʒ]_[ln]_</td>
          <td>E_ ⇌ Ø_E / a?[lmnvzɲʒ]_[ln]_</td>
          <td>ɔ̃ ⇌ E / [+cont]+X[flmnsvzŋɲʁʃʒ]_</td>
        </tr>
        <tr>
          <th>manger</th>
          <td>⇌ ɔ̃ / [mnɑ̃ɔ̃œ̃ɛ̃]?[+cont -haut][-haut][ŋɲʁʒ]_</td>
          <td>⇌ E / [mnɑ̃ɔ̃œ̃ɛ̃]?[+cont -haut][-haut][ŋɲʁʒ]_</td>
          <td>ɔ̃ ⇌ E / [+cont]+X[flmnsvzŋɲʁʃʒ]_</td>
        </tr>
        <tr>
          <th>geler</th>
          <td>E_ ⇌ Ø_ɔ̃ / a?[lmnvzɲʒ]_[ln]_</td>
          <td>E_ ⇌ Ø_E / a?[lmnvzɲʒ]_[ln]_</td>
          <td>ɔ̃ ⇌ E / [+cont]+X[flmnsvzŋɲʁʃʒ]_</td>
        </tr>
        <tr>
          <th>montrer</th>
          <td>⇌ ɔ̃ / [mnɑ̃ɔ̃œ̃ɛ̃]?[+cont -haut][-haut][ŋɲʁʒ]_</td>
          <td>⇌ E / [mnɑ̃ɔ̃œ̃ɛ̃]?[+cont -haut][-haut][ŋɲʁʒ]_</td>
          <td>ɔ̃ ⇌ E / [+cont]+X[flmnsvzŋɲʁʃʒ]_</td>
        </tr>
        <tr>
          <th>finir</th>
          <td>⇌ sɔ̃ / fini_</td>
          <td>⇌ sE / fini_</td>
          <td>ɔ̃ ⇌ E / [+cont]+X[flmnsvzŋɲʁʃʒ]_</td>
        </tr>
        <tr>
          <th>enseigner</th>
          <td>⇌ ɔ̃ / [mnɑ̃ɔ̃œ̃ɛ̃]?[+cont -haut][-haut][ŋɲʁʒ]_</td>
          <td>⇌ E / [mnɑ̃ɔ̃œ̃ɛ̃]?[+cont -haut][-haut][ŋɲʁʒ]_</td>
          <td>ɔ̃ ⇌ E / [+cont]+X[flmnsvzŋɲʁʃʒ]_</td>
        </tr>
      </tbody>
    </table></center>

Four other functions are also available to find alternations:


-  ``patterns.find_endings`` finds endings by removing the common prefix
   to each row. It does not use ``Patterns`` and returns a DataFrame of
   strings. ``patterns.find_endings_pairs`` does the same thing but
   gathers the endings two by two so the output has one column for each
   pairs on the paradigm's columns and looks like alternation patterns.
-  ``patterns.find_global_alternations_pairs`` uses ``Pattern`` with all
   the forms of each row as an input, then distributes the alternation
   in pairs. It differs from ``find_endings`` and can find any number of
   changes in left aligned forms.
-  ``patterns.find_local_alternations_pairs`` does the same but with
   pairs of forms as its input. It is similar to
   ``find_suffixal_patterns`` followed by extracting just the
   alternation string from each pattern.

.. code:: python 
    
    >>> p = patterns.find_endings(paradigms)    
    >>> HTML(p.to_html())



.. parsed-literal::
    :class: output

    ▕██████████████████████████████████████████████████▏100% (6 of 6) complete
    

.. raw:: html 

    <center><table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>prs.3.sg</th>
          <th>prs.1.pl</th>
          <th>prs.2.pl</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>amener</th>
          <td>En</td>
          <td>Ønõ</td>
          <td>ØnE</td>
        </tr>
        <tr>
          <th>manger</th>
          <td></td>
          <td>õ</td>
          <td>E</td>
        </tr>
        <tr>
          <th>geler</th>
          <td>El</td>
          <td>Ølõ</td>
          <td>ØlE</td>
        </tr>
        <tr>
          <th>montrer</th>
          <td></td>
          <td>õ</td>
          <td>E</td>
        </tr>
        <tr>
          <th>finir</th>
          <td></td>
          <td>sõ</td>
          <td>sE</td>
        </tr>
        <tr>
          <th>enseigner</th>
          <td></td>
          <td>õ</td>
          <td>E</td>
        </tr>
      </tbody>
    </table></center>

-  `Back to the documentation's index <../doc/index.html>`__
