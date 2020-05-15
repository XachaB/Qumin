The morphological paradigms file
=================================

This file relates phonological forms to their lexemes and paradigm cells. As an example of valid data, Qumin is shipped with a paradigm table from the French inflectional lexicon `Flexique <http://www.llf.cnrs.fr/fr/flexique-fr.php>`__. Here is a sample of the first 10 columns for 10 randomly picked verbs from Flexique:

 =========== ================ ========= ========= ========= ========== ========== ========= ========== ========== ========== 
  lexeme      variants         prs.1sg   prs.2sg   prs.3sg   prs.1pl    prs.2pl    prs.3pl   ipfv.1sg   ipfv.2sg   ipfv.3sg  
 =========== ================ ========= ========= ========= ========== ========== ========= ========== ========== ========== 
  peler       peler            pɛl       pɛl       pɛl       pəlɔ̃       pəle       pɛl       pəlE       pəlE       pəlE      
  soudoyer    soudoyer         sudwa     sudwa     sudwa     sudwajɔ̃    sudwaje    sudwa     sudwajE    sudwajE    sudwajE   
  inféoder    inféoder         ɛ̃fEɔd     ɛ̃fEɔd     ɛ̃fEɔd     ɛ̃fEOdɔ̃     ɛ̃fEOde     ɛ̃fEɔd     ɛ̃fEOdE     ɛ̃fEOdE     ɛ̃fEOdE    
  débiller    débiller         dEbij     dEbij     dEbij     dEbijɔ̃     dEbije     dEbij     dEbijE     dEbijE     dEbijE    
  désigner    désigner         dEziɲ     dEziɲ     dEziɲ     dEziɲɔ̃     dEziɲe     dEziɲ     dEziɲE     dEziɲE     dEziɲE    
  crachoter   crachoter        kʁaʃɔt    kʁaʃɔt    kʁaʃɔt    kʁaʃOtɔ̃    kʁaʃOte    kʁaʃɔt    kʁaʃOtE    kʁaʃOtE    kʁaʃOtE   
  saouler     saouler:soûler   sul       sul       sul       sulɔ̃       sule       sul       sulE       sulE       sulE      
  caserner    caserner         kazɛʁn    kazɛʁn    kazɛʁn    kazɛʁnɔ̃    kazɛʁne    kazɛʁn    kazɛʁnE    kazɛʁnE    kazɛʁnE   
  parrainer   parrainer        paʁɛn     paʁɛn     paʁɛn     paʁEnɔ̃     paʁEne     paʁɛn     paʁEnE     paʁEnE     paʁEnE    
  souscrire   souscrire        suskʁi    suskʁi    suskʁi    suskʁivɔ̃   suskʁive   suskʁiv   suskʁivE   suskʁivE   suskʁivE  
 =========== ================ ========= ========= ========= ========== ========== ========= ========== ========== ========== 

Paradigm files are written in `wide format <https://en.wikipedia.org/wiki/Wide_and_narrow_data>`__:

-  each row represents a lexeme, and each column represents a cell.
-  The first column indicates a unique identifier for each lexeme. It is usually convenient to use orthographic citation forms for this purpose (e.g. infinitive for verbs).
-  In Vlexique, there is a second column with orthographic variants for lexeme names, which is called “variants”. You do not need to add a “variant” column, and if it is there, it will be ignored.
-  the very first row indicates the names of the cells as column headers. Columns headers shouldn’t contain the character “#”.

While Qumin assumes that inflected forms are written in some phonemic notation (we suggest to be as close to the IPA as possible), you do not need to explicitely segment them into phonemes in the paradigms file.

The file itself is a ``csv``, meaning that the values are written as plain text, in utf-8 format, separated by spaces. This format can be read by spreadsheet programs as well as programmatically:

.. code:: sh

   %%sh
   head -n 3 "../Data/Vlexique/vlexique-20171031.csv"

::

   lexeme,variants,prs.1sg,prs.2sg,prs.3sg,prs.1pl,prs.2pl,prs.3pl,ipfv.1sg,ipfv.2sg,ipfv.3sg,ipfv.1pl,ipfv.2pl,ipfv.3pl,fut.1sg,fut.2sg,fut.3sg,fut.1pl,fut.2pl,fut.3pl,cond.1sg,cond.2sg,cond.3sg,cond.1pl,cond.2pl,cond.3pl,sbjv.1sg,sbjv.2sg,sbjv.3sg,sbjv.1pl,sbjv.2pl,sbjv.3pl,pst.1sg,pst.2sg,pst.3sg,pst.1pl,pst.2pl,pst.3pl,pst.sbjv.1sg,pst.sbjv.2sg,pst.sbjv.3sg,pst.sbjv.1pl,pst.sbjv.2pl,pst.sbjv.3pl,imp.2sg,imp.1pl,imp.2pl,inf,prs.ptcp,pst.ptcp.m.sg,pst.ptcp.m.pl,pst.ptcp.f.sg,pst.ptcp.f.pl
   accroire,accroire,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#,akʁwaʁ,#DEF#,#DEF#,#DEF#,#DEF#,#DEF#
   advenir,advenir,#DEF#,#DEF#,advjɛ̃,#DEF#,#DEF#,advjɛn,#DEF#,#DEF#,advənE,#DEF#,#DEF#,advənE,#DEF#,#DEF#,advjɛ̃dʁa,#DEF#,#DEF#,advjɛ̃dʁɔ̃,#DEF#,#DEF#,advjɛ̃dʁE,#DEF#,#DEF#,advjɛ̃dʁE,#DEF#,#DEF#,advjɛn,#DEF#,#DEF#,advjɛn,#DEF#,#DEF#,advɛ̃,#DEF#,#DEF#,advɛ̃ʁ,#DEF#,#DEF#,advɛ̃,#DEF#,#DEF#,advɛ̃s,#DEF#,#DEF#,#DEF#,advəniʁ,advənɑ̃,advəny,advəny,advəny,advəny

Overabundance
~~~~~~~~~~~~~

Inflectional paradigms sometimes have some overabundant forms, where the same lexeme and paradigm cell can be realized in various ways, as in “dreamed” vs “dreamt” for the English past of “to dream”. Concurrent forms can be written in the same cell, separated by “;”. Only some scripts can make use of this information, the other scripts will use the first value only. Here is an example for English verbs:

 =========== ==================== ========= ========== ======== ======== ============ ==================== ==================== 
  lexeme      ppart                pres3s    prespart   inf      pres1s   presothers   past13               pastnot13           
 =========== ==================== ========= ========== ======== ======== ============ ==================== ==================== 
  bind        baˑɪnd;baˑʊnd        baˑɪndz   baˑɪndɪŋ   baˑɪnd   baˑɪnd   baˑɪnd       baˑɪnd;baˑʊnd        baˑɪnd;baˑʊnd       
  wind(air)   waˑʊnd;waˑɪndɪd      waˑɪndz   waˑɪndɪŋ   waˑɪnd   waˑɪnd   waˑɪnd       waˑʊnd;waˑɪndɪd      waˑʊnd;waˑɪndɪd     
  weave       wəˑʊvn̩;wiːvd         wiːvz     wiːvɪŋ     wiːv     wiːv     wiːv         wəˑʊv;wiːvd          wəˑʊv;wiːvd         
  slink       slʌŋk;slæŋk;slɪŋkt   slɪŋks    slɪŋkɪŋ    slɪŋk    slɪŋk    slɪŋk        slʌŋk;slæŋk;slɪŋkt   slʌŋk;slæŋk;slɪŋkt  
  dream       driːmd               driːmz    driːmɪŋ    driːm    driːm    driːm        driːmd;drɛmt         driːmd;drɛmt     
 =========== ==================== ========= ========== ======== ======== ============ ==================== ====================    

Defectivity
~~~~~~~~~~~

On the contrary, some lexemes might be defective for some cells, and have no values whatsoever for these cells. The most explicit way to indicate these missing values is to write “#DEF#” in the cell. The cell can also be left empty. Note that some scripts ignore all lines with defective values.

Here are some examples from French verbs:

 ============== ========= ========= ========= ========= ========= ========= ========== ==========  
  lexeme         prs.1sg   prs.2sg   prs.3sg   prs.1pl   prs.2pl   prs.3pl   ipfv.1sg   ipfv.2sg  
 ============== ========= ========= ========= ========= ========= ========= ========== ========== 
  accroire       #DEF#     #DEF#     #DEF#     #DEF#     #DEF#     #DEF#     #DEF#      #DEF#     
  advenir        #DEF#     #DEF#     advjɛ̃     #DEF#     #DEF#     advjɛn    #DEF#      #DEF#     
  ardre          #DEF#     #DEF#     #DEF#     #DEF#     #DEF#     #DEF#     aʁdE       aʁdE      
  braire         #DEF#     #DEF#     bʁE       #DEF#     #DEF#     bʁE       #DEF#      #DEF#     
  chaloir        #DEF#     #DEF#     ʃo        #DEF#     #DEF#     #DEF#     #DEF#      #DEF#     
  comparoir      #DEF#     #DEF#     #DEF#     #DEF#     #DEF#     #DEF#     #DEF#      #DEF#     
  discontinuer   #DEF#     #DEF#     #DEF#     #DEF#     #DEF#     #DEF#     #DEF#      #DEF#     
  douer          #DEF#     #DEF#     #DEF#     #DEF#     #DEF#     #DEF#     #DEF#      #DEF#     
  échoir         #DEF#     #DEF#     eʃwa      #DEF#     #DEF#     #DEF#     #DEF#      #DEF#     
  endêver        #DEF#     #DEF#     #DEF#     #DEF#     #DEF#     #DEF#     #DEF#      #DEF#     
 ============== ========= ========= ========= ========= ========= ========= ========== ========== 
