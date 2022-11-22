Quickstart
==========

Installation
------------

First, you need to install *pyndl*. The easiest way to do this is using
`pip <https://pip.pypa.io/en/stable/>`_:

.. code:: bash

    pip install --user pyndl


.. warning::

    If you are using any other operating system than Linux this process can be
    more difficult. Check out :doc:`installation` for more detailed installation
    instruction.
    However, currently we can only ensure the expected behaviour on Linux
    system. Be aware that on other operating system some functionality may not
    work


Naive Discriminative Learning
-----------------------------

Naive Discriminative Learning, henceforth NDL, is an incremental learning
algorithm based on the learning rule of Rescorla and Wagner [1]_, which
describes the learning of direct associations between cues and outcomes.
The learning is thereby structured in events where each event consists of a
set of cues which give hints to outcomes. Outcomes can be seen as the result of
an event, where each outcome can be either present or absent. NDL is naive in
the sense that cue-outcome associations are estimated separately for each
outcome.

The Rescorla-Wagner learning rule describes how the association strength
:math:`\Delta V_{i}^{t}` at time :math:`t` changes over time. Time is here
described in form of learning events. For each event the association strength
is updated as

.. math::

    V_{i}^{t+1} = V_{i}^{t} + \Delta V_{i}^{t}

Thereby, the change in association strength :math:`\Delta V_{i}^{t}` is defined
as

.. math::

   \Delta V_{i}^{t} =
   \begin{array}{ll}
   \begin{cases}
   \displaystyle 0 & \: \textrm{if ABSENT}(C_{i}, t)\\ \alpha_{i}\beta_{1} \:
   (\lambda - \sum_{\textrm{PRESENT}(C_{j}, t)} \: V_{j}) & \:
   \textrm{if PRESENT}(C_{j}, t) \: \& \: \textrm{PRESENT}(O, t)\\
   \alpha_{i}\beta_{2} \: (0 - \sum_{\textrm{PRESENT}(C_{j}, t)} \: V_{j}) & \:
   \textrm{if PRESENT}(C_{j}, t) \: \& \: \textrm{ABSENT}(O, t)
   \end{cases}
   \end{array}

with

  * :math:`\alpha_{i}` being the salience of the cue :math:`i`
  * :math:`\beta_{1}` being the salience of the situation in which the outcome
    occurs
  * :math:`\beta_{2}` being the salience of the situation in which the outcome
    does not occur
  * :math:`\lambda` being the the maximum level of associative strength possible

.. note::

    Usually, the parameters are set to :math:`\alpha_{i} = \alpha_{j} \:
    \forall i, j`, :math:`\beta_{1} = \beta_{2}` and :math:`\lambda = 1`


Usage
-----

Analyzing data with *pyndl* involves three steps

    1. The data has to be preprocessed into the correct format
    2. One of the learning methods of *pyndl* is used to learn the desired associations
    3. The learned association (commonly also called weights) can be stored or directly 
       be analyzed further.

In the following, a usage example of *pyndl* is provided, in which the first two of the 
three steps are described for learning the associations between bigrams and meanings. The 
first section of this example focuses on the correct preparation of the data with inbuilt 
methods. However, it is worth to note that the learning algorithm itself does not require 
the data to be preprocessed by *pyndl*, nor it is limited by that. The 
:py:mod:`pyndl.preprocess` module should rather be seen as a collection of established and 
commonly used preprocessing methods within the context of NDL. Custom preprocessing can 
be used as long as the created event files follow the structure as outlined in the next
section. The second section, describes how the associations can be learned using *pyndl*,
while the last section describes how this can be exported and, for instance, loaded in R 
for further investigation.

Data Preparation
````````````````

To analyse any data using *pyndl* requires them to be in the long format as an
utf-8 encoded tab delimited gzipped text file with a header in the first line
and two columns:

1. the first column contains an underscore delimited list of all cues
2. the second column contains an underscore delimited list of all outcomes
3. each line therefore represents an event with a pair of a cue and an outcome
   (occurring one time)
4. the events (lines) are ordered chronologically

The algorithm itself is agnostic to the actual domain as long as the data is tokenized 
as Unicode character strings. While *pyndl* provides some basic preprocessing for grapheme
tokenization (see for instance the following examples), the tokenization of ideograms, 
pictograms, logograms, and speech has to be implemented manually. However, generic 
implementations are welcome as a contribution.


Creating Grapheme Clusters From Wide Format Data
''''''''''''''''''''''''''''''''''''''''''''''''

Often data which should be analysed is not in the right format to be processed
with *pyndl*. To illustrate how to get the data in the right format we use data
from Baayen, Milin, Đurđević, Hendrix & Marelli [2]_ as an example:

..  include:: lexicon.rst

Table 1 shows some words, their frequencies of occurrence and their meanings as
an artificial lexicon in the wide format. In the following, the letters
(unigrams and bigrams) of the words constitute the cues, whereas the meanings
represent the outcomes.

As the data in table 1 are artificial we can generate such a file for this
example by expanding table 1 randomly regarding the frequency of occurrence of
each event. The resulting event file `lexample.tab.gz`_
consists of 420 lines (419 = sum of frequencies + 1 header) and looks like the
following (nevertheless you are encouraged to take a closer look at this file
using any text editor of your choice):

=================  =============
Cues               Outcomes
=================  =============
#h_ha_an_nd_ds_s#  hand_plural
#l_la_ad_d#        lad
#l_la_as_ss_s#     lass
=================  =============


Creating Grapheme Clusters From Corpus Data
'''''''''''''''''''''''''''''''''''''''''''

Often the corpus which should be analysed is only a raw utf-8 encoded text file
that contains huge amounts of text. From here on we will refer to such a file
as a corpus file. In the corpus files several documents can be stored with  a
``---end.of.document---`` or ``---END.OF.DOCUMENT---`` string marking
where an old document finished and a new document starts.

The :py:mod:`pyndl.preprocess` module (besides other things)
provides the functionality to directly generate an event file based on a raw
corpus file and filter it:

.. code-block:: python

    >>> from pyndl import preprocess
    >>> preprocess.create_event_file(corpus_file='docs/data/lcorpus.txt',
    ...                              event_file='docs/data/levent.tab.gz',
    ...                              allowed_symbols='a-zA-Z',
    ...                              context_structure='document',
    ...                              event_structure='consecutive_words',
    ...                              event_options=(1, ),
    ...                              cue_structure='bigrams_to_word')

Here we use the example corpus `lcorpus.txt`_ to
produce an event file ``levent.tab.gz`` which (uncompressed) looks like this:

=================  ========
Cues               Outcomes
=================  ========
an_#h_ha_d#_nd     hand
ot_fo_oo_#f_t#     foot
ds_s#_an_#h_ha_nd  hands
=================  ========

.. note::

    :py:mod:`pyndl.corpus` allows you to generate such a corpus file from a
    bunch of gunzipped xml subtitle files filled with words.


Learn the associations
``````````````````````

The strength of the associations for the data can now easily be computed using
the :py:mod:`pyndl.ndl.ndl` function from the :py:mod:`pyndl.ndl` module:

.. code-block:: python

   >>> from pyndl import ndl
   >>> weights = ndl.ndl(events='docs/data/levent.tab.gz',
   ...                   alpha=0.1, betas=(0.1, 0.1), method="threading")


Save and load a weight matrix
`````````````````````````````

To save time in the future, we recommend saving the weights. For compatibility
reasons we recommend saving the weight matrix in the netCDF format [3]_:

.. code-block:: python

    >>> weights.to_netcdf('docs/data/weights.nc')  # doctest: +SKIP

Now, the saved weights can later be reused or be analysed in Python or R. In
Python the weights can simply be loaded with the `xarray module
<http://xarray.pydata.org/en/stable/>`_:

.. code-block:: python

    >>> import xarray  # doctest: +SKIP
    >>> with xarray.open_dataarray('docs/data/weights.nc') as weights_read:  # doctest: +SKIP
    ...     weights_read

In R you need the `ncdf4 package <https://cran.r-project.org/package=ncdf4>`_
to load a in netCDF format saved matrix:

.. code-block:: R

   > #install.packages("ncdf4") # uncomment to install
   > library(ncdf4)
   > weights_nc <- nc_open(filename = "docs/data/weights.nc")
   > weights_read <- t(as.matrix(ncvar_get(nc = weights_nc, varid = "__xarray_dataarray_variable__")))
   > rownames(weights_read) <- ncvar_get(nc = weights_nc, varid = "outcomes")
   > colnames(weights_read) <- ncvar_get(nc = weights_nc, varid = "cues")
   > nc_close(nc = weights_nc)
   > rm(weights_nc)

Clean up
````````

In order to keep everything clean we might want to remove all the files we
created in this tutorial:

.. code-block:: python

  >>> import os
  >>> os.remove('docs/data/levent.tab.gz')


.. _lexample.tab.gz:
     https://github.com/quantling/pyndl/blob/main/docs/data/lexample.tab.gz

.. _lcorpus.txt:
     https://github.com/quantling/pyndl/blob/main/docs/data/lcorpus.txt

----

.. [1] Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian
       conditioning: Variations in the effectiveness of reinforcement and
       non-reinforcement. *Classical conditioning II: Current research and
       theory*, 2, 64-99.

.. [2] Baayen, R. H., Milin, P., Đurđević, D. F., Hendrix, P., & Marelli, M.
       (2011). An amorphous model for morphological processing in visual
       comprehension based on naive discriminative learning.
       *Psychological review*, 118(3), 438.

.. [3] Unidata (2012). NetCDF. doi:10.5065/D6H70CW6. Retrieved from
       http://doi.org/10.5065/D6RN35XM)
