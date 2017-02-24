========
Examples
========

--------------------------------------------------------------------------------------------------
Lexical example data illustrating the Rescorla-Wagner[@rescorlawagner1972] equations [@baayen2011]
--------------------------------------------------------------------------------------------------

Premises
========
1. Cues are associated with outcomes and both can be present or absent
2. Cues are segment (letter) unigrams, bigrams, ...
3. Outcomes are meanings (word meanings, inflectional meanings, affixal
   meanings), ...
4. :math:`\textrm{PRESENT}(X, t)` denotes the presence of cue or outcome
   :math:`X` at time :math:`t`
5. :math:`\textrm{ABSENT}(X, t)` denotes the absence of cue or outcome
   :math:`X` at time :math:`t`
6. The association strength :math:`V_{i}^{t+1}` of cue :math:`C_{i}` with
   outcome :math:`O` at time :math:`t+1` is defined as :math:`V_{i}^{t+1} =
   V_{i}^{t} + \Delta V_{i}^{t}`
7. The change in association strength :math:`\Delta V_{i}^{t}` is defined as in
   :eq:`rw` with

   * :math:`\alpha_{i}` being the salience of the cue :math:`i`
   * :math:`\beta_{1}` being the salience of the situation in which the outcome occurs
   * :math:`\beta_{2}` being the salience of the situation in which the outcome does not occur
   * :math:`\lambda` being the the maximum level of associative strength possible

8. Default settings for the parameters are: :math:`\alpha_{i} = \alpha_{j} \:
   \forall i, j`, :math:`\beta_{1} = \beta_{2}` and :math:`\lambda = 1`

.. math::

    \Delta V_{i}^{t} =
    \begin{array}{ll}
    \begin{cases}
    \displaystyle 0 & \: \textrm{if ABSENT}(C_{i}, t)\\ \alpha_{i}\beta_{1} \: (\lambda - \sum_{\textrm{PRESENT}(C_{j}, t)} \: V_{j}) & \: \textrm{if PRESENT}(C_{j}, t) \: \& \: \textrm{PRESENT}(O, t)\\ \alpha_{i}\beta_{2} \: (0 - \sum_{\textrm{PRESENT}(C_{j}, t)} \: V_{j}) & \: \textrm{if PRESENT}(C_{j}, t) \: \& \: \textrm{ABSENT}(O, t)
    \end{cases}
    \end{array}
    :label: rw

Data
====

+-----------------+-----------------+-----------------+-----------------+
| Table 1                                                               |
+-----------------+-----------------+-----------------+-----------------+
| Word            | Frequency       | Lexical Meaning | Number          |
+=================+=================+=================+=================+
| hand            | 10              | HAND            |                 |
+-----------------+-----------------+-----------------+-----------------+
| hands           | 20              | HAND            | PLURAL          |
+-----------------+-----------------+-----------------+-----------------+
| land            | 8               | LAND            |                 |
+-----------------+-----------------+-----------------+-----------------+
| lands           | 3               | LAND            | PLURAL          |
+-----------------+-----------------+-----------------+-----------------+
| and             | 35              | AND             |                 |
+-----------------+-----------------+-----------------+-----------------+
| sad             | 18              | SAD             |                 |
+-----------------+-----------------+-----------------+-----------------+
| as              | 35              | AS              |                 |
+-----------------+-----------------+-----------------+-----------------+
| lad             | 102             | LAD             |                 |
+-----------------+-----------------+-----------------+-----------------+
| lads            | 54              | LADS            | PLURAL          |
+-----------------+-----------------+-----------------+-----------------+
| lass            | 134             | LASS            |                 |
+-----------------+-----------------+-----------------+-----------------+

Table 1 shows some words, their frequencies of occurrence and their meanings as
an artificial lexicon in the wide format. In the following, the letters
(unigrams and bigrams) of the words constitute the cues, the meanings represent
the outcomes.

Analyzing any data using **pyndl** requires them to be in the long format as an
utf-8 encoded tab delimitered text file with a header in the first line and two
columns:

1. the first column contains an underscore delimitered list of all cues
2. the second column contains an underscore delimitered list of all outcomes
3. each line therefore represents an event with a pair of a cue and an outcome
   (occuring one time)
4. the events (lines) are ordered chronologically

As the data in table 1 are artifical we can generate such a file for this
example by expanding table 1 randomly regarding the frequency of occurence of
each event. The resulting event file ``lexample.tab`` consists of 420 lines
(419 = sum of frequencies + 1 header) and looks like the following
(nevertheless you are encouraged to take a closer look at this file using any
text editor of your choice):

=================  =============
Cues               Outcomes
=================  =============
#h_ha_an_nd_ds_s#  hand_plural
#l_la_ad_d#        lad
#l_la_as_ss_s#     lass
=================  =============


pyndl.ndl module
================
We can now compute the strength of associations (weights or weight matrix)
after the  presentation of the 419 tokens of the 10 words using **pyndl**.
**pyndl** provides the two functions ``ndl.ndl`` and ``ndl.dict_ndl`` to
calculate the weights for all outcomes over all events. ``ndl.ndl`` itself
provides to methods regarding estimation, ``openmp`` and ``threading``. We
have to specify the path of our event file ``lexample.tab`` and for this
example set :math:`\alpha = 0.1`, :math:`\beta_{1} = 0.1`, :math:`\beta_{2} =
0.1` with leaving :math:`\lambda = 1.0` at its default value. You can use
**pyndl** directly in a Python3 Shell or you can write an executable script,
this is up to you. For educational purposes we use a Python3 Shell in this
example.


ndl.ndl
-------
``ndl.ndl`` is a parallel Python implementation using numpy, multithreading
and a binary format which is created automatically. It allows you to choose
between the two methods ``openmp`` and ``threading``, with the former one
using openMP and therefore being expected to be much faster when analyzing
larger data.  Besides, you can set three technical arguments which we will not
change here:

1. ``number_of_threads`` (int) giving the number of threads in which the job
   should be executed (default = 2)
2. ``sequence`` (int) giving the length of sublists generated from all outcomes
   (default = 10)
3. ``remove_duplicates`` (logical) to make cues and outcomes unique (default =
   None; which will raise an error if the same cue is present multiple times in
   the same event)

Let's start:

.. code-block:: python

    >>> import pyndl
    >>> from pyndl import ndl
    >>> weights = ndl.ndl(event_path='doc/data/lexample.tab', alpha=0.1, betas=(0.1, 0.1), method='openmp')
    >>> weights

``weights`` is an ``xarray.DataArray`` of dimension ``len(outcomes)``,
``len(cues)``. Our unique, chronologically ordered outcomes are 'hand',
'plural', 'lass', 'lad', 'land', 'as', 'sad', 'and'. Our unique,
chronologically ordered cues are '#h', 'ha', 'an', 'nd', 'ds', 's#', '#l',
'la', 'as', 'ss', 'ad', 'd#', '#a', '#s', 'sa'. Therefore all three indexing
methods

.. code-block:: python

    >>> weights[1, 5]
    >>> weights.loc[{'outcomes': 'plural', 'cues': 's#'}]
    >>> weights.loc['plural'].loc['s#']

return the weight of the cue 's#' (the unigram 's' being the word-final) for
the outcome 'plural' (remember counting in Python does start at 0) as ca. 0.077
and hence indicate 's#' being a marker for plurality.


ndl.dict_ndl
------------
``ndl.dict_ndl`` is a pure Python implementation, however, it differs from
``ndl.ndl`` regarding the following:

1. there are only two technical arguments: ``remove_duplicates`` and
   ``make_data_array`` (logical)
2. no longer an ``xarray.DataArray`` is returned but a ``dict`` of dicts
3. you can set initial weights by specifying the ``weights`` argument
4. the case :math:`\alpha_{i} \neq \alpha_{j} \:` can be handled by specifying
   a ``dict`` consisting of the cues as keys and corresponding :math:`\alpha`'s

Therefore

.. code-block:: python

    >>> weights = ndl.dict_ndl(event_list='doc/data/lexample.tab', alphas=0.1, betas=(0.1, 0.1))
    >>> weights['plural']['s#']

yields approximately the same results as before, however, you now can specify
initial weights or different :math:`\alpha`'s per cue or do both:

.. code-block:: python

    >>> alphas_cues = dict(zip(['#h', 'ha', 'an', 'nd', 'ds', 's#', '#l', 'la', 'as', 'ss', 'ad', 'd#', '#a', '#s', 'sa'], [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.2, 0.1, 0.3, 0.1, 0.2]))
    >>> weights_ini = ndl.dict_ndl(event_list = 'doc/data/lexample.tab', alphas = alphas_cues, betas = (0.1, 0.1))
    >>> weights = ndl.dict_ndl(event_list = 'doc/data/lexample.tab', alphas = alphas_cues, betas = (0.1, 0.1), weights = weights_ini)

Instead of a ``dict`` of dicts, it is also possible to get an
``xarray.DataArray`` returned:

.. code-block:: python

   >>> weights = ndl.dict_ndl(event_list='doc/data/lexample.tab', alphas=0.1, betas=(0.1, 0.1), make_data_array=True)

--------------------------
A minimal workflow example
--------------------------
As you should have a basic understanding of ``pyndl.ndl`` by now, the following
example will show you how to:

1. generate an event file based on a raw corpus file
2. count cues and outcomes
3. filter the events
4. learn the weights as already shown in the lexical learning example
5. save and load a weight matrix (netCDF format)
6. load a weight matrix (netCDF format) into R for further analyses


Generate an event file based on a raw corpus file
=================================================
Suppose you have a raw utf-8 encoded corpus file (by the way, ``pyndl.corpus``
allows you to generate such a corpus file from a bunch of gunzipped xml
subtitle files filled with words, which we will not cover here). For example
take a look at ``lcorpus.txt``.

To analyse the data, you need to convert the file into an event file similar to
``lexample.tab`` in our lexical learning example, as currently there is only
one word per line and neither is there the column for cues nor for outcomes::

   hand
   foot
   hands


pyndl.preprocess module
-----------------------
This module (besides other things) allows you to generate an event file based
on a raw corpus file and filter it:

.. code-block:: python

    >>> import pyndl
    >>> from pyndl import preprocess
    >>> preprocess.create_event_file(corpus_file = 'doc/data/lcorpus.txt', event_file = 'doc/data/levent.tab', context_structure = 'document', event_structure = 'consecutive_words', event_options = (1, ), cue_structure = 'bigrams_to_word')

The function ``preprocess.create_event_file`` has several arguments which you
might have to change to suit them your data, so you are strongly recommened to
read its documentation. We set ``context_structure = 'document'`` as in this
case the context is the whole document, ``event_structure =
'consecutive_words'`` as these are our events, ``event_options = (1, )`` as we
define an event to be one word and ``cue_structure = 'bigrams_to_word'`` to set
cues being bigrams. There are also several technical arguments you can
specifiy, which we will not change here. Our generated event file
``levent.tab`` now looks like this:

=================  ========
Cues               Outcomes
=================  ========
an_#h_ha_d#_nd     hand
ot_fo_oo_#f_t#     foot
ds_s#_an_#h_ha_nd  hands
=================  ========


Count cues and outcomes
=======================
We can now count the cues and outcomes in our event file using the


pyndl.count module
------------------
and also generate id maps for cues and outcomes:

.. code-block:: python

    >>> from pyndl import count
    >>> cue_freq_map, outcome_freq_map = count.cues_outcomes(event_file_name='doc/data/levent.tab')
    >>> cue_freq_map
    >>> outcome_freq_map
    >>> cues = list(cue_freq_map.keys())
    >>> cues.sort()
    >>> cue_id_map = {cue: ii for ii, cue in enumerate(cues)}
    >>> cue_id_map
    >>> outcomes = list(outcome_freq_map.keys())
    >>> outcomes.sort()
    >>> outcome_id_map = {outcome: nn for nn, outcome in enumerate(outcomes)}
    >>> outcome_id_map


Filter the events
=================
As we do not want to include the outcomes 'foot' and 'feet' in this example
aswell as their cues '#f', 'fo' 'oo', 'ot', 't#', 'fe', 'ee' 'et', we use the

pyndl.preprocess module
-----------------------
again, filtering our event file and update the id maps for cues and outcomes:

.. code-block:: python

    >>> preprocess.filter_event_file(input_event_file='doc/data/levent.tab',
    ...                              output_event_file='doc/data/levent.tab.filtered',
    ...                              remove_cues=('#f', 'fo', 'oo', 'ot', 't#', 'fe', 'ee', 'et'),
    ...                              remove_outcomes=('foot', 'feet'))
    >>> cue_freq_map, outcome_freq_map = count.cues_outcomes(event_file_name='doc/data/levent.tab.filtered')
    >>> cue_freq_map
    >>> outcome_freq_map
    >>> cues = list(cue_freq_map.keys())
    >>> cues.sort()
    >>> cue_id_map = {cue: ii for ii, cue in enumerate(cues)}
    >>> cue_id_map
    >>> outcomes = list(outcome_freq_map.keys())
    >>> outcomes.sort()
    >>> outcome_id_map = {outcome: nn for nn, outcome in enumerate(outcomes)}
    >>> outcome_id_map

Alternatively, using ``preprocess.filter_event_file`` you can also specify
which cues and outcomes to keep (``keep_cues`` and ``keep_outcomes``) or remap
cues and outcomes (``cue_map`` and ``outcomes_map``). Besides, there are also
some technical arguments you can specify, which will not discuss here.

Last but not least ``pyndl.preprocess`` does provide some other very useful
functions regarding preprocessing of which we did not make any use here, so
make sure to go through its documentation.


Learn the weights
=================
Computing the strength of associations for the data is now easy, using for
example ``ndl.ndl`` from the


pyndl.ndl module
----------------
like in the lexical learning example:

.. code-block:: python

   >>> from pyndl import ndl
   >>> weights_1 = ndl.ndl(event_path='doc/data/levent.tab.filtered', alpha=0.1, betas=(0.1, 0.1), method="threading")


Save and load a weight matrix
=============================
is straight forward using the netCDF format [@netCDF]

.. code-block:: python

   >>> import xarray
   >>> weights_1.to_netcdf('doc/data/weights_1.nc')
   >>> with xarray.open_dataarray('doc/data/weights_1.nc') as weights_1_read:
   >>>     weights_1_read[0, 0]

the same applies to


Load a weight matrix to R[@R2016]
=================================

.. code-block:: R

   > #install.packages("ncdf4") # uncomment to install
   > library(ncdf4)
   > weights_1_nc <- nc_open(filename = "doc/data/weights_1.nc")
   > weights_1_read <- t(as.matrix(ncvar_get(nc = weights_1_nc, varid = "__xarray_dataarray_variable__")))
   > rownames(weights_1_read) <- ncvar_get(nc = weights_1_nc, varid = "outcomes")
   > colnames(weights_1_read) <- ncvar_get(nc = weights_1_nc, varid = "cues")
   > nc_close(nc = weights_1_nc)
   > rm(weights_1_nc)
