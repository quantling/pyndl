==============
Usage Examples
==============

Lexical example
===============

The lexical example illustrates the Rescorla-Wagner equations [1]_.
This example is taken from Baayen, Milin, Đurđević, Hendrix and Marelli [2]_.

Premises
--------

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
7. The change in association strength :math:`\Delta V_{i}^{t}` is defined as
   in :eq:`RW` with

   * :math:`\alpha_{i}` being the salience of the cue :math:`i`
   * :math:`\beta_{1}` being the salience of the situation in which the outcome occurs
   * :math:`\beta_{2}` being the salience of the situation in which the outcome does not occur
   * :math:`\lambda` being the the maximum level of associative strength possible

8. Default settings for the parameters are: :math:`\alpha_{i} = \alpha_{j} \:
   \forall i, j`, :math:`\beta_{1} = \beta_{2}` and :math:`\lambda = 1`

.. math::
    :label: RW

    \Delta V_{i}^{t} =
    \begin{array}{ll}
    \begin{cases}
    \displaystyle 0 & \: \textrm{if ABSENT}(C_{i}, t)\\ \alpha_{i}\beta_{1} \: (\lambda - \sum_{\textrm{PRESENT}(C_{j}, t)} \: V_{j}) & \: \textrm{if PRESENT}(C_{j}, t) \: \& \: \textrm{PRESENT}(O, t)\\ \alpha_{i}\beta_{2} \: (0 - \sum_{\textrm{PRESENT}(C_{j}, t)} \: V_{j}) & \: \textrm{if PRESENT}(C_{j}, t) \: \& \: \textrm{ABSENT}(O, t)
    \end{cases}
    \end{array}


See :ref:`comparison_of_algorithms` for alternative formulations of the
Rescorla Wagner learning rule.


Data
----

..  include:: lexicon.rst

Table 1 shows some words, their frequencies of occurrence and their meanings as
an artificial lexicon in the wide format. In the following, the letters
(unigrams and bigrams) of the words constitute the cues, the meanings represent
the outcomes.

Analyzing any data using *pyndl* requires them to be in the long format as an
utf-8 encoded tab delimited gzipped text file with a header in the first line
and two columns:

1. the first column contains an underscore delimited list of all cues
2. the second column contains an underscore delimited list of all outcomes
3. each line therefore represents an event with a pair of a cue and an outcome
   (occurring one time)
4. the events (lines) are ordered chronologically

As the data in table 1 are artificial we can generate such a file for this
example by expanding table 1 randomly regarding the frequency of occurrence of
each event. The resulting event file `lexample.tab.gz`_ consists of 420 lines
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
----------------
We can now compute the strength of associations (weights or weight matrix)
after the  presentation of the 419 tokens of the 10 words using
:py:mod:`pyndl.ndl`. :py:mod:`pyndl.ndl` provides the two functions
:py:mod:`pyndl.ndl.ndl` and :py:mod:`pyndl.ndl.dict_ndl` to calculate the
weights for all outcomes over all events. :py:mod:`pyndl.ndl.ndl` itself
provides to methods regarding estimation, ``openmp`` and ``threading``. We have
to specify the path of our event file `lexample.tab.gz`_ and
for this example set :math:`\alpha = 0.1`, :math:`\beta_{1} = 0.1`,
:math:`\beta_{2} = 0.1` with leaving :math:`\lambda = 1.0` at its default
value. You can use *pyndl* directly in a Python3 Shell or you can write an
executable script, this is up to you. For educational purposes we use a Python3
Shell in this example.


pyndl.ndl.ndl
^^^^^^^^^^^^^
:py:mod:`pyndl.ndl.ndl` is a parallel Python implementation using numpy,
multithreading and a binary format which is created automatically. It allows
you to choose between the two methods ``openmp`` and ``threading``, with the
former one using `openMP <http://www.openmp.org/>`_ and therefore being
expected to be faster when analyzing larger data. Unfortunately, ``openmp`` is
only available on Linux right now, therefore all examples use ``threading``.
Besides, you can set three technical arguments which we will not change here:

1. ``n_jobs`` (int) giving the number of threads in which the job
   should be executed (default=2)
2. ``sequence`` (int) giving the length of sublists generated from all outcomes
   (default=10)
3. ``remove_duplicates`` (logical) to make cues and outcomes unique
   (default=None; which will raise an error if the same cue is present multiple
   times in the same event)

Let's start:

.. code-block:: python

    >>> from pyndl import ndl
    >>> weights = ndl.ndl(events='docs/data/lexample.tab.gz', alpha=0.1,
    ...                   betas=(0.1, 0.1), method='threading')
    >>> weights  # doctest: +ELLIPSIS
    <xarray.DataArray (outcomes: 8, cues: 15)>...
    ...

``weights`` is an ``xarray.DataArray`` of dimension ``len(outcomes)``,
``len(cues)``. Our unique, chronologically ordered outcomes are 'hand',
'plural', 'lass', 'lad', 'land', 'as', 'sad', 'and'. Our unique,
chronologically ordered cues are '#h', 'ha', 'an', 'nd', 'ds', 's#', '#l',
'la', 'as', 'ss', 'ad', 'd#', '#a', '#s', 'sa'. Therefore all three indexing
methods

.. code-block:: python

    >>> weights[1, 5]  # doctest: +ELLIPSIS
    <xarray.DataArray ()>...
    ...
    >>> weights.loc[{'outcomes': 'plural', 'cues': 's#'}]  # doctest: +ELLIPSIS
    <xarray.DataArray ()>...
    array(0.076988...)
    Coordinates:
        outcomes  <U6 ... 'plural'
        cues      <U2 ... 's#'
    ...
    >>> weights.loc['plural'].loc['s#']  # doctest: +ELLIPSIS
    <xarray.DataArray ()>...
    array(0.076988...)
    Coordinates:
        outcomes  <U6 ... 'plural'
        cues      <U2 ... 's#'
    ...

return the weight of the cue 's#' (the unigram 's' being the word-final) for
the outcome 'plural' (remember counting in Python does start at 0) as ca. 0.077
and hence indicate 's#' being a marker for plurality.

:py:mod:`pyndl.ndl.ndl` also allows you to continue learning from a previous
weight matrix by specifying the ``weight`` argument:

.. code-block:: python

    >>> weights2 = ndl.ndl(events='docs/data/lexample.tab.gz', alpha=0.1,
    ...                    betas=(0.1, 0.1), method='threading', weights=weights)
    >>> weights2  # doctest: +ELLIPSIS
    <xarray.DataArray (outcomes: 8, cues: 15)>...
    array([[ 0.24...
    ...
    ...]])
    Coordinates:
      * outcomes  (outcomes) <U6 ... 'hand' 'plural'...
      * cues      (cues) <U2 ... '#h' 'ha' 'an' 'nd'...
    Attributes:...
        date:...
        event_path:...
    ...

As you may have noticed already, :py:mod:`pyndl.ndl.ndl` provides you with meta
data organized in a ``dict`` which was collected during your calculations. Each
entry of each ``list`` of this meta data therefore references one specific
moment of your calculations:

.. code-block:: python

   >>> print('Attributes: ' + str(weights2.attrs))  # doctest: +ELLIPSIS
   Attributes: ...


pyndl.ndl.dict_ndl
^^^^^^^^^^^^^^^^^^
:py:mod:`pyndl.ndl.dict_ndl` is a pure Python implementation, however, it
differs from :py:mod:`pyndl.ndl.ndl` regarding the following:

1. there are only two technical arguments: ``remove_duplicates`` (logical) and
   ``make_data_array`` (logical)
2. by default, no longer an ``xarray.DataArray`` is returned but a ``dict`` of dicts
3. however, you are still able to get an ``xarray.DataArray`` by setting
   ``make_data_array=True``
4. the case :math:`\alpha_{i} \neq \alpha_{j} \:` can be handled by specifying
   a ``dict`` consisting of the cues as keys and corresponding :math:`\alpha`'s

Therefore

.. code-block:: python

    >>> weights = ndl.dict_ndl(events='docs/data/lexample.tab.gz',
    ...                        alphas=0.1, betas=(0.1, 0.1))
    >>> weights['plural']['s#'] # doctes: +ELLIPSIS
    0.076988227...

yields approximately the same results as before, however, you now can specify
different :math:`\alpha`'s per cue and as in :py:mod:`pyndl.ndl.ndl` continue
learning or do both:

.. code-block:: python

    >>> alphas_cues = dict(zip(['#h', 'ha', 'an', 'nd', 'ds', 's#', '#l', 'la', 'as', 'ss', 'ad', 'd#', '#a', '#s', 'sa'],
    ...                        [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.2, 0.1, 0.3, 0.1, 0.2]))
    >>> weights = ndl.dict_ndl(events='docs/data/lexample.tab.gz',
    ...                        alphas=alphas_cues, betas=(0.1, 0.1))
    >>> weights2 = ndl.dict_ndl(events='docs/data/lexample.tab.gz',
    ...                         alphas=alphas_cues, betas=(0.1, 0.1),
    ...                         weights=weights)

If you prefer to get a ``xarray.DataArray`` returned you can set the flag ``make_data_array=True``:

.. code-block:: python

    >>> weights = ndl.dict_ndl(events='docs/data/lexample.tab.gz',
    ...                        alphas=alphas_cues, betas=(0.1, 0.1),
    ...                        make_data_array=True)
    >>> weights  # doctest: +ELLIPSIS
    <xarray.DataArray (outcomes: 8, cues: 15)>...
    ...


A minimal workflow example
==========================
As you should have a basic understanding of :py:mod:`pyndl.ndl` by now, the
following example will show you how to:

1. generate an event file based on a raw corpus file
2. count cues and outcomes
3. filter the events
4. learn the weights as already shown in the lexical learning example
5. save and load a weight matrix (netCDF format)
6. load a weight matrix (netCDF format) into R for further analyses


Generate an event file based on a raw corpus file
-------------------------------------------------
Suppose you have a raw utf-8 encoded corpus file (by the way,
:py:mod:`pyndl.corpus` allows you to generate such a corpus file from a bunch of
gunzipped xml subtitle files filled with words, which we will not cover here).
For example take a look at `lcorpus.txt_`.

To analyse the data, you need to convert the file into an event file similar to
`lexample.tab.gz`_ in our lexical learning example, as currently there is only
one word per line and neither is there the column for cues nor for outcomes::

   hand
   foot
   hands


The :py:mod:`pyndl.preprocess` module (besides other things) allows you to
generate an event file based on a raw corpus file and filter it:

.. code-block:: python

    >>> import pyndl
    >>> from pyndl import preprocess
    >>> preprocess.create_event_file(corpus_file='docs/data/lcorpus.txt',
    ...                              event_file='docs/data/levent.tab.gz',
    ...                              allowed_symbols='a-zA-Z',
    ...                              context_structure='document',
    ...                              event_structure='consecutive_words',
    ...                              event_options=(1, ),
    ...                              cue_structure='bigrams_to_word')

The function :py:mod:`pyndl.preprocess.create_event_file` has several arguments
which you might have to change to suit them your data, so you are strongly
recommended to read its documentation. We set ``context_structure='document'``
as in this case the context is the whole document,
``event_structure='consecutive_words'`` as these are our events,
``event_options=(1, )`` as we define an event to be one word and
``cue_structure='bigrams_to_word'`` to set cues being bigrams.
There are also several technical arguments you can specify, which we will not
change here. Our generated event file ``levent.tab.gz`` now looks
(uncompressed) like this:

=================  ========
Cues               Outcomes
=================  ========
an_#h_ha_d#_nd     hand
ot_fo_oo_#f_t#     foot
ds_s#_an_#h_ha_nd  hands
=================  ========


Count cues and outcomes
-----------------------
We can now count the cues and outcomes in our event file using the
:py:mod:`pyndl.count` module and also generate id maps for cues and outcomes:

.. code-block:: python

    >>> from pyndl import count
    >>> freq, cue_freq_map, outcome_freq_map = count.cues_outcomes(event_file_name='docs/data/levent.tab.gz')
    >>> freq
    12
    >>> cue_freq_map  # doctest: +ELLIPSIS
    Counter({...})
    >>> outcome_freq_map  # doctest: +ELLIPSIS
    Counter({...})
    >>> cues = list(cue_freq_map.keys())
    >>> cues.sort()
    >>> cue_id_map = {cue: ii for ii, cue in enumerate(cues)}
    >>> cue_id_map  # doctest: +ELLIPSIS
    {...}
    >>> outcomes = list(outcome_freq_map.keys())
    >>> outcomes.sort()
    >>> outcome_id_map = {outcome: nn for nn, outcome in enumerate(outcomes)}
    >>> outcome_id_map  # doctest: +ELLIPSIS
    {...}


Filter the events
-----------------
As we do not want to include the outcomes 'foot' and 'feet' in this example
as well as their cues '#f', 'fo' 'oo', 'ot', 't#', 'fe', 'ee' 'et', we use the
:py:mod:`pyndl.preprocess` module again, filtering our event file and update
the id maps for cues and outcomes:

.. code-block:: python

    >>> preprocess.filter_event_file(input_event_file='docs/data/levent.tab.gz',
    ...                              output_event_file='docs/data/levent.tab.gz.filtered',
    ...                              remove_cues=('#f', 'fo', 'oo', 'ot', 't#', 'fe', 'ee', 'et'),
    ...                              remove_outcomes=('foot', 'feet'))
    >>> freq, cue_freq_map, outcome_freq_map = count.cues_outcomes(event_file_name='docs/data/levent.tab.gz.filtered')
    >>> cues = list(cue_freq_map.keys())
    >>> cues.sort()
    >>> cue_id_map = {cue: ii for ii, cue in enumerate(cues)}
    >>> cue_id_map  # doctest: +ELLIPSIS
    {...}
    >>> outcomes = list(outcome_freq_map.keys())
    >>> outcomes.sort()
    >>> outcome_id_map = {outcome: nn for nn, outcome in enumerate(outcomes)}
    >>> outcome_id_map  # doctest: +ELLIPSIS
    {...}

Alternatively, using :py:mod:`pyndl.preprocess.filter_event_file` you can also
specify which cues and outcomes to keep (``keep_cues`` and ``keep_outcomes``)
or remap cues and outcomes (``cue_map`` and ``outcomes_map``). Besides, there
are also some technical arguments you can specify, which will not discuss here.

Last but not least :py:mod:`pyndl.preprocess` does provide some other very
useful functions regarding preprocessing of which we did not make any use here,
so make sure to go through its documentation.


Learn the weights
-----------------
Computing the strength of associations for the data is now easy, using for
example :py:mod:`pyndl.ndl.ndl` from the :py:mod:`pyndl.ndl` module like in the lexical learning
example:

.. code-block:: python

   >>> from pyndl import ndl
   >>> weights = ndl.ndl(events='docs/data/levent.tab.gz.filtered',
   ...                   alpha=0.1, betas=(0.1, 0.1), method="threading")


Save and load a weight matrix
-----------------------------
is straight forward using the netCDF format [3]_

.. code-block:: python

    >>> import xarray  # doctest: +SKIP
    >>> weights.to_netcdf('docs/data/weights.nc')  # doctest: +SKIP
    >>> with xarray.open_dataarray('docs/data/weights.nc') as weights_read:  # doctest: +SKIP
    ...     weights_read

In order to keep everything clean we might want to remove all the files we
created in this tutorial:

.. code-block:: python

   >>> import os
   >>> os.remove('docs/data/levent.tab.gz')
   >>> os.remove('docs/data/levent.tab.gz.filtered')


Widrow-Hoff (WH) learning
=========================
There is a Widrow-Hoff learning module called `wh` now in `pyndl`, which uses
the same event files and nearly the same function parameters as the `ndl.ndl`
function. The main function to call is `wh.wh`.  Compared to `ndl.ndl` the
`wh.wh` function adds two look-up tables, one for cues and one for outcomes, to
its keyword arguments.  Each of this look-up tables  maps each cue and / or
outcome in your event file to a vector. This look-up table has to be an
instance `xarray.DataArray` and is passed with the keyword argument
`cue_vectors` or `outcome_vectors`. The second dimension of the look-up table
needs to be named `cue_vector_dimensions` and `outcome_vector_dimensions`
respectively. For more information have a look at the function doc string.

WH example
----------
This example shows that WH learning mimics RW learning, if the cue and outcome
vectors are containing unit vectors. Note that WH learning in contrast to the RW
learning only has one learning parameter, which is called `eta`. The assumption
is that `beta1` equals `beta2`.

.. code-block:: python

   >>> from pyndl import wh, ndl
   >>> import xarray as xr
   >>> import numpy as np
   >>> events = 'docs/data/event_file_wh.tab.gz'
   >>> eta = 0.01  # learning rate
   >>> cue_vectors = xr.DataArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
   ...                            dims=('cues', 'cue_vector_dimensions'),
   ...                            coords={'cues': ['a', 'b', 'c'], 'cue_vector_dimensions': ['dim1', 'dim2', 'dim3']})
   >>> outcome_vectors = xr.DataArray(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),
   ...                                dims=('outcomes', 'outcome_vector_dimensions'),
   ...                                coords={'outcomes': ['A', 'B', 'C', 'D'],
   ...                                        'outcome_vector_dimensions': ['dim1', 'dim2', 'dim3', 'dim4']})
   >>> weights_wh = wh.wh(events, eta, cue_vectors=cue_vectors, outcome_vectors=outcome_vectors, method='numpy')
   >>> weights_ndl = ndl.ndl(events, alpha=1.0, betas=(eta, eta), method='threading')

The weights returned by `wh.wh` have dimensions `outcome_vector_dimensions` and
`cue_vector_dimensions`. Therefore, a direct comparison is not possible. But
as the vectors used are unit vectors the first `cue_vector_dimension` `"dim1"`
corresponds to the first cue `"a"` and the second vector dimension corresponds
to the second cue etc. If the dimensions are ordered by their names, the equality
gets apparent.

.. code-block:: python

    >>> weights_wh = weights_wh.loc[{'outcome_vector_dimensions': ['dim1', 'dim2', 'dim3', 'dim4'],
    ...                              'cue_vector_dimensions': ['dim1', 'dim2', 'dim3']}]
    >>> weights_ndl = weights_ndl.loc[{'outcomes': ['A', 'B', 'C', 'D'], 'cues': ['a', 'b', 'c']}]
    >>> print(weights_wh)  # doctest: +ELLIPSIS
    <xarray.DataArray (outcome_vector_dimensions: 4, cue_vector_dimensions: 3)>...
    array([[0.06706..., 0.        , 0.        ],
           [0.        , 0.03940..., 0.        ],
           [0.0094... , 0.        , 0.03940...],
           [0.01      , 0.        , 0.        ]])
    Coordinates:
      * outcome_vector_dimensions  (outcome_vector_dimensions) <U4 ... 'dim1' ...
      * cue_vector_dimensions      (cue_vector_dimensions) <U4 ... 'dim1' ...
        outcomes                   <U1 ... 'A'
        cues                       <U1 ... 'a'
    Attributes: (12/15)
    ...
    >>> print(weights_ndl)  # doctest: +ELLIPSIS
    <xarray.DataArray (outcomes: 4, cues: 3)>...
    array([[0.06706..., 0.        , 0.        ],
           [0.        , 0.03940..., 0.        ],
           [0.0094... , 0.        , 0.03940...],
           [0.01      , 0.        , 0.        ]])
    Coordinates:
      * outcomes  (outcomes) <U1 ... 'A' 'B' 'C' 'D'
      * cues      (cues) <U1 ... 'a' 'b' 'c'
    Attributes: (12/17)
    ...

Furthermore, it is possible to only use either `cue_vectors` or
`outcome_vectors`. This functionality is Linux only at the moment.

.. code-block:: python

   >>> weights_wh_cv_only = wh.wh(events, eta, cue_vectors=cue_vectors, method='openmp')  # doctest: +SKIP
   >>> weights_wh_ov_only = wh.wh(events, eta, outcome_vectors=outcome_vectors, method='openmp')  # doctest: +SKIP

For this example the content of the resulting weights matches the content of
the `weights_wh` and `weights_ndl`.


Load a weight matrix to R [4]_
==============================
We can load a in netCDF format saved matrix into R:

.. code-block:: R

   > #install.packages("ncdf4") # uncomment to install
   > library(ncdf4)
   > weights_nc <- nc_open(filename = "docs/data/weights.nc")
   > weights_read <- t(as.matrix(ncvar_get(nc = weights_nc, varid = "__xarray_dataarray_variable__")))
   > rownames(weights_read) <- ncvar_get(nc = weights_nc, varid = "outcomes")
   > colnames(weights_read) <- ncvar_get(nc = weights_nc, varid = "cues")
   > nc_close(nc = weights_nc)
   > rm(weights_nc)

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

.. [4] R Core Team (2013). R: A language and environment for statistical
      computing. R Foundation for Statistical Computing, Vienna, Austria.
      URL https://www.R-project.org/.
