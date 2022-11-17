Tips & Tricks
=============

This is collection of more or less unrelated tips and tricks that can be helpful
during development and maintanance.


Running ``pyndl`` within ``R`` code
-----------------------------------

In order to run ``pyndl`` within ``R`` code first install Python and ``pyndl``
as described in the install instructions. Make sure ``pyndl`` runs for your
user within Python.

Now we can switch to ``R`` and install the ``reticulate`` package
(https://cran.r-project.org/web/packages/reticulate/vignettes/introduction.html)
After having the ``reticulate`` package installed we can run within R the following code:

.. code:: r

    library(reticulate)

    learn_weights <- function(event_file) {
        py_env <- py_run_string(
            paste(
                  "from pyndl import ndl",
                  paste0("weights = ndl.ndl('", event_file, "', alpha=0.01, betas=(1.0, 1.0), remove_duplicates=True)"),
                  "weight_matrix = weights.data",
                  "outcome_names = weights.coords['outcomes'].values",
                  "cue_names = weights.coords['cues'].values",
                  sep = "\n"
            ),
            convert = FALSE
        )
        wm <- py_to_r(py_env$weight_matrix)
        rownames(wm) <- py_to_r(py_env$outcome_names)
        colnames(wm) <- py_to_r(py_env$cue_names)
        py_run_string(
            paste(
                  "del cue_names",
                  "del outcome_names",
                  "del weight_matrix",
                  "del weights",
                  sep = "\n"
            ),
            convert = FALSE
        )
        wm
    }

After having defined this funtion a gzipped tab seperated event file can be learned using:

.. code:: r

    wm <- learn_weights('event_file.tab.gz')

Note that this code needs at the moment slightly more than two times the size
of the weights matrix.

There might be a way to learn the weight matrix without any copying between R and Python, but this needs to be elaborated a bit further. The basic idea is

1. to create the the matrix in R (in Fortran mode),
2. borrow / make the matrix available in Python,
3. transpose the matrix in Python to get it into C mode
4. learn the weights in place,
5. Check that the matrix in R has the weights learned as a side effect of the
   Python code.

Further reading:

- https://cran.r-project.org/web/packages/reticulate/vignettes/introduction.html
- https://cran.r-project.org/web/packages/reticulate/vignettes/arrays.html
- https://stackoverflow.com/questions/44379525/r-reticulate-how-do-i-clear-a-python-object-from-memory

