Tips and Tricks
===============

A collection of more or less unrelated tips and tricks that can be helpful
during development and maintanance.


Local testing with conda
------------------------

Sometimes it might be useful to test if ``pyndl`` works in a clean python
environment. Besides ``tox`` this is possible with ``conda`` as well. The
commands are as follows:

.. code:: bash

    conda create -n testpyndl
    conda activate testpyndl
    conda install python
    python -c 'from pyndl import ndl; print("success")'  # this should fail
    git clone https://github.com/quantling/pyndl.git
    pip install pyndl
    python -c 'from pyndl import ndl; print("success")'  # this should succeed
    conda deactivate
    conda env remove -n testpyndl


Memory profiling
----------------

Sometimes it is useful to monitory the memory footprint of the python process.
This can be achieved by using ``memory_profiler``
(https://pypi.python.org/pypi/memory_profiler).

