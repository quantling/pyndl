"""
pyndl.wh
---------

*pyndl.wh* provides functions in order to train WH models

"""
from collections import defaultdict, OrderedDict
import copy
import getpass
import os
from queue import Queue
import socket
import sys
import tempfile
import threading
import time

import cython
import pandas as pd
import numpy as np
import xarray as xr

from . import __version__
from . import count
from . import preprocess
from . import wh_parallel
from . import io
from . import ndl


def wh(events, eta, outcome_vectors, *,
        method='openmp', weights=None,
        n_jobs=8, n_outcomes_per_job=10, remove_duplicates=None,
        verbose=False, temporary_directory=None,
        events_per_temporary_file=10000000):
    """
    Calculate the weights for all events using the Widrow-Hoff learning rule
    and training as outcomes on sematic vectors in semantics.

    This is a parallel python implementation using numpy, multithreading and
    the binary format defined in preprocess.py.

    Parameters
    ----------
    events : str
        path to the event file
    eta : float
        learning rate
    outcome_vectors : xarray.DataArray
        matrix that contains the target vectors for each outcome

    method : {'openmp', 'threading'}
    weights : None or xarray.DataArray
        the xarray.DataArray needs to have the dimensions 'cues' and 'outcomes'
    n_jobs : int
        an integer giving the number of threads in which the job should be
        executed
    n_outcomes_per_job : int
        an integer giving the number of outcomes that are processed in one job
    remove_duplicates : {None, True, False}
        if None though a ValueError when the same cue is present multiple times
        in the same event; True make cues and outcomes unique per event; False
        keep multiple instances of the same cue or outcome (this is usually not
        preferred!)
    verbose : bool
        print some output if True.
    temporary_directory : str
        path to directory to use for storing temporary files created;
        if none is provided, the operating system's default will
        be used (/tmp on unix)
    events_per_temporary_file: int
        Number of events in each temporary binary file. Has to be larger than 1

    Returns
    -------
    weights : xarray.DataArray
        with dimensions 'outcomes' and 'cues'. You can lookup the weights
        between a cue and an outcome with ``weights.loc[{'outcomes': outcome,
        'cues': cue}]`` or ``weights.loc[outcome].loc[cue]``.

    """

    if not (remove_duplicates is None or isinstance(remove_duplicates, bool)):
        raise ValueError("remove_duplicates must be None, True or False")
    if not isinstance(events, str):
        raise ValueError("'events' need to be the path to a gzipped event file not {}".format(type(events)))

    weights_ini = weights
    wall_time_start = time.perf_counter()
    cpu_time_start = time.process_time()

    # preprocessing
    n_events, cues, outcomes_from_events = count.cues_outcomes(events,
                                                   number_of_processes=n_jobs,
                                                   verbose=verbose)
    cues = list(cues.keys())
    outcomes_from_events = list(outcomes_from_events.keys())
    outcomes = list(outcome_vectors.coords['outcomes'].data)
    cue_map = OrderedDict(((cue, ii) for ii, cue in enumerate(cues)))
    outcome_map = OrderedDict(((outcome, ii) for ii, outcome in enumerate(outcomes)))

    # check for unseen outcomes in events
    if set(outcomes_from_events) - set(outcomes):
        raise ValueError("all outcomes in events need to be specified as rows in outcome_vectors")

    if weights is not None and weights.shape[1] != outcome_vectors.shape[1]:
        raise ValueError("outcome dimensions in weights need to match dimensions in outcome_vectors")

    # TODO: check for having exactly one legal outcome in each event

    all_outcome_indices = [outcome_map[outcome] for outcome in outcomes]

    shape = (len(outcome_map), len(cue_map))

    # initialize weights
    if weights is None:
        weights = np.ascontiguousarray(np.zeros(shape, dtype=np.float64, order='C'))
    elif isinstance(weights, xr.DataArray):
        raise NotImplementedError("This needs some more refinement.")
        #old_cues = weights.coords["cues"].values.tolist()
        #new_cues = list(set(cues) - set(old_cues))
        #old_vector_dimensions = weights.coords["outcomes"].values.tolist()
        #new_vector_dimensions = outcome_vectors.coords["vector_dimensions"].values.tolist()

        #cues = old_cues + new_cues
        #
        #if old_vector_dimensions != new_vector_dimensions:
        #    raise ValueError("Vector dimensions need to match in continued learning!")

        #cue_map = OrderedDict(((cue, ii) for ii, cue in enumerate(cues)))
        #outcome_map = OrderedDict(((outcome, ii) for ii, outcome in enumerate(outcome_vectors.coords['outcomes'].data)))

        #all_outcome_indices = [outcome_map[outcome] for outcome in outcomes]

        #weights_tmp = np.concatenate((weights.values,
        #                              np.zeros((len(new_outcomes), len(old_cues)),
        #                                       dtype=np.float64, order='C')),
        #                             axis=0)
        #weights_tmp = np.concatenate((weights_tmp,
        #                              np.zeros((len(outcomes), len(new_cues)),
        #                                       dtype=np.float64, order='C')),
        #                             axis=1)

        #weights = np.ascontiguousarray(weights_tmp)

        #del weights_tmp, old_cues, new_cues, old_outcomes, new_outcomes
    else:
        raise ValueError('weights need to be None or xarray.DataArray with method=%s' % method)

    with tempfile.TemporaryDirectory(prefix="pyndl", dir=temporary_directory) as binary_path:
        number_events = preprocess.create_binary_event_files(events, binary_path, cue_map,
                                                             outcome_map, overwrite=True,
                                                             number_of_processes=n_jobs,
                                                             events_per_file=events_per_temporary_file,
                                                             remove_duplicates=remove_duplicates,
                                                             verbose=verbose)
        assert n_events == number_events, (str(n_events) + ' ' + str(number_events))
        binary_files = [os.path.join(binary_path, binary_file)
                        for binary_file in os.listdir(binary_path)
                        if os.path.isfile(os.path.join(binary_path, binary_file))]
        # sort binary files as they were created
        binary_files.sort(key=lambda filename: int(os.path.basename(filename)[9:-4]))
        if verbose:
            print('start learning...')
        # learning
        if method == 'openmp':
            wh_parallel.learn_inplace(binary_files, outcome_vectors.data, eta,
                                      np.array(all_outcome_indices, dtype=np.uint32),
                                      weights, 
                                      n_outcomes_per_job, n_jobs)
        #elif method == 'threading':
        #    part_lists = ndl.slice_list(all_outcome_indices, n_outcomes_per_job)

        #    working_queue = Queue(len(part_lists))
        #    threads = []
        #    queue_lock = threading.Lock()

        #    def worker():
        #        while True:
        #            with queue_lock:
        #                if working_queue.empty():
        #                    break
        #                data = working_queue.get()
        #            ndl_parallel.learn_inplace_2(binary_files, outcome_vectors,
        #                                         eta, weights, data)

        #    with queue_lock:
        #        for partlist in part_lists:
        #            working_queue.put(np.array(partlist, dtype=np.uint32))

        #    for _ in range(number_of_threads):
        #        thread = threading.Thread(target=worker)
        #        thread.start()
        #        threads.append(thread)

        #    for thread in threads:
        #        thread.join()
        else:
            raise ValueError('method needs to be either "threading" or "openmp"')

    cpu_time_stop = time.process_time()
    wall_time_stop = time.perf_counter()
    cpu_time = cpu_time_stop - cpu_time_start
    wall_time = wall_time_stop - wall_time_start

    if weights_ini is not None:
        attrs_to_be_updated = weights_ini.attrs
    else:
        attrs_to_be_updated = None

    attrs = _attributes(events, number_events, eta, cpu_time, wall_time,
                        __name__ + "." + ndl.__name__, method=method, attrs=attrs_to_be_updated)

    # post-processing
    weights = xr.DataArray(weights, [('outcomes', outcomes), ('cues', cues)],
                           attrs=attrs)
    return weights


def _attributes(event_path, number_events, eta, cpu_time,
                wall_time, function, method=None, attrs=None):

    width = max([len(ss) for ss in (event_path,
                                    str(number_events),
                                    str(eta),
                                    function,
                                    str(method),
                                    socket.gethostname(),
                                    getpass.getuser())])
    width = max(19, width)

    def _format(value):
        return '{0: <{width}}'.format(value, width=width)

    new_attrs = {'date': _format(time.strftime("%Y-%m-%d %H:%M:%S")),
                 'event_path': _format(event_path),
                 'number_events': _format(number_events),
                 'lambda': _format(str(eta)),
                 'function': _format(function),
                 'method': _format(str(method)),
                 'cpu_time': _format(str(cpu_time)),
                 'wall_time': _format(str(wall_time)),
                 'hostname': _format(socket.gethostname()),
                 'username': _format(getpass.getuser()),
                 'pyndl': _format(__version__),
                 'numpy': _format(np.__version__),
                 'pandas': _format(pd.__version__),
                 'xarray': _format(xr.__version__),
                 'cython': _format(cython.__version__)}

    if attrs is not None:
        for key in set(attrs.keys()) | set(new_attrs.keys()):
            if key in attrs:
                old_val = attrs[key]
            else:
                old_val = ''
            if key in new_attrs:
                new_val = new_attrs[key]
            else:
                new_val = ''
            new_attrs[key] = old_val + ' | ' + new_val
    return new_attrs
