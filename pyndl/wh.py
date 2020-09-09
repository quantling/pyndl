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

WeightDict = ndl.WeightDict


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
        with dimensions 'vector dimensions' and 'cues'. You can lookup the weights
        between a vector dimension and a cue with ``weights.loc[{'vector_dimensions': vector_dimension,
        'cues': cue}]`` or ``weights.loc[vector_dimension].loc[cue]``.

    """

    if not (remove_duplicates is None or isinstance(remove_duplicates, bool)):
        raise ValueError("remove_duplicates must be None, True or False")
    if not isinstance(events, str):
        raise ValueError("'events' need to be the path to a gzipped event file not {}".format(type(events)))

    weights_ini = weights
    wall_time_start = time.perf_counter()
    cpu_time_start = time.process_time()

    if type(outcome_vectors) == dict:
        # TODO: convert dict to xarray here
        raise NotImplementedError('dicts are not supported yet.')

    # preprocessing
    n_events, cues, outcomes_from_events = count.cues_outcomes(events,
                                                   number_of_processes=n_jobs,
                                                   verbose=verbose)

    # TODO: check for having exactly one legal outcome in each event
    # for now: crudely, just check the number of events and number of outcomes are equal
    assert n_events == sum(outcomes_from_events.values()), "there should be exactly one outcome per event"

    cues = list(cues.keys())
    outcomes_from_events = list(outcomes_from_events.keys())
    outcomes = list(outcome_vectors.coords['outcomes'].data)
    cue_map = OrderedDict(((cue, ii) for ii, cue in enumerate(cues)))
    outcome_map = OrderedDict(((outcome, ii) for ii, outcome in enumerate(outcomes)))

    # check for unseen outcomes in events
    if set(outcomes_from_events) - set(outcomes):
        raise ValueError("all outcomes in events need to be specified as rows in outcome_vectors")

    if weights is not None and weights.shape[0] != outcome_vectors.shape[1]:
        raise ValueError("outcome dimensions in weights need to match dimensions in outcome_vectors")

    all_outcome_indices = [outcome_map[outcome] for outcome in outcomes]

    shape = (outcome_vectors.shape[1], len(cue_map))

    # initialize weights
    if weights is None:
        weights = np.ascontiguousarray(np.zeros(shape, dtype=np.float64, order='C'))
    elif isinstance(weights, xr.DataArray):
        # raise NotImplementedError("This needs some more refinement.")
        old_cues = weights.coords["cues"].values.tolist()
        new_cues = list(set(cues) - set(old_cues))
        old_vector_dimensions = weights.coords["vector_dimensions"].values.tolist()
        new_vector_dimensions = outcome_vectors.coords["vector_dimensions"].values.tolist()

        cues = old_cues + new_cues

        if old_vector_dimensions != new_vector_dimensions:
            raise ValueError("Vector dimensions need to match in continued learning!")

        vector_dimensions = new_vector_dimensions

        cue_map = OrderedDict(((cue, ii) for ii, cue in enumerate(cues)))

        # weights_tmp = np.concatenate((weights.values,
        #                               np.zeros((len(new_outcomes), len(old_cues)),
        #                                        dtype=np.float64, order='C')),
        #                              axis=0)
        # weights_tmp = np.concatenate((weights_tmp,
        #                               np.zeros((len(outcomes), len(new_cues)),
        #                                        dtype=np.float64, order='C')),
        #                              axis=1)

        weights_tmp = np.concatenate((weights.values,
                                      np.zeros((len(vector_dimensions), len(new_cues)),
                                               dtype=np.float64, order='C')),
                                     axis=1)

        weights = np.ascontiguousarray(weights_tmp)

        del weights_tmp, old_cues, new_cues, old_vector_dimensions, new_vector_dimensions
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
    weights = xr.DataArray(weights, [('vector_dimensions', outcome_vectors.coords['vector_dimensions']), ('cues', cues)],
                           attrs=attrs)
    return weights


def dict_wh(events, eta, cue_vectors, outcome_vectors, *,
            weights=None, inplace=False, remove_duplicates=None,
            make_data_array=False, verbose=False):
    """
    Calculate the weights for all_outcomes over all events in events.

    This is a pure python implementation using dicts.

    Notes
    -----
    The metadata will only be stored when `make_data_array` is True and then
    `dict_ndl` cannot be used to continue learning. At the moment there is no
    proper way to automatically store the meta data into the default dict.

    Parameters
    ----------
    events : generator or str
        generates cues, outcomes pairs or the path to the event file
    eta : float
        learning rate
    cue_vectors : xarray.DataArray
        matrix that contains the cue vectors for each cue
    outcome_vectors : xarray.DataArray
        matrix that contains the target vectors for each outcome
    weights : dict of dicts or xarray.DataArray or None
        initial weights
    inplace: {True, False}
        if True calculates the weightmatrix inplace
        if False creates a new weightmatrix to learn on
    remove_duplicates : {None, True, False}
        if None though a ValueError when the same cue is present multiple times
        in the same event; True make cues and outcomes unique per event; False
        keep multiple instances of the same cue or outcome (this is usually not
        preferred!)
    make_data_array : {False, True}
        if True makes a xarray.DataArray out of the dict of dicts.
    verbose : bool
        print some output if True.

    Returns
    -------
    weights : dict of dicts of floats
        the first dict has outcomes as keys and dicts as values
        the second dict has cues as keys and weights as values
        weights[outcome][cue] gives the weight between outcome and cue.

    or

    weights : xarray.DataArray
        with dimensions 'outcomes' and 'cues'. You can lookup the weights
        between a cue and an outcome with ``weights.loc[{'outcomes': outcome,
        'cues': cue}]`` or ``weights.loc[outcome].loc[cue]``.

    """

    if not isinstance(make_data_array, bool):
        raise ValueError("make_data_array must be True or False")

    if not (remove_duplicates is None or isinstance(remove_duplicates, bool)):
        raise ValueError("remove_duplicates must be None, True or False")

    wall_time_start = time.perf_counter()
    cpu_time_start = time.process_time()
    if isinstance(events, str):
        event_path = events
    else:
        event_path = ""
    attrs_to_update = None

    # weights can be seen as an infinite outcome by cue matrix
    # weights[outcome][cue]
    if weights is None:
        weights = WeightDict()
    elif isinstance(weights, WeightDict):
        attrs_to_update = weights.attrs
    elif isinstance(weights, xr.DataArray):
        raise NotImplementedError('initilizing with a xr.DataArray is not supported yet.')
        weights_ini = weights
        attrs_to_update = weights_ini.attrs
        coords = weights_ini.coords
        weights = WeightDict()
        for outcome_index, outcome in enumerate(coords['outcomes'].values):
            for cue_index, cue in enumerate(coords['cues'].values):
                weights[outcome][cue] = weights_ini.item((outcome_index, cue_index))
    elif not isinstance(weights, defaultdict):
        raise ValueError('weights needs to be either defaultdict or None')

    if not inplace:
        weights = copy.deepcopy(weights)

    if outcome_vectors is None:
        all_outcomes = set(weights.keys())

    if isinstance(events, str):
        events = io.events_from_file(events)
    number_events = 0

    for cues, outcomes in events:
        number_events += 1
        if verbose and number_events % 1000:
            print('.', end='')
            sys.stdout.flush()
        if remove_duplicates is None:
            if (len(cues) != len(set(cues)) or
                    len(outcomes) != len(set(outcomes))):
                raise ValueError('cues or outcomes needs to be unique: cues '
                                 '"%s"; outcomes "%s"; use '
                                 'remove_duplicates=True' %
                                 (' '.join(cues), ' '.join(outcomes)))
        elif remove_duplicates:
            cues = set(cues)
            outcomes = set(outcomes)
        else:
            pass

        assert len(outcomes) == 1, 'for real_wh only one outcome is allowed per event'
        assert len(cues) == 1, 'for real_wh only one cue is allowed per event'

        cue_vec = cue_vectors.loc[cues[0]]
        outcome_vec = outcome_vectors.loc[outcomes[0]]

        for outcome_index in range(len(outcome_vec)):
            prediction_strength = 0
            for cue_index in range(len(cue_vec)):
                prediction_strength += float(cue_vec[cue_index] * weights[outcome_index][cue_index])
            error = float(outcome_vec[outcome_index] - prediction_strength)

            for cue_index in range(len(cue_vec)):
                weights[outcome_index][cue_index] += float(eta * error * cue_vec[cue_index])

    cpu_time_stop = time.process_time()
    wall_time_stop = time.perf_counter()
    cpu_time = cpu_time_stop - cpu_time_start
    wall_time = wall_time_stop - wall_time_start
    attrs = _attributes(event_path, number_events, eta, cpu_time, wall_time,
                        __name__ + "." + dict_wh.__name__, attrs=attrs_to_update)

    if make_data_array:
        outcome_dims = list(weights.keys())
        cue_dims = set()
        for outcome_dim in outcome_dims:
            cue_dims.update(set(weights[outcome_dim].keys()))

        cue_dims = list(cue_dims)

        weights_dict = weights
        shape = (len(outcome_dims), len(cue_dims))
        weights = xr.DataArray(np.zeros(shape), attrs=attrs,
                               coords={'outcome_dims': outcome_dims, 'cue_dims': cue_dims},
                               dims=('outcome_dims', 'cue_dims'))

        for outcome_dim in outcome_dims:
            for cue_dim in cue_dims:
                weights.loc[{"outcome_dims": outcome_dim, "cue_dims": cue_dim}] = weights_dict[outcome_dim][cue_dim]
    else:
        weights.attrs = attrs

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
