"""
pyndl.wh
--------

*pyndl.wh* provides functions in order to train Widrow-Hoff (WH) models. In contrast
to the Rescorla-Wagner (RW) models, the WH models can not only have binary
cues and outcomes, but can encode gradual intensities in the cues and outcomes.
This is done by associating a vector of continues values (real numbers) to each
cue and outcome. The size of the vector has to be the same for all cues and for
all outcomes, but can differ between cues and outcomes.

It is possible to calculate weights for continuous cues or continues outcomes,
while keeping the outcomes respectively cues binary. Finally, it is possible to
have both sides, cues and outcomes, to be continues and calculate the
Widrow-Hoff learning rule between them.

"""
from collections import defaultdict, OrderedDict
import copy
import getpass
import os
# from queue import Queue
import socket
import sys
import tempfile
# import threading
import time

import cython
import pandas as pd
import numpy as np
import xarray as xr

from . import __version__ as pyndl_version
from . import count
from . import preprocess
from . import io
from . import ndl

# conditional import as openmp is only compiled for linux
if sys.platform.startswith('linux'):
    from . import ndl_openmp
elif sys.platform.startswith('win32'):
    pass
elif sys.platform.startswith('darwin'):
    pass


WeightDict = ndl.WeightDict


def wh(events, eta, *, cue_vectors=None, outcome_vectors=None,
        method='openmp', weights=None,
        n_jobs=8, n_outcomes_per_job=10, remove_duplicates=None,
        verbose=False, temporary_directory=None,
        events_per_temporary_file=10000000):
    """
    Calculate the weights for all events using the Widrow-Hoff learning rule in
    three different flavors.

    In the first flavor, cues and outcomes both are vectors and the names in
    the eventfiles refer to these vectors. The vectors for all cues and
    outcomes are given as an xarray.DataArray with the arguments `cue_vectors`
    and `outcome_vectors'.

    In the second and third flavor, only the cues or only the outcomes are
    treated as vectors and the ones not being treated as vectors are still
    considered being present or not being present in a binary way.

    This is a parallel python implementation using cython, numpy,
    multithreading and the binary format defined in preprocess.py.

    Parameters
    ----------
    events : str
        path to the event file
    eta : float
        learning rate
    cue_vectors : xarray.DataArray
        matrix that contains the cue vectors for each cue
    outcome_vectors : xarray.DataArray
        matrix that contains the target vectors for each outcome
    method : {'openmp', 'threading', 'numpy'}
        'numpy' works only for real to real Widrow-Hoff.
    weights : None or xarray.DataArray
        the xarray.DataArray needs to have the named dimensions 'cues' or
        'cue_vector_dimensions' and 'outcomes' or 'outcome_vector_dimensions'
    n_jobs : int
        an integer giving the number of threads in which the job should be
        executed
    n_outcomes_per_job : int
        an integer giving the number of outcomes that are processed in one job
    remove_duplicates : {None, True, False}
        if None raise a ValueError when the same cue is present multiple times
        in the same event; True make cues and outcomes unique per event; False
        keep multiple instances of the same cue or outcome (this is usually not
        preferred!)
    verbose : bool
        print some output if True
    temporary_directory : str
        path to directory to use for storing temporary files created;
        if none is provided, the operating system's default will
        be used like '/tmp' on unix
    events_per_temporary_file: int
        Number of events in each temporary binary file. Has to be larger than 1

    Returns
    -------
    weights : xarray.DataArray
        the dimensions of the weights reflect the type of Widrow-Hoff that was
        run (real to real, binary to real, real to binary or binary to binary).
        The dimension names reflect this in the weights. They are a combination
        of 'outcomes' x 'outcome_vector_dimensions' and 'cues' x
        'cue_vector_dimensions' with dimensions 'outcome_vector dimensions' and
        'cue_vector_dimensions'. You can lookup the weights between a vector
        dimension and a cue with ``weights.loc[{'outcome_vector_dimensions':
        outcome_vector_dimension, 'cue_vector_dimensions':
        cue_vector_dimension}]`` or
        ``weights.loc[vector_dimension].loc[cue_vector_dimension]``.

    """
    if cue_vectors is None and outcome_vectors is None:
        lambda_ = 1.0
        alpha = 1.0
        betas = (eta, eta)
        return ndl.ndl(events, alpha, betas, lambda_,
                       method=method, weights=weights, n_jobs=n_jobs,
                       n_outcomes_per_job=n_outcomes_per_job,
                       remove_duplicates=remove_duplicates, verbose=verbose,
                       temporary_directory=temporary_directory,
                       events_per_temporary_file=events_per_temporary_file)
    elif cue_vectors is not None and outcome_vectors is not None:
        return _wh_real_to_real(events, eta, cue_vectors, outcome_vectors,
                                method=method, weights=weights, n_jobs=n_jobs,
                                n_outcomes_per_job=n_outcomes_per_job,
                                remove_duplicates=remove_duplicates, verbose=verbose,
                                temporary_directory=temporary_directory,
                                events_per_temporary_file=events_per_temporary_file)
    elif cue_vectors is not None and outcome_vectors is None:
        lambda_ = 1.0
        betas = (eta, eta)
        return _wh_real_to_binary(events, betas, lambda_, cue_vectors,
                                  method=method, weights=weights, n_jobs=n_jobs,
                                  n_outcomes_per_job=n_outcomes_per_job,
                                  remove_duplicates=remove_duplicates, verbose=verbose,
                                  temporary_directory=temporary_directory,
                                  events_per_temporary_file=events_per_temporary_file)
    elif cue_vectors is None and outcome_vectors is not None:
        return _wh_binary_to_real(events, eta, outcome_vectors,
                                  method=method, weights=weights, n_jobs=n_jobs,
                                  n_outcomes_per_job=n_outcomes_per_job,
                                  remove_duplicates=remove_duplicates, verbose=verbose,
                                  temporary_directory=temporary_directory,
                                  events_per_temporary_file=events_per_temporary_file)
    # The if statements above are covering all cases.


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

    Furthermore, this implementation only supports the 'real to real' case
    where cue vectors are learned on outcome vectors. For the 'binary to real'
    or 'real to binary' cases the `wh.wh` function needs to be used which uses
    a fast cython implementation.

    The main purpose of this function is to have a reference implementation
    which is used to validate the faster cython version against. Additionally, this
    function can be a good starting point to develop different flavors of the
    Widrow-Hoff learning rule.

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
        with dimensions 'outcome_vector_dimensions' and
        'cue_vector_dimensions'. You can lookup the weights
        between a cue dimension and an outcome dimension with
        ``weights.loc[{'outcome_vector_dimensions': outcome_vector_dimension,
        'cue_vector_dimensions': cue_vector_dimension}]`` or
        ``weights.loc[outcome_vector_dimension].loc[cue_vector_dimension]``.

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
        # weights_ini = weights
        # attrs_to_update = weights_ini.attrs
        # coords = weights_ini.coords
        # weights = WeightDict()
        # for outcome_index, outcome in enumerate(coords['outcomes'].values):
        #     for cue_index, cue in enumerate(coords['cues'].values):
        #         weights[outcome][cue] = weights_ini.item((outcome_index, cue_index))
    elif not isinstance(weights, defaultdict):
        raise ValueError('weights needs to be either defaultdict or None')

    if not inplace:
        weights = copy.deepcopy(weights)

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

        for outcome_index in outcome_vec.coords['outcome_vector_dimensions']:
            outcome_index = str(outcome_index.values)
            prediction_strength = 0
            for cue_index in cue_vec.coords['cue_vector_dimensions']:
                cue_index = str(cue_index.values)
                prediction_strength += float(cue_vec.loc[cue_index] * weights[outcome_index][cue_index])
            error = float(outcome_vec.loc[outcome_index] - prediction_strength)

            for cue_index in cue_vec.coords['cue_vector_dimensions']:
                cue_index = str(cue_index.values)
                weights[outcome_index][cue_index] += float(eta * error * cue_vec.loc[cue_index])

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
                               coords={'outcome_vector_dimensions': outcome_dims,
                                       'cue_vector_dimensions': cue_dims},
                               dims=('outcome_vector_dimensions', 'cue_vector_dimensions'))

        for outcome_dim in outcome_dims:
            for cue_dim in cue_dims:
                weights.loc[{"outcome_vector_dimensions": outcome_dim,
                             "cue_vector_dimensions": cue_dim}] = weights_dict[outcome_dim][cue_dim]
    else:
        weights.attrs = attrs

    return weights


def _wh_binary_to_real(events, eta, outcome_vectors, *,
                       method='openmp', weights=None,
                       n_jobs=8, n_outcomes_per_job=10, remove_duplicates=None,
                       verbose=False, temporary_directory=None,
                       events_per_temporary_file=10000000):
    """
    Calculate the weights for all events using the Widrow-Hoff learning rule
    from binary cues onto continuous outcome vectors.

    This is a parallel python implementation using cython, numpy,
    multithreading and the binary format defined in preprocess.py.

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
        the xarray.DataArray needs to have the dimensions 'cues' and
        'outcome_vector_dimensions'
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
        be used like '/tmp' on unix
    events_per_temporary_file: int
        Number of events in each temporary binary file. Has to be larger than 1

    Returns
    -------
    weights : xarray.DataArray
        with dimensions 'outcome_vector dimensions' and 'cues'. You can lookup
        the weights between a vector dimension and a cue with
        ``weights.loc[{'outcome_vector_dimensions': outcome_vector_dimension,
        'cues': cue}]`` or ``weights.loc[outcome_vector_dimension].loc[cue]``.

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

    if not outcome_vectors.data.data.c_contiguous:
        raise ValueError('outcome_vectors have to be c_contiguous')
    if not outcome_vectors.dtype == np.float64:
        raise ValueError('outcome_vectors have to be of dtype np.float64')

    # preprocessing
    n_events, cues, outcomes_from_events = count.cues_outcomes(events,
                                                               n_jobs=n_jobs,
                                                               verbose=verbose)
    cues = list(cues.keys())
    outcomes_from_events = list(outcomes_from_events.keys())
    outcomes = outcome_vectors.coords['outcomes'].values.tolist()
    cue_map = OrderedDict(((cue, ii) for ii, cue in enumerate(cues)))
    outcome_map = OrderedDict(((outcome, ii) for ii, outcome in enumerate(outcomes)))

    # check for unseen outcomes in events
    if set(outcomes_from_events) - set(outcomes):
        raise ValueError("all outcomes in events need to be specified as rows in outcome_vectors")

    if weights is not None and weights.shape[0] != outcome_vectors.shape[1]:
        raise ValueError("outcome dimensions in weights need to match dimensions in outcome_vectors")

    shape = (outcome_vectors.shape[1], len(cue_map))

    # initialize weights
    if weights is None:
        weights = np.ascontiguousarray(np.zeros(shape, dtype=np.float64, order='C'))
    elif isinstance(weights, xr.DataArray):
        outcome_vector_dimensions = outcome_vectors.coords["outcome_vector_dimensions"].values.tolist()
        if not outcome_vector_dimensions == weights.coords["outcome_vector_dimensions"].values.tolist():
            raise ValueError('Outcome vector dimensions in weights and outcome_vectors do not match!')

        old_cues = weights.coords["cues"].values.tolist()
        new_cues = list(set(cues) - set(old_cues))
        cues = old_cues + new_cues
        cue_map = OrderedDict(((cue, ii) for ii, cue in enumerate(cues)))

        weights_tmp = np.concatenate((weights.values,
                                      np.zeros((len(outcome_vector_dimensions), len(new_cues)),
                                               dtype=np.float64, order='C')),
                                     axis=1)
        weights = np.ascontiguousarray(weights_tmp)

        del weights_tmp, old_cues, new_cues, outcome_vector_dimensions
    else:
        raise ValueError('weights need to be None or xarray.DataArray with method=%s' % method)

    with tempfile.TemporaryDirectory(prefix="pyndl", dir=temporary_directory) as binary_path:
        number_events = preprocess.create_binary_event_files(events, binary_path, cue_map,
                                                             outcome_map, overwrite=True,
                                                             n_jobs=n_jobs,
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
        if not weights.data.c_contiguous:
            raise ValueError('weights has to be c_contiguous')
        if method == 'openmp':
            if not sys.platform.startswith('linux'):
                raise NotImplementedError("OpenMP is linux only at the moment."
                                          "Use method='threading' instead.")
            ndl_openmp.learn_inplace_binary_to_real(binary_files,
                                                    eta,
                                                    outcome_vectors.data,
                                                    weights,
                                                    n_outcomes_per_job,
                                                    n_jobs)
        elif method == 'threading':
            raise ValueError('TODO: for now: method needs "openmp"')
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

        #    for _ in range(n_jobs):
        #        thread = threading.Thread(target=worker)
        #        thread.start()
        #        threads.append(thread)

        #    for thread in threads:
        #        thread.join()
        else:
            raise ValueError('method needs to be either "openmp"')

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
    weights = xr.DataArray(weights, coords=[('outcome_vector_dimensions',
                                             outcome_vectors.coords['outcome_vector_dimensions'].data),
                                            ('cues', cues)], attrs=attrs)
    return weights


def _wh_real_to_binary(events, betas, lambda_, cue_vectors, *,
                       method='openmp', weights=None,
                       n_jobs=8, n_outcomes_per_job=10, remove_duplicates=None,
                       verbose=False, temporary_directory=None,
                       events_per_temporary_file=10000000):
    """
    Calculate the weights for all events using the Widrow-Hoff learning rule
    from continuous cue vectors onto binary outcomes.

    This is a parallel python implementation using cython and the binary format
    defined in preprocess.py.

    Parameters
    ----------
    events : str
        path to the event file
    betas : (float, float)
        one value for successful prediction (reward) one for punishment
    lambda\\_ : float
    cue_vectors : xarray.DataArray
        matrix that contains the cue vectors for each cue
    method : {'openmp', 'threading'}
    weights : None or xarray.DataArray
        the xarray.DataArray needs to have the dimensions
        'cue_vector_dimensions' and 'outcomes'
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
        be used like '/tmp' on unix
    events_per_temporary_file: int
        Number of events in each temporary binary file. Has to be larger than 1

    Returns
    -------
    weights : xarray.DataArray
        with dimensions outcomes and cue_vector_dimensions. You can lookup the
        weights between an outcome and a cue_vector_dimension with
        ``weights.loc[{'outcome': outcome, 'cue_vector_dimensions':
        cue_vector_dimension}]`` or
        ``weights.loc[outcome].loc[cue_vector_dimension]``.

    """
    if not (remove_duplicates is None or isinstance(remove_duplicates, bool)):
        raise ValueError("remove_duplicates must be None, True or False")

    if not isinstance(events, str):
        raise ValueError("'events' need to be the path to a gzipped event file not {}".format(type(events)))

    if type(cue_vectors) == dict:
        # TODO: convert dict to xarray here
        raise NotImplementedError('dicts are not supported yet.')

    if not cue_vectors.data.data.c_contiguous:
        raise ValueError('cue_vectors have to be c_contiguous')
    if not cue_vectors.dtype == np.float64:
        raise ValueError('cue_vectors have to be of dtype np.float64')

    weights_ini = weights
    wall_time_start = time.perf_counter()
    cpu_time_start = time.process_time()

    # preprocessing
    n_events, cues_from_events, outcomes_from_events = count.cues_outcomes(events,
                                                                           n_jobs=n_jobs,
                                                                           verbose=verbose)

    cues_from_events = list(cues_from_events.keys())
    cues = list(cue_vectors.coords['cues'].data)
    outcomes = list(outcomes_from_events.keys())

    cue_map = OrderedDict(((cue, ii) for ii, cue in enumerate(cues)))
    outcome_map = OrderedDict(((outcome, ii) for ii, outcome in enumerate(outcomes)))

    if set(cues_from_events) - set(cues):
        raise ValueError("all cues in events need to be specified as rows in cue_vectors")

    del outcomes_from_events, cues_from_events

    shape = (len(outcomes), cue_vectors.shape[1])

    if not cue_vectors.dims[1] == 'cue_vector_dimensions':
        raise ValueError("The second dimension of the 'cue_vectors' has to be named 'cue_vector_dimensions'.")

    cue_vector_dimensions = cue_vectors['cue_vector_dimensions']

    # initialize weights
    if weights is None:
        weights = np.ascontiguousarray(np.zeros(shape, dtype=np.float64, order='C'))
    elif isinstance(weights, xr.DataArray):
        if not all(cue_vector_dimensions == weights['cue_vector_dimensions']):
            raise ValueError("Cue vector dimensions names do not match in weights and cue_vectors")

        old_outcomes = weights.coords["outcomes"].values.tolist()
        new_outcomes = list(set(outcomes) - set(old_outcomes))
        outcomes = old_outcomes + new_outcomes

        outcome_map = OrderedDict(((outcome, ii) for ii, outcome in enumerate(outcomes)))
        weights_tmp = np.concatenate((weights.values,
                                      np.zeros((len(new_outcomes), len(cue_vector_dimensions)),
                                               dtype=np.float64, order='C')),
                                     axis=0)
        weights = np.ascontiguousarray(weights_tmp)
        del weights_tmp, old_outcomes, new_outcomes
    else:
        raise ValueError('weights need to be None or xarray.DataArray with method=%s' % method)

    weights = xr.DataArray(weights,
                           coords={'outcomes': outcomes, 'cue_vector_dimensions': cue_vector_dimensions},
                           dims=('outcomes', 'cue_vector_dimensions'))
    del shape, cue_vector_dimensions

    with tempfile.TemporaryDirectory(prefix="pyndl", dir=temporary_directory) as binary_path:
        number_events = preprocess.create_binary_event_files(events, binary_path, cue_map,
                                                             outcome_map, overwrite=True,
                                                             n_jobs=n_jobs,
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
        if not weights.data.data.c_contiguous:
            raise ValueError('weights has to be c_contiguous')
        if method == 'openmp':
            if not sys.platform.startswith('linux'):
                raise NotImplementedError("OpenMP is linux only at the moment."
                                          "Use method='threading' instead.")
            beta1, beta2 = betas
            ndl_openmp.learn_inplace_real_to_binary(binary_files,
                                                    beta1,
                                                    beta2,
                                                    lambda_,
                                                    cue_vectors.data,
                                                    weights.data,
                                                    n_outcomes_per_job,
                                                    n_jobs)
        else:
            # TODO: implement threading
            raise ValueError('TODO: for now: method needs to be "openmp"')

        weights = weights.reset_coords(drop=True)

    cpu_time_stop = time.process_time()
    wall_time_stop = time.perf_counter()
    cpu_time = cpu_time_stop - cpu_time_start
    wall_time = wall_time_stop - wall_time_start

    if weights_ini is not None:
        attrs_to_be_updated = weights_ini.attrs
    else:
        attrs_to_be_updated = None

    attrs = ndl._attributes(events, number_events, 'cue_vectors', betas, lambda_, cpu_time, wall_time,
                            __name__ + "." + ndl.__name__, method=method, attrs=attrs_to_be_updated)

    # post-processing
    weights.attrs = attrs
    return weights


def _wh_real_to_real(events, eta, cue_vectors, outcome_vectors, *,
                     method='openmp', weights=None,
                     n_jobs=8, n_outcomes_per_job=10, remove_duplicates=None,
                     verbose=False, temporary_directory=None,
                     events_per_temporary_file=10000000):
    """
    Calculate the weights for all events using the Widrow-Hoff learning rule
    from cue vectors onto outcome vectors.

    This is a parallel python implementation using cython, numpy,
    multithreading and the binary format defined in preprocess.py.

    Parameters
    ----------
    events : str
        path to the event file
    eta : float
        learning rate
    cue_vectors : xarray.DataArray
        matrix that contains the cue vectors for each cue
    outcome_vectors : xarray.DataArray
        matrix that contains the target vectors for each outcome
    method : {'openmp', 'threading', 'numpy'}
    weights : None or xarray.DataArray
        the xarray.DataArray needs to have the dimensions
        'cue_vector_dimensions' and 'outcome_vector_dimensions'
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
        be used like '/tmp' on unix
    events_per_temporary_file: int
        Number of events in each temporary binary file. Has to be larger than 1

    Returns
    -------
    weights : xarray.DataArray
        with dimensions 'outcome_vector_dimensions' and
        'cue_vector_dimensions'. You can lookup the weights between a
        outcome_vector_dimension and a cue_vector_dimension with
        ``weights.loc[{'outcome_vector_dimensions': outcome_vector_dimension,
            'cue_vector_dimensions': cue_vector_dimension}]`` or
        ``weights.loc[outcome_vector_dimension].loc[cue_vector_dimension]``.

    """

    if not (remove_duplicates is None or isinstance(remove_duplicates, bool)):
        raise ValueError("remove_duplicates must be None, True or False")
    if not isinstance(events, str):
        raise ValueError("'events' need to be the path to a gzipped event file not {}".format(type(events)))

    weights_ini = weights
    wall_time_start = time.perf_counter()
    cpu_time_start = time.process_time()

    if type(cue_vectors) == dict:
        # TODO: convert dict to xarray here
        raise NotImplementedError('dicts are not supported yet.')

    if type(outcome_vectors) == dict:
        # TODO: convert dict to xarray here
        raise NotImplementedError('dicts are not supported yet.')

    if not cue_vectors.data.data.c_contiguous:
        raise ValueError('cue_vectors have to be c_contiguous')
    if not cue_vectors.dtype == np.float64:
        raise ValueError('cue_vectors have to be of dtype np.float64')

    if not outcome_vectors.data.data.c_contiguous:
        raise ValueError('outcome_vectors have to be c_contiguous')
    if not outcome_vectors.dtype == np.float64:
        raise ValueError('outcome_vectors have to be of dtype np.float64')

    # preprocessing
    n_events, cues_from_events, outcomes_from_events = count.cues_outcomes(events,
                                                                           n_jobs=n_jobs,
                                                                           verbose=verbose)

    cues_from_events = list(cues_from_events.keys())
    cues = list(cue_vectors.coords['cues'].data)
    outcomes_from_events = list(outcomes_from_events.keys())
    outcomes = list(outcome_vectors.coords['outcomes'].data)
    cue_map = OrderedDict(((cue, ii) for ii, cue in enumerate(cues)))
    outcome_map = OrderedDict(((outcome, ii) for ii, outcome in enumerate(outcomes)))

    # check for unseen outcomes in events
    if set(outcomes_from_events) - set(outcomes):
        raise ValueError("all outcomes in events need to be specified as rows in outcome_vectors")
    if set(cues_from_events) - set(cues):
        raise ValueError("all cues in events need to be specified as rows in cue_vectors")

    del outcomes_from_events, cues_from_events

    shape = (outcome_vectors.shape[1], cue_vectors.shape[1])

    if not outcome_vectors.dims[1] == 'outcome_vector_dimensions':
        raise ValueError("The second dimension of the 'outcome_vectors' has to be named 'outcome_vector_dimensions'.")
    if not cue_vectors.dims[1] == 'cue_vector_dimensions':
        raise ValueError("The second dimension of the 'cue_vectors' has to be named 'cue_vector_dimensions'.")

    cue_dims = cue_vectors['cue_vector_dimensions']
    outcome_dims = outcome_vectors['outcome_vector_dimensions']

    # initialize weights
    if weights is None:
        weights = np.ascontiguousarray(np.zeros(shape, dtype=np.float64, order='C'))
        weights = xr.DataArray(weights,
                               coords={'outcome_vector_dimensions': outcome_dims, 'cue_vector_dimensions': cue_dims},
                               dims=('outcome_vector_dimensions', 'cue_vector_dimensions'))
        del cue_dims, outcome_dims
    elif isinstance(weights, xr.DataArray):
        if not weights.shape == shape:
            raise ValueError("Vector dimensions need to match in continued learning!")
        if not all(outcome_dims == weights['outcome_vector_dimensions']):
            raise ValueError("Outcome vector dimensions names do not match in weights and outcome_vectors")
        if not all(cue_dims == weights['cue_vector_dimensions']):
            raise ValueError("Cue vector dimensions names do not match in weights and cue_vectors")
        # align the cue and outcome vector dimension names between the old weights
        # and the cue_vectors / outcome_vectors
        weights = weights.loc[{'outcome_vector_dimensions': outcome_dims, 'cue_vector_dimensions': cue_dims}]
        weights = weights.copy()
    else:
        raise ValueError('weights need to be None or xarray.DataArray with method=%s' % method)
    del shape

    if not weights.data.data.c_contiguous:
        raise ValueError('weights has to be c_contiguous')
    if method == 'numpy':
        event_generator = io.events_from_file(events)
        number_events = 0

        if verbose:
            print('start learning...')
        for cues, outcomes in event_generator:
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

            # TODO: implement multiple cues / outcomes in numpy
            assert len(outcomes) == 1, 'for method=numpy only one outcome is allowed per event'
            assert len(cues) == 1, 'for method_numpy only one cue is allowed per event'

            cue_vec = cue_vectors.loc[cues[0]]
            outcome_vec = outcome_vectors.loc[outcomes[0]]

            prediction_vec = weights.dot(cue_vec)  # why is the @ not working?
            error = outcome_vec - prediction_vec
            weights += eta * error * cue_vec  # broadcasted array multiplication
            # NOTE: we could calculate the same weights first on the first half
            # of the outcome vector dimensions and then on the second half and
            # row bind both in the end. Do we? Yes, and we can use this to
            # multiprocess the numpy computation.
    elif method in ('openmp', 'threading'):
        with tempfile.TemporaryDirectory(prefix="pyndl", dir=temporary_directory) as binary_path:
            number_events = preprocess.create_binary_event_files(events, binary_path, cue_map,
                                                                 outcome_map, overwrite=True,
                                                                 n_jobs=n_jobs,
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
                if not sys.platform.startswith('linux'):
                    raise NotImplementedError("OpenMP is linux only at the moment."
                                              "Use method='threading' instead.")
                ndl_openmp.learn_inplace_real_to_real(binary_files,
                                                      eta,
                                                      cue_vectors.data,
                                                      outcome_vectors.data,
                                                      weights.data,
                                                      n_outcomes_per_job,
                                                      n_jobs)
            else:
                # TODO: implement threading
                raise ValueError('TODO: for now: method needs to be "numpy" or "openmp"')

            weights = weights.reset_coords(drop=True)
    else:
        raise ValueError('method needs to be "numpy" or "openmp"')

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
                 'pyndl': _format(pyndl_version),
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
