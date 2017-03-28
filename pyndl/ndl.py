from collections import defaultdict, OrderedDict
import os
import tempfile
import time
import getpass
import socket
import copy

import threading
from queue import Queue

import numpy as np
import pandas as pd
import xarray as xr
import cython

from . import __version__
from . import count
from . import preprocess
from . import ndl_parallel

BINARY_PATH = tempfile.mkdtemp()


def events(event_path):
    """
    Yields events for all events in event_file.

    Parameters
    ----------
    event_path : str
        path to event file

    Yields
    ------
    cues, outcomes : list, list
        a tuple of two lists containing cues and outcomes

    """
    with open(event_path, 'rt') as event_file:
        # skip header
        event_file.readline()
        for line in event_file:
            cues, outcomes = line.strip('\n').split('\t')
            cues = cues.split('_')
            outcomes = outcomes.split('_')
            yield (cues, outcomes)


def ndl(event_path, alpha, betas, lambda_=1.0, *,
        method='openmp', weights=None,
        number_of_threads=8, sequence=10, remove_duplicates=None):
    """
    Calculate the weights for all_outcomes over all events in event_file
    given by the files path.

    This is a parallel python implementation using numpy, multithreading and
    the binary format defined in preprocess.py.

    Parameters
    ----------
    event_path : str
        path to the event file
    alpha : float
        saliency of all cues
    betas : (float, float)
        one value for successful prediction (reward) one for punishment
    lambda\\_ : float

    method : {'openmp', 'threading'}
    weights : None or xarray.DataArray
        the xarray.DataArray needs to have the dimensions 'cues' and 'outcomes'
    number_of_threads : int
        a integer giving the number of threads in which the job should
        executed
    sequence : int
        a integer giving the length of sublists generated from all outcomes
    remove_duplicates : {None, True, False}
        if None though a ValueError when the same cue is present multiple times
        in the same event; True make cues and outcomes unique per event; False
        keep multiple instances of the same cue or outcome (this is usually not
        preferred!)

    Returns
    -------
    weights : xarray.DataArray
        with dimensions 'cues' and 'outcomes'. You can lookup the weights
        between a cue and an outcome with ``weights.loc[{'outcomes': outcome,
        'cues': cue}]`` or ``weights.loc[outcome].loc[cue]``.

    """

    if not (remove_duplicates is None or isinstance(remove_duplicates, bool)):
        raise ValueError("remove_duplicates must be None, True or False")

    weights_ini = weights
    wall_time_start = time.perf_counter()
    cpu_time_start = time.process_time()

    # preprocessing
    cues, outcomes = count.cues_outcomes(event_path, number_of_processes=number_of_threads)
    cues = list(cues.keys())
    outcomes = list(outcomes.keys())
    cue_map = OrderedDict(((cue, ii) for ii, cue in enumerate(cues)))
    outcome_map = OrderedDict(((outcome, ii) for ii, outcome in enumerate(outcomes)))

    all_outcome_indices = [outcome_map[outcome] for outcome in outcomes]

    shape = (len(outcome_map), len(cue_map))

    # initialize weights
    if weights is None:
        weights = np.ascontiguousarray(np.zeros(shape, dtype=np.float64, order='C'))
    elif isinstance(weights, xr.DataArray):
        old_cues = weights.coords["cues"].values.tolist()
        new_cues = list(set(cues) - set(old_cues))
        old_outcomes = weights.coords["outcomes"].values.tolist()
        new_outcomes = list(set(outcomes) - set(old_outcomes))

        cues = old_cues + new_cues
        outcomes = old_outcomes + new_outcomes

        cue_map = OrderedDict(((cue, ii) for ii, cue in enumerate(cues)))
        outcome_map = OrderedDict(((outcome, ii) for ii, outcome in enumerate(outcomes)))

        all_outcome_indices = [outcome_map[outcome] for outcome in outcomes]

        weights_tmp = np.concatenate((weights.values,
                                      np.zeros((len(new_outcomes), len(old_cues)),
                                               dtype=np.float64, order='C')),
                                     axis=0)
        weights_tmp = np.concatenate((weights_tmp,
                                      np.zeros((len(outcomes), len(new_cues)),
                                               dtype=np.float64, order='C')),
                                     axis=1)

        weights = np.ascontiguousarray(weights_tmp)

        del weights_tmp, old_cues, new_cues, old_outcomes, new_outcomes
    else:
        raise ValueError('weights need to be None or xarray.DataArray with method=%s' % method)

    beta1, beta2 = betas

    preprocess.create_binary_event_files(event_path, BINARY_PATH, cue_map,
                                         outcome_map, overwrite=True,
                                         number_of_processes=number_of_threads,
                                         remove_duplicates=remove_duplicates)
    binary_files = [os.path.join(BINARY_PATH, binary_file)
                    for binary_file in os.listdir(BINARY_PATH)
                    if os.path.isfile(os.path.join(BINARY_PATH, binary_file))]
    # learning
    if method == 'openmp':
        ndl_parallel.learn_inplace(binary_files, weights, alpha,
                                   beta1, beta2, lambda_,
                                   np.array(all_outcome_indices, dtype=np.uint32),
                                   sequence, number_of_threads)
    elif method == 'threading':
        part_lists = slice_list(all_outcome_indices, sequence)

        working_queue = Queue(len(part_lists))
        threads = []
        queue_lock = threading.Lock()

        def _worker():
            while True:
                with queue_lock:
                    if working_queue.empty():
                        break
                    data = working_queue.get()
                ndl_parallel.learn_inplace_2(binary_files, weights, alpha,
                                             beta1, beta2, lambda_, data)

        with queue_lock:
            for partlist in part_lists:
                working_queue.put(np.array(partlist, dtype=np.uint32))

        for _ in range(number_of_threads):
            thread = threading.Thread(target=_worker)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
    else:
        raise ValueError('method needs to be either "threading" or "openmp"')

    cpu_time_stop = time.process_time()
    wall_time_stop = time.perf_counter()
    cpu_time = cpu_time_stop - cpu_time_start
    wall_time = wall_time_stop - wall_time_start

    attrs = _attributes(event_path, alpha, betas, lambda_, cpu_time, wall_time,
                        __name__ + "." + ndl.__name__, method=method)

    if weights_ini is not None:
        attrs_to_be_updated = weights_ini.attrs
        for key in attrs_to_be_updated.keys():
            attrs_to_be_updated[key] += ' | ' + attrs[key]
        attrs = attrs_to_be_updated

    # post-processing
    weights = xr.DataArray(weights, [('outcomes', outcomes), ('cues', cues)],
                           attrs=attrs)
    return weights


def _attributes(event_path, alpha, betas, lambda_, cpu_time, wall_time,
                function, method=None):
    width = max([len(value) for value in (event_path,
                                          str(alpha),
                                          str(betas),
                                          str(lambda_),
                                          function,
                                          str(method),
                                          socket.gethostname(),
                                          getpass.getuser())])
    width = max(19, width)

    def _format(value):
        return '{0: <{width}}'.format(value, width=width)

    attrs = {'date': _format(time.strftime("%Y-%m-%d %H:%M:%S")),
             'event_path': _format(event_path),
             'alpha': _format(str(alpha)),
             'betas': _format(str(betas)),
             'lambda': _format(str(lambda_)),
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
    return attrs


def dict_ndl(event_list, alphas, betas, lambda_=1.0, *,
             weights=None, inplace=False, remove_duplicates=None, make_data_array=False):
    """
    Calculate the weights for all_outcomes over all events in event_file.

    This is a pure python implementation using dicts.

    Notes
    -----
    The metadata will only be stored when `make_data_array` is True and then
    `dict_ndl` cannot be used to continue learning. At the moment there is no
    proper way to automatically store the meta data into the default dict.

    Parameters
    ----------
    event_list : generator or str
        generates cues, outcomes pairs or the path to the event file
    alphas : dict or float
        a (default)dict having cues as keys and a value below 1 as value
    betas : (float, float)
        one value for successful prediction (reward) one for punishment
    lambda\\_ : float
    weights : dict of dicts or None
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

    Returns
    -------
    weights : dict of dicts of floats
        the first dict has outcomes as keys and dicts as values
        the second dict has cues as keys and weights as values
        weights[outcome][cue] gives the weight between outcome and cue.

    or

    weights : xarray.DataArray
        with dimensions 'cues' and 'outcomes'. You can lookup the weights
        between a cue and an outcome with ``weights.loc[{'outcomes': outcome,
        'cues': cue}]`` or ``weights.loc[outcome].loc[cue]``.

    """

    if not isinstance(make_data_array, bool):
        raise ValueError("make_data_array must be True or False")

    if not (remove_duplicates is None or isinstance(remove_duplicates, bool)):
        raise ValueError("remove_duplicates must be None, True or False")

    if make_data_array:
        wall_time_start = time.perf_counter()
        cpu_time_start = time.process_time()
        if isinstance(event_list, str):
            event_path = event_list
        else:
            event_path = None

    # weights can be seen as an infinite outcome by cue matrix
    # weights[outcome][cue]
    if weights is None:
        weights = defaultdict(lambda: defaultdict(float))
    elif not isinstance(weights, defaultdict):
        raise ValueError('weights needs to be either defaultdict or None')

    if not inplace:
        weights = copy.deepcopy(weights)

    beta1, beta2 = betas
    all_outcomes = set(weights.keys())

    if isinstance(event_list, str):
        event_list = events(event_list)
    if isinstance(alphas, float):
        alpha = alphas
        alphas = defaultdict(lambda: alpha)

    for cues, outcomes in event_list:
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

        all_outcomes.update(outcomes)
        for outcome in all_outcomes:
            association_strength = sum(weights[outcome][cue] for cue in cues)
            if outcome in outcomes:
                update = beta1 * (lambda_ - association_strength)
            else:
                update = beta2 * (0 - association_strength)
            for cue in cues:
                weights[outcome][cue] += alphas[cue] * update

    if make_data_array:
        cpu_time_stop = time.process_time()
        wall_time_stop = time.perf_counter()
        cpu_time = cpu_time_stop - cpu_time_start
        wall_time = wall_time_stop - wall_time_start

        attrs = _attributes(event_path, alphas, betas, lambda_, cpu_time, wall_time,
                            __name__ + "." + dict_ndl.__name__)

        # post-processing
        weights = pd.DataFrame(weights)
        # weights.fillna(0.0, inplace=True)  # TODO make sure to not remove real NaNs
        weights = xr.DataArray(weights.T, dims=('outcomes', 'cues'), attrs=attrs)

    return weights


def slice_list(list_, sequence):
    r"""
    Slices a list in sublists with the length sequence.

    Parameters
    ----------
    list\_ : list
         list which should be sliced in sublists
    sequence : int
         integer which determines the length of the sublists

    Returns
    -------
    seq_list : list of lists
        a list of sublists with the length sequence

    """
    if sequence < 1:
        raise ValueError("sequence must be larger then one")
    assert len(list_) == len(set(list_))
    ii = 0
    seq_list = list()
    while ii < len(list_):
        seq_list.append(list_[ii:ii+sequence])
        ii = ii+sequence

    return seq_list
