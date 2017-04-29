from collections import defaultdict, OrderedDict
import copy
import getpass
import gzip
import os
from queue import Queue
import socket
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
from . import ndl_parallel


def events(event_path):
    """
    Yields events for all events in a gzipped event_file.

    Parameters
    ----------
    event_path : str
        path to gzipped event file

    Yields
    ------
    cues, outcomes : list, list
        a tuple of two lists containing cues and outcomes

    """
    with gzip.open(event_path, 'rt') as event_file:
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
        with dimensions 'outcomes' and 'cues'. You can lookup the weights
        between a cue and an outcome with ``weights.loc[{'outcomes': outcome,
        'cues': cue}]`` or ``weights.loc[outcome].loc[cue]``.

    """

    if not (remove_duplicates is None or isinstance(remove_duplicates, bool)):
        raise ValueError("remove_duplicates must be None, True or False")

    weights_ini = weights
    wall_time_start = time.perf_counter()
    cpu_time_start = time.process_time()

    # preprocessing
    n_events, cues, outcomes = count.cues_outcomes(event_path,
                                                   number_of_processes=number_of_threads)
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

    with tempfile.TemporaryDirectory(prefix="pyndl") as binary_path:
        number_events = preprocess.create_binary_event_files(event_path, binary_path, cue_map,
                                                             outcome_map, overwrite=True,
                                                             number_of_processes=number_of_threads,
                                                             remove_duplicates=remove_duplicates)
        assert n_events == number_events, (str(n_events) + ' ' + str(number_events))
        binary_files = [os.path.join(binary_path, binary_file)
                        for binary_file in os.listdir(binary_path)
                        if os.path.isfile(os.path.join(binary_path, binary_file))]
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

            def worker():
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

            for thread_id in range(number_of_threads):
                thread = threading.Thread(target=worker)
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

    if weights_ini is not None:
        attrs_to_be_updated = weights_ini.attrs
    else:
        attrs_to_be_updated = None

    attrs = _attributes(event_path, number_events, alpha, betas, lambda_, cpu_time, wall_time,
                        __name__ + "." + ndl.__name__, method=method, attrs=attrs_to_be_updated)

    # post-processing
    weights = xr.DataArray(weights, [('outcomes', outcomes), ('cues', cues)],
                           attrs=attrs)
    return weights


def _attributes(event_path, number_events, alpha, betas, lambda_, cpu_time,
                wall_time, function, method=None, attrs=None):
    width = max([len(ss) for ss in (event_path,
                                    str(number_events),
                                    str(alpha),
                                    str(betas),
                                    str(lambda_),
                                    function,
                                    str(method),
                                    socket.gethostname(),
                                    getpass.getuser())])
    width = max(19, width)

    def format_(ss):
        return '{0: <{width}}'.format(ss, width=width)

    new_attrs = {'date': format_(time.strftime("%Y-%m-%d %H:%M:%S")),
                 'event_path': format_(event_path),
                 'number_events': format_(number_events),
                 'alpha': format_(str(alpha)),
                 'betas': format_(str(betas)),
                 'lambda': format_(str(lambda_)),
                 'function': format_(function),
                 'method': format_(str(method)),
                 'cpu_time': format_(str(cpu_time)),
                 'wall_time': format_(str(wall_time)),
                 'hostname': format_(socket.gethostname()),
                 'username': format_(getpass.getuser()),
                 'pyndl': format_(__version__),
                 'numpy': format_(np.__version__),
                 'pandas': format_(pd.__version__),
                 'xarray': format_(xr.__version__),
                 'cython': format_(cython.__version__)}

    if attrs is not None:
        for key in set(attrs.keys()) | set(new_attrs.keys()):
            if key in attrs:
                old_val = attrs[key]
            else:
                old_val = ''
            if key in new_attrs:
                new_val = new_attrs[key]
            else:
                new_val = format_('')
            new_attrs[key] = old_val + ' | ' + new_val
    return new_attrs


class WeightDict(defaultdict):
    # pylint: disable=missing-docstring

    """
    Subclass of defaultdict to represent outcome-cue weights.

    Notes
    -----
    Weight for each outcome-cue combination is 0 per default.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(lambda: defaultdict(float))

        if 'attrs' in kwargs:
            self.attrs = kwargs['attrs']
        else:
            self.attrs = {}

    @property
    def attrs(self):
        return self._attrs

    @attrs.setter
    def attrs(self, attrs):
        self._attrs = OrderedDict(attrs)


def dict_ndl(event_list, alphas, betas, lambda_=1.0, *,
             weights=None, inplace=False, remove_duplicates=None,
             make_data_array=False):
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
    if isinstance(event_list, str):
        event_path = event_list
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
        weights_ini = weights
        attrs_to_update = weights_ini.attrs
        coords = weights_ini.coords
        weights = WeightDict()
        for oi, outcome in enumerate(coords['outcomes'].values):
            for ci, cue in enumerate(coords['cues'].values):
                weights[outcome][cue] = weights_ini.item((oi, ci))
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
    number_events = 0

    for cues, outcomes in event_list:
        number_events += 1
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

    cpu_time_stop = time.process_time()
    wall_time_stop = time.perf_counter()
    cpu_time = cpu_time_stop - cpu_time_start
    wall_time = wall_time_stop - wall_time_start
    attrs = _attributes(event_path, number_events, alphas, betas, lambda_, cpu_time, wall_time,
                        __name__ + "." + dict_ndl.__name__, attrs=attrs_to_update)

    if make_data_array:
        outcomes = list(weights.keys())
        cues = set()
        for outcome in outcomes:
            cues.update(set(weights[outcome].keys()))

        cues = list(cues)

        weights_dict = weights
        shape = (len(outcomes), len(cues))
        weights = xr.DataArray(np.zeros(shape), attrs=attrs,
                               coords={'outcomes': outcomes, 'cues': cues},
                               dims=('outcomes', 'cues'))

        for outcome in outcomes:
            for cue in cues:
                weights.loc[{"outcomes": outcome, "cues": cue}] = weights_dict[outcome][cue]
    else:
        weights.attrs = attrs

    return weights


def slice_list(li, sequence):
    """
    Slices a list in sublists with the length sequence.

    Parameters
    ----------
    li : list
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
    assert len(li) == len(set(li))
    ii = 0
    seq_list = list()
    while ii < len(li):
        seq_list.append(li[ii:ii+sequence])
        ii = ii+sequence

    return seq_list
