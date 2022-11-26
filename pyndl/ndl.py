"""
pyndl.ndl
---------

*pyndl.ndl* provides functions in order to train NDL models

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
import warnings
import types

import cython
import pandas as pd
import numpy as np
import xarray as xr

from . import __version__ as pyndl_version
from . import count
from . import preprocess
from . import ndl_parallel
from . import io

# conditional import as openmp is only compiled for linux
if sys.platform.startswith('linux'):
    from . import ndl_openmp
elif sys.platform.startswith('win32'):
    pass
elif sys.platform.startswith('darwin'):
    pass


warnings.simplefilter('always', DeprecationWarning)


class WeightDict(defaultdict):
    # pylint: disable=missing-docstring

    """
    Subclass of defaultdict to represent outcome-cue weights.

    Notes
    -----
    Weight for each outcome-cue combination is 0 per default.

    """

    # pylint: disable=W0613
    def __init__(self, *args, **kwargs):
        super().__init__(lambda: defaultdict(float))

        self._attrs = OrderedDict()

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


def ndl(events, alpha, betas, lambda_=1.0, *,
        method='openmp', weights=None,
        number_of_threads=None, n_jobs=8, len_sublists=None, n_outcomes_per_job=10,
        remove_duplicates=None,
        verbose=False, temporary_directory=None,
        events_per_temporary_file=10000000):
    """
    Calculate the weights for all_outcomes over all events in event_file
    given by the files path.

    This is a parallel python implementation using numpy, multithreading and
    the binary format defined in preprocess.py.

    Parameters
    ----------
    events : generator or str
        generates cues, outcomes pairs or the path to the event file
    alpha : float
        saliency of all cues
    betas : (float, float)
        one value for successful prediction (reward) one for punishment
    lambda\\_ : float

    method : {'openmp', 'threading'}
    weights : None or xarray.DataArray
        the xarray.DataArray needs to have the named dimensions 'cues' and 'outcomes'
    n_jobs : int
        a integer giving the number of threads in which the job should
        executed
    n_outcomes_per_job : int
        a integer giving the length of sublists generated from all outcomes
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

    # Create temporary file if events is a generator
    if isinstance(events, types.GeneratorType):
        file_path = tempfile.NamedTemporaryFile().name
        io.events_to_file(events, file_path)
        events = file_path
        del file_path

    if number_of_threads is not None:
        warnings.warn("Parameter `number_of_threads` is renamed to `n_jobs`. The old name "
                      "will stop working with v0.9.0.",
                      DeprecationWarning, stacklevel=2)
        n_jobs = number_of_threads
    if len_sublists is not None:
        warnings.warn("Parameter `len_sublists` is renamed to `n_outcomes_per_job`. The old name "
                      "will stop working with v0.9.0.",
                      DeprecationWarning, stacklevel=2)
        n_outcomes_per_job = len_sublists

    if not (remove_duplicates is None or isinstance(remove_duplicates, bool)):
        raise ValueError("remove_duplicates must be None, True or False")
    if not isinstance(events, (str, os.PathLike)):
        raise ValueError("'events' need to be the path to a gzipped event file not {}".format(type(events)))

    weights_ini = weights
    wall_time_start = time.perf_counter()
    cpu_time_start = time.process_time()

    # preprocessing
    n_events, cues, outcomes = count.cues_outcomes(events,
                                                   n_jobs=n_jobs,
                                                   verbose=verbose)
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

    if any(length > 4294967295 for length in weights.shape):
        raise ValueError("Neither number of cues nor outcomes shall exceed 4294967295 "
                         "for now. See https://github.com/quantling/pyndl/issues/169")

    beta1, beta2 = betas

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
            ndl_openmp.learn_inplace_binary_to_binary(binary_files,
                                                      alpha,
                                                      beta1,
                                                      beta2,
                                                      lambda_,
                                                      weights,
                                                      np.array(all_outcome_indices,
                                                               dtype=np.uint32),
                                                      n_outcomes_per_job,
                                                      n_jobs)
        elif method == 'threading':
            part_lists = slice_list(all_outcome_indices, n_outcomes_per_job)

            working_queue = Queue(len(part_lists))
            threads = []
            queue_lock = threading.Lock()

            def worker():
                while True:
                    with queue_lock:
                        if working_queue.empty():
                            break
                        data = working_queue.get()
                    ndl_parallel.learn_inplace_binary_to_binary(binary_files,
                                                                alpha,
                                                                beta1,
                                                                beta2,
                                                                lambda_,
                                                                weights,
                                                                data)

            with queue_lock:
                for partlist in part_lists:
                    working_queue.put(np.array(partlist, dtype=np.uint32))

            for _ in range(n_jobs):
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

    attrs = _attributes(events, number_events, alpha, betas, lambda_, cpu_time, wall_time,
                        __name__ + "." + ndl.__name__, method=method, attrs=attrs_to_be_updated)

    # post-processing
    weights = xr.DataArray(weights, [('outcomes', outcomes), ('cues', cues)],
                           attrs=attrs)
    return weights


def _attributes(event_path, number_events, alpha, betas, lambda_, cpu_time,
                wall_time, function, method=None, attrs=None):
    if not isinstance(alpha, (float, int)):
        alpha_str = 'varying'
    else:
        alpha_str = str(alpha)

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

    def _format(value):
        return '{0: <{width}}'.format(value, width=width)

    new_attrs = {'date': _format(time.strftime("%Y-%m-%d %H:%M:%S")),
                 'event_path': _format(event_path),
                 'number_events': _format(number_events),
                 'alpha': _format(alpha_str),
                 'betas': _format(str(betas)),
                 'lambda': _format(str(lambda_)),
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


def dict_ndl(events, alphas, betas, lambda_=1.0, *,
             weights=None, inplace=False, remove_duplicates=None,
             make_data_array=False, verbose=False):
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
    events : generator or str
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

    beta1, beta2 = betas
    all_outcomes = set(weights.keys())

    if isinstance(events, str):
        events = io.events_from_file(events)
    if isinstance(alphas, float):
        alpha = alphas
        alphas = defaultdict(lambda: alpha)
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
        weights = data_array(weights, attrs=attrs)
    else:
        weights.attrs = attrs

    return weights


def data_array(weights, *, attrs=None):
    """
    Calculate the weights for all_outcomes over all events in event_file.

    Parameters
    ----------
    weights : dict of dicts of floats or WeightDict
        the first dict has outcomes as keys and dicts as values
        the second dict has cues as keys and weights as values
        weights[outcome][cue] gives the weight between outcome and cue.
        If a dict of dicts is given, attrs is required. If a WeightDict is
        given, attrs is optional
    attrs : dict
        A dictionary of attributes

    Returns
    -------
    weights : xarray.DataArray
        with dimensions 'outcomes' and 'cues'. You can lookup the weights
        between a cue and an outcome with ``weights.loc[{'outcomes': outcome,
        'cues': cue}]`` or ``weights.loc[outcome].loc[cue]``.
    """

    if isinstance(weights, xr.DataArray) and weights.dims == ('outcomes', 'cues'):
        return weights

    if attrs is None:
        try:
            attrs = weights.attrs
        except AttributeError:
            raise AttributeError("weights does not have attributes and no attrs "
                                 "argument is given.")

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

    return weights


def slice_list(list_, len_sublists):
    r"""
    Slices a list in sublists with the length len_sublists.

    Parameters
    ----------
    list\_ : list
         list which should be sliced in sublists
    len_sublists : int
         integer which determines the length of the sublists

    Returns
    -------
    seq_list : list of lists
        a list of sublists with the length len_sublists

    """
    if len_sublists < 1:
        raise ValueError("'len_sublists' must be larger then one")
    assert len(list_) == len(set(list_))
    ii = 0
    seq_list = list()
    while ii < len(list_):
        seq_list.append(list_[ii:ii + len_sublists])
        ii = ii + len_sublists

    return seq_list
