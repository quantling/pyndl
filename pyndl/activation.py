"""
pyndl.activation
----------------

*pyndl.activation* provides the functionality to estimate activation of a
trained ndl model for given events. The trained ndl model is thereby
represented as the outcome-cue weights.
"""
import multiprocessing as mp
import ctypes
from collections import defaultdict, OrderedDict
from typing import Iterable, List, Dict, Optional, Tuple, Union

import numpy as np
import xarray as xr


from numpy import ndarray
from xarray.core.dataarray import DataArray

from . import io
from .types import AnyWeights, CollectionEvent, AnyEvent, Path, CueCollection, Collection


# pylint: disable=W0621
def activation(events: Union[Path, Iterable[AnyEvent]],
               weights: AnyWeights,
               number_of_threads: int = 1,
               remove_duplicates: Optional[bool] = None,
               ignore_missing_cues: bool = False) -> Union[DataArray, Dict[str, ndarray]]:
    """
    Estimate activations for given events in event file and outcome-cue weights.

    Memory overhead for multiprocessing is one copy of weights
    plus a copy of cues for each thread.

    Parameters
    ----------
    events : generator or str
        generates cues, outcomes pairs or the path to the event file
    weights : xarray.DataArray or dict[dict[float]]
        the xarray.DataArray needs to have the dimensions 'outcomes' and 'cues'
        the dictionaries hold weight[outcome][cue].
    number_of_threads : int
        a integer giving the number of threads in which the job should
        executed
    remove_duplicates : {None, True, False}
        if None raise a ValueError when the same cue is present multiple times
        in the same event; True make cues unique per event; False
        keep multiple instances of the same cue (this is usually not
        preferred!)
    ignore_missing_cues : {True, False}
        if True function ignores cues which are in the test dataset but not in
        the weight matrix
        if False raises a KeyError for cues which are not in the weight matrix

    Returns
    -------
    activations : xarray.DataArray
        with dimensions 'outcomes' and 'events'. Contains coords for the outcomes.
        returned if weights is instance of xarray.DataArray

    or

    activations : dict of numpy.arrays
        the first dict has outcomes as keys and dicts as values
        the list has a activation value per event
        returned if weights is instance of dict

    """
    event_list = []  # type: Iterable[CollectionEvent]
    if isinstance(events, Path):
        event_list = io.events_from_file(events)
    else:
        event_list = events

    cues_gen = (cues for cues, outcomes in event_list)  # type: Iterable[CueCollection]
    if remove_duplicates is None:
        def check_no_duplicates(cues):
            if len(cues) != len(set(cues)):
                raise ValueError('cues needs to be unique: "{}"; use '
                                 'remove_duplicates=True'.format(' '.join(cues)))
            else:
                return set(cues)
        cues_gen = (check_no_duplicates(cues) for cues in cues_gen)
    elif remove_duplicates is True:
        cues_gen = (set(cues) for cues in cues_gen)

    if isinstance(weights, xr.DataArray):
        cues = weights.coords["cues"].values.tolist()
        outcomes = weights.coords["outcomes"].values.tolist()
        if not weights.values.shape == (len(outcomes), len(cues)):
            raise ValueError('dimensions of weights are wrong. Probably you need to transpose the matrix')
        cue_map = OrderedDict(((cue, ii) for ii, cue in enumerate(cues)))
        if ignore_missing_cues:
            event_cue_indices_list = (tuple(cue_map[cue] for cue in event_cues if cue in cues)
                                      for event_cues in cues_gen)
        else:
            event_cue_indices_list = (tuple(cue_map[cue] for cue in event_cues)
                                      for event_cues in cues_gen)
        # pylint: disable=W0621
        activations = _activation_matrix(list(event_cue_indices_list),
                                         weights.values, number_of_threads)
        return xr.DataArray(activations,
                            coords={
                                'outcomes': outcomes
                            },
                            dims=('outcomes', 'events'))
    elif isinstance(weights, dict):
        assert number_of_threads == 1, "Estimating activations with multiprocessing is not implemented for dicts."
        cues_list = list(cues_gen)
        activation_dict = defaultdict(lambda: np.zeros(len(cues_list)))  # type: Dict[str, ndarray]
        for outcome, cue_dict in weights.items():
            _activations = activation_dict[outcome]
            for row, cues in enumerate(cues_list):
                for cue in cues:
                    _activations[row] += cue_dict[cue]  # type: ignore
        return activation_dict
    else:
        raise ValueError("Weights other than xarray.DataArray or dicts are not supported.")


def _init_mp_activation_matrix(weights_, weights_shape_, activations_, activations_shape_):
    """
    Private helper function for multiprocessing in _activation_matrix.
    Initializes shared variables weights and activations.

    """
    # pylint: disable=C0103, W0621, W0601
    global weights, activations
    weights = np.ctypeslib.as_array(weights_)
    weights.shape = weights_shape_
    activations = np.ctypeslib.as_array(activations_)
    activations.shape = activations_shape_


def _run_mp_activation_matrix(event_index, cue_indices):
    """
    Private helper function for multiprocessing in _activation_matrix.
    Calculate activation for all outcomes while a event.

    """
    activations[:, event_index] = weights[:, cue_indices].sum(axis=1)


def _activation_matrix(indices_list: List[Tuple[int, ...]],
                       weights: ndarray, number_of_threads: int) -> ndarray:
    """
    Estimate activation for indices in weights

    Memory overhead for multiprocessing is one copy of weights
    plus a copy of cues for each thread.

    Parameters
    ----------
    indices_list : list[int]
        events as cue indices in weights
    weights : numpy.array
        weight matrix with shape (outcomes, cues)
    number_of_threads : int

    Returns
    -------
    activation_matrix : numpy.array
        estimated activations as matrix with shape (outcomes, events)

    """
    assert number_of_threads >= 1, "Can't run with less than 1 thread"

    activations_dim = (weights.shape[0], len(indices_list))
    if number_of_threads == 1:
        activations = np.empty(activations_dim, dtype=np.float64)
        for row, event_cues in enumerate(indices_list):
            activations[:, row] = weights[:, event_cues].sum(axis=1)
        return activations
    else:
        #  type stubs seem to be incorrect for multiprocessing lib. 2018-05-16
        shared_activations = mp.RawArray(ctypes.c_double, int(np.prod(activations_dim)))  # type: ignore
        weights = np.ascontiguousarray(weights)
        shared_weights = mp.sharedctypes.copy(np.ctypeslib.as_ctypes(np.float64(weights)))  # type: ignore
        initargs = (shared_weights, weights.shape, shared_activations, activations_dim)
        with mp.Pool(number_of_threads, initializer=_init_mp_activation_matrix, initargs=initargs) as pool:
            pool.starmap(_run_mp_activation_matrix, enumerate(indices_list))
        activations = np.ctypeslib.as_array(shared_activations)  # type: ignore
        activations.shape = activations_dim
        return activations
