#!/bin/python

import multiprocessing as mp
import ctypes
from collections import defaultdict, OrderedDict

import numpy as np
import xarray as xr

from . import ndl


def activation(event_list, weights, number_of_threads=1, remove_duplicates=None, ignore_missing_cues=True):
    """
    Estimate activations for given events in event file and cue-outcome weights.

    Memory overhead for multiprocessing is one copy of weights
    plus a copy of cues for each thread.

    Parameters
    ----------
    event_list : generator or str
        generates cues, outcomes pairs or the path to the event file
    weights : xarray.DataArray or dict[dict[float]]
        the xarray.DataArray needs to have the dimensions 'cues' and 'outcomes'
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
        the weightmatrix

    Returns
    -------
    activations : xarray.DataArray
        with dimensions 'events' and 'outcomes'. Contains coords for the outcomes.
        returned if weights is instance of xarray.DataArray

    or

    activations : dict of numpy.arrays
        the first dict has outcomes as keys and dicts as values
        the list has a activation value per event
        returned if weights is instance of dict

    """
    if isinstance(event_list, str):
        event_list = ndl.events(event_list)

    event_cues_list = (cues for cues, outcomes in event_list)
    if remove_duplicates is None:
        def enforce_no_duplicates(cues):
            if len(cues) != len(set(cues)):
                raise ValueError('cues needs to be unique: "%s"; use '
                                 'remove_duplicates=True' %
                                 (' '.join(cues)))
            else:
                return set(cues)
        event_cues_list = (enforce_no_duplicates(cues) for cues in event_cues_list)
    elif remove_duplicates is True:
        event_cues_list = (set(cues) for cues in event_cues_list)

    if isinstance(weights, xr.DataArray):
        cues = weights.coords["cues"].values.tolist()
        outcomes = weights.coords["outcomes"].values.tolist()
        cue_map = OrderedDict(((cue, ii) for ii, cue in enumerate(cues)))
        event_cue_indices_list = (tuple(cue_map[cue] for cue in event_cues if cue in cues)
                                  for event_cues in event_cues_list)
        activations = _activation_matrix(list(event_cue_indices_list), weights.values, number_of_threads)
        return xr.DataArray(activations,
                            coords={
                                'outcomes': outcomes
                            },
                            dims=('events', 'outcomes'))
    elif isinstance(weights, dict):
        assert number_of_threads == 1, "Estimating activations with multiprocessing is not implemented for dicts."
        activations = defaultdict(lambda: np.zeros(len(event_cues_list)))
        event_cues_list = list(event_cues_list)
        for outcome, cue_dict in weights.items():
            _activations = activations[outcome]
            for row, cues in enumerate(event_cues_list):
                for cue in cues:
                    _activations[row] += cue_dict[cue]
        return activations
    else:
        raise ValueError("Weights other than xarray.DataArray or dicts are not supported.")


def _init_mp_activation_matrix(weights_, weights_shape_, activations_, activations_shape_):
    """
    Private helper function for multiprocessing in _activation_matrix.
    Initializes shared variables weights and activations.

    """
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
    activations[event_index, :] = weights[cue_indices, :].sum(axis=0)


def _activation_matrix(indices_list, weights, number_of_threads):
    """
    Estimate activation for indices in weights

    Memory overhead for multiprocessing is one copy of weights
    plus a copy of cues for each thread.

    Parameters
    ----------
    indices_list : list with iteratables containing the indices of the cues in weight matrix.
    weights : Weight matrix as 2d numpy.array with shape (cues, weights)
    number_of_threads : int
        a integer giving the number of threads in which the job should
        executed

    Returns
    -------
    activation_matrix : 2d numpy.array
        activations for the events and all outcomes in the weights and

    """
    assert number_of_threads >= 1, "Can't run with less than 1 thread"

    activations_dim = (len(indices_list), weights.shape[1])
    if number_of_threads == 1:
        activations = np.empty(activations_dim, dtype=np.float64)
        for row, event_cues in enumerate(indices_list):
            activations[row, :] = weights[event_cues, :].sum(axis=0)
        return activations
    else:
        shared_activations = mp.RawArray(ctypes.c_double, int(np.prod(activations_dim)))
        weights = np.ascontiguousarray(weights)
        shared_weights = mp.sharedctypes.copy(np.ctypeslib.as_ctypes(np.float64(weights)))
        initargs = (shared_weights, weights.shape, shared_activations, activations_dim)
        with mp.Pool(number_of_threads, initializer=_init_mp_activation_matrix, initargs=initargs) as pool:
            pool.starmap(_run_mp_activation_matrix, enumerate(indices_list))
        activations = np.ctypeslib.as_array(shared_activations)
        activations.shape = activations_dim
        return activations
