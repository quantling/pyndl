#!/bin/python

import multiprocessing as mp
import numpy as np
from functools import partial
import ctypes


def _sp_activation_matrix(events_cues, weights, cues):
    act_m = np.empty((len(events_cues), weights.shape[1]), dtype=np.float64)
    for row, event_cues in enumerate(events_cues):
        cue_indices = [cues[cue] for cue in event_cues]
        act_m[row, :] = weights[cue_indices, :].sum(axis=0)
    return act_m


def _init_mp_activation_matrix(weights_, weights_shape_, activations_, activations_shape_):
    global weights, activations
    weights = np.ctypeslib.as_array(weights_)
    weights.shape = weights_shape_
    activations = np.ctypeslib.as_array(activations_)
    activations.shape = activations_shape_


def _run_mp_activation_matrix(row, event_cues, cues):
    cue_indices = [cues[cue] for cue in event_cues]
    activations[row, :] = weights[cue_indices, :].sum(axis=0)


def _mp_activation_matrix(events_cues, weights, cues, numThreads):
    activations_dim = (len(events_cues), weights.shape[1])
    shared_activations = mp.RawArray(ctypes.c_double, int(np.prod(activations_dim)))
    weights_dim = weights.shape
    shared_weights = mp.sharedctypes.copy(np.ctypeslib.as_ctypes(np.float64(weights)))

    initargs = (shared_weights, weights_dim, shared_activations, activations_dim)
    with mp.Pool(numThreads, initializer=_init_mp_activation_matrix, initargs=initargs) as pool:
        pool.starmap(partial(_run_mp_activation_matrix, cues=cues), enumerate(events_cues))
    activations = np.ctypeslib.as_array(shared_activations)
    activations.shape = activations_dim
    return activations


def activation_matrix(events, weights, cues, numThreads=1):
    """Estimate activations for given event cues

    Memory overhead for multiprocessing is one copy of weights
    plus a copy of cues for each thread.

    events: Iteratable of events as cues lists or strings separated by underline
    weights: Weight matrix as 2d numpy.array with shape (cues, weights)
    cues: List of cue strings labeling weights axis 0 or dict with cue_string: row_index
    numThreads: number of cores for multiprocessing. Has memory overhead if > 1 (see above).
    """
    assert len(weights) == len(cues), "Cues label and weight matrix rows differ."
    assert numThreads >= 1, "Can't run with less than 1 thread"

    events = [event.split("_") if isinstance(event, str) else event for event in events]
    if not isinstance(cues, dict):
        cues = {x: i for i, x in enumerate(cues)}

    if numThreads == 1:
        activations = _sp_activation_matrix(events, weights, cues)
    else:
        activations = _mp_activation_matrix(events, weights, cues, numThreads=numThreads)
    return activations
