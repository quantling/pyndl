#!/bin/python

import multiprocessing as mp
import numpy as np
import ctypes


def _sp_activation_matrix(events_cues, weights):
    act_m = np.empty((len(events_cues), weights.shape[1]), dtype=np.float64)
    for row, event_cues in enumerate(events_cues):
        act_m[row, :] = weights[event_cues, :].sum(axis=0)
    return act_m


def _init_mp_activation_matrix(weights_, weights_shape_, activations_, activations_shape_):
    global weights, activations
    weights = np.ctypeslib.as_array(weights_)
    weights.shape = weights_shape_
    activations = np.ctypeslib.as_array(activations_)
    activations.shape = activations_shape_


def _run_mp_activation_matrix(row, event_cues):
    activations[row, :] = weights[event_cues, :].sum(axis=0)


def _mp_activation_matrix(events_cues, weights, numThreads):
    activations_dim = (len(events_cues), weights.shape[1])
    shared_activations = mp.RawArray(ctypes.c_double, int(np.prod(activations_dim)))
    weights_dim = weights.shape
    if not weights.flags.contiguous:
        weights = weights.copy()
    shared_weights = mp.sharedctypes.copy(np.ctypeslib.as_ctypes(np.float64(weights)))

    initargs = (shared_weights, weights_dim, shared_activations, activations_dim)
    with mp.Pool(numThreads, initializer=_init_mp_activation_matrix, initargs=initargs) as pool:
        pool.starmap(_run_mp_activation_matrix, enumerate(events_cues))
    activations = np.ctypeslib.as_array(shared_activations)
    activations.shape = activations_dim
    return activations


def activation_matrix(events, weights, cues, numThreads=1):
    """Estimate activations for given event cues

    Memory overhead for multiprocessing is one copy of weights
    plus a copy of cues for each thread.

    Args:
        events: Iteratable of events as cues lists or strings separated by underline
        weights: Weight matrix as 2d numpy.array with shape (cues, weights)
        cues: List of cue strings labeling weights axis 0 or dict with cue_string: row_index
        numThreads: number of cores for multiprocessing. Has memory overhead if > 1 (see above).
    Returns:
        activation: matrix
        new_cues: cues not present in weight matrix and ignored
    """
    assert len(weights) == len(cues), "Cues label and weight matrix rows differ."
    assert numThreads >= 1, "Can't run with less than 1 thread"

    if not isinstance(cues, dict):
        cues = {x: i for i, x in enumerate(cues)}

    new_cues = set()

    def prepare_event(event, new_cues):
        if isinstance(event, str):
            event = event.split("_")
        event_indexed = [cues.get(cue) for cue in event]
        new_cues_event = {event[i] for i, c in enumerate(event_indexed) if c is None}
        if len(new_cues_event) > 0:
            new_cues |= new_cues_event
            return [cue for cue in event_indexed if cue is not None]
        else:
            return event_indexed
    events = [prepare_event(event, new_cues) for event in events]

    if numThreads == 1:
        activations = _sp_activation_matrix(events, weights)
    else:
        activations = _mp_activation_matrix(events, weights, numThreads=numThreads)
    return activations, new_cues
