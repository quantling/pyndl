from collections import defaultdict, OrderedDict
import multiprocessing
import time
import os
import pyximport #; pyximport.install()

import threading
from queue import Queue

import numpy as np

from . import count
from . import preprocess
from . import ndl_c
from . import ndl_parallel


try:
    from numba import jit
except ImportError:
    jit = lambda x: x

# Path where the binary resources are temporaly stored
BINARY_PATH = os.path.join(os.path.dirname(__file__), "tests/binary_resources/")

def events(event_path, *, frequency=False):
    """
    Yields events for all events in event_file.

    Parameters
    ==========
    event_path : str
        path to event file
    frequency : bool
        frequency should be in the event_file

    Yields
    ======
    cues, outcomes : list, list
        a tuple of two lists containing cues and outcomes

    """
    with open(event_path, 'rt') as event_file:
        # skip header
        event_file.readline()
        if not frequency:
            for line in event_file:
                cues, outcomes = line.strip('\n').split('\t')
                cues = cues.split('_')
                outcomes = outcomes.split('_')
                yield (cues, outcomes)
        else:
            for line in event_file:
                cues, outcomes, frequency = line.strip('\n').split('\t')
                cues = cues.split('_')
                outcomes = outcomes.split('_')
                frequency = int(frequency)
                for _ in range(frequency):
                    yield (cues, outcomes)

def thread_ndl_simple(event_path, alpha, betas, lambda_, *,
                                       number_of_threads=2, sequence=10):
    """
    Calculate the weights for all_outcomes over all events in event_file
    given by the files path.

    This is a parallel python implementation using numpy, multiprocessing and
    the binary format defined in preprocess.py.

    Parameters
    ==========
    event_path : str
        path to the event file
    alpha : float
        saliency of all cues
    betas : (float, float)
        one value for successful prediction (reward) one for punishment
    lambda_ : float

    number_of_threads : int
        a integer giving the number of threads in which the job should
        executed
    sequence : int
        a integer giving the length of sublists generated from all outcomes

    Returns
    =======
    weights : numpy.array of shape len(outcomes), len(cues)
        weights[outcome_index][cue_index] gives the weight between outcome and cue.

    """

    # preprocessing
    cue_map, outcome_map, all_outcome_indices = generate_mapping(
                                                    event_path,
                                                    number_of_processes=2,
                                                    binary=True)

    preprocess.create_binary_event_files(event_path, BINARY_PATH, cue_map,
                                         outcome_map, overwrite=True,
                                         number_of_processes=2)

    shape = (len(outcome_map), len(cue_map))
    weights = np.ascontiguousarray(np.zeros(shape, dtype=np.float64, order='C'))
    beta1, beta2 = betas
    binary_files = [os.path.join(BINARY_PATH, binary_file)
                    for binary_file in os.listdir(BINARY_PATH)
                    if os.path.isfile(os.path.join(BINARY_PATH, binary_file))]

    part_lists = slice_list(all_outcome_indices,sequence)

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
                                            beta1, beta2, lambda_,
                                            data)

    with queue_lock:
        for partlist in part_lists:
            working_queue.put(np.array(partlist, dtype=np.uint32))

    for thread_id in range(number_of_threads):
        thread = threading.Thread(target=worker)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return weights

def openmp_ndl_simple(event_path, alpha, betas, lambda_, *,
                                             number_of_threads=8, sequence=10):
    """
    Calculate the weights for all_outcomes over all events in event_file
    given by the files path.

    This is a parallel python implementation using numpy, multiprocessing and
    the binary format defined in preprocess.py.

    Parameters
    ==========
    event_path : str
        path to the event file
    alpha : float
        saliency of all cues
    betas : (float, float)
        one value for successful prediction (reward) one for punishment
    lambda_ : float

    number_of_threads : int
        a integer giving the number of threads in which the job should
        executed
    sequence : int
        a integer giving the length of sublists generated from all outcomes

    Returns
    =======
    weights : numpy.array of shape len(outcomes), len(cues)
        weights[outcome_index][cue_index] gives the weight between outcome and cue.

    """

    # preprocessing
    cue_map, outcome_map, all_outcome_indices = generate_mapping(
                                                    event_path,
                                                    number_of_processes=2,
                                                    binary=True)

    preprocess.create_binary_event_files(event_path, BINARY_PATH, cue_map,
                                         outcome_map, overwrite=True,
                                         number_of_processes=2)

    shape = (len(outcome_map), len(cue_map))
    weights = np.ascontiguousarray(np.zeros(shape, dtype=np.float64, order='C'))
    beta1, beta2 = betas
    binary_files = [os.path.join(BINARY_PATH, binary_file)
                    for binary_file in os.listdir(BINARY_PATH)
                    if os.path.isfile(os.path.join(BINARY_PATH, binary_file))]

    ndl_parallel.learn_inplace(binary_files, weights, alpha,
                                beta1, beta2, lambda_,
                                np.array(all_outcome_indices, dtype=np.uint32),
                                sequence,
                                number_of_threads)

    return weights

def dict_ndl(event_list, alphas, betas, all_outcomes):
    """
    Calculate the weigths for all_outcomes over all events in event_file.

    This is a pure python implementation using dicts.

    Parameters
    ==========
    events : generator or str
        generates cues, outcomes pairs or the path to the event file
    alphas : dict
        a (default)dict having cues as keys and a value below 1 as value
    betas : dict
        a (default)dict having outcomes as keys and a value below 1 as value
    all_outcomes : list
        a list of all outcomes of interest

    Returns
    =======
    weights : dict of dicts of floats
        the first dict has outcomes as keys and dicts as values
        the second dict has cues as keys and weights as values
        weights[outcome][cue] gives the weight between outcome and cue.

    """
    lambda_ = 1.0
    beta1, beta2 = betas
    # weights can be seen as an infinite outcome by cue matrix
    # weights[outcome][cue]
    weights = defaultdict(lambda: defaultdict(float))

    if isinstance(event_list, str):
        event_list = events(event_list)

    for cues, outcomes in event_list:
        for outcome in all_outcomes:
            association_strength = sum(weights[outcome][cue] for cue in cues)
            if outcome in outcomes:
                update = beta1 * (lambda_ - association_strength)
            else:
                update = beta2 * (0 - association_strength)
            for cue in cues:
                weights[outcome][cue] += alphas[cue] * update

    return weights

def dict_ndl_simple(event_path, alpha, betas, lambda_):
    """
    Calculate the weigths for all_outcomes over all events in event_file.

    This is a pure python implementation using dicts.

    Parameters
    ==========
    events : generator or str
        generates cues, outcomes pairs or the path to the event file
    alpha : floatall_outcomes
    betas : (float, float)
        one value for successful prediction (reward) one for punishment
    all_outcomes : list
        a list of all outcomes of interest

    Returns
    =======
    weights : dict of dicts of floats
        the first dict has outcomes as keys and dicts as values
        the second dict has cues as keys and weights as values
        weights[outcome][cue] gives the weight between outcome and cue.

    """
    # weights can be seen as an infinite outcome by cue matrix
    # weights[outcome][cue]
    weights = defaultdict(lambda: defaultdict(float))

    beta1, beta2 = betas

    all_outcomes = generate_all_outcomes(event_path)
    event_list = events(event_path, frequency=True)

    for cues, outcomes in event_list:
        for outcome in all_outcomes:
            association_strength = sum(weights[outcome][cue] for cue in cues)
            if outcome in outcomes:
                update = beta1 * (lambda_ - association_strength)
            else:
                update = beta2 * (0 - association_strength)
            for cue in cues:
                weights[outcome][cue] += alpha * update

    return weights

def activations(cues, weights):
    if isinstance(weights, dict):
        activations_ = defaultdict(float)
        for outcome, cue_dict in weights.items():
            for cue in cues:
                activations_[outcome] += cue_dict[cue]
        return activations_

# NOTE: In the original code some stuff was differently handled for multiple
# cues and multiple outcomes.

def generate_all_outcomes(event_path):
    """
    Generates a list of all outcomes of the event_file.

    Parameters
    ==========
    event_path : str
        path to the event_file for which the mapping should be generated

    Returns
    =======
    all_outcomes : list
        a list of all outcomes in the event file
    """
    cues, outcomes = count.cues_outcomes(event_path, number_of_processes=2)
    all_outcomes = list(outcomes.keys())

    return all_outcomes

def generate_mapping(event_path, number_of_processes=2, binary=False): # TODO find better name
    """
    Generates OrderedDicts of all cues and outcomes to use indizes in the numpy
    implementation.

    Parameters
    ==========
    event_path : str
        path to the event_file for which the mapping should be generated
    number_of_processes : int
         integer of how many processes should be used

    Returns
    =======
    cue_map: OrderedDict
        a OrderedDict mapping all cues to indizes
    outcome_map: OrderedDict
        a OrderedDict mapping all outcomes to indizes
    all_outcomes : list
        a list of all outcomes in the event file
    """
    cues, outcomes = count.cues_outcomes(event_path, number_of_processes=number_of_processes)
    all_cues = list(cues.keys())
    all_outcomes = list(outcomes.keys())
    cue_map = OrderedDict(((cue, ii) for ii, cue in enumerate(all_cues)))
    outcome_map = OrderedDict(((outcome, ii) for ii, outcome in enumerate(all_outcomes)))

    if binary:
        all_outcome_indices = [outcome_map[outcome] for outcome in all_outcomes]
        return (cue_map, outcome_map, all_outcome_indices)
    else:
        return (cue_map, outcome_map, all_outcomes)


def slice_list(li, sequence):
    """
    Slices a list in sublists with the length sequence.

    Parameters
    ==========
    li : list
         list which should be sliced in sublists
    sequence : int
         integer which determines the length of the sublists

    Returns
    =======
    seq_list : list of lists
        a list of sublists with the length sequence

    """
    assert len(li) == len(set(li))
    ii = 0
    seq_list = list()
    while ii < len(li):
        seq_list.append(li[ii:ii+sequence])
        ii = ii+sequence

    return seq_list

if __name__ == '__main__':
    with open('tests/resources/event_file.tab', 'rt') as event_file:
        events_ = events(event_file)
        all_outcomes = ('była', 'tak', 'skupiona', 'brzucha', 'botoksem',
                        'kolagenem')
        weights = dict_ndl(events_, defaultdict(lambda: 0.01), defaultdict(lambda: 0.01), all_outcomes)
        for outcome, cues in weights.items():
            print('Outcome: %s' % str(outcome))
            for cue, value in cues.items():
                print('  %s = %f' % (str(cue), value))
