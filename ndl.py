from collections import defaultdict, OrderedDict
import multiprocessing
import time
import os
import pyximport; pyximport.install()

import threading
from queue import Queue

import numpy as np

from . import count
from . import preprocess
from . import ndl_c


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

def dict_ndl_parrallel(event_path, alphas, betas, all_outcomes, *,
                       number_of_processes=2, sequence=10,
                       frequency_in_event_file=False):
    """
    Calculate the weights for all_outcomes over all events in event_file
    given by the files path.

    This is a parallel python implementation using dicts and multiprocessing.

    Parameters
    ==========
    event_path : path to the event file
    alpha : dict
        saliency dict of all cues
    betas : dict
        a (default)dict having outcomes as keys and a value below 1 as value
    all_outcomes : list
        a list of all outcomes of interest
    number_of_processes : int
        a integer giving the number of processes in which the job should
        executed
    sequence : int
        a integer giving the length of sublists generated from all outcomes
    frequency_in_event_file : bool
        is the frequency column in the event file present?

    Returns
    =======
    weights : dict of dicts of floats
        the first dict has outcomes as keys and dicts as values
        the second dict has cues as keys and weights as values
        weights[outcome][cue] gives the weight between outcome and cue.

    """
    with multiprocessing.Pool(number_of_processes) as pool:

        job = JobCalculateWeights(event_path,
                                  alphas,
                                  betas,
                                  frequency_in_event_file=frequency_in_event_file)

        weights = defaultdict(lambda:defaultdict(float))

        partlists_of_outcomes = slice_list(all_outcomes,sequence)

        for result in pool.imap_unordered(job.dict_ndl_weight_calculator,
                                          partlists_of_outcomes):
            for outcome, cues in result:
                weights[outcome] = cues

        return weights

def numpy_ndl_parrallel(event_path, alphas, betas, all_outcomes, *, cue_map,
                        outcome_map, number_of_processes=2, sequence=10,
                        frequency_in_event_file=False):
    """
    Calculate the weights for all_outcomes over all events in event_file
    given by the files path.

    This is a parallel python implementation using numpy and multiprocessing.

    Parameters
    ==========
    event_path : path to the event file
    alphas : numpy.array
        saliency array of all cues
    betas : (float, float)
    all_outcomes : list
        a list of all outcomes of interest
    cue_map: dict
        a mapping between cues and index for the numpy array
    outcome_map: dict
        a mapping between outcomes and index for the numpy array
    number_of_processes : int
        a integer giving the number of processes in which the job should
        executed
    sequence : int
        a integer giving the length of sublists generated from all outcomes
    frequency_in_event_file : bool
        is the frequency column in the event file present?

    Returns
    =======
    weights : numpy.array of shape len(outcomes), len(cues)
        weights[outcome_index][cue_index] gives the weight between outcome and cue.

    """


    with multiprocessing.Pool(number_of_processes) as pool:

        job = JobCalculateWeights(event_path,
                                  alphas,
                                  betas,
                                  frequency_in_event_file=frequency_in_event_file,
                                  cue_map=cue_map,
                                  outcome_map=outcome_map)

        weights = np.zeros((len(outcome_map), len(cue_map)), dtype=float)

        partlists_of_outcomes = slice_list(all_outcomes,sequence)

        for result in pool.imap_unordered(job.numpy_ndl_weight_calculator, partlists_of_outcomes):
            weights = np.add(weights, result)


        return weights

def binary_numpy_ndl_parrallel(event_path, alpha, betas, lambda_, *,
                                number_of_processes=2, sequence=10):
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

    number_of_processes : int
        a integer giving the number of processes in which the job should
        executed
    sequence : int
        a integer giving the length of sublists generated from all outcomes

    Returns
    =======
    weights : numpy.array of shape len(outcomes), len(cues)
        weights[outcome_index][cue_index] gives the weight between outcome and cue.

    """

    cue_map, outcome_map, all_outcome_indices = generate_mapping(
                                                    event_path,
                                                    number_of_processes=number_of_processes,
                                                    binary=True)

    preprocess.create_binary_event_files(event_path, BINARY_PATH, cue_map,
                                         outcome_map, overwrite=True,
                                         number_of_processes=number_of_processes)


    with multiprocessing.Pool(number_of_processes) as pool:
        shape = (len(outcome_map),len(cue_map))
        weights = np.zeros(shape, dtype=float)
        job = JobCalculateWeightsInplace(BINARY_PATH,
                                         alpha,
                                         betas,
                                         lambda_,
                                         shape)

        partlists_of_outcome_indices = slice_list(all_outcome_indices,sequence)

        for result in pool.imap_unordered(job.binary_ndl_weight_calculator, partlists_of_outcome_indices):
            weights = np.add(weights, result)


        return weights

def binary_inplace_numpy_ndl(event_path, alpha, betas, lambda_, *,
                                       number_of_processes=2):
    """
    Calculate the weights for all_outcomes over all events in event_file
    given by the files path.

    Parameters
    ==========
    event_path : str
        path to the event file
    alpha : float
        saliency of all cues
    betas : (float, float)
        one value for successful prediction (reward) one for punishment
    lambda_ : float

    number_of_processes : int
        a integer giving the number of processes in which the job should
        executed

    Returns
    =======
    weights : numpy.array of shape len(outcomes), len(cues)
        weights[outcome_index][cue_index] gives the weight between outcome and cue.

    """
    # preprocessing
    cue_map, outcome_map, all_outcome_indices = generate_mapping(
                                                    event_path,
                                                    number_of_processes=number_of_processes,
                                                    binary=True)

    preprocess.create_binary_event_files(event_path, BINARY_PATH, cue_map,
                                         outcome_map, overwrite=True,
                                         number_of_processes=number_of_processes)


    shape = (len(outcome_map),len(cue_map))
    weights = np.zeros(shape, dtype=float)

    binary_files = [os.path.join(BINARY_PATH, binary_file)
                    for binary_file in os.listdir(BINARY_PATH)
                    if os.path.isfile(os.path.join(BINARY_PATH, binary_file))]
    binary_files.reverse()

    beta1, beta2 = betas

    for binary_file in binary_files:
        ndl_c.learn_inplace(binary_file, weights, alpha, beta1, beta2, lambda_,
                                      np.array(all_outcome_indices))
    return weights

def binary_inplace_numpy_ndl_parrallel(event_path, alpha, betas, lambda_, *,
                                       number_of_processes=2, sequence=10):
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

    number_of_processes : int
        a integer giving the number of processes in which the job should
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
                                                    number_of_processes=number_of_processes,
                                                    binary=True)

    preprocess.create_binary_event_files(event_path, BINARY_PATH, cue_map,
                                         outcome_map, overwrite=True,
                                         number_of_processes=number_of_processes)

    shape = (len(outcome_map),len(cue_map))
    weights = np.zeros(shape, dtype=float)
    # multiprocessing
    with multiprocessing.Pool(number_of_processes) as pool:

        job_inplace = JobCalculateWeightsInplace(BINARY_PATH,
                                         alpha,
                                         betas,
                                         lambda_,
                                         shape)

        partlists_of_outcome_indices = slice_list(all_outcome_indices,sequence)

        for result in pool.imap_unordered(job_inplace.binary_inplace_ndl_weight_calculator, partlists_of_outcome_indices):
            weights = np.add(weights, result)

        return weights

def binary_inplace_numpy_ndl_parrallel_thread(event_path, alpha, betas, lambda_, *,
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
                                                    number_of_processes=number_of_threads,
                                                    binary=True)

    preprocess.create_binary_event_files(event_path, BINARY_PATH, cue_map,
                                         outcome_map, overwrite=True,
                                         number_of_processes=number_of_threads)

    shape = (len(outcome_map),len(cue_map))
    weights = np.zeros(shape, dtype=float)
    beta1, beta2 = betas
    binary_files = [os.path.join(BINARY_PATH, binary_file)
                    for binary_file in os.listdir(BINARY_PATH)
                    if os.path.isfile(os.path.join(BINARY_PATH, binary_file))]

    lock = threading.Lock()
    exit_flag = 0

    def do_work(part_list):
        weights = np.zeros(shape, dtype=float)
        for binary_file in binary_files:
            ndl_c.learn_inplace(binary_file, weights, alpha,
                                beta1, beta2, lambda_,
                                np.array(part_list))

        results.append(weights)

    def worker():
        while exit_flag != 1:
            part_list  = q.get()
            do_work(part_list)
            q.task_done()

    q = Queue()
    results = []
    threads = []

    # Init Threads
    for i in range(number_of_threads):
         t = threading.Thread(target=worker)
         t.daemon = True
         t.start()
         threads.append(t)

    # Fill Queue
    partlists_of_outcome_indices = slice_list(all_outcome_indices,sequence)
    for part_list in partlists_of_outcome_indices:
        q.put(part_list)

    q.join()
    exit_flag = 1

    for result in results:
        weights = np.add(weights, result)

    return weights


class JobCalculateWeightsInplace():
    """
        Methods are used as workers for the multiprocessed implementations
    """
    def __init__(self, binary_event_path, alpha, betas, lambda_, shape):
        self.binary_files = [os.path.join(binary_event_path, binary_file)
                        for binary_file in os.listdir(binary_event_path)
                        if os.path.isfile(os.path.join(binary_event_path, binary_file))]
        self.binary_files.reverse()
        self.event_path = binary_event_path
        self.alpha = alpha
        beta1, beta2 = betas
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda_ = lambda_
        self.shape = shape

    def binary_inplace_ndl_weight_calculator(self,part_outcome_indices):
        weights = np.zeros(self.shape, dtype=float)

        for binary_file in self.binary_files:
            ndl_c.learn_inplace(binary_file, weights, self.alpha,
                                          self.beta1, self.beta2, self.lambda_,
                                          np.array(part_outcome_indices))

        return weights

    def binary_ndl_weight_calculator(self,part_outcome_indices):
        weights = np.zeros(self.shape, dtype=float)
        for binary_file in self.binary_files:
            binary_events = preprocess.read_binary_file(binary_file)
            for cue_indices, outcome_indices in binary_events:
                _update_numpy_array_inplace(weights,
                                            cue_indices, outcome_indices,
                                            part_outcome_indices, self.alpha,
                                            self.beta1, self.beta2,
                                            self.lambda_)

#                ndl_c._update_numpy_array_inplace(weights,
#                                                  np.array(cue_indices),
#                                                  np.array(outcome_indices),
#                                                  np.array(part_outcome_indices),
#                                                  self.alpha, self.beta1,
#                                                  self.beta2, self.lambda_)
        return weights

class JobCalculateWeights():
    """
    Stores the values of alphas and betas an the path to the event file

    Method is used as a worker for the multiprocessed dict_ndl implementation

    """

    def __init__(self, event_path, alphas, betas, *,
                 frequency_in_event_file=False, cue_map=None, outcome_map=None):
        self.event_path = event_path
        self.alphas = alphas
        self.betas = betas
        self.frequency_in_event_file = frequency_in_event_file
        self.cue_map = cue_map
        self.outcome_map = outcome_map

    def dict_ndl_weight_calculator(self,part_outcomes):
        events_ = events(self.event_path, frequency=self.frequency_in_event_file)
        weights = dict_ndl(events_, self.alphas, self.betas, part_outcomes)
        return [(outcome,cues) for outcome, cues in weights.items()]

    def numpy_ndl_weight_calculator(self,part_outcomes):
        events_ = events(self.event_path, frequency=self.frequency_in_event_file)
        weigths = numpy_ndl(events_, self.alphas, self.betas, part_outcomes,
                            cue_map=self.cue_map, outcome_map=self.outcome_map)
        return weigths


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

def dict_ndl_simple(event_list, alpha, betas, lambda_, all_outcomes):
    """
    Calculate the weigths for all_outcomes over all events in event_file.

    This is a pure python implementation using dicts.

    Parameters
    ==========
    events : generator or str
        generates cues, outcomes pairs or the path to the event file
    alpha : float
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
                weights[outcome][cue] += alpha * update

    return weights

@jit
def _update_numpy_array_inplace(weights, cue_indices, outcome_indices,
                                all_outcome_indices, alpha, beta1, beta2,
                                lambda_):
    for outcome_index in all_outcome_indices:
        association_strength = np.sum(weights[outcome_index][cue_indices])
        if outcome_index in outcome_indices:
            update = beta1 * (lambda_ - association_strength)
        else:
            update = beta2 * (0 - association_strength)
        for cue_index in cue_indices:
            weights[outcome_index][cue_index] += alpha * update

    return weights

def numpy_ndl(event_list, alphas, betas, all_outcomes, *, cue_map, outcome_map):
    """
    Calculate the weigths for all_outcomes over all events in event_file.

    This is a python implementation using numpy.

    Parameters
    ==========
    event_path : str
        generates cues, outcomes pairs or the path to the event file
    alphas : numpy array
        saliency array of all cues
    betas : (float, float)
        one value for successful prediction (reward) one for punishment
    all_outcomes : list
        a list of all outcomes of interest
    cue_map : dict
        has cues as keys and int as values, where the int represents the index in the
        numpy array.
    outcome_map : dict
        has outcomes as keys and int as values, where the int represents the index in the
        numpy array.

    Returns
    =======
    weights : numpy.array of shape len(outcomes), len(cues)
        weights[outcome_index][cue_index] gives the weight between outcome and cue.

    """
    lambda_ = 1.0

    all_outcome_indices = [outcome_map[outcome] for outcome in all_outcomes]

    weights = np.zeros((len(outcome_map), len(cue_map)), dtype=float)

    beta1, beta2 = betas

    if isinstance(event_list, str):
        event_list = events(event_list)

    for cues, outcomes in event_list:
        cue_indices = [cue_map[cue] for cue in cues]
        outcome_indices = [outcome_map[outcome] for outcome in outcomes]
        for outcome_index in all_outcome_indices:
            association_strength = np.sum(weights[outcome_index][cue_indices])
            if outcome_index in outcome_indices:
                update = beta1 * (lambda_ - association_strength)
            else:
                update = beta2 * (0 - association_strength)
            for cue_index in cue_indices:
                weights[outcome_index][cue_index] += alphas[cue_index] * update
    return weights

def binary_numpy_ndl(event_path, alpha, betas, lambda_, *,
                     number_of_processes=2):
    """
    Calculate the weigths for all_outcomes over all events in a binary_event_file.

    This is a python implementation using numpy.

    Parameters
    ==========
    event_path : str
        path to the event file
    weights: numpy array
        weight matrix on which the algortihm learns
    alphas : numpy array
        saliency array of all cues
    betas : (float, float)
        one value for successful prediction (reward) one for punishment
    all_outcomes : list
        a list of all outcomes of interest

    Returns
    =======
    weights : numpy.array of shape len(outcomes), len(cues)
        weights[outcome_index][cue_index] gives the weight between outcome and cue.

    """

    cue_map, outcome_map, all_outcome_indices = generate_mapping(
                                                    event_path,
                                                    number_of_processes=number_of_processes,
                                                    binary=True)

    preprocess.create_binary_event_files(event_path, BINARY_PATH, cue_map,
                                         outcome_map, overwrite=True,
                                         number_of_processes=number_of_processes)

    weights = np.zeros((len(outcome_map),len(cue_map)), dtype=float)

    binary_files = [os.path.join(BINARY_PATH, binary_file)
                    for binary_file in os.listdir(BINARY_PATH)
                    if os.path.isfile(os.path.join(BINARY_PATH, binary_file))]
    binary_files.reverse()

    beta1, beta2 = betas

    for binary_file in binary_files:
        binary_events = preprocess.read_binary_file(binary_file)
        for cue_indices, outcome_indices in binary_events:
            ndl_c._update_numpy_array_inplace(weights,
                                              np.array(cue_indices),
                                              np.array(outcome_indices),
                                              np.array(all_outcome_indices),
                                              alpha, beta1, beta2, lambda_)

    return weights

def numpy_ndl_simple(event_list, alpha, betas, all_outcomes, *, cue_map, outcome_map):
    """
    Calculate the weigths for all_outcomes over all events in event_file.

    This is a python implementation using numpy.

    Parameters
    ==========
    event_path : str
        generates cues, outcomes pairs or the path to the event file
    alpha : float
        saliency of all cues
    betas : (float, float)
        one value for successful prediction (reward) one for punishment
    all_outcomes : list
        a list of all outcomes of interest
    cue_map : dict
        has cues as keys and int as values, where the int represents the index in the
        numpy array.
    outcome_map : dict
        has outcomes as keys and int as values, where the int represents the index in the
        numpy array.

    Returns
    =======
    weights : numpy.array of shape len(outcomes), len(cues)
        weights[outcome_index][cue_index] gives the weight between outcome and cue.

    """
    lambda_ = 1.0

    weights = np.zeros((len(outcome_map), len(cue_map)), dtype=float)

    beta1, beta2 = betas

    all_outcome_indices = [outcome_map[outcome] for outcome in all_outcomes]

    if isinstance(event_list, str):
        event_list = events(event_list)

    for cues, outcomes in event_list:
        cue_indices = [cue_map[cue] for cue in cues]
        outcome_indices = [outcome_map[outcome] for outcome in outcomes]
        _update_numpy_array_inplace(weights, cue_indices, outcome_indices,
                                    all_outcome_indices, alpha, beta1, beta2,
                                    lambda_)
    return weights

def activations(cues, weights):
    if isinstance(weights, dict):
        activations_ = defaultdict(float)
        for outcome, cue_dict in weights.items():
            for cue in cues:
                activations_[outcome] += cue_dict[cue]
        return activations_

def binary_ndl(events, outcomes, number_of_cues, alphas, betas):
    """
    Calculate the weights for the outcomes over all events.

    Parameters
    ==========
    events : generator
         generates the binary events
    outcome_indices : list
         list of indexes of outcomes one is interested in
    number_of_cues: int
         maximal number of cues (highest index + 1)

    """
    # create dict {index: column} pair for each outcome
    outcomes = {index: np.zeros(number_of_cues) for index in outcome_indices}

    beta1, beta2 = betas

    for present_cues, present_outcomes in events:
        for outcome_index in outcome_indices:
            outcome = outcomes[outcome_index]
            association_strength = sum(outcome[cue_index] for cue_index in present_cues)

            for cue_index in present_cues:
                alpha = alphas[cue_index]
                if outcome_index in present_outcomes:
                    what_to_add = beta1 * (lambda_ - association_strength)
                else:
                    what_to_add = beta2 * (0 - association_strength)
                what_to_add *= alpha
                outcome[cue_index] += what_to_add
    return outcomes

# NOTE: In the original code some stuff was differently handled for multiple
# cues and multiple outcomes.

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
