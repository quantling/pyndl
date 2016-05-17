from collections import defaultdict
import multiprocessing
import time

import numpy as np

def events(event_path, *, frequency=True):
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

def dict_ndl_parrallel(event_path, alphas, betas, all_outcomes, *, number_of_processes=2, sequence=10):
    """
    Calculate the weigths for all_outcomes over all events in event_file
    given by the files path.

    This is a parrallel python implementation using dicts and multiprocessing.

    Parameters
    ==========
    event_path : path to the event file
    alphas : dict
        a (default)dict having cues as keys and a value below 1 as value
    betas : dict
        a (default)dict having outcomes as keys and a value below 1 as value
    all_outcomes : list
        a list of all outcomes of interest
    number_of_processes : int
        a integer giving the number of processes in which the job should
        executed
    sequence : int
        a integer giving the length of sublists generated from all outcomes

    Returns
    =======
    weights : dict of dicts of floats
        the first dict has outcomes as keys and dicts as values
        the second dict has cues as keys and weights as values
        weights[outcome][cue] gives the weight between outcome and cue.

    """
    with multiprocessing.Pool(number_of_processes) as pool:

        job = JobCalculateWeights(alphas, betas, event_path)

        weights = defaultdict(lambda:defaultdict(float))

        partlists_of_outcomes = slice_list(all_outcomes,sequence)

        for result in pool.imap_unordered(job.dict_ndl_weight_calculator, partlists_of_outcomes):
            for outcome, cues in result:
                weights[outcome] = cues

        return weights


class JobCalculateWeights():
    """
    Stores the values of alphas and betas an the path to the event file

    Method is used as a worker for the multiprocessed dict_ndl implementation
    """

    def __init__(self, alphas, betas, event_path):
        self.alphas = alphas
        self.betas = betas
        self.event_path = event_path

    def dict_ndl_weight_calculator(self,part_outcomes):

        weights = dict_ndl(self.event_path, self.alphas, self.betas, part_outcomes)

        return [(outcome,cues) for outcome, cues in weights.items()]




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
    # weights can be seen as an infinite outcome by cue matrix
    # weights[outcome][cue]
    weights = defaultdict(lambda: defaultdict(float))

    if isinstance(event_list, str):
        event_list = events(event_list)

    for cues, outcomes in event_list:
        for outcome in all_outcomes:
            beta = betas[outcome]
            association_strength = sum(weights[outcome][cue] for cue in cues)
            if outcome in outcomes:
                update = lambda_ - association_strength
            else:
                update = 0 - association_strength
            for cue in cues:
                weights[outcome][cue] += alphas[cue] * beta * update

    return weights

# NOTE: In the original code some stuff was differently handled for multiple
# cues and multiple outcomes.


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

    for present_cues, present_outcomes in events:
        for outcome_index in outcome_indices:
            beta = betas[present_outcome_index]
            outcome = outcomes[outcome_index]
            association_strength = sum(outcome[cue_index] for cue_index in present_cues)

            for cue_index in present_cues:
                alpha = alphas[cue_index]
                if outcome_index in present_outcomes:
                    what_to_add = lambda_ - association_strength
                else:
                    what_to_add = 0 - association_strength
                what_to_add *= alpha * beta
                outcome[cue_index] += what_to_add
    return outcomes

# NOTE: In the original code some stuff was differently handled for multiple
# cues and multiple outcomes.

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
