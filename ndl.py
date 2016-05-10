from collections import defaultdict
import time

import numpy as np


def events(event_file, *, frequency=True):
    """
    Yields events for all events in event_file.

    Parameters
    ==========
    event_file : file_handle
    frequency : bool
        frequency should be in the event_file

    Yields
    ======
    cues, outcomes : list, list
        a tuple of two lists containing cues and outcomes

    """
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


def dict_ndl(events, alphas, betas, all_outcomes):
    """
    Calculate the weiths for all_outcomes over all events in event_file.

    This is a pure python implementation using dicts.

    Parameters
    ==========
    events : generator yielding cues, outcomes pairs
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

    for cues, outcomes in events:
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

