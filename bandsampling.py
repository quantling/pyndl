
import sys
import random
from collections import Counter


def bandsample(population, sample_size=50000, *, cutoff=5, verbose=False):
    """
    Creates a sample of size sample_size out of the population using
    bandsampling.

    """
    # make a copy of the population
    # filter all words with freq < cutoff
    population = [(word, freq) for word, freq in population.items() if freq >=
                  cutoff]
    # shuffle words with same frequency
    random.shuffle(population)
    population.sort(key=lambda x: x[1])  # lowest -> highest freq

    step = sum(freq for word, freq in population) / sample_size
    if verbose:
        print("step %.2f" % step)

    accumulator = 0
    index = 0
    sample = list()
    while 0 <= index < len(population):
        word, freq = population[index]
        accumulator += freq
        if verbose:
            print("%s\t%i\t%.2f" % (word, freq, accumulator))
        if accumulator >= step:
            sample.append((word, freq))
            accumulator -= step
            if verbose:
                print("add\t%s\t%.2f" % (word, accumulator))
            del population[index]
            while accumulator >= step and index >= 1:
                index -= 1
                sample.append(population[index])
                accumulator -= step
                if verbose:
                    word, freq = population[index]
                    print("  add\t%s\t%.2f" % (word, accumulator))
                del population[index]
        else:
            # only add to index if no element was removed
            # if element was removed, index points at next element already
            index += 1
            if index % 10000 == 0:
                print(".", end="")
                sys.stdout.flush()
    sample = Counter({key: value for key, value in sample})
    return sample


