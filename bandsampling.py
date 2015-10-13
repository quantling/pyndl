
import sys
import random

WORD_FREQ_FILE = "sr.subtitles.utf8.words.freq"
SAMPLESIZE = 60000
SAMPLE_FILE = "sr.subtitles.utf8.words.freq.used"
WORD_ONLY_FILE =  "sr.subtitles.utf8.words.used"


def bandsampling(population, sample_size=50000, cutoff=5, verbose=False):
    # filter all words with freq < 5
    population = [(word, freq) for word, freq in population if freq >= cutoff]
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
    return sample


def main():
    population = list()
    with open(WORD_FREQ_FILE, "rt") as infile:
        for line in infile:
            if not line.strip():
                continue
            word, freq = line.split(" ")
            population.append((word, int(freq)))
    sample = bandsampling(population, SAMPLESIZE)
    sample.sort(key=lambda x: x[1], reverse=True)
    with open(SAMPLE_FILE, "wt") as outfile:
        with open(WORD_ONLY_FILE, "wt") as outfile2:
            for word, freq in sample:
                outfile.write("%s %i\n" % (word, freq))
                outfile2.write("%s\n" % word)


if __name__ == "__main__":
    #test_population = [("A", 10), ("B", 100), ("C", 10), ("D", 3), ("E", 12),
    #                   ("F", 6)]
    #sample = bandsampling(test_population, 4, verbose=True)
    #print(sample)

    main()


