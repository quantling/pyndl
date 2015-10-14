from .. import counting

def test_cues_outcomes():
    cues, outcomes = counting.cues_outcomes("test_events.tab")
    cues3, outcomes3 = counting.cues_outcomes("test_events.tab",
                                              number_of_processes=3,
                                              verbose=False)
    assert cues == cues3
    assert outcomes == outcomes3

