from pathlib import Path
from pyndl import io
from pyndl import preprocess

txt_file = 'graphem_wiki_de.txt'
event_dir = Path('events')
full_event_file = Path('graphem_wiki_de_full_event.tab.gz')
ascii_event_file = Path('graphem_wiki_de_ascii_event.tab.gz')

event_dir.mkdir(exist_ok=True)
full_event_file.unlink(missing_ok=True)
ascii_event_file.unlink(missing_ok=True)

print(f"Create events and save to {full_event_file}.")
preprocess.create_event_file(corpus_file=txt_file,
                             event_file=full_event_file,
                             allowed_symbols=(lambda c: c not in '()[]-|,'),
                             context_structure='document',
                             event_structure='consecutive_words',
                             event_options=(1,),  # number of words,
                             cue_structure="trigrams_to_word",
                             lower_case=True,
                             remove_duplicates=True)


events = list(io.events_from_file(full_event_file))
print(f"{len(events)} events")
for cues, outcome in events[0:3]:
    print(f"Cues: {cues}, Outcome: {outcome}")

for cues, outcome in events[5554:5557]:
    print(f"Cues: {cues}, Outcome: {outcome}")

print(f"Create events with ASCII only characters and save to {ascii_event_file}.")
preprocess.create_event_file(corpus_file=txt_file,
                             event_file=ascii_event_file,
                             allowed_symbols='a-zA-Z0-9',
                             context_structure='document',
                             event_structure='consecutive_words',
                             event_options=(1,),  # number of words,
                             cue_structure="trigrams_to_word",
                             lower_case=True,
                             remove_duplicates=True)
ascii_events = list(io.events_from_file(ascii_event_file))

for times in (1, 10, 20, 30, 40, 50):
    # R ndl2 fails on the full event files
    #io.events_to_file(events * times, event_dir / f"{times}_times_{full_event_file.name}", compatible=True)
    #io.events_to_file(events * times, event_dir / f"{times}_times_{full_event_file.stem}", compression=None, compatible=True)
    io.events_to_file(ascii_events * times, event_dir / f"{times}_times_{ascii_event_file.name}", compatible=True)
    io.events_to_file(ascii_events * times, event_dir / f"{times}_times_{ascii_event_file.stem}", compression=None, compatible=True)
