from typing import Dict, Iterator, Tuple, TypeVar

try:
    from typing import Collection
except ImportError:
    from collections.abc import Collection

from numpy import ndarray
from xarray.core.dataarray import DataArray


Path = str
Cue = str
Outcome = str
Id = int

IdCollection = Collection[Id]
CueCollection = Collection[Cue]
AnyCues = TypeVar('AnyCues', ndarray, CueCollection)
OutcomeCollection = Collection[Outcome]
AnyOutcomes = TypeVar('AnyOutcomes', ndarray, OutcomeCollection)
CollectionEvent = Tuple[CueCollection, OutcomeCollection]
IdCollectionEvent = Tuple[IdCollection, IdCollection]
StringEvent = Tuple[str, str]
AnyEvent = Tuple[AnyCues, AnyOutcomes]
AnyEvents = TypeVar('AnyEvents', Path, Iterator[AnyEvent])
WeightDict = Dict[str, Dict[str, float]]
AnyWeights = TypeVar('AnyWeights', DataArray, WeightDict)
