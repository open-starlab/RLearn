from typing import Any, Iterator, List

from rlearn.sports.soccer.dataclass import (
    Events,
    SimpleObservationActionSequence,
)
from rlearn.sports.soccer.env import DATA_DIR
from rlearn.sports.soccer.utils.file_utils import load_jsonlines


class DataSource:
    def collect(self) -> Iterator[Any]:
        raise NotImplementedError


class JLeagueDataSource(DataSource):
    def __init__(self, data_name: str, subset: str) -> None:
        self.data_name = data_name
        self.subset = subset
        self._data: List[Events]
        self.__build_data()

    def __build_data(self) -> None:
        for events in load_jsonlines(DATA_DIR / "preprocessed" / f"{self.data_name} / {self.subset}.jsonl"):
            self._data.append(Events.from_dict(events))

    def collect(self) -> Iterator[Any]:
        yield from self._data


class JLeagueSimpleObservationActionSequenceDataSource(DataSource):
    def __init__(self, data_name: str, subset: str) -> None:
        self.data_name = data_name
        self.subset = subset
        self._data: List[SimpleObservationActionSequence] = []
        self.__build_data()

    def __build_data(self) -> None:
        for observation_action in load_jsonlines(
            DATA_DIR / "preprocessed" / f"{self.data_name}" / f"{self.subset}.jsonl"
        ):
            self._data.append(SimpleObservationActionSequence.from_dict(observation_action))

    def collect(self) -> Iterator[SimpleObservationActionSequence]:
        yield from self._data

