from typing import Mapping, List, Optional
from src.utils.keygen import get_key
import pandas as pd
from singleton import Singleton


class ExperimentStorage(metaclass=Singleton):
    def __init__(self):
        self.storage = []
        # Hashmap for mapping process_id to list index in storage, so we get ~constant time read operation for storage item.
        self.storage_index = {}

    @property
    def storage(self) -> List:
        return self.storage

    @storage.setter
    def storage(self, value: List[Mapping]):
        self.storage = value

    def get_item(self, process_id: str) -> Mapping:
        index = self.storage_index[process_id]
        return self.storage[index]

    def _has_key(self, process_id: str) -> bool:
        if process_id in self.storage_index:
            return True
        else:
            return False

    def store(self, process_id: str, dict: Mapping) -> None:
        if self._has_key(process_id):
            item: dict = self.get_item(process_id)
            item.update(dict)
        else:
            key = get_key()
            self.storage.append(dict)
            self.storage_index[key] = len(self.storage) - 1

    def to_df(self) -> Optional[pd.DataFrame]:
        if len(self.storage) == 0:
            return None
        else:
            df = pd.DataFrame(self.storage)
            return df

    def to_csv(self, file_path: str) -> None:
        df = self.to_df()

        if df is None:
            raise Exception("Storage has not recorded any items")
        else:
            df.to_csv(file_path)
