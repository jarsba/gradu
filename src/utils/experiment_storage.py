from typing import Mapping, List, Optional
import pandas as pd
from src.utils.singleton import Singleton
import contextvars

experiment_id_ctx = contextvars.ContextVar('experiment_id')


class ExperimentStorage(metaclass=Singleton):
    def __init__(self):
        self.storage = []
        # Hashmap for mapping process_id to list index in storage, so we get ~constant time read operation for storage item.
        self.storage_index = {}

    def get_item(self, experiment_id: str) -> Mapping:
        index = self.storage_index[experiment_id]
        return self.storage[index]

    def _has_key(self, experiment_id: str) -> bool:
        if experiment_id in self.storage_index:
            return True
        else:
            return False

    def store(self, experiment_id: str, dict: Mapping) -> None:
        if self._has_key(experiment_id):
            item: dict = self.get_item(experiment_id)
            item.update(dict)
        else:
            self.storage.append(dict)
            self.storage_index[experiment_id] = len(self.storage) - 1

    def to_df(self) -> Optional[pd.DataFrame]:
        if len(self.storage) == 0:
            return None
        else:
            df = pd.DataFrame(self.storage)
            return df

    def to_csv(self, file_path: str, **kwargs) -> None:
        df = self.to_df()

        if df is None:
            raise Exception("Storage has not recorded any items")
        else:
            df.to_csv(file_path, **kwargs)
