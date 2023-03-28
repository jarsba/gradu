from typing import Mapping, Optional, Literal, Union
import pandas as pd
from src.utils.singleton import Singleton
import contextvars

experiment_id_ctx = contextvars.ContextVar('experiment_id')


class ExperimentStorage(metaclass=Singleton):
    def __init__(self, file_path: Optional[str] = None, mode: Optional[Literal["replace", "append"]] = None):
        self.storage = {}
        self.file_path = file_path
        self.mode = mode

    def get_item(self, experiment_id: str) -> dict:
        item = self.storage[experiment_id]
        return item

    def _has_key(self, experiment_id: str) -> bool:
        if experiment_id in self.storage:
            return True
        else:
            return False

    def store(self, experiment_id: Union[str, int], dict_to_store: Mapping) -> None:
        if self._has_key(experiment_id):
            item = self.get_item(experiment_id)
            item.update(dict_to_store)
        else:
            self.storage[experiment_id] = dict_to_store

    def to_df(self) -> Optional[pd.DataFrame]:
        if len(self.storage) == 0:
            return None
        else:
            df = pd.DataFrame().from_records(self.storage)
            return df

    def to_csv(self, file_path: Optional[str], **kwargs) -> None:
        if file_path is None and self.file_path is None:
            raise Exception("File path is not defined.")

        df = self.to_df()

        if df is None:
            raise Exception("Storage has not recorded any items")
        else:
            df.to_csv(file_path, **kwargs)

    def save(self, file_path: str = None, mode: Literal["replace", "append"] = None) -> None:
        """Save to contents of the experiment storage to a csv file. Convenience wrapper for to_csv-function

        Args:
            file_path: Path to the file where the timer contents are saved.
            mode: If the file already exists, should the contents be replaced or appended to the file. If mode is
                "replace", the file will be overwritten. If mode is "append", the contents will be appended to the file.
                With mode "replace" save can be called multiple times and the file will be overwritten each time. With
                mode "append" the same content might be written multiple times.

        Returns: None
        """

        if file_path is None and self.file_path is None:
            raise Exception("File path is not defined.")

        if mode is None and self.mode is None:
            raise Exception("Mode is not defined.")

        file_path = file_path if file_path is not None else self.file_path
        mode = mode if mode is not None else self.mode

        if mode == 'replace':
            self.to_csv(file_path, mode="w")
        elif mode == 'append':
            self.to_csv(file_path, mode="a")
        else:
            raise Exception("Invalid mode")
