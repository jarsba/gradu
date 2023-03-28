from timeit import default_timer as timer
from typing import Optional, Literal
import pandas as pd
from src.utils.keygen import generate_experiment_id
from src.utils.singleton import Singleton


class Timer(metaclass=Singleton):

    def __init__(self, file_path: Optional[str] = None, mode: Optional[Literal["replace", "append"]] = None):
        self.times = {}
        self.file_path = file_path
        self.mode = mode

    def get_times(self):
        return self.times

    def get_time(self, process_id: str) -> dict:
        return self.times[process_id]

    def start(self, task: str, **kwargs) -> str:
        print("Recording: ", task)
        if "experiment_id" in kwargs:
            experiment_id = kwargs['experiment_id']
        else:
            experiment_id = None
        start = timer()

        process_id = generate_experiment_id()

        self.times[process_id] = {
            'experiment_id': experiment_id,
            'start': start,
            'stop': None,
            'timedelta': None,
            'task': task,
            **kwargs
        }

        return process_id

    def stop(self, process_id: str) -> None:
        stop = timer()
        times_obj: dict = self.times[process_id]
        times_obj['stop'] = stop
        times_obj['timedelta'] = stop - times_obj['start']

    def to_df(self) -> Optional[pd.DataFrame]:
        if len(self.times) == 0:
            return None
        else:
            df = pd.DataFrame.from_records(self.times)
            return df

    def to_csv(self, file_path: Optional[str], **kwargs) -> None:
        if file_path is None and self.file_path is None:
            raise Exception("File path is not defined.")

        df = self.to_df()

        file_path = file_path if file_path is not None else self.file_path

        if df is None:
            raise Exception("Timer has not recorded any timestamps")
        else:
            df.to_csv(file_path, **kwargs)

    def save(self, file_path: str = None, mode: Literal["replace", "append"] = None) -> None:
        """Save to contents of the timer to a csv file. Convenience wrapper for to_csv-function.

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
