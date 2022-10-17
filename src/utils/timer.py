from timeit import default_timer as timer
from typing import Optional, Tuple

import pandas as pd
from src.utils.keygen import get_key
from singleton import Singleton


class Timer(metaclass=Singleton):

    def __init__(self):
        self.times = []
        # Hashmap for mapping process_id to list index in times, so we get ~constant time read operation for times item.
        self.times_index = {}

    def get_times(self):
        return self.times

    def start(self, task: str, **kwargs) -> str:
        print("Recording: ", task)
        if "experiment_id" in kwargs:
            experiment_id = kwargs['experiment_id']
        else:
            experiment_id = None
        start = timer()
        self.times.append({
            'experiment_id': experiment_id,
            'start': start,
            'stop': None,
            'timedelta': None,
            'task': task,
            **kwargs
        })

        index = len(self.times) - 1
        process_id = get_key()

        self.times_index[process_id] = index

        return process_id

    def stop(self, process_id: str) -> None:
        stop = timer()

        index = self.times_index[process_id]

        times_obj: dict = self.times[index]
        times_obj['stop'] = stop
        times_obj['timedelta'] = stop - times_obj['start']

    def to_df(self) -> Optional[pd.DataFrame]:
        if len(self.times) == 0:
            return None
        else:
            df = pd.DataFrame(self.times)
            return df

    def to_csv(self, file_path: str) -> None:
        df = self.to_df()

        if df is None:
            raise Exception("Timer has not recorded any timestamps")
        else:
            df.to_csv(file_path)
