from timeit import default_timer as timer
import random
import string

import pandas as pd


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Timer(metaclass=Singleton):

    def __init__(self):
        self.times = []
        # Hashmap for mapping process_id to list index in times, so we get ~constant time read operation for times item.
        self.times_index = {}

    def get_times(self):
        return self.times

    def start(self, task: str, **kwargs):
        print("Recording: ", task)
        process_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        start = timer()
        self.times.append({
            'process_id': process_id,
            'start': start,
            'stop': None,
            'timedelta': None,
            'task': task,
            **kwargs
        })

        index = len(self.times) - 1

        self.times_index[process_id] = index

        return process_id

    def stop(self, process_id):
        stop = timer()

        index = self.times_index[process_id]

        times_obj: dict = self.times[index]
        times_obj['stop'] = stop
        times_obj['timedelta'] = stop - times_obj['start']

    def to_df(self):
        if len(self.times) == 0:
            return None
        else:
            df = pd.DataFrame(self.times)
            return df

    def to_csv(self, file_path):
        df = self.to_df()

        if df is None:
            raise Exception("Timer has not recorded any timestamps")
        else:
            df.to_csv(file_path)
