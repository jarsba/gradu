import os
import string
import random

from time_utils import get_formatted_datetime

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.getcwd()))
DATA_FOLDER = os.path.join(PROJECT_ROOT_PATH, "data")
DATASETS_FOLDER = os.path.join(DATA_FOLDER, "datasets")
SYNT_DATASETS_FOLDER = os.path.join(DATA_FOLDER, "synt_datasets")
MODELS_FOLDER = os.path.join(PROJECT_ROOT_PATH, "models")
NAPSU_FOLDER = os.path.join(PROJECT_ROOT_PATH, "napsu_mq")


def get_dataset_name(dataset_name: str, number: int, epsilon: float):
    id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    datetime_str = get_formatted_datetime()
    return f'dataframe_synthetic_{dataset_name}_{datetime_str}_{number}_{epsilon}_{id}.csv'

