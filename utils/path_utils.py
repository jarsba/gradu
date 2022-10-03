import os
import string
import random
from pathlib import Path
from time_utils import get_formatted_datetime

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.getcwd()))
DATA_FOLDER = os.path.join(PROJECT_ROOT_PATH, "data")
DATASETS_FOLDER = os.path.join(DATA_FOLDER, "datasets")
SYNT_DATASETS_FOLDER = os.path.join(DATA_FOLDER, "synt_datasets")
MODELS_FOLDER = os.path.join(PROJECT_ROOT_PATH, "models")
NAPSU_FOLDER = os.path.join(PROJECT_ROOT_PATH, "napsu_mq")


def get_dataset_name(path):
    filename = get_filename(path)
    synthetic_dataset = "synthetic_dataset" in filename

    if synthetic_dataset:
        dataset_name = filename.split("_")[3]
        return dataset_name

    else:
        if "adult" in filename:
            return "adult"
        elif "binary4d" in filename:
            return "binary4d"
        else:
            raise ValueError(f"Cannot determine dataset name from filename {filename}")


def get_filename(path: str, with_suffix=False) -> str:
    path = Path(path)

    if with_suffix:
        return path.name

    else:
        return path.stem


def get_metadata_from_synthetic_path(synthetic_path: str):
    filename = get_filename(synthetic_path)
    filename_paths = filename.split("_")
    dataset_index = filename_paths[2]
    dataset_name = filename_paths[3]
    epsilon_str = filename_paths[4][0:2]
    epsilon = float(f"{epsilon_str[0]}.{epsilon_str[1]}")
    MCMC_algorithm = filename_paths[5]

    return dataset_index, dataset_name, epsilon, MCMC_algorithm
