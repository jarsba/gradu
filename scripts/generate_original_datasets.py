import sys
import os
sys.path.append(snakemake.config['workdir'])

from src.utils.preprocess_dataset import \
    get_adult_train_no_discretization, get_adult_test_no_discretization, \
    get_adult_train_low_discretization, get_adult_test_low_discretization, \
    get_adult_train_high_discretization, get_adult_test_high_discretization, \
    get_adult_train_small, get_adult_test_small, \
    get_adult_train_large, get_adult_test_large
from src.utils.path_utils import DATASETS_FOLDER

datasets = snakemake.input
target_files = snakemake.output


PREPROCESSING_FUNCTIONS = {
    "adult_small": get_adult_train_small,
    "adult_large": get_adult_train_large,
    "adult_no_discretization": get_adult_train_no_discretization,
    "adult_low_discretization": get_adult_train_low_discretization,
    "adult_high_discretization": get_adult_train_high_discretization,
    "adult_small_test": get_adult_test_small,
    "adult_large_test": get_adult_test_large,
    "adult_no_discretization_test": get_adult_test_no_discretization,
    "adult_low_discretization_test": get_adult_test_low_discretization,
    "adult_high_discretization_test": get_adult_test_high_discretization,
}

for dataset, target_file in zip(datasets, target_files):

    # Skip if the dataset is not in the preprocessing functions like for "adult", "binary3d" and "binary4d"
    if dataset not in PREPROCESSING_FUNCTIONS:
        continue

    df = PREPROCESSING_FUNCTIONS[dataset](dataset_folder=DATASETS_FOLDER)
    df.to_csv(target_file, index=False)
