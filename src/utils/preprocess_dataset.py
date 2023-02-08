from typing import List

import numpy as np
import pandas as pd
import os
from src.utils.path_utils import DATASETS_FOLDER
from pandas.api.types import is_integer_dtype, is_categorical_dtype

ADULT_COLUMNS_TO_DROP = ['had-capital-gains', 'had-capital-losses', 'capital-gain', 'capital-loss']
ADULT_COLUMNS = ['age', 'workclass', 'education-num', 'marital-status',
                 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                 'hours-per-week', 'native-country', 'had-capital-gains', 'had-capital-losses', 'compensation']

ADULT_COLUMNS_DISCRETIZATION = ['age', 'sex', 'education-num', 'hours-per-week', 'workclass', 'marital-status',
                                'had-capital-gains', 'had-capital-losses', 'compensation']


def convert_to_int_array(df: pd.DataFrame) -> np.ndarray:
    int_df = df.copy()
    for col in df.columns:
        if is_categorical_dtype(df[col]):
            int_df[col] = df[col].cat.codes
        elif is_integer_dtype(df[col]):
            int_df[col] = df[col]
        else:
            raise ValueError(f"DataFrame contains unsupported column type: {df[col].dtype}")

    int_array = int_df.to_numpy()

    return int_array


def clean_dataset(df, dataset_name):
    if dataset_name == 'adult':
        df = clean_adult(df)
    elif dataset_name == 'binary3d':
        df = clean_synthetic_binary(df)
    elif dataset_name == 'binary4d':
        df = clean_synthetic_binary(df)

    return df


def clean_synthetic_binary(data: pd.DataFrame) -> pd.DataFrame:
    data = data.apply(pd.to_numeric)
    return data


def clean_adult(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaned Adult dataset with all columns and without discretization.
    """

    data['age'] = data['age'].astype(int)
    data['workclass'] = pd.Categorical(data['workclass'])
    data['education-num'] = pd.Categorical(data['education-num'])
    data['marital-status'] = pd.Categorical(data['marital-status'])
    data['occupation'] = pd.Categorical(data['occupation'])
    data['relationship'] = pd.Categorical(data['relationship'])
    data['race'] = pd.Categorical(data['race'])
    data['sex'] = pd.Categorical(data['sex'])
    data['had-capital-gains'] = pd.Categorical(data['had-capital-gains'])
    data['had-capital-losses'] = pd.Categorical(data['had-capital-losses'])
    data['capital-gain'] = data['capital-gain'].astype(int)
    data['capital-loss'] = data['capital-loss'].astype(int)
    data['hours-per-week'] = data['hours-per-week'].astype(int)
    data['native-country'] = pd.Categorical(data['native-country'])
    data['compensation'] = pd.Categorical(data['compensation'])

    # Move "compensation" column to last
    data = data.reindex(columns=[col for col in data.columns if col != 'compensation'] + ['compensation'])

    return data


def clean_adult_with_discretization(data: pd.DataFrame, bucket_size: int = 10,
                                    columns: List[str] = None) -> pd.DataFrame:
    """ Clean Adult dataset with fixed level of discretization and by default following columns dropped:
        - `capital-gains`
        - `capital-losses`
        - `had-capital-gains`
        - `had-capital-losses`
    """

    if columns is None:
        columns_to_drop = ADULT_COLUMNS_TO_DROP
    else:
        columns_to_drop = list(set(ADULT_COLUMNS) - set(columns))

    age_labels = ["{0} - {1}".format(i, i + bucket_size) for i in range(0, 100, bucket_size)]
    data['age'] = pd.Categorical(pd.cut(data.age, range(0, 105, bucket_size), right=False, labels=age_labels))
    data['workclass'] = pd.Categorical(data['workclass'])
    data['education-num'] = pd.Categorical(data['education-num'])
    data['marital-status'] = pd.Categorical(data['marital-status'])
    data['occupation'] = pd.Categorical(data['occupation'])
    data['relationship'] = pd.Categorical(data['relationship'])
    data['race'] = pd.Categorical(data['race'])
    data['sex'] = pd.Categorical(data['sex'])
    data['capital-gain'] = data['capital-gain'].astype(int)
    data['capital-loss'] = data['capital-loss'].astype(int)
    data['had-capital-gains'] = pd.Categorical(data['had-capital-gains'])
    data['had-capital-losses'] = pd.Categorical(data['had-capital-losses'])
    hours_labels = ["{0} - {1}".format(i, i + bucket_size) for i in range(0, 100, bucket_size)]
    data['hours-per-week'] = pd.Categorical(
        pd.cut(data['hours-per-week'], range(0, 105, bucket_size), right=False, labels=hours_labels)
    )
    data['native-country'] = pd.Categorical(data['native-country'])
    data['compensation'] = pd.Categorical(data['compensation'])

    data.drop(columns=columns_to_drop, inplace=True)

    # Move "compensation" column to last
    data = data.reindex(columns=[col for col in data.columns if col != 'compensation'] + ['compensation'])

    return data


def get_adult_train_raw(dataset_folder: str = None) -> pd.DataFrame:
    """
    Returns: Cleaned Adult train dataset without any other preprocessing.
    """

    if dataset_folder is None:
        dataset_folder = DATASETS_FOLDER

    data = pd.read_csv(os.path.join(dataset_folder, "cleaned_adult_train_data.csv"))
    cleaned_adult = clean_adult(data)

    return cleaned_adult


def get_adult_test_raw(dataset_folder: str = None) -> pd.DataFrame:
    """
    Returns: Cleaned Adult test dataset without any other preprocessing.
    """

    if dataset_folder is None:
        dataset_folder = DATASETS_FOLDER

    data = pd.read_csv(os.path.join(dataset_folder, "cleaned_adult_test_data.csv"))
    cleaned_adult = clean_adult(data)

    return cleaned_adult


def get_adult_train_no_discretization(dataset_folder: str = None) -> pd.DataFrame:
    """
    Returns: Subset of Adult train dataset that has columns discretized to buckets of 10 and that has following columns:
        - 'age'
        - 'sex'
        - 'education-num'
        - 'hours-per-week'
        - 'workclass'
        - 'marital-status'
        - 'had-capital-gains'
        - 'had-capital-losses'
        - 'compensation'
    """

    if dataset_folder is None:
        dataset_folder = DATASETS_FOLDER

    data = pd.read_csv(os.path.join(dataset_folder, "cleaned_adult_train_data.csv"))
    cleaned_adult = clean_adult(data)

    columns_to_drop = list(set(ADULT_COLUMNS) - set(ADULT_COLUMNS_DISCRETIZATION))

    cleaned_adult.drop(columns=columns_to_drop, inplace=True)

    return cleaned_adult


def get_adult_test_no_discretization(dataset_folder: str = None) -> pd.DataFrame:
    """
    Returns: Subset of Adult test dataset that has columns discretized to buckets of 10 and that has following columns:
        - 'age'
        - 'sex'
        - 'education-num'
        - 'hours-per-week'
        - 'workclass'
        - 'marital-status'
        - 'had-capital-gains'
        - 'had-capital-losses'
        - 'compensation'
    """

    if dataset_folder is None:
        dataset_folder = DATASETS_FOLDER

    data = pd.read_csv(os.path.join(dataset_folder, "cleaned_adult_test_data.csv"))
    cleaned_adult = clean_adult(data)

    columns_to_drop = list(set(ADULT_COLUMNS) - set(ADULT_COLUMNS_DISCRETIZATION))

    cleaned_adult.drop(columns=columns_to_drop, inplace=True)

    return cleaned_adult


def get_adult_train_low_discretization(dataset_folder: str = None) -> pd.DataFrame:
    """
    Returns: Subset of Adult train dataset that has following columns:
        - 'age'
        - 'sex'
        - 'education-num'
        - 'hours-per-week'
        - 'workclass'
        - 'marital-status'
        - 'had-capital-gains'
        - 'had-capital-losses'
        - 'compensation'
    """

    if dataset_folder is None:
        dataset_folder = DATASETS_FOLDER

    data = pd.read_csv(os.path.join(dataset_folder, "cleaned_adult_train_data.csv"))
    cleaned_adult = clean_adult_with_discretization(data, bucket_size=10,
                                                    columns=ADULT_COLUMNS_DISCRETIZATION)

    return cleaned_adult


def get_adult_test_low_discretization(dataset_folder: str = None) -> pd.DataFrame:
    """
    Returns: Subset of Adult test dataset that has columns discretized to buckets of 10 and that has following columns:
        - 'age'
        - 'sex'
        - 'education-num'
        - 'hours-per-week'
        - 'workclass'
        - 'marital-status'
        - 'had-capital-gains'
        - 'had-capital-losses'
        - 'compensation'
    """

    if dataset_folder is None:
        dataset_folder = DATASETS_FOLDER

    data = pd.read_csv(os.path.join(dataset_folder, "cleaned_adult_test_data.csv"))
    cleaned_adult = clean_adult_with_discretization(data, bucket_size=10,
                                                    columns=ADULT_COLUMNS_DISCRETIZATION)

    return cleaned_adult


def get_adult_train_high_discretization(dataset_folder: str = None) -> pd.DataFrame:
    """
    Returns: Subtset of Adult train dataset that has columns discretized to buckets of 5 and that has following columns:
        - 'age'
        - 'sex'
        - 'education-num'
        - 'hours-per-week'
        - 'workclass'
        - 'marital-status'
        - 'had-capital-gains'
        - 'had-capital-losses'
        - 'compensation'
    """

    if dataset_folder is None:
        dataset_folder = DATASETS_FOLDER

    data = pd.read_csv(os.path.join(dataset_folder, "cleaned_adult_train_data.csv"))
    cleaned_adult = clean_adult_with_discretization(data, bucket_size=5,
                                                    columns=ADULT_COLUMNS_DISCRETIZATION)

    return cleaned_adult


def get_adult_test_high_discretization(dataset_folder: str = None) -> pd.DataFrame:
    """
    Returns: Subtset of Adult test dataset that has columns discretized to buckets of 5 and that has following columns:
        - 'age'
        - 'sex'
        - 'education-num'
        - 'hours-per-week'
        - 'workclass'
        - 'marital-status'
        - 'had-capital-gains'
        - 'had-capital-losses'
        - 'compensation'
    """

    if dataset_folder is None:
        dataset_folder = DATASETS_FOLDER

    data = pd.read_csv(os.path.join(dataset_folder, "cleaned_adult_test_data.csv"))
    cleaned_adult = clean_adult_with_discretization(data, bucket_size=5,
                                                    columns=ADULT_COLUMNS_DISCRETIZATION)

    return cleaned_adult


def get_adult_train(dataset_folder: str = None) -> pd.DataFrame:
    """
    Returns: Discretized Adult train dataset that has following columns removed:
        - `capital-gains`
        - `capital-losses`
        - `had-capital-gains`
        - `had-capital-losses`
    """

    if dataset_folder is None:
        dataset_folder = DATASETS_FOLDER

    data = pd.read_csv(os.path.join(dataset_folder, "cleaned_adult_train_data.csv"))
    cleaned_adult = clean_adult_with_discretization(data, bucket_size=10)

    return cleaned_adult


def get_adult_test(dataset_folder: str = None) -> pd.DataFrame:
    """
        Returns: Discretized Adult test dataset that has following columns removed:
            - `capital-gains`
            - `capital-losses`
            - `had-capital-gains`
            - `had-capital-losses`
    """

    if dataset_folder is None:
        dataset_folder = DATASETS_FOLDER

    data = pd.read_csv(os.path.join(dataset_folder, "cleaned_adult_test_data.csv"))
    cleaned_adult = clean_adult_with_discretization(data, bucket_size=10)

    return cleaned_adult


def get_binary3d_train(dataset_folder: str = None) -> pd.DataFrame:
    """
    Returns: Binary 3d dataset with 2 feature columns ('A', 'B') and one prediction column 'C' for logistic regression
    """

    if dataset_folder is None:
        dataset_folder = DATASETS_FOLDER

    data = pd.read_csv(os.path.join(dataset_folder, "binary3d.csv"))
    data['A'] = pd.Categorical(data['A'])
    data['B'] = pd.Categorical(data['B'])
    data['C'] = pd.Categorical(data['C'])

    return data


def get_binary3d_test(dataset_folder: str = None) -> pd.DataFrame:
    """
    Returns: Binary 3d dataset with 2 feature columns ('A', 'B') and one prediction column 'C' for logistic regression
    """

    if dataset_folder is None:
        dataset_folder = DATASETS_FOLDER

    data = pd.read_csv(os.path.join(dataset_folder, "binary3d_test.csv"))
    data['A'] = pd.Categorical(data['A'])
    data['B'] = pd.Categorical(data['B'])
    data['C'] = pd.Categorical(data['C'])

    return data


def get_binary4d_train(dataset_folder: str = None) -> pd.DataFrame:
    """
    Returns: Binary 4d dataset with 3 feature columns ('A', 'B', 'C') and one prediction column 'D' for logistic regression
    """

    if dataset_folder is None:
        dataset_folder = DATASETS_FOLDER

    data = pd.read_csv(os.path.join(dataset_folder, "binary4d.csv"))
    data['A'] = pd.Categorical(data['A'])
    data['B'] = pd.Categorical(data['B'])
    data['C'] = pd.Categorical(data['C'])
    data['D'] = pd.Categorical(data['D'])

    return data


def get_binary4d_test(dataset_folder: str = None) -> pd.DataFrame:
    """
    Returns: Binary 4d dataset with 3 feature columns ('A', 'B', 'C') and one prediction column 'D' for logistic regression
    """

    if dataset_folder is None:
        dataset_folder = DATASETS_FOLDER

    data = pd.read_csv(os.path.join(dataset_folder, "binary4d_test.csv"))
    data['A'] = pd.Categorical(data['A'])
    data['B'] = pd.Categorical(data['B'])
    data['C'] = pd.Categorical(data['C'])
    data['D'] = pd.Categorical(data['D'])

    return data
