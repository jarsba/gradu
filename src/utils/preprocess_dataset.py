import numpy as np
import pandas as pd
import os
from src.utils.path_utils import DATASETS_FOLDER
from pandas.api.types import is_integer_dtype, is_categorical_dtype

ADULT_COLUMNS_TO_DROP = ['had-capital-gains', 'had-capital-losses']


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

    age_labels = ["{0} - {1}".format(i, i + 9) for i in range(0, 100, 10)]

    data['age'] = pd.Categorical(pd.cut(data.age, range(0, 105, 10), right=False, labels=age_labels))
    data['workclass'] = pd.Categorical(data['workclass'])
    data['education-num'] = pd.Categorical(data['education-num'])
    data['marital-status'] = pd.Categorical(data['marital-status'])
    data['occupation'] = pd.Categorical(data['occupation'])
    data['relationship'] = pd.Categorical(data['relationship'])
    data['race'] = pd.Categorical(data['race'])
    data['sex'] = pd.Categorical(data['sex'])
    data['had-capital-gains'] = pd.Categorical(data['had-capital-gains'])
    data['had-capital-losses'] = pd.Categorical(data['had-capital-losses'])
    hours_labels = ["{0} - {1}".format(i, i + 9) for i in range(0, 100, 10)]
    data['hours-per-week'] = pd.Categorical(
        pd.cut(data['hours-per-week'], range(0, 105, 10), right=False, labels=hours_labels)
    )
    data['native-country'] = pd.Categorical(data['native-country'])
    data['compensation'] = pd.Categorical(data['compensation'])

    data.drop(columns=ADULT_COLUMNS_TO_DROP, inplace=True)

    # Move "compensation" column to last
    data = data.reindex(columns=[col for col in data.columns if col != 'compensation'] + ['compensation'])

    return data


def get_adult_train() -> pd.DataFrame:
    """
    Returns: Discretized Adult train dataset that has following columns removed:
        - `capital-gains`
        - `capital-losses`
        - `had-capital-gains`
        - `had-capital-losses`
    """

    data = pd.read_csv(os.path.join(DATASETS_FOLDER, "cleaned_adult_train_data.csv"))
    cleaned_adult = clean_adult(data)

    return cleaned_adult


def get_adult_test() -> pd.DataFrame:
    """
        Returns: Discretized Adult test dataset that has following columns removed:
            - `capital-gains`
            - `capital-losses`
            - `had-capital-gains`
            - `had-capital-losses`
    """

    data = pd.read_csv(os.path.join(DATASETS_FOLDER, "cleaned_adult_test_data.csv"))
    cleaned_adult = clean_adult(data)

    return cleaned_adult


def get_binary3d_train() -> pd.DataFrame:
    """
    Returns: Binary 3d dataset with 2 feature columns ('A', 'B') and one prediction column 'C' for logistic regression
    """
    data = pd.read_csv(os.path.join(DATASETS_FOLDER, "binary3d.csv"))
    data['A'] = pd.Categorical(data['A'])
    data['B'] = pd.Categorical(data['B'])
    data['C'] = pd.Categorical(data['C'])

    return data


def get_binary3d_test() -> pd.DataFrame:
    """
    Returns: Binary 3d dataset with 2 feature columns ('A', 'B') and one prediction column 'C' for logistic regression
    """
    data = pd.read_csv(os.path.join(DATASETS_FOLDER, "binary3d_test.csv"))
    data['A'] = pd.Categorical(data['A'])
    data['B'] = pd.Categorical(data['B'])
    data['C'] = pd.Categorical(data['C'])

    return data


def get_binary4d_train() -> pd.DataFrame:
    """
    Returns: Binary 4d dataset with 3 feature columns ('A', 'B', 'C') and one prediction column 'D' for logistic regression
    """
    data = pd.read_csv(os.path.join(DATASETS_FOLDER, "binary4d.csv"))
    data['A'] = pd.Categorical(data['A'])
    data['B'] = pd.Categorical(data['B'])
    data['C'] = pd.Categorical(data['C'])
    data['D'] = pd.Categorical(data['D'])

    return data


def get_binary4d_test() -> pd.DataFrame:
    """
    Returns: Binary 4d dataset with 3 feature columns ('A', 'B', 'C') and one prediction column 'D' for logistic regression
    """
    data = pd.read_csv(os.path.join(DATASETS_FOLDER, "binary4d_test.csv"))
    data['A'] = pd.Categorical(data['A'])
    data['B'] = pd.Categorical(data['B'])
    data['C'] = pd.Categorical(data['C'])
    data['D'] = pd.Categorical(data['D'])

    return data
