from typing import List, Iterable, Optional

import numpy as np
import pandas as pd

DATASET_COLUMN_TYPES = {
    'adult': {
        'age': 'int',
        'workclass': 'category',
        'gender': 'category',
        'education-num': 'int',
        'marital-status': 'category',
        'occupation': 'category',
        'relationship': 'category',
        'race': 'category',
        'sex': 'category',
        'capital-gain': 'int',
        'capital-loss': 'int',
        'hours-per-week': 'int',
        'native-country': 'category',
        'had-capital-gains': 'int',
        'had-capital-losses': 'int',
        'compensation': 'int'
    }
}


def dataframe_list_to_tensor(df_list: List[pd.DataFrame]) -> np.ndarray:
    """Return 3d tensor from list of Pandas DataFrames. Tensor dimension (number of dataframes, rows, cols)

    Args:
        df_list: List of Pandas DataFrames

    Returns:
        Numpy tensor (n, rows, cols)
    """

    np_list = [df.to_numpy() for df in df_list]
    np_tensor = np.stack(np_list)

    return np_tensor


def numpy_tensor_to_dataframe_list(np_tensor, columns: Optional[Iterable[str]] = None) -> List[pd.DataFrame]:
    """Return list of Pandas DataFrames from 3d tensor. Tensor dimension (number of dataframes, rows, cols)

    Args:
        np_tensor (np.ndarray): Numpy tensor (n, rows, cols)
        columns (List[str]): List of columns for Pandas dataframe

    Returns:
        List of Pandas DataFrames
    """

    df_list = []

    for i in range(np_tensor.shape[0]):
        array_slice = np_tensor[i]
        df = pd.DataFrame(array_slice, columns=columns)
        df_list.append(df)

    return df_list


def transform_for_classification(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    columns = df.columns

    df_copy = df.copy()

    if 'adult' in dataset_name:
        ADULT_TYPES = DATASET_COLUMN_TYPES['adult']
        for column in columns:
            dtype = ADULT_TYPES[column]
            if dtype == 'category':
                df_copy[column] = df_copy[column].astype('category')
            elif dtype == 'int':
                df_copy[column] = df_copy[column].astype('int')

        df_copy = pd.get_dummies(df_copy)

        df_copy = df_copy.reindex(columns=[col for col in df_copy.columns if col != 'compensation'] + ['compensation'])

    elif 'binary' in dataset_name:
        df_copy = df_copy.astype('int')
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return df_copy


def transform_for_modeling(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    columns = df.columns

    df_copy = df.copy()

    if 'adult' in dataset_name:
        ADULT_TYPES = DATASET_COLUMN_TYPES['adult']
        for column in columns:
            dtype = ADULT_TYPES[column]
            if dtype == 'category':
                df_copy[column] = df_copy[column].astype('category')
            elif dtype == 'int':
                df_copy[column] = df_copy[column].astype('int')

        df_copy = df_copy.reindex(columns=[col for col in df_copy.columns if col != 'compensation'] + ['compensation'])

    elif 'binary' in dataset_name:
        df_copy = df_copy.astype('int')
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return df_copy
