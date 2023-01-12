from typing import List, Iterable, Optional

import numpy as np
import pandas as pd


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
