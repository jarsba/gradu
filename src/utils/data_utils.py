from typing import List

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
