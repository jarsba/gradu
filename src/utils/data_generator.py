import numpy as np
import pandas as pd


def create_dummy_dataset(n_categories, n_columns, n_rows):
    dataframe = []

    for i in range(n_columns):
        column = np.round(np.random.uniform(1, n_categories, size=n_rows)).astype(int)
        dataframe.append(column)

    df = pd.DataFrame.from_records(dataframe).T.astype("category")

    # Columns are N first capital letters based on n_columns
    columns = [chr(i) for i in range(65, 65 + n_columns)]

    df.columns = columns

    return df
