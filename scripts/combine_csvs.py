import sys

sys.path.append(snakemake.config['workdir'])

import pandas as pd

if __name__ == '__main__':

    file_paths = snakemake.input
    target_file = snakemake.output[0]

    dataframes = []

    for path in file_paths:
        df = pd.read_csv(path)
        dataframes.append(df)

    combined_csv = pd.concat(dataframes, ignore_index=True)
    combined_csv.to_csv(target_file, index=False)
