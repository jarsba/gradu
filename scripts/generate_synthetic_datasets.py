import sys
import os
sys.path.append(snakemake.config['workdir'])

from typing import List

import jax
import numpy as np
import pandas as pd
import pickle
from src.utils.data_utils import dataframe_list_to_tensor
from src.utils.string_utils import epsilon_float_to_str
from src.napsu_mq.napsu_mq import NapsuMQResult
from src.utils.synthetic_data_object import SynthDataObject
from constants import TRAIN_DATASET_SIZE_MAP
from src.utils.path_utils import SYNT_DATASETS_FOLDER

models = snakemake.input
output_files = snakemake.output
n_synt_datasets = snakemake.config['n_synt_datasets']

print(models)

sampling_rng = jax.random.PRNGKey(86933526)

for model_path, output_path in zip(models, output_files):
    print(f"Generating data for model {model_path}")
    print(f"Output path: {output_path}")

    napsu_result_read_file = open(f"{model_path}", "rb")
    model: NapsuMQResult = NapsuMQResult.load(napsu_result_read_file)
    meta_info = model.meta

    experiment_id = meta_info['experiment_id']
    dataset_name = meta_info['dataset_name']
    epsilon = meta_info['epsilon']
    delta = meta_info['delta']
    MCMC_algorithm = meta_info['MCMC_algo']
    query = meta_info['query_list']
    query_str = meta_info['query_str']
    n_canonical_queries = meta_info['n_canonical_queries']

    n_synt_samples = TRAIN_DATASET_SIZE_MAP[dataset_name]

    synt_datasets: List[pd.DataFrame] = model.generate_extended(sampling_rng, n_synt_samples, n_synt_datasets,
                                                                single_dataframe=False)
    np_tensor: np.ndarray = dataframe_list_to_tensor(synt_datasets)

    n_datasets, n_rows, n_cols = np_tensor.shape

    synth_data_object = SynthDataObject(
        synth_data=np_tensor,
        experiment_id=experiment_id,
        n_datasets=n_datasets,
        n_rows=n_rows,
        n_cols=n_cols,
        original_dataset=dataset_name,
        queries=query,
        n_canonical_queries=n_canonical_queries,
        inference_algorithm=MCMC_algorithm,
        epsilon=epsilon,
        delta=delta
    )

    epsilon_str = epsilon_float_to_str(epsilon)

    with open(output_path, "wb") as file:
        pickle.dump(synth_data_object, file)
        file.close()
