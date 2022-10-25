import os
from typing import List

import jax
import numpy as np
import pandas as pd
import pickle
from data_utils import dataframe_list_to_tensor
from src.napsu_mq.napsu_mq import NapsuMQResult
from synthetic_data_object import SynthDataObject
from .constants import TRAIN_DATASET_SIZE_MAP
from src.utils.path_utils import SYNT_DATASETS_FOLDER

models = snakemake.input[0]
n_datasets = snakemake.config['n_datasets']

rng = jax.random.PRNGKey(86933526)

for model_path in models:
    napsu_result_read_file = open(f"{model_path}", "rb")
    model: NapsuMQResult = NapsuMQResult.load(napsu_result_read_file)
    meta_info = model.meta

    experiment_id = meta_info['experiment_id']
    dataset_name = meta_info['dataset_name']
    epsilon = meta_info['epsilon']
    delta = meta_info['delta']
    MCMC_algorithm = meta_info['MCMC_algo']
    query = meta_info['query']
    n_canonical_queries = meta_info['n_canonical_queries']

    n_samples = TRAIN_DATASET_SIZE_MAP[dataset_name]

    synt_datasets: List[pd.DataFrame] = model.generate_extended(rng, n_samples, n_datasets)
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
        inference_algorithm=MCMC_algorithm,
        epsilon=epsilon,
        delta=delta
    )

    path = os.path.join(SYNT_DATASETS_FOLDER,
                        f"synthetic_dataset_{experiment_id}_{dataset_name}_{query}_{epsilon}e_{MCMC_algorithm}.pickle")

    with open(path, "wb") as file:
        pickle.dump(synth_data_object, file)
