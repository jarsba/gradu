import os
import jax
from src.napsu_mq.napsu_mq import NapsuMQResult
from .constants import TRAIN_DATASET_SIZE_MAP
from src.utils.path_utils import SYNT_DATASETS_FOLDER

models = snakemake.input[0]
n_datasets = snakemake.config['n_datasets']

rng = jax.random.PRNGKey(86933526)

for model_path in models:
    napsu_result_read_file = open(f"{model_path}", "rb")
    model: NapsuMQResult = NapsuMQResult.load(napsu_result_read_file)
    meta_info = model.meta

    dataset_name = meta_info['dataset_name']
    epsilon = meta_info['epsilon']
    MCMC_algorithm = meta_info['MCMC_algo']
    query = meta_info['query']

    n_samples = TRAIN_DATASET_SIZE_MAP[dataset_name]

    synt_datasets = model.generate_extended(rng, n_samples, n_datasets)

    for i, df in enumerate(synt_datasets):
        path = os.path.join(SYNT_DATASETS_FOLDER,
                            f"synthetic_dataset_{i}_{dataset_name}_{query}_{epsilon}e_{MCMC_algorithm}.csv")
        df.to_csv(path, index=False)
