import os
import jax
from napsu_mq.napsu_mq import NapsuMQResult

models = snakemake.input[0]
n_datasets = snakemake.config['n_datasets']

rng = jax.random.PRNGKey(86933526)

SYNT_DATASET_DIRECTORY = os.path.join("data", "synt_datasets")

DATASET_SIZE_MAP = {
    "adult": 45222,
    "binary4d": 100000
}

for model_path in models:
    print(model_path)
    napsu_result_read_file = open(f"{model_path}", "rb")
    model: NapsuMQResult = NapsuMQResult.load(napsu_result_read_file)
    meta_info = model.meta

    dataset_name = meta_info['dataset_name']
    epsilon = meta_info['epsilon']
    MCMC_algorithm = meta_info['MCMC_algo']

    n_samples = DATASET_SIZE_MAP[dataset_name]

    synt_datasets = model.generate_extended(rng, n_samples, n_datasets)

    for i, df in enumerate(synt_datasets):
        path = os.path.join(SYNT_DATASET_DIRECTORY, f"synthetic_dataset_{i}_{dataset_name}_{epsilon}e_{MCMC_algorithm}.csv")
        df.to_csv(path, index=False)