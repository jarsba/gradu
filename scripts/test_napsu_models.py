import sys
import os


import jax
jax.config.update("jax_enable_x64", True)

import pandas as pd
from arviz.data.inference_data import InferenceDataT

from src.utils.path_utils import MODELS_FOLDER
from src.napsu_mq.napsu_mq import NapsuMQModel, NapsuMQResult
from src.utils.query_utils import join_query_list
from src.utils.data_utils import transform_for_modeling

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
DATA_FOLDER = os.path.join(ROOT_FOLDER, "data", "datasets")

epsilon = 0.1
query_list = []
algo = 'NUTS'
epsilon_str = "0.1"

dataframe = pd.read_csv(os.path.join(DATA_FOLDER, "adult_small.csv"))
dataset_name = "adult_small"

dataframe = transform_for_modeling(dataset_name, dataframe)

n, d = dataframe.shape
print(f"Rows: {n}")
print(f"Columns: {d}")
query_str = join_query_list(query_list)
delta = (n ** (-2))

dataset_query_str = f"{dataset_name}_{query_str}"

print(
    f"PARAMS: \n\tdataset name {dataset_name}\n\tcliques {query_str}\n\tMCMC algo {algo}\n\tepsilon {epsilon_str}\n\tdelta: {delta}\n\tLaplace approximation {True}")

print("Initializing NapsuMQModel")
rng = jax.random.PRNGKey(6473286482)

model = NapsuMQModel()

result: NapsuMQResult
inf_data: InferenceDataT

result, inf_data = model.fit(
    data=dataframe,
    dataset_name=dataset_name,
    rng=rng,
    epsilon=epsilon,
    delta=delta,
    column_feature_set=query_list,
    MCMC_algo=algo,
    use_laplace_approximation=True,
    return_inference_data=True,
    enable_profiling=True,
    laplace_approximation_algorithm="torch_LBFGS",
)

model_file_path = os.path.join(MODELS_FOLDER, f"napsu_{dataset_query_str}_{epsilon_str}e_{algo}.dill")
result.store(model_file_path)

inf_data.to_netcdf(f"logs/inf_data_{dataset_query_str}_{epsilon_str}e_{algo}.nc")