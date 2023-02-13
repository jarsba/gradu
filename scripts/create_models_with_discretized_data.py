import sys
import os
sys.path.append(snakemake.config['workdir'])


from arviz.data.inference_data import InferenceDataT
import jax

jax.config.update("jax_enable_x64", True)

import pandas as pd

from src.utils.path_utils import MODELS_FOLDER
from src.utils.experiment_storage import ExperimentStorage, experiment_id_ctx
from src.utils.keygen import get_key
from src.napsu_mq.napsu_mq import NapsuMQModel, NapsuMQResult
from src.utils.string_utils import epsilon_float_to_str
from src.utils.timer import Timer
from src.utils.data_utils import transform_for_modeling

"""

Steps:

Create NAPSU-MQ model for Adult datasets with columns ['age', 'sex', 'education-num' , 'hours-per-week', 'workclass', 'marital-status', 'had-capital-gains', 'had-capital-losses', 'compensation'] with "small" levels of discretization and with "high" levels
of discretization.

Low level of discretization:

age -> buckets of 5
hours-per-week -> buckets of 5
capital-gains -> binary 0/1
capital-losses -> binary 0/1

High level of discretization:

age -> buckets of 10
hours-per-week -> buckets of 10
capital-gains -> binary 0/1
capital-losses -> binary 0/1
"""

dataset_map = snakemake.config['discretization_datasets']
inverted_dataset_map = {v: k for k, v in dataset_map.items()}

datasets = snakemake.input
target_files = snakemake.output

epsilons = snakemake.config["epsilons"]

storage = ExperimentStorage(file_path="napsu_discretization_test_storage.csv", mode="replace")
timer = Timer(file_path="napsu_discretization_test_timer.csv", mode="replace")

input_output_map = list(zip(datasets, target_files))

print(input_output_map)

for dataset, target_file in input_output_map:

    dataset_name = inverted_dataset_map[dataset]

    dataframe = pd.read_csv(dataset)

    dataframe = transform_for_modeling(dataset_name, dataframe)

    discretization_level = "low" if "low" in dataset else "high"

    for epsilon in epsilons:

        epsilon_str = epsilon_float_to_str(epsilon)
        n, d = dataframe.shape
        query = []
        delta = (n ** (-2))

        experiment_id = get_key()
        experiment_id_ctx.set(experiment_id)

        timer_meta = {
            "experiment_id": experiment_id,
            "dataset_name": "adult",
            "query": query,
            "epsilon": epsilon,
            "delta": delta,
            "MCMC_algo": "NUTS",
            "laplace_approximation": True,
            "discretization": discretization_level
        }

        pid = timer.start(f"Main run", **timer_meta)

        print(
            f"PARAMS: \n\tdataset name ADULT\n\tdiscretization {discretization_level}\n\tMCMC algo: NUTS\n\tepsilon {epsilon}\n\tdelta: {delta}\n\tLaplace approximation {True}")

        print("Initializing NapsuMQModel")
        rng = jax.random.PRNGKey(6473286482)

        model = NapsuMQModel()

        result: NapsuMQResult
        inf_data: InferenceDataT

        result, inf_data = model.fit(
            data=dataframe,
            dataset_name=f"adult_{discretization_level}_discretization",
            rng=rng,
            epsilon=epsilon,
            delta=delta,
            column_feature_set=query,
            MCMC_algo="NUTS",
            use_laplace_approximation=True,
            return_inference_data=True,
            discretization=discretization_level
        )

        timer.stop(pid)

        print("Writing model to file")
        model_file_path = os.path.join(MODELS_FOLDER,
                                       f"napsu_discretization_{discretization_level}_{epsilon_str}e.dill")
        result.store(model_file_path)

        inf_data.to_netcdf(f"logs/inf_data_discretization_{discretization_level}_{epsilon_str}e.nc")

        # Save storage and timer results every iteration
        storage.save()
        timer.save()

storage.save()
timer.save()
