import sys
import os

from arviz.data.inference_data import InferenceDataT

sys.path.append("/home/jarlehti/projects/gradu")
import jax

jax.config.update("jax_enable_x64", True)

import pandas as pd

from src.utils.path_utils import DATASETS_FOLDER, MODELS_FOLDER
from src.utils.preprocess_dataset import get_adult_train_low_discretization, get_adult_train_high_discretization, \
    get_adult_train_no_discretization
from src.utils.experiment_storage import ExperimentStorage, experiment_id_ctx
from src.utils.keygen import get_key
from src.napsu_mq.napsu_mq import NapsuMQModel, NapsuMQResult
from src.utils.string_utils import epsilon_float_to_str
from src.utils.timer import Timer

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

adult_dataset = pd.read_csv(os.path.join(DATASETS_FOLDER, "cleaned_adult_train_data.csv"))

epsilons = [0.1, 0.3, 1.0, 3.0, 8.0]

adult_train_no_discretization = get_adult_train_no_discretization()
adult_train_discretized_low = get_adult_train_low_discretization()
adult_train_discretized_high = get_adult_train_high_discretization()

storage = ExperimentStorage(file_path="napsu_discretization_test_storage.csv", mode="replace")
timer = Timer(file_path="napsu_discretization_test_timer.csv", mode="replace")

dataset_dict = {
    'no_discretization': adult_train_no_discretization,
    'discretized_low': adult_train_discretized_low,
    'discretized_high': adult_train_discretized_high
}

for epsilon in epsilons:

    epsilon_str = epsilon_float_to_str(epsilon)
    n, d = adult_train_no_discretization.shape
    query = []
    delta = (n ** (-2))

    for name, dataset in dataset_dict.items():
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
            "discretization": name
        }

        pid = timer.start(f"Main run", **timer_meta)

        print(
            f"PARAMS: \n\tdataset name ADULT\n\tdiscretization {name}\n\tMCMC algo: NUTS\n\tepsilon {epsilon}\n\tdelta: {delta}\n\tLaplace approximation {True}")

        print("Initializing NapsuMQModel")
        rng = jax.random.PRNGKey(6473286482)

        model = NapsuMQModel()

        result: NapsuMQResult
        inf_data: InferenceDataT

        result, inf_data = model.fit(
            data=dataset,
            dataset_name="adult",
            rng=rng,
            epsilon=epsilon,
            delta=delta,
            column_feature_set=query,
            MCMC_algo="NUTS",
            use_laplace_approximation=True,
            return_inference_data=True,
        )

        timer.stop(pid)

        dataset_discretization_str = f"adult_{name}"

        print("Writing model to file")
        model_file_path = os.path.join(MODELS_FOLDER,
                                       f"napsu_discretization_test_{dataset_discretization_str}_{epsilon_str}e.dill")
        result.store(model_file_path)

        inf_data.to_netcdf(f"logs/inf_data_discretization_test_{dataset_discretization_str}_{epsilon_str}e.nc")

        # Save storage and timer results every iteration
        storage.save()
        timer.save()

storage.save()
timer.save()
