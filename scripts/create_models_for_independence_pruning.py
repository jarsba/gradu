import sys
import os

#sys.path.append(snakemake.config['workdir'])
#sys.path.append("/home/jarlehti/projects/gradu")
import jax

jax.config.update("jax_enable_x64", True)

import itertools
from arviz.data.inference_data import InferenceDataT
import pandas as pd

from src.utils.path_utils import MODELS_FOLDER, DATASETS_FOLDER
from src.utils.timer import Timer
from src.utils.preprocess_dataset import convert_to_categorical
from src.utils.keygen import get_key
from src.utils.experiment_storage import ExperimentStorage, experiment_id_ctx
from src.napsu_mq.napsu_mq import NapsuMQModel, NapsuMQResult
from src.utils.query_utils import join_query_list
from src.utils.string_utils import epsilon_float_to_str
from src.utils.data_utils import transform_for_modeling

"""

Steps:

Start with Adult dataset with 6 variables and determine the independence ranking for each pair (15 combinations) using 
Chi-Square test. Run Napsu-MQ with full set of marginals and with each full set with one pair removed. Look at the 
downstream results like confidence interval coverage and compare the predicted result with the real results. Continue 
with 5 variables using the set with of variables with least harm to downstream accuracy and repeat for 5 and 4 variables. 

"""

dataset_map = snakemake.config['independence_pruning_datasets']
inverted_dataset_map = {v: k for k, v in dataset_map.items()}
dataset = snakemake.input
dataset_name = inverted_dataset_map[dataset]

adult_dataset = pd.read_csv(dataset)

epsilons = snakemake.config["epsilons"]

six_adult_columns = list(adult_dataset.columns)

# Make set of sets
marginal_pairs = list(itertools.combinations(six_adult_columns, 2))
full_set_of_marginals = marginal_pairs

# Removes set from list of tuples that are interpreted as list of sets
immutable_set_remove = lambda element, list_obj: list(filter(lambda x: set(x) != set(element), list_obj))

# List of lists with tuples of all 2-way marginals
test_queries = [immutable_set_remove(pair, full_set_of_marginals) for pair in marginal_pairs]
# Add also full and no queries
test_queries.append(full_set_of_marginals)
test_queries.append([])

adult_train_df = transform_for_modeling("adult_small", adult_dataset)

storage = ExperimentStorage(file_path="napsu_independence_pruning_storage.csv", mode="replace")
timer = Timer(file_path="napsu_independence_pruning_timer.csv", mode="replace")

for epsilon in epsilons:

    epsilon_str = epsilon_float_to_str(epsilon)

    for query_list in test_queries:

        n, d = adult_train_df.shape
        query_str = join_query_list(query_list)

        if len(query_list) == 0:
            missing_query = "all"
            query_removed = full_set_of_marginals

        elif len(query_list) == len(full_set_of_marginals):
            missing_query = "none"
            query_removed = []
        else:
            missing_query = list(set(full_set_of_marginals) - set(test_queries[0]))
            query_removed = ["+".join(map(str, pair)) for pair in missing_query][0]

            if len(missing_query) != 1:
                print(f"Missing too many queries! Queries missing: {missing_query}")
                sys.exit(1)

        delta = (n ** (-2))

        experiment_id = get_key()
        experiment_id_ctx.set(experiment_id)

        timer_meta = {
            "experiment_id": experiment_id,
            "dataset_name": dataset_name,
            "query": query_str,
            "missing_query": query_removed,
            "epsilon": epsilon,
            "delta": delta,
            "MCMC_algo": "NUTS",
            "laplace_approximation": True
        }

        pid = timer.start(f"Main run", **timer_meta)

        print(
            f"PARAMS: \n\tdataset name {dataset_name}\n\tmissing query {missing_query}\n\tMCMC algo: NUTS\n\tepsilon {epsilon}\n\tdelta: {delta}\n\tLaplace approximation {True}")

        print("Initializing NapsuMQModel")
        rng = jax.random.PRNGKey(6473286482)

        model = NapsuMQModel()

        result: NapsuMQResult
        inf_data: InferenceDataT

        result, inf_data = model.fit(
            data=adult_train_df,
            dataset_name=dataset_name,
            rng=rng,
            epsilon=epsilon,
            delta=delta,
            column_feature_set=query_list,
            MCMC_algo="NUTS",
            use_laplace_approximation=True,
            return_inference_data=True,
            missing_query=missing_query,
        )

        timer.stop(pid)

        dataset_query_str = f"{dataset_name}_{query_removed}"

        print("Writing model to file")
        model_file_path = os.path.join(MODELS_FOLDER, f"napsu_independence_pruning_{dataset_query_str}_missing_{epsilon_str}e.dill")
        result.store(model_file_path)

        inf_data.to_netcdf(f"logs/inf_data_independence_pruning_{dataset_query_str}_missing_{epsilon_str}e.nc")

        timer.save()
        storage.save()

timer.save()
storage.save()
