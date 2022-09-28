# Copyright 2022 twinify Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from pathlib import Path
from typing import TypeVar, Optional, List, Tuple, Iterable
import sys
from jax.config import config

import numpy as np
import binary_logistic_regression_generator as binary_lr_data

config.update("jax_enable_x64", True)
import jax
import pandas as pd
import logging
import time
from datetime import datetime
import random
import string
from napsu_mq.napsu_mq import NapsuMQModel, NapsuMQResult

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

InputDataType = TypeVar("InputDataType", pd.DataFrame, str)

current_path = Path().resolve()

times = []


def format_td():
    t0, t1 = times[-2:]
    return t1 - t0


def create_model(input: pd.DataFrame, dataset_name: str, epsilon: float, delta: float,
                 cliques: List[Tuple[str, ...]] = None, MCMC_algo: str = 'NUTS',
                 use_laplace_approximation: bool = True, rng: jax.random.PRNGKey = None) -> NapsuMQResult:
    if rng is None:
        rng = jax.random.PRNGKey(6473286482)

    print(
        f"PARAMS: \n\tepsilon {epsilon}\n\tdelta: {delta}")

    print("Initializing NapsuMQModel")
    model = NapsuMQModel()
    result = model.fit(input, dataset_name, rng, epsilon, delta, column_feature_set=cliques, MCMC_algo=MCMC_algo,
                       use_laplace_approximation=use_laplace_approximation)

    return result


def sample(model: NapsuMQResult, n_datasets: int, n_samples: int, rng: jax.random.PRNGKey = None) -> Iterable[pd.DataFrame]:
    if rng is None:
        rng = jax.random.PRNGKey(86933526)

    print("Generating datasets")
    datasets = model.generate_extended(rng, n_samples, n_datasets)

    return datasets


if __name__ == "__main__":

    rng = jax.random.PRNGKey(6473286482)
    np.random.seed(42)
    jax.numpy.set_printoptions(threshold=10000)

    epsilon = float(sys.argv[1])
    dataset = sys.argv[2]
    print(f"Epsilon: {epsilon}")
    if dataset == 'adult':
        ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))

        data = pd.read_csv(os.path.join(ROOT_FOLDER, "data/cleaned_adult_data_v2.bak"))
        age_labels = ["{0} - {1}".format(i, i + 9) for i in range(0, 100, 10)]

        data['age'] = pd.Categorical(pd.cut(data.age, range(0, 105, 10), right=False, labels=age_labels))
        data['workclass'] = pd.Categorical(data['workclass'])
        data['education-num'] = pd.Categorical(data['education-num'])
        data['marital-status'] = pd.Categorical(data['marital-status'])
        data['occupation'] = pd.Categorical(data['occupation'])
        data['relationship'] = pd.Categorical(data['relationship'])
        data['race'] = pd.Categorical(data['race'])
        data['sex'] = pd.Categorical(data['sex'])
        data['had-capital-gains'] = pd.Categorical(data['had-capital-gains'])
        data['had-capital-losses'] = pd.Categorical(data['had-capital-losses'])
        hours_labels = ["{0} - {1}".format(i, i + 9) for i in range(0, 100, 10)]
        data['hours-per-week'] = pd.Categorical(
            pd.cut(data['hours-per-week'], range(0, 105, 10), right=False, labels=hours_labels)
        )
        data['native-country'] = pd.Categorical(data['native-country'])
        data['compensation'] = pd.Categorical(data['compensation'])

        data.drop(columns=['had-capital-gains', 'had-capital-losses'], inplace=True)

        column_feature_set = [
            ("age", "compensation"), ("race", "compensation"), ("sex", "compensation"), ("race", "sex"),
            ("hours-per-week", "compensation")
        ]

        n = len(data)

        print(f"Running NAPSU-MQ for adult dataset with epsilon {epsilon}")
        start = time.time()

        name = 'adult'
        id = f'{str(epsilon).replace(".", "")}_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

        result = create_model(rng=rng, name=name, id=id, input=data, epsilon=epsilon, delta=(n ** (-2)),
                              cliques=column_feature_set, use_laplace_approximation=True, output="output")
        finish = time.time()
        print(f"Total time for NAPSU-MQ with epsilon {epsilon}: {finish - start}")

        print(f"Sampling NAPSU-MQ for adult dataset")
        start = time.time()

        sample(rng=rng, name=name, id=id, model=result, n_datasets=10, n_samples=n, output="output")
        finish = time.time()
        print(f"Total time for sampling: {finish - start}")

    elif dataset == 'dummy':
        data_gen = binary_lr_data.BinaryLogisticRegressionDataGenerator(np.array([1.0, 0.0]))
        data = data_gen.generate_data(2000)
        dataframe = pd.DataFrame(data, columns=['A', 'B', 'C'])
        x_values = data_gen.x_values
        n, d = data.shape

        column_feature_set = [
            ('A', 'B'), ('B', 'C'), ('A', 'C')
        ]

        print(f"Running NAPSU-MQ for adult dataset with epsilon {epsilon}")
        start = time.time()

        name = 'binary_lr'
        id = f'{str(epsilon).replace(".", "")}_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

        result = create_model(rng=rng, name=name, id=id, input=dataframe, epsilon=epsilon, delta=(n ** (-2)),
                              cliques=column_feature_set, use_laplace_approximation=True, output="output")
        finish = time.time()
        print(f"Total time for NAPSU-MQ with epsilon {epsilon}: {finish - start}")

        print(f"Sampling NAPSU-MQ for adult dataset")
        start = time.time()

        sample(rng=rng, name=name, id=id, model=result, n_datasets=10, n_samples=n, output="output")
        finish = time.time()
        print(f"Total time for sampling: {finish - start}")
    else:
        print(f"Unrecognized dataset: {dataset}")
