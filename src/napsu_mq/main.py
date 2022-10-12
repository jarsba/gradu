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


import sys
import time
from pathlib import Path
from typing import TypeVar, List, Tuple, Iterable

import numpy as np
from jax.config import config

from .binary_logistic_regression_generator import BinaryLogisticRegressionDataGenerator

config.update("jax_enable_x64", True)
import jax
import pandas as pd
import logging
from .napsu_mq import NapsuMQModel, NapsuMQResult

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

InputDataType = TypeVar("InputDataType", pd.DataFrame, str)

current_path = Path().resolve()

times = []

rng = jax.random.PRNGKey(6473286482)
np.random.seed(42)


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


def sample(model: NapsuMQResult, n_datasets: int, n_samples: int, rng: jax.random.PRNGKey = None) -> Iterable[
    pd.DataFrame]:
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
    print(f"Epsilon: {epsilon}")

    data_gen = BinaryLogisticRegressionDataGenerator(np.array([1.0, 0.0]))
    data = data_gen.generate_data(2000)
    dataframe = pd.DataFrame(data, columns=['A', 'B', 'C'])
    x_values = data_gen.x_values
    n, d = data.shape

    column_feature_set = [
        ('A', 'B'), ('B', 'C'), ('A', 'C')
    ]

    print(f"Running NAPSU-MQ for adult dataset with epsilon {epsilon}")
    start = time.time()

    result = create_model(input=dataframe, dataset_name="dummy", epsilon=epsilon, delta=(n ** (-2)),
                          cliques=column_feature_set, use_laplace_approximation=True)
    finish = time.time()
    print(f"Total time for NAPSU-MQ with epsilon {epsilon}: {finish - start}")

    print(f"Sampling NAPSU-MQ for adult dataset")
    start = time.time()

    # sample(model=result, n_datasets=10, n_samples=n)
    finish = time.time()
    print(f"Total time for sampling: {finish - start}")
