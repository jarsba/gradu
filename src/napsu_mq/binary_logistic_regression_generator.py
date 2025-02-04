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
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.special import expit

TRUE_PARAMS_MAP = {
    "3d": [1.0, 0.0],
    "4d": [1.0, 0.0, 2.0]
}


class BinaryLogisticRegressionDataGenerator:
    """
    Generator for d-dimensional binary datasets where all variables but the
    last are generated with independent coin flips, and the last variable
    is generated from logistic regression on the others.

    Used for testing purposes.
    """

    def __init__(self, true_params: jnp.ndarray):
        """Initialise the generator.
        Args:
            true_params (jnp.ndarray): Coefficients for the logistic regression.
        """
        self.true_params = true_params
        self.d = true_params.shape[0] + 1
        self.x_values = self.compute_x_values()
        self.values_by_feature = {i: [0, 1] for i in range(self.d)}

    def generate_data(self, n: int, rng_key: jax.random.PRNGKey = None, probability=0.5) -> jnp.ndarray:
        """Generate the output d-dimensional binary output.
        Args:
            n (int): Number of datapoints to generate.
            rng_key (dp3 RNG key, optional): Random number generator key for d3p. Defaults to None.
            probability (float): Probability p for the Bernoulli distribution.
        Returns:
            jnp.ndarray: The generated output.
        """

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        x = jax.random.bernoulli(key=rng_key, p=probability, shape=(n, self.d - 1)).astype(int)
        alpha = x @ self.true_params.reshape((-1, 1))
        probs = expit(alpha)
        y = jax.random.bernoulli(key=rng_key, p=probs).astype(int)
        result = jnp.concatenate((x, y), axis=1)

        return result

    def compute_x_values(self) -> np.ndarray:
        """Enumerate all possible datapoints.
        Returns:
            np.ndarray: 2-d array enumerating all possible datapoints.
        """
        x_values = np.zeros((2 ** self.d, self.d))
        for i, val in enumerate(itertools.product(range(2), repeat=self.d)):
            x_values[i, :] = np.array(val)

        return x_values


if __name__ == '__main__':
    data_generator4d = BinaryLogisticRegressionDataGenerator(np.array(TRUE_PARAMS_MAP["4d"]))
    rng_key = jax.random.PRNGKey(42)
    data: np.ndarray = jnp.asarray(data_generator4d.generate_data(150000, rng_key=rng_key))

    train_data, test_data = data[0:100000], data[100000:]

    train_dataframe = pd.DataFrame(train_data, columns=['A', 'B', 'C', 'D'])
    train_dataframe.to_csv("binary4d.csv", index=False)

    test_dataframe = pd.DataFrame(test_data, columns=['A', 'B', 'C', 'D'])
    test_dataframe.to_csv("binary4d_test.csv", index=False)
