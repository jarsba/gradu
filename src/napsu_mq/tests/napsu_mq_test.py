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

from jax.config import config

config.update("jax_enable_x64", True)

import unittest
import pytest
import random
import string
import numpy as np
import pandas as pd
import jax
from src.napsu_mq.napsu_mq import NapsuMQResult, NapsuMQModel
from src.napsu_mq.binary_logistic_regression_generator import BinaryLogisticRegressionDataGenerator
from src.napsu_mq.tests.test_utils import create_test_directory, file_exists, TEST_DIRECTORY_PATH, purge_test_directory
from src.utils.keygen import get_key
from src.utils.experiment_storage import experiment_id_ctx


class TestNapsuMQ(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = jax.random.PRNGKey(16778774)
        data_gen = BinaryLogisticRegressionDataGenerator(np.array([1.0, 0.0]))
        cls.data = data_gen.generate_data(n=2000, rng_key=rng)
        cls.dataframe = pd.DataFrame(cls.data, columns=['A', 'B', 'C'], dtype=int)
        cls.n, cls.d = cls.data.shape

        experiment_id = get_key()
        experiment_id_ctx.set(experiment_id)

    def setUp(self):
        self.data = self.__class__.data
        self.dataframe = self.__class__.dataframe
        self.n = self.__class__.n
        self.d = self.__class__.d

    # Takes about ~ 1 minute to run
    @pytest.mark.slow
    def test_NAPSUMQ_model_without_IO(self):
        column_feature_set = [
            ('A', 'B'), ('B', 'C'), ('A', 'C')
        ]

        rng = jax.random.PRNGKey(6473286482)
        inference_rng, sampling_rng = jax.random.split(rng)
        model = NapsuMQModel()
        result = model.fit(data=self.dataframe, dataset_name="binary3d", rng=inference_rng, epsilon=1,
                           delta=(self.n ** (-2)),
                           column_feature_set=column_feature_set,
                           use_laplace_approximation=False)

        datasets = result.generate_extended(rng=sampling_rng, num_data_per_parameter_sample=2000,
                                            num_parameter_samples=5)

        self.assertEqual(len(datasets), 5)
        self.assertEqual(datasets[0].shape, (2000, 3))

        original_means = self.dataframe.mean()
        original_stds = self.dataframe.std()

        for i, df in enumerate(datasets):
            means = df.mean()
            stds = df.std()
            pd.testing.assert_series_equal(means, original_means, check_exact=False, rtol=0.3)
            pd.testing.assert_series_equal(stds, original_stds, check_exact=False, rtol=0.3)

    # Takes about ~ 1 minute to run
    @pytest.mark.slow
    def test_NAPSUMQ_model_with_IO(self):
        column_feature_set = [
            ('A', 'B'), ('B', 'C'), ('A', 'C')
        ]

        rng = jax.random.PRNGKey(6473286482)
        inference_rng, sampling_rng = jax.random.split(rng)

        model = NapsuMQModel()
        result = model.fit(data=self.dataframe, dataset_name="binary3d", rng=inference_rng, epsilon=1,
                           delta=(self.n ** (-2)),
                           column_feature_set=column_feature_set,
                           use_laplace_approximation=False)

        create_test_directory()

        napsu_result_file = open(f"{TEST_DIRECTORY_PATH}/napsu_test_result.dill", "wb")
        result.store(napsu_result_file)

        self.assertTrue(file_exists(f"{TEST_DIRECTORY_PATH}/napsu_test_result.dill"))

        napsu_result_read_file = open(f"{TEST_DIRECTORY_PATH}/napsu_test_result.dill", "rb")
        loaded_result: NapsuMQResult = NapsuMQResult.load(napsu_result_read_file)
        datasets = loaded_result.generate_extended(rng=sampling_rng, num_data_per_parameter_sample=2000,
                                                   num_parameter_samples=5)

        self.assertEqual(len(datasets), 5)
        self.assertEqual(datasets[0].shape, (2000, 3))

        original_means = self.dataframe.mean()
        original_stds = self.dataframe.std()

        for i, df in enumerate(datasets):
            means = df.mean()
            stds = df.std()

            pd.testing.assert_series_equal(means, original_means, check_exact=False, rtol=0.3)
            pd.testing.assert_series_equal(stds, original_stds, check_exact=False, rtol=0.3)

            random_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

            df.to_csv(f'{TEST_DIRECTORY_PATH}/napsumq_test_df_{random_id}.csv')

            self.assertTrue(file_exists(f'{TEST_DIRECTORY_PATH}/napsumq_test_df_{random_id}.csv'))

        purge_test_directory(TEST_DIRECTORY_PATH)

    # Takes about ~ 1 minute to run
    @pytest.mark.slow
    def test_NAPSUMQ_model_without_IO_with_LA(self):
        column_feature_set = [
            ('A', 'B'), ('B', 'C'), ('A', 'C')
        ]

        rng = jax.random.PRNGKey(6473286482)
        inference_rng, sampling_rng = jax.random.split(rng)

        model = NapsuMQModel()
        result, inf_data = model.fit(data=self.dataframe, dataset_name="binary3d", rng=inference_rng, epsilon=1,
                                     delta=(self.n ** (-2)),
                                     column_feature_set=column_feature_set,
                                     use_laplace_approximation=True, return_inference_data=True, enable_profiling=True)

        datasets = result.generate_extended(rng=sampling_rng, num_data_per_parameter_sample=2000,
                                            num_parameter_samples=5)

        self.assertEqual(len(datasets), 5)
        self.assertEqual(datasets[0].shape, (2000, 3))

        original_means = self.dataframe.mean()
        original_stds = self.dataframe.std()

        for i, df in enumerate(datasets):
            means = df.mean()
            stds = df.std()
            pd.testing.assert_series_equal(means, original_means, check_exact=False, rtol=0.3)
            pd.testing.assert_series_equal(stds, original_stds, check_exact=False, rtol=0.3)
