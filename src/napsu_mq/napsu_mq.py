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
from typing import Optional, Union, Iterable, BinaryIO, Mapping, Tuple, List, Literal

import arviz as az
import dill
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import jaxopt
from jax.random import PRNGKey
from mbi import Domain, Dataset
from arviz.data.inference_data import InferenceDataT

from . import maximum_entropy_inference as mei
from . import privacy_accounting as privacy_accounting
from .base import InferenceModel, InferenceResult, InvalidFileFormatException
from .dataframe_data import DataFrameData
from .marginal_query import FullMarginalQuerySet
from .markov_network_jax import MarkovNetworkJax
from .mst import MST_selection
from src.utils.timer import Timer
from src.utils.query_utils import calculate_query_number, join_query_list
from src.utils.experiment_storage import experiment_id_ctx
from src.utils.keygen import get_key

timer = Timer()

laplace_approximation_algorithms = Literal['jax_minimize', 'jaxopt_LBFGS', 'jaxopt_BFGS', 'tfp_LBFGS', 'tfp_BFGS']


def check_kwargs(kwargs, name, default_value):
    if name in kwargs:
        return kwargs[name]
    else:
        return default_value


laplace_approximation_mapping = {
    'jax_minimize': mei.run_numpyro_laplace_approximation,
    'jaxopt_LBGFS': mei.laplace_approximation_with_jaxopt,
}


class NapsuMQModel(InferenceModel):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, data: pd.DataFrame, dataset_name: str, rng: PRNGKey, epsilon: float, delta: float,
            **kwargs) -> Union[
        'NapsuMQResult', Tuple['NapsuMQResult', InferenceDataT], Mapping, Tuple['NapsuMQResult', Timer], Tuple[
            'NapsuMQResult', InferenceDataT, Timer]]:
        column_feature_set = check_kwargs(kwargs, 'column_feature_set', [])
        MCMC_algo = check_kwargs(kwargs, 'MCMC_algo', 'NUTS')
        use_laplace_approximation = check_kwargs(kwargs, 'use_laplace_approximation', True)
        return_inference_data = check_kwargs(kwargs, 'return_inference_data', False)
        missing_query = check_kwargs(kwargs, "missing_query", None)
        enable_profiling = check_kwargs(kwargs, "enable_profiling", False)
        dry_run = check_kwargs(kwargs, "dry_run", False)
        return_MST_weights = check_kwargs(kwargs, "return_MST_weights", False)
        discretization = check_kwargs(kwargs, "discretization", None)
        laplace_approximation_algorithm = check_kwargs(kwargs, "laplace_approximation_algorithm", "jax_minimize")
        return_timer = check_kwargs(kwargs, "return_timer", False)
        laplace_approximation_forward_mode = check_kwargs(kwargs, "laplace_approximation_forward_mode", False)

        try:
            experiment_id = experiment_id_ctx.get()
        except Exception as exc:
            print(f"No experiment_id found: {exc}")
            experiment_id = get_key()
            print(f"Setting experiment_id to {experiment_id}")
            experiment_id_ctx.set(experiment_id)

        query_str = join_query_list(column_feature_set)

        timer_meta = {
            "experiment_id": experiment_id,
            "dataset_name": dataset_name,
            "query_str": query_str,
            "query_list": column_feature_set,
            "epsilon": epsilon,
            "delta": delta,
            "MCMC_algo": MCMC_algo,
            "laplace_approximation": use_laplace_approximation,
            "missing_query": missing_query,
            "discretization": discretization,
        }

        dataframe = DataFrameData(data)
        print(f"Domain size: {dataframe.get_domain_size()}")
        category_mapping = DataFrameData.get_category_mapping(data)
        n, d = dataframe.int_array.shape
        print(n)
        print(d)

        domain_key_list = list(dataframe.values_by_col.keys())
        domain_value_count_list = [len(dataframe.values_by_col[key]) for key in domain_key_list]

        pid = timer.start(f"Query selection", **timer_meta)

        domain = Domain(domain_key_list, domain_value_count_list)

        print("start MST selection")
        if return_MST_weights is True:
            query_sets, weights = MST_selection(Dataset(dataframe.int_df, domain), epsilon, delta,
                                                cliques_to_include=column_feature_set,
                                                return_MST_weights=return_MST_weights)

        else:
            query_sets = MST_selection(Dataset(dataframe.int_df, domain), epsilon, delta,
                                       cliques_to_include=column_feature_set)

        print("end MST selection")

        timer.stop(pid)

        pid = timer.start(f"Calculating full marginal query", **timer_meta)

        queries = FullMarginalQuerySet(query_sets, dataframe.values_by_col)
        timer.stop(pid)

        # query_list = queries.flatten()
        # print(query_list)

        pid = timer.start(f"Calculating canonical query set", **timer_meta)

        queries = queries.get_canonical_queries()

        query_number = calculate_query_number(queries.queries)
        timer_meta['n_canonical_queries'] = query_number

        print(f"Canonical queries: {query_number}")

        timer.stop(pid)

        mnjax = MarkovNetworkJax(dataframe.values_by_col, queries)

        pid = timer.start(f"Calculating lambda0", **timer_meta)
        mnjax.lambda0(jnp.ones(mnjax.lambda_d))
        timer.stop(pid)
        print(timer.get_time(pid))

        pid = timer.start(f"Calculating suff stat mean and cov", **timer_meta)
        mnjax.suff_stat_mean_and_cov(jnp.ones(mnjax.lambda_d))
        timer.stop(pid)
        print(timer.get_time(pid))

        junction_tree_width = mnjax.junction_tree.calculate_max_tree_width()
        print(f"Junction tree width: {junction_tree_width}")
        timer_meta['junction_tree_width'] = junction_tree_width

        suff_stat = np.sum(queries.flatten()(dataframe.int_array), axis=0)

        suff_stat_dim = suff_stat.shape
        print(suff_stat_dim)
        timer_meta['suff_stat_dim'] = suff_stat_dim

        sensitivity = np.sqrt(2 * len(query_sets))
        inference_rng, dp_rng = jax.random.split(rng, 2)

        sigma_DP = privacy_accounting.sigma(epsilon, delta, sensitivity)

        print(sigma_DP)

        dp_noise = jax.random.normal(dp_rng, suff_stat.shape) * sigma_DP
        dp_suff_stat = suff_stat + dp_noise

        if dry_run is True:
            meta = {
                'original_queries': column_feature_set,
                'canonical_query_number': query_number,
                'tree_width': junction_tree_width,
                'sigma_DP': sigma_DP,
                'suff_stat_dim': suff_stat_dim,
            }

            if return_MST_weights is True:
                meta['MST_weights'] = weights

            return meta

        if use_laplace_approximation is True:

            timer_meta['laplace_approximation_algorithm'] = laplace_approximation_algorithm

            approx_rng, mcmc_rng = jax.random.split(inference_rng, 2)

            pid = timer.start(f"Laplace approximation", **timer_meta)

            if laplace_approximation_algorithm == 'jax_minimize':
                laplace_approx, success = mei.run_numpyro_laplace_approximation(approx_rng, dp_suff_stat, n, sigma_DP,
                                                                                mnjax,
                                                                                use_forward_mode=laplace_approximation_forward_mode)
            elif laplace_approximation_algorithm == 'jaxopt_LBFGS':
                laplace_approx, success = mei.laplace_approximation_with_jaxopt(approx_rng, dp_suff_stat, n, sigma_DP,
                                                                                mnjax,
                                                                                use_forward_mode=laplace_approximation_forward_mode)
            else:
                raise ValueError(f"Unknown laplace approximation algorithm: {laplace_approximation_algorithm}")

            timer.stop(pid)

            pid = timer.start(f"MCMC", **timer_meta)

            mcmc, backtransform = mei.run_numpyro_mcmc_normalised(
                mcmc_rng, dp_suff_stat, n, sigma_DP, mnjax, laplace_approx, num_samples=2000, num_warmup=800,
                num_chains=4, enable_profiling=enable_profiling
            )

            timer.stop(pid)

            inf_data = az.from_numpyro(mcmc, log_likelihood=False)
            posterior_values = inf_data.posterior.stack(draws=("chain", "draw"))
            posterior_values = backtransform(posterior_values.norm_lambdas.values.transpose())

        else:

            pid = timer.start(f"MCMC", **timer_meta)

            mcmc = mei.run_numpyro_mcmc(
                inference_rng, dp_suff_stat, n, sigma_DP, mnjax, MCMC_algo, num_samples=2000, num_warmup=800,
                num_chains=4, enable_profiling=enable_profiling
            )

            timer.stop(pid)

            inf_data = az.from_numpyro(mcmc, log_likelihood=False)
            posterior_values = inf_data.posterior.stack(draws=("chain", "draw"))
            posterior_values = posterior_values.lambdas.values.transpose()

        if return_timer is True:
            if return_inference_data is True:
                return NapsuMQResult(mnjax, posterior_values, category_mapping, timer_meta), inf_data, timer
            else:
                return NapsuMQResult(mnjax, posterior_values, category_mapping, timer_meta), timer
        else:
            if return_inference_data is True:
                return NapsuMQResult(mnjax, posterior_values, category_mapping, timer_meta), inf_data
            else:
                return NapsuMQResult(mnjax, posterior_values, category_mapping, timer_meta)


class NapsuMQResult(InferenceResult):

    def __init__(self, markov_network: MarkovNetworkJax, posterior_values: jnp.ndarray,
                 category_mapping: Mapping, meta: Mapping) -> None:
        super().__init__()
        self._markov_network = markov_network
        self._posterior_values = posterior_values
        self._category_mapping = category_mapping
        self._meta = meta

    @property
    def markov_network(self) -> MarkovNetworkJax:
        return self._markov_network

    @property
    def posterior_values(self) -> jnp.ndarray:
        return self._posterior_values

    @property
    def category_mapping(self) -> Mapping:
        return self._category_mapping

    @property
    def meta(self) -> Mapping:
        return self._meta

    def _store_to_io(self, write_io: BinaryIO) -> None:
        assert write_io.writable()
        result = dill.dumps(self)
        return NapsuMQResultIO.store_to_io(write_io, result)

    @classmethod
    def _load_from_io(cls, read_io: BinaryIO) -> 'NapsuMQResult':
        return NapsuMQResultIO.load_from_io(read_io)

    def generate_extended(self, rng: jax.random.PRNGKey, num_data_per_parameter_sample: int,
                          num_parameter_samples: int,
                          single_dataframe: Optional[bool] = False) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """Generate new synthetic datasets from NAPSU-MQ model

        Args:
            rng (d3p.random.PRNGState): d3p PRNG key
            num_data_per_parameter_sample: Number of synthetic datapoints for each dataset
            num_parameter_samples: Number of datasets
            single_dataframe: Return generated data in single dataframe vs list of dataframes

        Returns:
            List[pd.Dataframe] or pd.Dataframe: Synthetic dataset/datasets
        """

        print(f"Generating data with {num_data_per_parameter_sample} points and {num_parameter_samples} datasets")

        mnjax = self._markov_network
        posterior_values = self.posterior_values
        inds = jax.random.choice(key=rng, a=posterior_values.shape[0], shape=[num_parameter_samples])
        posterior_sample = posterior_values[inds, :]
        rng, *data_keys = jax.random.split(rng, num_parameter_samples + 1)
        syn_datasets = [mnjax.sample(syn_data_key, jnp.array(posterior_value), num_data_per_parameter_sample) for
                        syn_data_key, posterior_value
                        in list(zip(data_keys, posterior_sample))]

        dataframes = [DataFrameData.apply_category_mapping(syn_data, self._category_mapping) for syn_data
                      in syn_datasets]

        if single_dataframe is True:
            combined_dataframe = pd.concat(dataframes, ignore_index=True)
            return combined_dataframe
        else:
            return dataframes


class NapsuMQResultIO:
    IDENTIFIER = "NapsuMQ".encode("utf8")
    CURRENT_IO_VERSION = 1
    # Replace with twinify.serialization.ENDIANESS with merge
    CURRENT_IO_VERSION_BYTES = CURRENT_IO_VERSION.to_bytes(1, 'big')

    @staticmethod
    def load_from_io(read_io: BinaryIO) -> NapsuMQResult:
        assert read_io.readable()

        if not NapsuMQResultIO.is_file_stored_result_from_io(read_io, reset_cursor=False):
            raise InvalidFileFormatException(NapsuMQResult, "Stored data does not have correct type identifier.")

        current_version = int.from_bytes(read_io.read(1), 'big')
        if current_version != NapsuMQResultIO.CURRENT_IO_VERSION:
            raise InvalidFileFormatException(NapsuMQResult, "Stored data uses an unknown storage format version.")

        result_binary = read_io.read()
        result = dill.loads(result_binary)

        return result

    @staticmethod
    def is_file_stored_result_from_io(read_io: BinaryIO, reset_cursor: bool) -> bool:
        assert read_io.readable()
        assert read_io.seekable()

        identifier = read_io.read(len(NapsuMQResultIO.IDENTIFIER))
        if reset_cursor:
            read_io.seek(-len(identifier), os.SEEK_CUR)

        if identifier == NapsuMQResultIO.IDENTIFIER:
            return True

        return False

    @staticmethod
    def store_to_io(write_io: BinaryIO, result) -> None:
        assert write_io.writable()

        write_io.write(NapsuMQResultIO.IDENTIFIER)
        write_io.write(NapsuMQResultIO.CURRENT_IO_VERSION_BYTES)
        write_io.write(result)
        write_io.close()
