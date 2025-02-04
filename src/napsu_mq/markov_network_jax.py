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

import functools
from typing import Optional, List, Dict, Iterable
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .log_factor import LogFactorJax
from .marginal_query import FullMarginalQuerySet
from .markov_network import MarkovNetwork


class MarkovNetworkJax(MarkovNetwork):
    """Jax implementation of MarkovNetwork."""

    def __init__(self, domain: Dict, queries: FullMarginalQuerySet, elimination_order: Optional[Iterable] = None,
                 debug_checks: Optional[bool] = True):
        super().__init__(domain, queries, elimination_order, debug_checks)
        self.suff_stat_mean = jax.jit(jax.grad(self.lambda0))
        self.suff_stat_cov = jax.jit(jax.hessian(self.lambda0))
        self.suff_stat_mean_bp = jax.jit(self.suff_stat_mean_bp)
        self.suff_stat_cov_bp = jax.jit(jax.jacrev(self.suff_stat_mean_bp))
        self.log_factor_class = LogFactorJax

    @functools.partial(jax.jit, static_argnums=(0,))
    def lambda0(self, lambdas: jnp.ndarray) -> jnp.ndarray:
        factors = self.compute_factors(lambdas)
        result_factor = self.variable_elimination(factors, self.elimination_order)
        return result_factor.values

    def suff_stat_mean_bp(self, lambdas: jnp.ndarray) -> jnp.ndarray:
        factors = self.compute_factors(lambdas)
        result_factors = self.belief_propagation(factors)
        result: jnp.ndarray = jnp.zeros(self.suff_stat_d)
        for clique, indices in self.variable_associations.items():
            node_variables = self.junction_tree.node_for_factor[clique]
            factor = result_factors[node_variables]
            for variable in set(node_variables).difference(clique):
                factor = factor.marginalise(variable)
            result = result.at[jnp.array(indices)].set(factor.query(self.queries.queries[clique]))
        return result

    def sample(self, rng: jax.random.PRNGKey, lambdas: jnp.ndarray, n_sample: Optional[int] = 1) -> pd.DataFrame:
        n_cols = len(self.domain.keys())
        cols = self.domain.keys()
        data = np.zeros((n_sample, n_cols), dtype=np.int64)
        df = pd.DataFrame(data, columns=cols, dtype=int)

        order = self.elimination_order[::-1]
        batch_factors = [factor.add_batch_dim(n_sample) for factor in self.compute_factors(lambdas)]
        for variable in order:
            marginal = self.marginal_distribution_logits(batch_factors, [variable])
            rng, key = jax.random.split(rng)
            values = jax.random.categorical(key, marginal)
            batch_factors = [factor.batch_condition(variable, values) if variable in factor.scope else factor for factor
                             in batch_factors]
            # Change to use non-depricated function
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message=(
                        ".*will attempt to set the values inplace instead of always setting a new array. "
                        "To retain the old behavior, use either.*"
                    ),
                )
                df.loc[:, variable] = np.array(values)

        return df

    def log_factor_vector(self, lambdas: jnp.ndarray, variables: Iterable) -> jnp.ndarray:
        vec = jnp.zeros(tuple(len(self.domain[var]) for var in variables))
        for query_ind in self.variable_associations[variables]:
            query_val = jnp.array(self.flat_queries.queries[query_ind].value, dtype=int)
            vec = vec.at[tuple(query_val)].set(lambdas[query_ind])
        return vec

    def log_factor_vector_vmapped(self, lambdas: jnp.ndarray, variables: Iterable) -> jnp.ndarray:
        shape = tuple(len(self.domain[var]) for var in variables)
        vec = jnp.zeros(shape)
        indices = jnp.array(np.ndindex(shape))

        # get the query indices and values for the given variables
        query_indices = self.variable_associations[variables]
        query_values = jnp.array([self.flat_queries.queries[ind].value for ind in query_indices], dtype=int)

        # update the empty array with the corresponding lambda values
        vec = vec.at[indices[query_values]].set(lambdas[query_indices])

        return vec

    def compute_factors(self, lambdas: jnp.ndarray) -> List[LogFactorJax]:
        return [
                   LogFactorJax(factor_scope, self.log_factor_vector(lambdas, factor_scope), self.debug_checks)
                   for factor_scope in self.variable_associations.keys()
               ] + [
                   LogFactorJax((variable,), jnp.zeros(len(self.domain[variable])), self.debug_checks)
                   for variable in self.variables_not_in_queries
               ]

    def compute_factors_vmapped(self, lambdas: jnp.ndarray) -> List[LogFactorJax]:
        factor_scopes = list(self.variable_associations.keys())

        log_factors_batch = jax.vmap(self.log_factor_vector)
        log_factor_vectors = [log_factors_batch(lambdas, factor_scope) for factor_scope in factor_scopes]

        first_list = [LogFactorJax(scope, log_factor, self.debug_checks) for scope, log_factor in zip(factor_scopes, log_factor_vectors)]
        second_list = [LogFactorJax((variable,), jnp.zeros(len(self.domain[variable])), self.debug_checks)
                       for variable in self.variables_not_in_queries]

        return first_list + second_list
