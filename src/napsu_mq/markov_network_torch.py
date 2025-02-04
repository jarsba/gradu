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
from typing import Optional, Dict, List

import functorch
import pandas as pd
import torch

from .log_factor_torch import LogFactorTorch
from .marginal_query import FullMarginalQuerySet
from .markov_network import MarkovNetwork


class MarkovNetworkTorch(MarkovNetwork):
    """PyTorch implementation of MarkovNetwork.
    """
    def __init__(self, domain: Dict, queries: FullMarginalQuerySet, elimination_order: Optional[List] = None, debug_checks: Optional[bool] = True):
        super().__init__(domain, queries, elimination_order, debug_checks)
        self.suff_stat_mean = torch.func.grad(self.lambda0)
        self.suff_stat_cov = torch.func.jacrev(torch.func.jacrev(self.lambda0))
        self.suff_stat_cov_bp = torch.func.jacrev(self.suff_stat_mean_bp)
        self.log_factor_class = LogFactorTorch

    def lambda0(self, lambdas: torch.Tensor) -> torch.Tensor:
        factors = self.compute_factors(lambdas)
        result_factor = self.variable_elimination(factors, self.elimination_order)
        return result_factor.values

    def suff_stat_mean_bp(self, lambdas: torch.Tensor) -> torch.Tensor:
        factors = self.compute_factors(lambdas)
        result_factors = self.belief_propagation(factors)
        result = torch.zeros(self.suff_stat_d)
        for clique, indices in self.variable_associations.items():
            node_variables = self.junction_tree.node_for_factor[clique]
            factor = result_factors[node_variables]
            for variable in set(node_variables).difference(clique):
                factor = factor.marginalise(variable)
            result[torch.tensor(indices)] = factor.query(self.queries.queries[clique])
        return result

    def sample(self, lambdas: torch.Tensor, n_sample: Optional[int] = 1) -> pd.DataFrame:
        n_cols = len(self.domain.keys())
        cols = self.domain.keys()
        data = torch.zeros((n_sample, n_cols), dtype=torch.long)
        df = pd.DataFrame(data, columns=cols, dtype=int)

        order = self.elimination_order[::-1]
        batch_factors = [factor.add_batch_dim(n_sample) for factor in self.compute_factors(lambdas)]
        for variable in order:
            marginal = self.marginal_distribution_logits(batch_factors, [variable])
            values = torch.distributions.Categorical(logits=marginal).sample((1,)).flatten()
            batch_factors = [factor.batch_condition(variable, values) if variable in factor.scope else factor for factor in batch_factors]
            df.loc[:, variable] = values.numpy()

        return df

    def log_factor_vector(self, lambdas: torch.Tensor, variables: List) -> torch.Tensor:
        vec = torch.zeros(tuple(len(self.domain[var]) for var in variables))
        for query_ind in self.variable_associations[variables]:
            query_val = self.flat_queries.queries[query_ind].value
            vec[tuple(query_val)] = lambdas[query_ind]
        return vec

    def compute_factors(self, lambdas: torch.Tensor) -> List['LogFactorTorch']:
        return [
            LogFactorTorch(factor_scope, self.log_factor_vector(lambdas, factor_scope), self.debug_checks)
            for factor_scope in self.variable_associations.keys()
        ] + [
            LogFactorTorch((variable,), torch.zeros(len(self.domain[variable])), self.debug_checks)
            for variable in self.variables_not_in_queries
        ]