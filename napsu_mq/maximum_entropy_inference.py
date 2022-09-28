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
from typing import Union, Callable, Tuple, Literal

import jax
import jax.numpy as jnp
import numpyro
import numpyro.infer.util as nummcmc_util
from jax import random

import napsu_mq.maximum_entropy_model as mem
from napsu_mq.markov_network_jax import MarkovNetworkJax

kernels = Literal['NUTS', 'HMC',]

def get_kernel(MCMC_algo: kernels):
    if MCMC_algo == 'NUTS':
        return numpyro.infer.NUTS
    elif MCMC_algo == 'HMC':
        return numpyro.infer.HMC
    else:
        raise ValueError(f"{MCMC_algo} not recognized as MCMC inference algorithm")



def run_numpyro_mcmc(
        rng: random.PRNGKey, suff_stat: jnp.ndarray, n: int, sigma_DP: float, max_ent_dist: MarkovNetworkJax,
        MCMC_algo: kernels = "NUTS", prior_mu: Union[float, jnp.ndarray] = 0, prior_sigma: float = 10, num_samples: int = 1000,
        num_warmup: int = 500, num_chains: int = 1, disable_progressbar: bool = False,
) -> numpyro.infer.MCMC:

    MCMC_algorithm = get_kernel(MCMC_algo)

    kernel = MCMC_algorithm(model=mem.normal_prior_model_numpyro, max_tree_depth=12)
    mcmc = numpyro.infer.MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
        progress_bar=not disable_progressbar, jit_model_args=False, chain_method="sequential"
    )
    mcmc.run(rng, suff_stat, n, sigma_DP, prior_mu, prior_sigma, max_ent_dist)
    return mcmc


def run_numpyro_mcmc_normalised(
        rng: random.PRNGKey, suff_stat: jnp.ndarray, n: int, sigma_DP: float, max_ent_dist: MarkovNetworkJax,
        laplace_approx: numpyro.distributions.MultivariateNormal, MCMC_algo: kernels = "NUTS", prior_sigma: float = 10,
        num_samples: int = 1000, num_warmup: int = 500, num_chains: int = 1, disable_progressbar: bool = False,
) -> Tuple[numpyro.infer.MCMC, Callable]:

    MCMC_algorithm = get_kernel(MCMC_algo)

    kernel = MCMC_algorithm(model=mem.normal_prior_normalised_model_numpyro, max_tree_depth=12)
    mcmc = numpyro.infer.MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
        progress_bar=not disable_progressbar, jit_model_args=False, chain_method="sequential"
    )

    mean_guess = laplace_approx.mean
    L_guess = jnp.linalg.cholesky(laplace_approx.covariance_matrix)
    mcmc.run(rng, suff_stat, n, sigma_DP, prior_sigma, max_ent_dist, mean_guess, L_guess)

    def backtransform(lambdas: jnp.ndarray) -> jnp.ndarray:
        return (L_guess @ lambdas.transpose()).transpose() + mean_guess

    return mcmc, backtransform


class ConvergenceException(Exception):
    """Convergence error in optimization process"""


def run_numpyro_laplace_approximation(
        rng: random.PRNGKey, suff_stat: jnp.ndarray, n: int, sigma_DP: float, max_ent_dist: MarkovNetworkJax,
        prior_mu: Union[float, jnp.ndarray] = 0, prior_sigma: float = 10, max_retries=5
) -> Tuple[numpyro.distributions.MultivariateNormal, bool]:

    fail_count = 0

    for i in range(0, max_retries + 1):

        init_lambdas, potential_fn, t, mt = nummcmc_util.initialize_model(
            rng, mem.normal_prior_model_numpyro,
            model_args=(suff_stat, n, sigma_DP, prior_mu, prior_sigma, max_ent_dist)
        )

        lambdas = init_lambdas[0]["lambdas"]

        result = jax.scipy.optimize.minimize(lambda l: potential_fn({"lambdas": l}), lambdas, method="BFGS", tol=1e-2)
        if not result.success:
            fail_count += 1
        else:
            mean = result.x
            break

        if fail_count == max_retries:
            raise ConvergenceException(f"Minimize function failed to converge with {max_retries} retries")

    prec = jax.hessian(lambda l: potential_fn({"lambdas": l}))(mean)
    laplace_approx = numpyro.distributions.MultivariateNormal(loc=mean, precision_matrix=prec)
    return laplace_approx, result.success
