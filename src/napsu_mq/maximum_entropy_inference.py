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
from numpyro.diagnostics import summary
from jax import random
import numpy as np
from . import maximum_entropy_model as mem
from .markov_network_jax import MarkovNetworkJax
from src.utils.experiment_storage import ExperimentStorage
from scripts.run_napsu import experiment_id_ctx

storage = ExperimentStorage()

kernels = Literal['NUTS', 'HMC',]


def get_kernel(MCMC_algo: kernels):
    if MCMC_algo == 'NUTS':
        return numpyro.infer.NUTS
    elif MCMC_algo == 'HMC':
        return numpyro.infer.HMC
    else:
        raise ValueError(f"{MCMC_algo} not recognized as MCMC inference algorithm")


def get_stats_from_variable(trace, n_chains=4):
    n = trace.shape[0]
    samples_per_chain = n // n_chains
    stats = np.zeros(shape=(4, n_chains))
    for i in range(0, n_chains):
        slice_min = samples_per_chain * i
        slice_max = samples_per_chain * (i + 1)

        min = jnp.min(trace[slice_min:slice_max])
        if jnp.isscalar(min):
            min = round(min, 6)
        max = jnp.max(trace[slice_min:slice_max])
        if jnp.isscalar(max):
            max = round(max, 6)
        mean = jnp.mean(trace[slice_min:slice_max])
        if jnp.isscalar(mean):
            mean = round(mean, 6)
        std = jnp.std(trace[slice_min:slice_max])
        if jnp.isscalar(std):
            std = round(std, 6)

        stats[, :0] = np.array([min, max, mean, std])

    return stats


def print_MCMC_diagnostics(mcmc):
    mcmc.print_summary()
    summary_dict = summary(mcmc.get_samples(), group_by_chain=True)

    for key, value in summary_dict.items():
        print(f"[{key}]\t max r_hat: {jnp.max(value['r_hat']):.4f}")

    poten = mcmc.get_extra_fields()["potential_energy"]
    accept_prob = mcmc.get_extra_fields()["accept_prob"]
    mean_accept_prob = mcmc.get_extra_fields()["mean_accept_prob"]
    diverging = mcmc.get_extra_fields()["diverging"]

    min, max, mean, std = get_stats_from_variable(poten)
    print(f"Potential energy\tmin: {min}\tmax: {max}\tmean: {mean}\tstd: {std}")
    min, max, mean, std = get_stats_from_variable(accept_prob)
    print(f"Acceptance probability\tmin: {min}\tmax: {max}\tmean: {mean}\tstd: {std}")
    min, max, mean, std = get_stats_from_variable(mean_accept_prob)
    print(f"Mean acceptance probability\tmin: {min}\tmax: {max}\tmean: {mean}\tstd: {std}")
    min, max, mean, std = get_stats_from_variable(diverging)
    print(f"Diverging\tmin: {min}\tmax: {max}\tmean: {mean}\tstd: {std}")


def store_MCMC_diagnostics(mcmc):
    summary_dict = summary(mcmc.get_samples(), group_by_chain=True)
    MCMC_diagnostics = {**summary_dict}
    experiment_id = experiment_id_ctx.get()
    storage.store(experiment_id, MCMC_diagnostics)


def run_numpyro_mcmc(
        rng: random.PRNGKey, suff_stat: jnp.ndarray, n: int, sigma_DP: float, max_ent_dist: MarkovNetworkJax,
        MCMC_algo: kernels = "NUTS", prior_mu: Union[float, jnp.ndarray] = 0, prior_sigma: float = 10,
        num_samples: int = 1000,
        num_warmup: int = 500, num_chains: int = 4, disable_progressbar: bool = True,
) -> numpyro.infer.MCMC:
    MCMC_algorithm = get_kernel(MCMC_algo)

    kernel = MCMC_algorithm(model=mem.normal_prior_model_numpyro, max_tree_depth=12)
    mcmc = numpyro.infer.MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
        progress_bar=not disable_progressbar, jit_model_args=False, chain_method="sequential",
    )
    mcmc.run(rng, suff_stat, n, sigma_DP, prior_mu, prior_sigma, max_ent_dist,
             extra_fields=("potential_energy", "accept_prob", "mean_accept_prob", "diverging"))

    print_MCMC_diagnostics(mcmc)
    store_MCMC_diagnostics(mcmc)
    return mcmc


def run_numpyro_mcmc_normalised(
        rng: random.PRNGKey, suff_stat: jnp.ndarray, n: int, sigma_DP: float, max_ent_dist: MarkovNetworkJax,
        laplace_approx: numpyro.distributions.MultivariateNormal, MCMC_algo: kernels = "NUTS", prior_sigma: float = 10,
        num_samples: int = 1000, num_warmup: int = 500, num_chains: int = 4, disable_progressbar: bool = True,
) -> Tuple[numpyro.infer.MCMC, Callable]:
    MCMC_algorithm = get_kernel(MCMC_algo)

    kernel = MCMC_algorithm(model=mem.normal_prior_normalised_model_numpyro, max_tree_depth=12)
    mcmc = numpyro.infer.MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
        progress_bar=not disable_progressbar, jit_model_args=False, chain_method="sequential"
    )

    mean_guess = laplace_approx.mean
    L_guess = jnp.linalg.cholesky(laplace_approx.covariance_matrix)
    mcmc.run(rng, suff_stat, n, sigma_DP, prior_sigma, max_ent_dist, mean_guess, L_guess,
             extra_fields=("potential_energy", "potential_energy", "accept_prob", "mean_accept_prob", "diverging"))

    print_MCMC_diagnostics(mcmc)
    store_MCMC_diagnostics(mcmc)

    def backtransform(lambdas: jnp.ndarray) -> jnp.ndarray:
        return (L_guess @ lambdas.transpose()).transpose() + mean_guess

    return mcmc, backtransform


class ConvergenceException(Exception):
    """Convergence error in optimization process"""


def run_numpyro_laplace_approximation(
        rng: random.PRNGKey, suff_stat: jnp.ndarray, n: int, sigma_DP: float, max_ent_dist: MarkovNetworkJax,
        prior_mu: Union[float, jnp.ndarray] = 0, prior_sigma: float = 10, max_retries=5
) -> Tuple[numpyro.distributions.MultivariateNormal, bool]:
    key, *subkeys = random.split(rng, max_retries + 1)
    fail_count = 0

    for i in range(0, max_retries + 1):

        rng = subkeys[i]

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
