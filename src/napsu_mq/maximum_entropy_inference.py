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
from typing import Union, Callable, Tuple, Literal, Optional
import cProfile
from pstats import SortKey
import pstats
import io
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
import jaxopt
import numpyro
import numpyro.infer.util as nummcmc_util
from numpyro.diagnostics import summary
from jax import random
import numpy as np

import pyro.infer.mcmc.util as mcmc_util
import torch
import torch.autograd as autograd
import torch.optim as optim
from pyro.infer import MCMC, NUTS

from . import maximum_entropy_model as mem
from .markov_network_jax import MarkovNetworkJax
from src.utils.experiment_storage import ExperimentStorage, experiment_id_ctx

storage = ExperimentStorage()
kernels = Literal['NUTS', 'HMC']


def rng_state_set(generator: Optional[torch.Generator] = None) -> Optional[torch.Tensor]:
    if generator is not None:
        old_rng_state = torch.get_rng_state()
        torch.set_rng_state(generator.get_state())
        return old_rng_state
    return None


def rng_state_restore(old_rng_state: torch.Tensor) -> None:
    if old_rng_state is not None:
        torch.set_rng_state(old_rng_state)


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

        stats[:, 0] = np.array([min, max, mean, std])

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
        enable_profiling: bool = False,
) -> numpyro.infer.MCMC:
    MCMC_algorithm = get_kernel(MCMC_algo)

    kernel = MCMC_algorithm(model=mem.normal_prior_model_numpyro, max_tree_depth=12)
    mcmc = numpyro.infer.MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
        progress_bar=not disable_progressbar, jit_model_args=False, chain_method="parallel",
    )

    if enable_profiling:
        pr = cProfile.Profile()
        pr.enable()
        mcmc.run(rng, suff_stat, n, sigma_DP, prior_mu, prior_sigma, max_ent_dist,
                 extra_fields=("potential_energy", "accept_prob", "mean_accept_prob", "diverging", "num_steps"))
        pr.disable()
        experiment_id = experiment_id_ctx.get()
        pr.dump_stats(f"logs/MCMC_profile_{experiment_id}.prof")

        s = io.StringIO()
        sort_by = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
        ps.dump_stats(f"logs/MCMC_profile_stats_{experiment_id}.prof")
        s.close()
    else:
        mcmc.run(rng, suff_stat, n, sigma_DP, prior_mu, prior_sigma, max_ent_dist,
                 extra_fields=("potential_energy", "accept_prob", "mean_accept_prob", "diverging", "num_steps"))

    print_MCMC_diagnostics(mcmc)
    store_MCMC_diagnostics(mcmc)
    return mcmc


def run_numpyro_mcmc_normalised(
        rng: random.PRNGKey, suff_stat: jnp.ndarray, n: int, sigma_DP: float, max_ent_dist: MarkovNetworkJax,
        laplace_approx: numpyro.distributions.MultivariateNormal, MCMC_algo: kernels = "NUTS", prior_sigma: float = 10,
        num_samples: int = 1000, num_warmup: int = 500, num_chains: int = 4, disable_progressbar: bool = True,
        enable_profiling: bool = False
) -> Tuple[numpyro.infer.MCMC, Callable]:
    MCMC_algorithm = get_kernel(MCMC_algo)

    kernel = MCMC_algorithm(model=mem.normal_prior_normalised_model_numpyro, max_tree_depth=12)
    mcmc = numpyro.infer.MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
        progress_bar=not disable_progressbar, jit_model_args=False, chain_method="parallel"
    )

    if type(laplace_approx) is torch.distributions.MultivariateNormal:
        mean_guess = jnp.array(laplace_approx.mean.detach().numpy())
        L_guess = jnp.linalg.cholesky(jnp.array(laplace_approx.covariance_matrix.detach().numpy()))
    else:
        mean_guess = laplace_approx.mean
        L_guess = jnp.linalg.cholesky(laplace_approx.covariance_matrix)

    if enable_profiling is True:
        pr = cProfile.Profile()
        pr.enable()
        mcmc.run(rng, suff_stat, n, sigma_DP, prior_sigma, max_ent_dist, mean_guess, L_guess,
                 extra_fields=("potential_energy", "accept_prob", "mean_accept_prob", "diverging", "num_steps"))
        pr.disable()
        experiment_id = experiment_id_ctx.get()
        pr.dump_stats(f"logs/MCMC_{experiment_id}.prof")

        s = io.StringIO()
        sort_by = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
        ps.dump_stats(f"logs/MCMC_profile_stats_{experiment_id}.prof")
        s.close()

    else:
        mcmc.run(rng, suff_stat, n, sigma_DP, prior_sigma, max_ent_dist, mean_guess, L_guess,
                 extra_fields=("potential_energy", "accept_prob", "mean_accept_prob", "diverging", "num_steps"))

    print_MCMC_diagnostics(mcmc)
    store_MCMC_diagnostics(mcmc)

    def backtransform(lambdas: jnp.ndarray) -> jnp.ndarray:
        return (L_guess @ lambdas.transpose()).transpose() + mean_guess

    return mcmc, backtransform


class ConvergenceException(Exception):
    """Convergence error in optimization process"""


def run_numpyro_laplace_approximation(
        rng: random.PRNGKey, suff_stat: jnp.ndarray, n: int, sigma_DP: float, max_ent_dist: MarkovNetworkJax,
        prior_mu: Union[float, jnp.ndarray] = 0, prior_sigma: float = 10, max_retries=10, use_forward_mode=False
) -> Tuple[numpyro.distributions.MultivariateNormal, bool]:
    print("Started NumPyro Laplace approximation")
    key, *subkeys = random.split(rng, max_retries + 1)
    fail_count = 0

    for i in range(0, max_retries + 1):

        print(f"Attempting Laplace approximation, {i}th try")

        rng = subkeys[i]

        init_lambdas, potential_fn, t, mt = nummcmc_util.initialize_model(
            rng, mem.normal_prior_model_numpyro,
            model_args=(suff_stat, n, sigma_DP, prior_mu, prior_sigma, max_ent_dist),
            forward_mode_differentiation=use_forward_mode
        )

        print("Initialising model done")

        lambdas = init_lambdas[0]["lambdas"]

        print("Minimising potential function")

        result = jax.scipy.optimize.minimize(lambda l: potential_fn({"lambdas": l}), lambdas, method="BFGS", tol=1e-2)
        print(result)
        if not result.success:
            fail_count += 1
        else:
            mean = result.x
            break

        if fail_count == max_retries:
            raise ConvergenceException(f"Minimize function failed to converge with {max_retries} retries")

    print("Calculating Hessian")

    prec = jax.hessian(lambda l: potential_fn({"lambdas": l}))(mean)
    laplace_approx = numpyro.distributions.MultivariateNormal(loc=mean, precision_matrix=prec)
    return laplace_approx, result.success


def laplace_approximation_with_jaxopt(
        rng: random.PRNGKey, suff_stat: jnp.ndarray, n: int, sigma_DP: float, max_ent_dist: MarkovNetworkJax,
        prior_mu: Union[float, jnp.ndarray] = 0, prior_sigma: float = 10, max_retries=10, use_forward_mode=False) -> \
        Tuple[
            numpyro.distributions.MultivariateNormal, bool]:
    print("Started Jaxopt Laplace approximation")
    key, *subkeys = random.split(rng, max_retries + 1)

    fail_count = 0

    for i in range(0, max_retries + 1):
        print(f"Attempting Laplace approximation, {i}th try")

        rng = subkeys[i]

        start = timer()

        init_lambdas, potential_fn, t, mt = nummcmc_util.initialize_model(
            rng, mem.normal_prior_model_numpyro,
            model_args=(suff_stat, n, sigma_DP, prior_mu, prior_sigma, max_ent_dist),
            forward_mode_differentiation=use_forward_mode
        )

        print(f"Took: {timer() - start} seconds to initialise model")

        print("Initialising model done")

        lambdas = init_lambdas[0]["lambdas"]

        print("Minimising potential function")

        solver = jaxopt.LBFGS(potential_fn, maxiter=1000)
        params, state = solver.run(init_params={'lambdas': lambdas})

        failed = state.failed_linesearch

        if not failed:
            mean = params['lambdas']
            break
        else:
            print("Failed linesearch, try again")

        if fail_count == max_retries:
            raise ConvergenceException(f"Minimize function failed to converge with {max_retries} retries")

    print("Calculating Hessian")
    prec = jax.hessian(lambda l: potential_fn({"lambdas": l}))(mean)
    laplace_approx = numpyro.distributions.MultivariateNormal(loc=mean, precision_matrix=prec)
    return laplace_approx, True


def run_mcmc(
        suff_stat, n, sigma_DP, max_ent_dist,
        prior_mu=0, prior_sigma=10,
        num_samples=2000, warmup_steps=200, num_chains=4,
        disable_progressbar=False, generator=Optional[torch.Generator]
):
    ors = rng_state_set(generator)
    nuts_kernel = NUTS(mem.normal_prior_model, jit_compile=False)

    mcmc = MCMC(
        nuts_kernel, num_samples=num_samples,
        warmup_steps=warmup_steps, num_chains=num_chains,
        mp_context="forkserver", disable_progbar=disable_progressbar
    )
    mcmc.run(suff_stat, n, sigma_DP, prior_mu, prior_sigma, max_ent_dist)
    rng_state_restore(ors)
    return mcmc


def run_mcmc_normalised(
        suff_stat, n, sigma_DP, max_ent_dist, laplace_approx,
        prior_sigma=10, num_samples=2000, warmup_steps=200, num_chains=4,
        disable_progressbar=False, generator=Optional[torch.Generator]
):
    ors = rng_state_set(generator)
    nuts_kernel = NUTS(mem.normal_prior_normalised_model, jit_compile=False)

    mean_guess = laplace_approx.loc
    L_guess = torch.linalg.cholesky(laplace_approx.covariance_matrix)
    mcmc = MCMC(
        nuts_kernel, num_samples=num_samples,
        warmup_steps=warmup_steps, num_chains=num_chains,
        mp_context="spawn", disable_progbar=disable_progressbar
    )
    mcmc.run(suff_stat, n, sigma_DP, prior_sigma, max_ent_dist, mean_guess, L_guess)

    def backtransform(lambdas):
        return (L_guess.detach().numpy() @ lambdas.T).T + mean_guess.detach().numpy()

    rng_state_restore(ors)
    return mcmc, backtransform


def laplace_optimisation_torch(lambdas, potential_fn, max_iters, tol, max_loss_jump):
    print("Running Laplace optimization")
    opt = optim.LBFGS([lambdas])
    losses = []
    for i in range(max_iters):
        def closure():
            opt.zero_grad()
            output = potential_fn({"lambdas": lambdas})
            losses.append(output.item())
            output.backward()
            return output

        opt.step(closure)
        if len(losses) > 1 and abs(losses[-1] - losses[-2]) < tol:
            return True, lambdas, losses
        if len(losses) > 1 and (losses[-1] - losses[-2]) > max_loss_jump:
            return False, lambdas, losses

    return False, lambdas, losses


def laplace_approximation_normal_prior(
        suff_stat, n, sigma_DP, max_ent_dist, prior_mu=0, prior_sigma=10, max_iters=500,
        tol=1e-5, max_loss_jump=1e3, max_retries=10, generator: Optional[torch.Generator] = None
):
    ors = rng_state_set(generator)

    fail_count = 0
    for i in range(max_retries + 1):
        init_lambdas, potential_fn, t, mt = mcmc_util.initialize_model(
            mem.normal_prior_model, (suff_stat, n, sigma_DP, prior_mu, prior_sigma, max_ent_dist)
        )
        lambdas = init_lambdas["lambdas"].clone().requires_grad_(True)
        success, lambdas, losses = laplace_optimisation_torch(lambdas, potential_fn, max_iters, tol, max_loss_jump)
        if success is True:
            hess = autograd.functional.hessian(lambda l: potential_fn({"lambdas": l}), lambdas)
            try:
                laplace_approx = torch.distributions.MultivariateNormal(lambdas, precision_matrix=hess)
            except ValueError:
                success = False
        if success is True:
            break
        else:
            fail_count += 1

    rng_state_restore(ors)

    if success is False:
        raise ConvergenceException(f"Torch Laplace approximation L-BFGS failed to converge with {max_retries} retries")

    return laplace_approx, True
