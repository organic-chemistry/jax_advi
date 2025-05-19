import numpy as np
import jax.numpy as jnp
from .utils.flattening import flatten_and_summarise, reconstruct
from jax import vmap, jit, value_and_grad
from .constraints import apply_constraints
from .utils.misc import convert_decorator
from functools import partial
from scipy.optimize import minimize
from typing import Tuple, Dict, Callable, Any,List
from .optimization import optimize_with_jac, optimize_with_hvp

from jax_advi.guide import VariationalGuide,MeanFieldGuide
from typing import Protocol, NamedTuple
import jax

#@jax.jit()
#@partial(jax.jit, static_argnames=('verbose','constrain_fun_dict'))
def _calculate_log_posterior(
    flat_theta, log_lik_fun, log_prior_fun, constrain_fun_dict, summary,verbose=False
):

    cur_theta = reconstruct(flat_theta, summary, jnp.reshape)
    #if verbose:
    #    print(cur_theta)

    # Compute the log determinant of the constraints
    cur_theta, cur_log_det = apply_constraints(cur_theta, constrain_fun_dict)
    if verbose:
        print(cur_theta)
    cur_likelihood = log_lik_fun(cur_theta)
    cur_prior = log_prior_fun(cur_theta)
    if verbose:
        print("like",cur_likelihood)
        print("prior",cur_prior)
        print("logdet",cur_log_det)

    return cur_likelihood + cur_prior + cur_log_det



#@partial(jax.jit, static_argnames=('verbose','constrain_fun_dict','log_lik_fun', 'log_prior_fun',"guide"))
def _build_objective_fun(theta_shape_dict, constrain_fun_dict, log_lik_fun, 
                        log_prior_fun, seed, M, guide: VariationalGuide,verbose=False):
    
    theta = {k: jnp.empty(v) for k, v in theta_shape_dict.items()}
    flat_theta, summary,indices = flatten_and_summarise(**theta)
    var_params = guide.init_params(flat_theta)

    
    # Generate noise draws
    zs = jax.random.normal(jax.random.PRNGKey(seed), (M, guide.z_dim()))
    
    # Create objective
    to_minimize = partial(
        _calculate_objective,
        summary=summary,
        constrain_fun_dict=constrain_fun_dict,
        log_lik_fun=log_lik_fun,
        log_prior_fun=log_prior_fun,
        zs=zs,
        guide=guide,
        verbose=verbose
    )
    
    return flat_theta, summary, to_minimize,var_params

#@partial(jax.jit, static_argnames=('verbose','constrain_fun_dict'))
def _calculate_objective(var_params_flat, summary, constrain_fun_dict, 
                        log_lik_fun, log_prior_fun, zs, guide: VariationalGuide,verbose):
    
    cur_entropy = guide.entropy(var_params_flat)
    
    def calculate_log_posterior(z):
        flat_theta = guide.transform_draws(z, var_params_flat)
        return _calculate_log_posterior(
            flat_theta, log_lik_fun, log_prior_fun, constrain_fun_dict, summary
        )
    def calculate_log_posteriorv(z):
        flat_theta = guide.transform_draws(z, var_params_flat)
        return _calculate_log_posterior(
            flat_theta, log_lik_fun, log_prior_fun, constrain_fun_dict, summary,verbose=True
        )
    if verbose:
        print("Post",jnp.mean(jax.vmap(calculate_log_posterior)(zs)))
        print("Entropy",cur_entropy)
        print(jax.vmap(calculate_log_posterior)(zs))
        print(jax.vmap(calculate_log_posteriorv)(zs))
        print(jax.vmap(calculate_log_posteriorv)(jnp.zeros_like(zs)))

        print(zs)
    return -jnp.mean(jax.vmap(calculate_log_posterior)(zs)) - cur_entropy

# ======================== Modified ADVI Core ========================
def optimize_advi(
    theta_shape_dict: Dict[str, Tuple],
    log_prior_fun: Callable[[Dict[str, jax.Array]], float],
    log_lik_fun: Callable[[Dict[str, jax.Array]], float],
    guide: VariationalGuide = MeanFieldGuide(),
    M: int = 100,
    constrain_fun_dict: Dict = {},
    verbose: bool = False,
    seed: int = 2,
    n_draws: int = 1000,
    opt_method: str = "trust-ncg",
    minimize_kwargs = {}
) -> Dict[str, Any]:
    
    # Build objective function
    flat_theta, summary, to_minimize, var_params = _build_objective_fun(
        theta_shape_dict, constrain_fun_dict, log_lik_fun, log_prior_fun, 
        seed, M, guide
    )
    

    # Initialize variational parameters
    # Optimization
    #with_grad = partial(convert_decorator, verbose=verbose)(jax.value_and_grad(to_minimize))
    
    if opt_method == "L-BFGS-B":
        result = optimize_with_jac(to_minimize, var_params.reshape(-1), opt_method, verbose,minimize_kwargs=minimize_kwargs)
    else:
        result = optimize_with_hvp(to_minimize, var_params.reshape(-1), opt_method, verbose,minimize_kwargs=minimize_kwargs)[0]


    grads = jax.grad(to_minimize)(result.x)
    hessian = jax.hessian(to_minimize)(result.x)
    elbo = -result.fun
    free_params = result.x

    if guide.name == "LaplaceApproxGuide":
        d=np.diag(np.abs(hessian))
        log_sigma = -np.log(d)/2
        free_params = np.concatenate([free_params,log_sigma])
        guide = MeanFieldGuide()
        guide.init_params(flat_theta)
        guide.name = "LaplaceApproxGuide"
        delta_elbo = guide.inv_norm(log_sigma)
        elbo += delta_elbo




    #To be able to drow from the guide:
    guide.final_var_params_flat = free_params
    guide.summary = summary
    guide.constrain_fun_dict = constrain_fun_dict
    #print(to_minimize(result.x))



    return  {
        "free_params": result.x,
        "elbo" : elbo,
        "opt_result": result,
        "grads": grads,
        "hessian" : hessian,
        "guide":guide,
        "draws": guide.posterior_draw_and_transform(n_draws)
    }



def get_pickleable_subset(fit_results):

    # Everything except the objective function should be OK
    return {x: y for x, y in fit_results.items() if x != "objective_fun"}
