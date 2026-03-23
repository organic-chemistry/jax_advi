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
from .guide import CompositeGuide
from jax_advi.guide import VariationalGuide,MeanFieldGuide,LaplaceApproxGuide
from typing import Protocol, NamedTuple
import jax
from collections import namedtuple
#@jax.jit()
#@partial(jax.jit, static_argnames=('verbose'))



#@partial(jax.jit, static_argnames=('verbose','log_lik_fun', 'log_prior_fun',"guide","constrain_fun_dict","theta_shape_dict","M"))
def _build_objective_fun(theta_shape_dict, constrain_fun_dict, log_lik_fun,
                         log_prior_fun, seed, M, guide: VariationalGuide, verbose=False):

    theta = {k: jnp.empty(v) for k, v in theta_shape_dict.items()}
    flat_theta, summary, indices = flatten_and_summarise(**theta)


    # CompositeGuide needs summary before init_params
    if isinstance(guide, CompositeGuide):
        guide.summary = summary
        if not guide.param_mapping:
            assert set(guide.group_guides.keys()) == set(theta_shape_dict.keys()), (
                f"CompositeGuide keys {set(guide.group_guides.keys())} "
                f"must match theta_shape_dict keys {set(theta_shape_dict.keys())}"
            )
        else:
            assert set(guide.param_mapping.keys()) == set(theta_shape_dict.keys()), (
                f"CompositeGuide param_mapping keys {set(guide.param_mapping.keys())} "
                f"must match theta_shape_dict keys {set(theta_shape_dict.keys())}"
            )

    var_params = guide.init_params(flat_theta)
    zs = guide.sample_z(jax.random.PRNGKey(seed), M)

    to_minimize = partial(
        _calculate_objective,
        summary=summary,
        constrain_fun_dict=constrain_fun_dict,
        log_lik_fun=log_lik_fun,
        log_prior_fun=log_prior_fun,
        zs=zs,
        guide=guide,
        verbose=verbose,
    )

    return flat_theta, summary, to_minimize, var_params


def _calculate_objective(var_params_flat, summary, constrain_fun_dict,
                         log_lik_fun, log_prior_fun, zs, guide: VariationalGuide, verbose):

    cur_entropy = guide.entropy(var_params_flat)

    def calculate_log_posterior(z):
        # Get theta_dict either from composite or flat path
        if isinstance(guide, CompositeGuide):
            theta_dict = guide.transform_draws_to_dict(z, var_params_flat)
        else:
            flat_theta = guide.transform_draws(z, var_params_flat)
            theta_dict = reconstruct(flat_theta, summary, jnp.reshape)

        # Apply constraints
        theta_dict, log_det = apply_constraints(theta_dict, constrain_fun_dict)

        # Apply reparametrisation if present
        if guide.post_transform is not None:
            log_det = log_det + guide.post_transform.log_det(theta_dict)
            theta_dict = guide.post_transform.forward(theta_dict)

        if verbose:
            print("theta", theta_dict)
            print("like", log_lik_fun(theta_dict))
            print("prior", log_prior_fun(theta_dict))
            print("logdet", log_det)

        return log_lik_fun(theta_dict) + log_prior_fun(theta_dict) + log_det

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
    

    #convert to namptuple in order to try to jit the next functions
    #theta_shape_named = namedtuple('theta_shape', theta_shape_dict.keys())
    #theta_shape = theta_shape_named(*theta_shape_dict.values())
    #constrain_fun_named = namedtuple('constrain_fun', constrain_fun_dict.keys())
    #constrain_fun = constrain_fun_named(*constrain_fun_dict.values())

    #print(theta_shape_named.__name__)
    #aise

    #globals()[theta_shape_named.__name__] = theta_shape_named
    #globals()[constrain_fun_named.__name__] = constrain_fun_named

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

    if isinstance(guide, CompositeGuide):
        new_group_guides = {}
        new_free_params = []
        delta_elbo = 0.0

        for name, sub_guide in guide.group_guides.items():
            vp_s, vp_e = guide._var_param_slices[name]
            sub_params = result.x[vp_s:vp_e]

            if isinstance(sub_guide, LaplaceApproxGuide):
                sub_hessian = hessian[vp_s:vp_e, vp_s:vp_e]
                new_sub_guide, new_sub_params, d_elbo = sub_guide.to_meanfield(sub_params, sub_hessian)
                new_group_guides[name] = new_sub_guide
                new_free_params.append(new_sub_params)
                delta_elbo += d_elbo
            else:
                new_group_guides[name] = sub_guide
                new_free_params.append(sub_params)

        if delta_elbo != 0.0:
            free_params = np.concatenate(new_free_params)
            elbo += delta_elbo
            guide.group_guides = new_group_guides
            guide.init_params(flat_theta)  # recompute slices

    elif isinstance(guide, LaplaceApproxGuide):
        guide, free_params, delta_elbo = guide.to_meanfield(free_params, hessian)
        elbo += delta_elbo




    #To be able to drow from the guide:
    guide.final_var_params_flat = free_params
    if not isinstance(guide, CompositeGuide):
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
