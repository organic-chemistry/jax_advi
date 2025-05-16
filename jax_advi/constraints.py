import jax.numpy as jnp
import jax



def apply_constraints(theta_dict, constraint_dict,verbose=False):
    # theta_dict is a dictionary of "var_name -> unconstrained_value"
    # constraint_dict is a dictionary of "var_name -> constrain_fun", where the
    # constrain_fun has to take in the unconstrained value and return a tuple of
    # the constrained value and the log determinant of the Jacobian of the
    # transformation (see constrain_exp for an example).

    new_dict = {x: y for x, y in theta_dict.items()}

    log_det = 0.0

    for cur_var_name, cur_constrain_fun in constraint_dict.items():
        
            
        new_dict[cur_var_name], cur_log_det = cur_constrain_fun(
            theta_dict[cur_var_name]
        )
        if verbose:
            print(cur_var_name,cur_constrain_fun,cur_log_det)
        log_det = log_det + cur_log_det

    return new_dict, log_det


def constrain_exp(theta):
    # Computes the value and log determinant of the transformation y =
    # exp(theta). theta can be any shape.
    # TODO: Make docstring better.

    value = jnp.exp(theta)
    log_det = jnp.sum(theta)

    return value, log_det


def get_constraint_name(constraint_func):
    """Helper function to extract the transformation name from the constraint function."""
    if constraint_func is None:
        return ""
    name = getattr(constraint_func, '__name__', str(constraint_func))
    if name.startswith('constrain_'):
        return name[len('constrain_'):]
    return name

constrain_positive = constrain_exp


def make_constraint(transform_fn):
    """Creates a constraint function that uses JVP to compute value and log-det in one pass.
    
    Args:
        transform_fn: Element-wise transformation (e.g., `jnp.exp`, `jax.nn.sigmoid`).
        
    Returns:
        constraint_fn: Function returning `(transformed_value, log_det_jacobian)`.
    """
    def constraint_fn(theta):
        # Compute transformed value and Jacobian diagonal (via JVP)
        value, derivs = jax.jvp(
            transform_fn,  # The transformation function
            (theta,),      # Input primal values
            (jnp.ones_like(theta),)  # Tangent vector (ones to capture diagonal Jacobian)
        )
        # Log determinant: sum(log|derivatives|)
        log_det = jnp.sum(jnp.log(jnp.abs(derivs)))
        
        return value, log_det
    
    return constraint_fn


def constrain_sigmoid_stable(theta,v=1):
    value = v*jax.nn.sigmoid(theta)
    log_det_terms = v*jax.nn.log_sigmoid(theta) + v*jax.nn.log_sigmoid(-theta)
    log_det = jnp.sum(log_det_terms)
    return value, log_det

def constrain_range_stable(theta,low,high):
    delta = high-low
    value = low + delta*jax.nn.sigmoid(theta)
    log_det_terms = delta*jax.nn.log_sigmoid(theta) + delta*jax.nn.log_sigmoid(-theta)
    log_det = jnp.sum(log_det_terms)
    return value, log_det