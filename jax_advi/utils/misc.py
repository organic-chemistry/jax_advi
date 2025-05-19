import numpy as np
from functools import wraps
import jax
from functools import partial

#@partial(jax.jit, static_argnames=('verbose',))
def convert_decorator(fun, verbose=True):
    # A decorator that makes sure we return float64 dtypes, and optionally
    # prints the evaluation of the function.
    def result(x):

        value, grad = fun(x)
        #raise
        if verbose:
            print(value, np.linalg.norm(grad))

        return (
            np.array(value).astype(np.float64),
            np.array(grad).astype(np.float64),
        )

    return result
"""
def convert_decorator(fun, verbose=True):
    # Ensure JAX-compatible operations (no NumPy)
    @jax.jit
    def jax_wrapped(x):
        value, grad = fun(x)
        return value, grad

    def result(x):
        value, grad = jax_wrapped(x)
        if verbose:
            jax.debug.print("Value: {value}, Grad Norm: {grad_norm}", 
                            value=value, grad_norm=jax.numpy.linalg.norm(grad))
        return value, grad  # Return JAX arrays directly

    return result
"""
def print_decorator(fun, verbose=True):
    def result(x):

        value, grad = fun(x)

        if verbose:
            print(f"'f': {value}, ||grad(f)||: {np.linalg.norm(grad)}", flush=True)

        return value, grad

    return result


def count_decorator(function):
    # If wrapped around a function, the number of calls of the function can be
    # accessed by calling function.calls on the decorated result.
    @wraps(function)
    def new_fun(*args, **kwargs):
        new_fun.calls += 1
        return function(*args, **kwargs)

    new_fun.calls = 0
    return new_fun
