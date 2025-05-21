import numpy as np
import jax.numpy as jnp
from functools import partial
from collections import namedtuple, OrderedDict


def extract_info(array):

    return array.shape


def flatten_and_summarise(**input_arrays):
    input_arrays = OrderedDict(input_arrays)
    summaries = OrderedDict()
    indices = OrderedDict()
    current = 0
    flattened_parts = []
    
    for name, array in input_arrays.items():
        flat_part = array.reshape(-1)
        n_elements = flat_part.size
        summaries[name] = extract_info(array) 
        indices[name] = (current, current + n_elements)
        flattened_parts.append(flat_part)
        current += n_elements
    
    flattened = jnp.concatenate(flattened_parts)
    return flattened, summaries, indices


def apply_constraints_on_flat(flat_array, constraint_dict, summaries, indices):
    new_flat = flat_array.copy()
    log_det = 0.0
    
    for var_name, constrain_fun in constraint_dict.items():
        if var_name not in indices:
            continue  # Handle missing constraints if necessary
        start, end = indices[var_name]
        shape = summaries[var_name]
        
        # Extract, reshape, apply constraint
        var_slice = new_flat[start:end]
        var_array = var_slice.reshape(shape)
        constrained_array, cur_log_det = constrain_fun(var_array)
        
        # Update flat array and log determinant
        new_flat[start:end] = constrained_array.reshape(-1)
        log_det += cur_log_det
    
    return new_flat, log_det


def reconstruct(flat_array, summaries,fn_reshape):
    params = OrderedDict()
    current = 0
    
    for name, shape in summaries.items():
        n_elements = np.prod(shape).astype(int)
        params[name] = fn_reshape(flat_array[current:current + n_elements],shape)
        current += n_elements
    
    return params
