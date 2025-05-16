from typing import Protocol
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Callable, Any,List
from jax_advi.utils.flattening import reconstruct
from jax_advi.constraints import apply_constraints,get_constraint_name
import numpy as np

        
"""
class NNGaussianGuide:
    #
    #Possible NN imylementation #not tested
    #
    def __init__(self, latent_dim=10, hidden_sizes=(50,50)):
        self.latent_dim = latent_dim
        self.hidden_sizes = hidden_sizes

    def init_params(self, flat_theta: jax.Array) -> jax.Array:
        self.dim_param = flat_theta.shape[0]
        
        # Define network inside init to capture dim_param
        def net_fn(z):
            mlp = hk.Sequential([
                hk.Linear(size) for size in self.hidden_sizes
            ] + [
                hk.Linear(2*self.dim_param)
            ])
            return mlp(z)
            
        self.net = hk.without_apply_rng(hk.transform(net_fn))
        
        # Return both network params and noise params
        return {
            'net': self.net.init(jax.random.PRNGKey(0), 
            'noise': jnp.zeros(2*self.dim_param)
        }

    def make_draws(self, z, var_params):
        outputs = self.net.apply(var_params['net'], z)
        mean, log_sd = jnp.split(outputs + var_params['noise'], 2)
        return mean + jnp.exp(log_sd) * z[:self.dim_param]

    def entropy(self, var_params):
        log_sd = jnp.split(var_params['noise'], 2)[1]
        return 0.5*self.dim_param*(1 + jnp.log(2*jnp.pi)) + jnp.sum(log_sd)


class NNGaussianGuideF(nnx.Module):
    def __init__(self, 
                 latent_dim: int = 10,
                 hidden_sizes: Tuple[int, ...] = (50, 50),
                 init_scale: float = 0.01):
        super().__init__()
        
        # Define network components
        self.latent_dim = latent_dim
        self.hidden_sizes = hidden_sizes
        self.init_scale = init_scale
        self.dim_param = None  # Will be set during initialization
        
        # Define layers
        self.layers = [
            nnx.Linear(size, 
                      kernel_init=nnx.initializers.normal(self.init_scale))
            for size in self.hidden_sizes
        ]
        self.final_layer = nnx.Linear(None)  # Placeholder, set in setup
        
    def setup(self, dim_param: int):
        #Setup method needed for dynamic parameter sizing
        self.dim_param = dim_param
        self.final_layer = nnx.Linear(2 * dim_param,
                                     kernel_init=nnx.initializers.normal(self.init_scale))
        
    def __call__(self, z: jax.Array) -> jax.Array:
        for layer in self.layers:
            z = layer(z)
            z = jax.nn.relu(z)
        return self.final_layer(z)

class NNGaussianFlax:
    def __init__(self, **kwargs):
        self.guide = NNGaussianGuide(**kwargs)
        self.params = None
        self.noise = None
        
    def init_params(self, flat_theta: jax.Array) -> nnx.State:
        self.guide.setup(flat_theta.shape[0])
        state, _ = self.guide.split()
        
        # Initialize noise parameters
        self.noise = {
            'noise': nnx.Param(jnp.zeros(2 * flat_theta.shape[0]))
        }
        
        return nnx.merge(state, self.noise)
    
    def make_draws(self, z: jax.Array, var_params: nnx.State) -> jax.Array:
        # Split parameters
        state, noise = nnx.split(var_params, nnx.Param)
        
        # Merge into module
        _, module = self.guide.merge(state)
        
        # Forward pass
        outputs = module(z) + noise['noise']
        mean, log_sd = jnp.split(outputs, 2)
        return mean + jnp.exp(log_sd) * z[:self.guide.dim_param]
    
    def entropy(self, var_params: nnx.State) -> float:
        _, noise = nnx.split(var_params, nnx.Param)
        log_sd = jnp.split(noise['noise'], 2)[1]
        return 0.5 * self.guide.dim_param * (1 + jnp.log(2*jnp.pi)) + jnp.sum(log_sd)
    
    def z_dim(self) -> int:
        return self.guide.latent_dim
"""

# ======================== Guide Interface ========================
class VariationalGuide(Protocol):
    def init_params(self, flat_theta: jax.Array) -> jax.Array:
        """Initialize variational parameters"""
    def transform_draws(self, z: jax.Array, var_params: jax.Array) -> jax.Array:
        """Transform noise z into parameters using var_params (the pushing trick)"""
    def entropy(self, var_params: jax.Array) -> float:
        """Compute entropy of the guide distribution"""
    def z_dim(self) -> int:
        """Dimension where the reparametrisation trick takes place"""
   
# ======================== Guide Implementations ========================


class Guide(VariationalGuide):
    final_var_params_flat : jax.Array = jnp.array([])
    constrain_fun_dict: Dict = {}
    summary: Dict = {}

    def _posterior_draw_and_transform(self, 
                                       var_params_flat: jax.Array,
                                       n_draws: int = 1000,
                                       key: jax.random.PRNGKey = jax.random.PRNGKey(0)) -> Dict[str, jnp.ndarray]:
        """
        Generate constrained posterior draws from a variational approximation
        """
        # Generate standard normal draws
        zs = jax.random.normal(key, (n_draws, self.z_dim()))

        # Batch-transform through guide
        def transform_and_reconstruct(z,var_params_flat,summary,constrain_fun_dict):
            flat_draw = self.transform_draws(z, var_params_flat)  #, in_axes=(0, None))
            reconstructed_draw = reconstruct(flat_draw, summary, jnp.reshape)
            constrained_theta, _ = apply_constraints(reconstructed_draw,
                                                    constrain_fun_dict
                                                    )
            return constrained_theta
        
        batch_transform_and_reconstruct = jax.vmap(transform_and_reconstruct,
                                            in_axes=(0,None,None,None),
                                            out_axes={k:0 for k in self.summary.keys()})
        #print(summary)
        # Stack and format draws
        return batch_transform_and_reconstruct(zs,var_params_flat,self.summary,self.constrain_fun_dict)
    
    def posterior_draw_and_transform(self,
                                     n_draws: int = 1000,
                                     key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
        
        return self._posterior_draw_and_transform(self.final_var_params_flat,n_draws=n_draws,key=key)



class MAPGuide(Guide):
    """
    Convenient class to perform standard optimisation
    """
    name: str ="MAPGuide"
    init_mean_scale: float = 0.0
    dim_param: int = 0   # actual dimensiion of the original parameter space

    
    def init_params(self, flat_theta: jax.Array) -> jax.Array:
        self.dim_param = flat_theta.shape[0]
        return self.init_mean_scale * jnp.ones_like(flat_theta)
           
    
    def unpack_params(self, var_params_flat: jax.Array) -> Tuple[jax.Array,jax.Array]:  
        return var_params_flat
    

    def transform_draws(self, z: jax.Array, var_params: jax.Array) -> jax.Array:
        mean = self.unpack_params(var_params)
        return mean
    
    def entropy(self, var_params: jax.Array) -> float:
        return 0
    def z_dim(self):
        return self.dim_param
    
class LaplaceApproxGuide(MAPGuide):
    """
     Perform laplace approximation by doing MAP + estimating variance using the gaussian.
     To the elbo is added the inv_norm of the gaussian (cf bishop pattern recognition and ML pragraph(4.4.1))
    """
    name: str ="LaplaceApproxGuide"




class MeanFieldGuide(Guide):

    name: str ="MeanFieldGuide"
    init_mean_scale =  0.0  # Either float or an array of size nflat_theta.shape[0]
    init_log_sd_scale: float = 0.0
    dim_param: int = 0   # actual dimensiion of the original parameter space

    
    def init_params(self, flat_theta: jax.Array) -> jax.Array:
        self.dim_param = flat_theta.shape[0]
        return jnp.stack([
            self.init_mean_scale * jnp.ones_like(flat_theta),
            self.init_log_sd_scale * jnp.ones_like(flat_theta),
        ])
    
    def unpack_params(self, var_params_flat: jax.Array) -> Tuple[jax.Array,jax.Array]:
        d = var_params_flat.shape[0] // 2
        return var_params_flat[:d], var_params_flat[d:2*d]
    

    def transform_draws(self, z: jax.Array, var_params: jax.Array) -> jax.Array:
        mean, log_sd = self.unpack_params(var_params)
        return mean + jnp.exp(log_sd) * z
    
    def entropy(self, var_params: jax.Array) -> float:
        _, log_sd = self.unpack_params(var_params)
        d = log_sd.shape[0]
        return 0.5 * d * (1 + jnp.log(2 * jnp.pi)) + jnp.sum(log_sd)
    def z_dim(self):
        return self.dim_param
    
    def inv_norm(self,log_sd):
        d = log_sd.shape[0]
        return 0.5 * d * jnp.log(2 * jnp.pi) - 0.5* jnp.sum(jnp.abs(log_sd))

    
    def get_stuctured_mean_std_before_transformation(self):
        """
        Only needed for nice representation
        """
        if self.final_var_params_flat.size == 0:
            return [],[]
        
        mean_flat, log_sd_flat = self.unpack_params(self.final_var_params_flat)
        
        # Reconstruct mean and sigma into structured parameters
        mean_dict = reconstruct(mean_flat, self.summary, jnp.reshape)
        log_sd_dict = reconstruct(log_sd_flat, self.summary, jnp.reshape)
        sigma_dict = jax.tree_map(jnp.exp, log_sd_dict)

        return mean_dict,sigma_dict

    
    def __repr__(self):
        """
        Only needed for nice representation
        """
        if self.final_var_params_flat.size == 0:
            return "Guide parameters not initialized."

        mean_dict,sigma_dict  = self.get_stuctured_mean_std_before_transformation()
        sep ="____________________"
        lines = [sep,"Mean Field Guide:"]
        for param in self.summary.keys():
            param_mean = mean_dict[param]
            param_sigma = sigma_dict[param]
            
            # Convert to numpy arrays for printing
            mean_np = jax.device_get(param_mean)
            sigma_np = jax.device_get(param_sigma)
            
            # Get constraint function and its name
            constraint_func = self.constrain_fun_dict.get(param, None)
            trans_name = get_constraint_name(constraint_func)
            
            # Format the parameter line
            trans_part = f"{trans_name}(" if trans_name else ""
            closing = ")" if trans_name else ""
            
            mean_str = np.array2string(mean_np, precision=4, separator=', ')
            sigma_str = np.array2string(sigma_np, precision=4, separator=', ')
            lines.append(f"{param}: {trans_part}N(μ={mean_str}, σ={sigma_str}){closing}")
        
        return "\n".join(lines+[sep])
    
    
 

class LowRankGuide(Guide):
    name: str ="LowRankGuide"
    rank: int
    init_mean_scale: float = 0.0
    init_log_sd_scale: float = 0.0
    init_L_scale: float = 0.1
    dim_param: int = 0   # actual dimensiion of the original parameter space

    def __init__(self,rank):
        self.rank = rank

    def init_params(self, flat_theta: jax.Array) -> jax.Array:
        d = flat_theta.shape[0]
        self.dim_param = d
        return jnp.concatenate([
            self.init_mean_scale * jnp.ones(d),
            self.init_log_sd_scale * jnp.ones(d),
            self.init_L_scale * jax.random.normal(jax.random.PRNGKey(0), (d * self.rank,))
        ])

    def unpack_params(self, var_params_flat: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        d = self.dim_param
        mean = var_params_flat[:d]
        log_sd = var_params_flat[d:2*d]
        L = var_params_flat[2*d:].reshape(d, self.rank)
        return mean, log_sd, L,d
    
    def transform_draws(self, z: jax.Array, var_params_flat: jax.Array) -> jax.Array:
        mean, log_sd, L,d = self.unpack_params(var_params_flat)
        z_diag = z[:d]
        z_lowrank = z[d:]
        return mean + jnp.exp(log_sd)*z_diag + L @ z_lowrank  # jnp.exp(log_sd) ** 0.5 ?
    

    def entropy(self, var_params: jax.Array) -> float:
        _, log_sd, L,d = self.unpack_params(var_params)
        scaled_L = L / jnp.exp(log_sd[:, None])
        I_plus = jnp.eye(self.rank) + scaled_L.T @ scaled_L
        log_det = jnp.linalg.slogdet(I_plus)[1]
        return 0.5*d*(1 + jnp.log(2*jnp.pi)) + jnp.sum(log_sd) + 0.5*log_det
    
    def z_dim(self):
        return self.dim_param + self.rank


class FullRankGuide(Guide):
    name: str ="FullRankGuide"
    init_mean_scale = 0.0
    init_L_scale: float = 0.1
    dim_param: int = 0

    def init_params(self, flat_theta: jax.Array) -> jax.Array:
        d = flat_theta.shape[0]
        self.dim_param = d
        L_size = d * (d + 1) // 2  # Number of Cholesky elements
        
        return jnp.concatenate([
            self.init_mean_scale * jnp.ones(d),  # Means
            self.init_L_scale * jnp.ones(d),  # log_sd for diagonal
            jnp.zeros(L_size - d)  # Lower triangle (off-diagonal)
        ])

    def unpack_params(self, var_params_flat: jax.Array) -> Tuple[jax.Array, jax.Array]:
        d = self.dim_param
        mean = var_params_flat[:d]
        log_sd = var_params_flat[d:2*d]
        lower = var_params_flat[2*d:]
        
        # Build Cholesky factor with positive diagonal
        L = jnp.zeros((d, d))
        
        tril_indices = jnp.tril_indices(d,-1)  # -1 to remove the diagonal

        L = L.at[tril_indices].set(lower)
        diag_indices = jnp.diag_indices(d)
        L = L.at[diag_indices].set(jnp.exp(log_sd) + 1e-6) 
        return mean, L
    
    def transform_draws(self, z: jax.Array, var_params_flat: jax.Array) -> jax.Array:
        mean, L = self.unpack_params(var_params_flat)
        return mean + L @ z

    def entropy(self, var_params: jax.Array) -> float:
        _, L = self.unpack_params(var_params)
        return 0.5*self.dim_param*(1 + jnp.log(2*jnp.pi)) + jnp.sum(jnp.log(jnp.diag(L)))
    
    def z_dim(self):
        return self.dim_param