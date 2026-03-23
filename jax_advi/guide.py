from typing import Protocol, Dict, Tuple, Any,List
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import numpy as np
from jax_advi.utils.flattening import reconstruct
from jax_advi.constraints import apply_constraints, get_constraint_name


# ======================== Guide Protocol ========================

class VariationalGuide(Protocol):
    def init_params(self, flat_theta: jax.Array) -> jax.Array:
        """Initialize variational parameters given a flat parameter vector"""

    def transform_draws(self, z: jax.Array, var_params: jax.Array) -> jax.Array:
        """Transform noise z into parameters (reparametrisation trick)"""

    def entropy(self, var_params: jax.Array) -> float:
        """Compute entropy of the guide distribution"""

    def z_dim(self) -> int:
        """Dimension of the noise vector z"""

    def sample_z(self, key: jax.random.PRNGKey, n_draws: int) -> jax.Array:
        """
        Sample noise vectors. Override for non-Gaussian guides (e.g. Bernoulli).
        Default: standard normal, suitable for most continuous guides.
        """


# ======================== Base Guide (shared utilities) ========================
@dataclass
class Guide:
    """
    Base class providing:
      - default Gaussian sample_z
      - posterior draw + constrained transform
      - __repr__
    Subclasses must implement: init_params, transform_draws, entropy, z_dim
    """
    name: str = "Guide"
    final_var_params_flat: jax.Array = field(default_factory=lambda: jnp.array([]))
    constrain_fun_dict: Dict = field(default_factory=dict)
    summary: Dict = field(default_factory=dict)
    post_transform: Any = field(default=None)  # Reparametrisation or None

    def sample_z(self, key: jax.random.PRNGKey, n_draws: int) -> jax.Array:
        """Standard normal noise — override for discrete/other guides."""
        return jax.random.normal(key, (n_draws, self.z_dim()))

    def _apply_post_transform(self, theta_dict: Dict) -> Dict:
        """Apply post_transform if present, otherwise return as-is."""
        if self.post_transform is not None:
            return self.post_transform.forward(theta_dict)
        return theta_dict

    def _out_keys_after_transform(self, base_keys) -> set:
        """Get output keys accounting for post_transform renaming."""
        if self.post_transform is not None:
            dummy = {k: jnp.zeros(v) for k, v in self.summary.items()}
            return set(self.post_transform.forward(dummy).keys())
        return set(base_keys)

    def _posterior_draw_and_transform(self, var_params_flat, n_draws=1000,
                                    key=jax.random.PRNGKey(0)):
        zs = self.sample_z(key, n_draws)
        out_keys = self._out_keys_after_transform(self.summary.keys())

        def transform_and_reconstruct(z, var_params_flat, summary, constrain_fun_dict):
            flat_draw = self.transform_draws(z, var_params_flat)
            theta_dict = reconstruct(flat_draw, summary, jnp.reshape)
            constrained, _ = apply_constraints(theta_dict, constrain_fun_dict)
            return self._apply_post_transform(constrained)

        batch_fn = jax.vmap(
            transform_and_reconstruct,
            in_axes=(0, None, None, None),
            out_axes={k: 0 for k in out_keys},
        )
        return batch_fn(zs, var_params_flat, self.summary, self.constrain_fun_dict)
    
    def posterior_draw_and_transform(
            self,
            n_draws: int = 1000,
            key: jax.random.PRNGKey = jax.random.PRNGKey(0),
        ) -> Dict[str, jnp.ndarray]:
            return self._posterior_draw_and_transform(
                self.final_var_params_flat, n_draws=n_draws, key=key
        )



# ======================== Concrete Guides ========================

@dataclass
class MAPGuide(Guide):
    name: str = "MAPGuide"
    init_mean_scale: float = 0.0
    dim_param: int = field(default=0, init=False)

    def init_params(self, flat_theta: jax.Array) -> jax.Array:
        self.dim_param = flat_theta.shape[0]
        return self.init_mean_scale * jnp.ones_like(flat_theta)

    def transform_draws(self, z: jax.Array, var_params: jax.Array) -> jax.Array:
        return var_params  # MAP: just return the mean

    def entropy(self, var_params: jax.Array) -> float:
        return 0.0

    def z_dim(self) -> int:
        return self.dim_param


@dataclass
class LaplaceApproxGuide(MAPGuide):
    name: str = "LaplaceApproxGuide"

    def to_meanfield(self, free_params: jax.Array, hessian: jax.Array):# -> Tuple[MeanFieldGuide, jax.Array, float]:
        """Convert MAP solution to MeanField using Laplace approximation."""
        d = np.diag(np.abs(hessian))
        log_sigma = -np.log(d) / 2
        new_free_params = np.concatenate([free_params, log_sigma])
        
        new_guide = MeanFieldGuide()
        new_guide.init_params(jnp.zeros(self.dim_param))
        new_guide.name = "LaplaceApproxGuide"
        
        delta_elbo = new_guide.inv_norm(log_sigma)
        return new_guide, new_free_params, delta_elbo

@dataclass
class MeanFieldGuide(Guide):
    name: str = "MeanFieldGuide"
    init_mean_scale: float = 0.0
    init_log_sd_scale: float = 0.0
    dim_param: int = field(default=0, init=False)

    def init_params(self, flat_theta: jax.Array) -> jax.Array:
        self.dim_param = flat_theta.shape[0]
        return jnp.concatenate([
            self.init_mean_scale * jnp.ones_like(flat_theta),
            self.init_log_sd_scale * jnp.ones_like(flat_theta),
        ])

    def _unpack(self, var_params: jax.Array) -> Tuple[jax.Array, jax.Array]:
        d = var_params.shape[0] // 2
        return var_params[:d], var_params[d:]

    def transform_draws(self, z: jax.Array, var_params: jax.Array) -> jax.Array:
        mean, log_sd = self._unpack(var_params)
        return mean + jnp.exp(log_sd) * z

    def entropy(self, var_params: jax.Array) -> float:
        _, log_sd = self._unpack(var_params)
        d = log_sd.shape[0]
        return 0.5 * d * (1 + jnp.log(2 * jnp.pi)) + jnp.sum(log_sd)

    def z_dim(self) -> int:
        return self.dim_param

    def inv_norm(self, log_sd: jax.Array) -> float:
        d = log_sd.shape[0]
        return 0.5 * d * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(jnp.abs(log_sd))

    def get_structured_mean_std_before_transformation(self):
        if self.final_var_params_flat.size == 0:
            return [], []
        mean_flat, log_sd_flat = self._unpack(self.final_var_params_flat)
        mean_dict = reconstruct(mean_flat, self.summary, jnp.reshape)
        log_sd_dict = reconstruct(log_sd_flat, self.summary, jnp.reshape)
        sigma_dict = jax.tree_map(jnp.exp, log_sd_dict)
        return mean_dict, sigma_dict

    def __repr__(self):
        if self.final_var_params_flat.size == 0:
            return "Guide parameters not initialized."
        mean_dict, sigma_dict = self.get_structured_mean_std_before_transformation()
        sep = "____________________"
        lines = [sep, "Mean Field Guide:"]
        for param in self.summary.keys():
            mean_np = jax.device_get(mean_dict[param])
            sigma_np = jax.device_get(sigma_dict[param])
            constraint_func = self.constrain_fun_dict.get(param, None)
            trans_name = get_constraint_name(constraint_func)
            trans_part = f"{trans_name}(" if trans_name else ""
            closing = ")" if trans_name else ""
            mean_str = np.array2string(mean_np, precision=4, separator=", ")
            sigma_str = np.array2string(sigma_np, precision=4, separator=", ")
            lines.append(f"{param}: {trans_part}N(μ={mean_str}, σ={sigma_str}){closing}")
        return "\n".join(lines + [sep])


@dataclass
class LowRankGuide(Guide):
    name: str = "LowRankGuide"
    rank: int = 1
    init_mean_scale: float = 0.0
    init_log_sd_scale: float = 0.0
    init_L_scale: float = 0.1
    dim_param: int = field(default=0, init=False)

    def init_params(self, flat_theta: jax.Array) -> jax.Array:
        d = flat_theta.shape[0]
        self.dim_param = d
        return jnp.concatenate([
            self.init_mean_scale * jnp.ones(d),
            self.init_log_sd_scale * jnp.ones(d),
            self.init_L_scale * jax.random.normal(jax.random.PRNGKey(0), (d * self.rank,)),
        ])

    def _unpack(self, var_params: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        d = self.dim_param
        return var_params[:d], var_params[d:2*d], var_params[2*d:].reshape(d, self.rank)

    def transform_draws(self, z: jax.Array, var_params: jax.Array) -> jax.Array:
        mean, log_sd, L = self._unpack(var_params)
        d = self.dim_param
        return mean + jnp.exp(log_sd) * z[:d] + L @ z[d:]

    def entropy(self, var_params: jax.Array) -> float:
        _, log_sd, L = self._unpack(var_params)
        scaled_L = L / jnp.exp(log_sd[:, None])
        log_det = jnp.linalg.slogdet(jnp.eye(self.rank) + scaled_L.T @ scaled_L)[1]
        return 0.5*self.dim_param*(1 + jnp.log(2*jnp.pi)) + jnp.sum(log_sd) + 0.5*log_det

    def z_dim(self) -> int:
        return self.dim_param + self.rank


@dataclass
class FullRankGuide(Guide):
    name: str = "FullRankGuide"
    init_mean_scale: Any = 0.0
    init_log_sd_scale: Any = 0.0  # add this — maps to log diagonal of L
    init_L_scale: float = 0.1     # keep for off-diagonal init
    dim_param: int = field(default=0, init=False)

    def init_params(self, flat_theta: jax.Array) -> jax.Array:
        d = flat_theta.shape[0]
        self.dim_param = d
        L_size = d * (d + 1) // 2
        return jnp.concatenate([
            self.init_mean_scale * jnp.ones_like(flat_theta),
            self.init_log_sd_scale * jnp.ones_like(flat_theta),  # log diagonal
            self.init_L_scale * jnp.ones(L_size - d),            # off-diagonal
        ])

    def _unpack(self, var_params: jax.Array) -> Tuple[jax.Array, jax.Array]:
        d = self.dim_param
        mean = var_params[:d]
        log_sd = var_params[d:2*d]
        lower = var_params[2*d:]
        tril_idx = jnp.tril_indices(d, -1)
        diag_idx = jnp.diag_indices(d)
        L = jnp.zeros((d, d))
        L = L.at[tril_idx].set(lower)
        L = L.at[diag_idx].set(jnp.exp(log_sd) + 1e-6)
        return mean, L

    def transform_draws(self, z: jax.Array, var_params: jax.Array) -> jax.Array:
        mean, L = self._unpack(var_params)
        return mean + L @ z

    def entropy(self, var_params: jax.Array) -> float:
        _, L = self._unpack(var_params)
        return (
            0.5 * self.dim_param * (1 + jnp.log(2 * jnp.pi))
            + jnp.sum(jnp.log(jnp.diag(L)))
        )

    def z_dim(self) -> int:
        return self.dim_param


# ======================== Composite Guide ========================

@dataclass
class CompositeGuide(Guide):
    """
    Assign a different guide to each named parameter group.
    
    For independent groups matching theta_shape_dict keys directly:
        G = CompositeGuide(
            group_guides={"lambdai": FullRankGuide(), "extra_t": MeanFieldGuide()},
        )

    For paired/correlated groups across parameters:
        G = CompositeGuide(
            group_guides={f"pair_{i}": FullRankGuide() for i in range(n)},
            param_mapping={
                "lambdai": [(f"pair_{i}", 0) for i in range(n)],
                "extra_t":  [(f"pair_{i}", 1) for i in range(n)],
            }
        )
    """
    name: str = "CompositeGuide"
    group_guides: Dict[str, Guide] = field(default_factory=dict)
    param_mapping: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)

    _group_dims: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _var_param_slices: Dict[str, Tuple[int, int]] = field(default_factory=dict, init=False, repr=False)
    _z_slices: Dict[str, Tuple[int, int]] = field(default_factory=dict, init=False, repr=False)

    def __getitem__(self, key: str) -> Guide:
        return self.group_guides[key]

    def init_params(self, flat_theta: jax.Array) -> jax.Array:
        if self.param_mapping:
            # Derive per-group dims from param_mapping alone
            # flat_theta ordering is irrelevant for composite with mapping
            self._group_dims = {}
            for param_name, sources in self.param_mapping.items():
                for group_name, local_idx in sources:
                    self._group_dims[group_name] = max(
                        self._group_dims.get(group_name, 0), local_idx + 1
                    )
        else:
            # No mapping: group names match theta_shape_dict keys directly
            # summary must be set before init_params is called
            self._group_dims = {
                name: int(jnp.prod(jnp.array(self.summary[name])))
                for name in self.group_guides
            }

        all_params = []
        vp_cursor, z_cursor = 0, 0

        for name, guide in self.group_guides.items():
            dim = self._group_dims[name]
            # dummy flat_theta — sub-guides only need dim to initialise
            params = guide.init_params(jnp.zeros(dim))
            all_params.append(params)

            vp_end = vp_cursor + params.shape[0]
            z_end = z_cursor + guide.z_dim()
            self._var_param_slices[name] = (vp_cursor, vp_end)
            self._z_slices[name] = (z_cursor, z_end)
            vp_cursor, z_cursor = vp_end, z_end

        return jnp.concatenate(all_params)

    def transform_draws(self, z: jax.Array, var_params: jax.Array) -> jax.Array:
        """Flat version — kept for compatibility with base Guide machinery."""
        draws = []
        for name, guide in self.group_guides.items():
            vp_s, vp_e = self._var_param_slices[name]
            z_s, z_e = self._z_slices[name]
            draws.append(guide.transform_draws(z[z_s:z_e], var_params[vp_s:vp_e]))
        return jnp.concatenate(draws)

    def transform_draws_to_dict(
        self, z: jax.Array, var_params: jax.Array
    ) -> Dict[str, jax.Array]:
        """Dict version — bypasses flatten/summary entirely."""
        # Get draw for each group
        group_draws = {}
        for name, guide in self.group_guides.items():
            vp_s, vp_e = self._var_param_slices[name]
            z_s, z_e = self._z_slices[name]
            group_draws[name] = guide.transform_draws(z[z_s:z_e], var_params[vp_s:vp_e])

        if self.param_mapping:
            # Reassemble from pairs back into original parameter structure
            result = {}
            for param_name, sources in self.param_mapping.items():
                values = jnp.array([
                    group_draws[group_name][local_idx]
                    for group_name, local_idx in sources
                ])
                result[param_name] = values.reshape(self.summary[param_name])
            return result
        else:
            # Group names match param names directly
            return {
                name: draw.reshape(self.summary[name])
                for name, draw in group_draws.items()
            }

    def entropy(self, var_params: jax.Array) -> float:
        total = 0.0
        for name, guide in self.group_guides.items():
            vp_s, vp_e = self._var_param_slices[name]
            total += guide.entropy(var_params[vp_s:vp_e])
        return total

    def z_dim(self) -> int:
        return sum(g.z_dim() for g in self.group_guides.values())

    def sample_z(self, key: jax.random.PRNGKey, n_draws: int) -> jax.Array:
        """Each sub-guide samples its own z block independently."""
        keys = jax.random.split(key, len(self.group_guides))
        return jnp.concatenate([
            guide.sample_z(k, n_draws)
            for k, guide in zip(keys, self.group_guides.values())
        ], axis=-1)

    def _posterior_draw_and_transform(self, var_params_flat, n_draws=1000,
                                    key=jax.random.PRNGKey(0)):
        zs = self.sample_z(key, n_draws)
        base_keys = self.param_mapping.keys() if self.param_mapping else self.group_guides.keys()
        out_keys = self._out_keys_after_transform(base_keys)

        def transform_and_constrain(z, var_params_flat, constrain_fun_dict):
            theta_dict = self.transform_draws_to_dict(z, var_params_flat)
            constrained, _ = apply_constraints(theta_dict, constrain_fun_dict)
            return self._apply_post_transform(constrained)

        batch_fn = jax.vmap(
            transform_and_constrain,
            in_axes=(0, None, None),
            out_axes={k: 0 for k in out_keys},
        )
        return batch_fn(zs, var_params_flat, self.constrain_fun_dict)

