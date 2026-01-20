
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

# We use a fixed max number of parents to reduce recompilation.
MAX_PARENTS = 5

@partial(jit, static_argnums=(1, 2, 3, 4))
def jax_loglik_discrete_core(data, node_idx, parent_indices_padded, cardinalities, n_parents):
    """
    Core log-likelihood calculation with padded parents.
    data: dynamic array (N, D)
    node_idx: static int
    parent_indices_padded: static tuple
    cardinalities: static tuple
    n_parents: static int
    """
    n_obs = data.shape[0]
    node_data = data[:, node_idx]
    node_card = cardinalities[node_idx]
    
    if n_parents == 0:
        counts = jnp.bincount(node_data, length=node_card)
        mask = counts > 0
        ll = jnp.sum(jnp.where(mask, counts * jnp.log(jnp.where(mask, counts.astype(jnp.float32), 1.0)), 0.0)) - n_obs * jnp.log(float(n_obs))
        return ll
    
    p_idx = jnp.zeros(n_obs, dtype=jnp.int32)
    stride = 1
    for i in range(MAX_PARENTS):
        is_active = i < n_parents
        if is_active:
            idx = parent_indices_padded[i]
            p_idx += data[:, idx].astype(jnp.int32) * stride
            stride *= cardinalities[idx]
    
    joint_idx = p_idx * node_card + node_data
    joint_counts = jnp.bincount(joint_idx, length=stride * node_card)
    parent_counts = jnp.bincount(p_idx, length=stride)
    
    jc_mask = joint_counts > 0
    pc_mask = parent_counts > 0
    
    term1 = jnp.sum(jnp.where(jc_mask, joint_counts * jnp.log(jnp.where(jc_mask, joint_counts.astype(jnp.float32), 1.0)), 0.0))
    term2 = jnp.sum(jnp.where(pc_mask, parent_counts * jnp.log(jnp.where(pc_mask, parent_counts.astype(jnp.float32), 1.0)), 0.0))
    
    return term1 - term2

@partial(jit, static_argnums=(1, 2, 3))
def jax_loglik_discrete_simple(data, node_idx, parent_indices, cardinalities):
    n_obs = data.shape[0]
    node_data = data[:, node_idx]
    node_card = cardinalities[node_idx]
    
    if len(parent_indices) == 0:
        counts = jnp.bincount(node_data, length=node_card)
        mask = counts > 0
        ll = jnp.sum(jnp.where(mask, counts * jnp.log(jnp.where(mask, counts.astype(jnp.float32), 1.0)), 0.0)) - n_obs * jnp.log(float(n_obs))
        return ll
    
    p_idx = jnp.zeros(n_obs, dtype=jnp.int32)
    stride = 1
    for idx in parent_indices:
        p_idx += data[:, idx].astype(jnp.int32) * stride
        stride *= cardinalities[idx]
    
    joint_idx = p_idx * node_card + node_data
    joint_counts = jnp.bincount(joint_idx, length=stride * node_card)
    parent_counts = jnp.bincount(p_idx, length=stride)
    jc_mask = joint_counts > 0
    pc_mask = parent_counts > 0
    term1 = jnp.sum(jnp.where(jc_mask, joint_counts * jnp.log(jnp.where(jc_mask, joint_counts.astype(jnp.float32), 1.0)), 0.0))
    term2 = jnp.sum(jnp.where(pc_mask, parent_counts * jnp.log(jnp.where(pc_mask, parent_counts.astype(jnp.float32), 1.0)), 0.0))
    return term1 - term2

def jax_loglik_discrete(data, node_idx, parent_indices, cardinalities):
    n_parents = len(parent_indices)
    if n_parents > MAX_PARENTS:
        return jax_loglik_discrete_simple(data, node_idx, tuple(parent_indices), tuple(cardinalities))
    
    padded = list(parent_indices) + [0] * (MAX_PARENTS - n_parents)
    return jax_loglik_discrete_core(data, node_idx, tuple(padded), tuple(cardinalities), n_parents)

def jax_bic_discrete(data, node_idx, parent_indices, cardinalities, k_bic):
    ll = jax_loglik_discrete(data, node_idx, parent_indices, cardinalities)
    node_card = cardinalities[node_idx]
    parent_levels = 1.0
    for idx in parent_indices:
        parent_levels *= cardinalities[idx]
    n_params = (node_card - 1) * parent_levels
    return ll - k_bic * n_params
