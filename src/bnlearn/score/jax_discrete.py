
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

# Enable 64-bit precision for consistency with R
jax.config.update("jax_enable_x64", True)

# We use a fixed max number of parents to reduce recompilation.
MAX_PARENTS = 5
# Max size for batch counting buffer. Stride * NodeCard must be <= this.
BATCH_BUFFER_SIZE = 2048

@partial(jit, static_argnums=(1, 2, 3, 4))
def jax_loglik_discrete_core(data, node_idx, parent_indices_padded, cardinalities, n_parents):
    """
    Core log-likelihood calculation for a single node.
    """
    n_obs = data.shape[0]
    node_data = data[:, node_idx]
    node_card = cardinalities[node_idx]
    
    if n_parents == 0:
        counts = jnp.bincount(node_data, length=int(node_card))
        mask = counts > 0
        ll = jnp.sum(jnp.where(mask, counts * jnp.log(jnp.where(mask, counts.astype(jnp.float64), 1.0)), 0.0)) - n_obs * jnp.log(float(n_obs))
        return ll
    
    p_idx = jnp.zeros(n_obs, dtype=jnp.int32)
    stride = 1
    for i in range(MAX_PARENTS):
        if i < n_parents:
            idx = parent_indices_padded[i]
            p_idx += data[:, idx].astype(jnp.int32) * stride
            stride *= cardinalities[idx]
    
    joint_idx = p_idx * node_card + node_data
    joint_counts = jnp.bincount(joint_idx, length=int(stride * node_card))
    parent_counts = jnp.bincount(p_idx, length=int(stride))
    
    jc_mask = joint_counts > 0
    pc_mask = parent_counts > 0
    
    term1 = jnp.sum(jnp.where(jc_mask, joint_counts * jnp.log(jnp.where(jc_mask, joint_counts.astype(jnp.float64), 1.0)), 0.0))
    term2 = jnp.sum(jnp.where(pc_mask, parent_counts * jnp.log(jnp.where(pc_mask, parent_counts.astype(jnp.float64), 1.0)), 0.0))
    
    return term1 - term2

@partial(jit, static_argnums=(1, 2, 3))
def jax_loglik_discrete_simple(data, node_idx, parent_indices, cardinalities):
    n_obs = data.shape[0]
    node_data = data[:, node_idx]
    node_card = cardinalities[node_idx]
    p_idx = jnp.zeros(n_obs, dtype=jnp.int32)
    stride = 1
    for idx in parent_indices:
        p_idx += data[:, idx].astype(jnp.int32) * stride
        stride *= cardinalities[idx]
    joint_idx = p_idx * node_card + node_data
    joint_counts = jnp.bincount(joint_idx, length=int(stride * node_card))
    parent_counts = jnp.bincount(p_idx, length=int(stride))
    jc_mask = joint_counts > 0
    pc_mask = parent_counts > 0
    term1 = jnp.sum(jnp.where(jc_mask, joint_counts * jnp.log(jnp.where(jc_mask, joint_counts.astype(jnp.float64), 1.0)), 0.0))
    term2 = jnp.sum(jnp.where(pc_mask, parent_counts * jnp.log(jnp.where(pc_mask, parent_counts.astype(jnp.float64), 1.0)), 0.0))
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

# --- BATCH SCORING ---

@partial(jit, static_argnums=(1, 2, 3, 5, 6))
def jax_bic_add_batch_core(data, node_idx, current_parents_padded, cardinalities, candidate_indices, n_current, k_bic):
    n_obs = data.shape[0]
    node_data = data[:, node_idx]
    card_array = jnp.array(cardinalities)
    node_card = cardinalities[node_idx]
    
    # Precompute current parent index
    base_p_idx = jnp.zeros(n_obs, dtype=jnp.int32)
    base_stride = 1
    base_n_params = 1.0
    for i in range(MAX_PARENTS):
        if i < n_current:
            idx = current_parents_padded[i]
            base_p_idx += data[:, idx].astype(jnp.int32) * base_stride
            base_stride *= cardinalities[idx]
            base_n_params *= cardinalities[idx]
    
    def eval_one(u_idx):
        u_card = card_array[u_idx]
        new_p_idx = base_p_idx + data[:, u_idx].astype(jnp.int32) * base_stride
        new_stride = base_stride * u_card
        
        joint_idx = new_p_idx * node_card + node_data
        joint_counts = jnp.bincount(joint_idx, length=BATCH_BUFFER_SIZE)
        parent_counts = jnp.bincount(new_p_idx, length=BATCH_BUFFER_SIZE // 2)
        
        jc_mask = (jnp.arange(BATCH_BUFFER_SIZE) < new_stride * node_card) & (joint_counts > 0)
        pc_mask = (jnp.arange(BATCH_BUFFER_SIZE // 2) < new_stride) & (parent_counts > 0)
        
        ll_term1 = jnp.sum(jnp.where(jc_mask, joint_counts * jnp.log(jnp.where(jc_mask, joint_counts.astype(jnp.float64), 1.0)), 0.0))
        ll_term2 = jnp.sum(jnp.where(pc_mask, parent_counts * jnp.log(jnp.where(pc_mask, parent_counts.astype(jnp.float64), 1.0)), 0.0))
        ll = ll_term1 - ll_term2
        n_params = (node_card - 1) * base_n_params * u_card
        return ll - k_bic * n_params

    return vmap(eval_one)(candidate_indices)

def jax_bic_add_batch(data, node_idx, current_parent_indices, cardinalities, candidate_indices, k_bic):
    n_current = len(current_parent_indices)
    if n_current >= MAX_PARENTS: # >= because we add one more
        return None
    padded = list(current_parent_indices) + [0] * (MAX_PARENTS - n_current)
    return jax_bic_add_batch_core(data, node_idx, tuple(padded), tuple(cardinalities), candidate_indices, n_current, k_bic)
