
import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import vmap, jit

# Mock data generation
def generate_data(n_obs=10000, n_cols=5, n_levels=3):
    data = np.random.randint(0, n_levels, (n_obs, n_cols))
    cols = [f"X{i}" for i in range(n_cols)]
    df = pd.DataFrame(data, columns=cols)
    for col in cols:
        df[col] = df[col].astype('category')
    return df, data, cols

# Current (Pandas-based) implementation logic
def pandas_score(df, node, parents):
    cols = [node] + parents
    joint_counts = df.value_counts(subset=cols, sort=False).values
    parent_counts = df.value_counts(subset=parents, sort=False).values
    ll = np.sum(joint_counts * np.log(joint_counts)) - np.sum(parent_counts * np.log(parent_counts))
    return ll

# JAX-based implementation logic
@jit
def jax_loglik(data, node_idx, parent_indices, cardinalities):
    # Map configuration to flat index
    # We assume data is (N, D) integer array
    n_obs = data.shape[0]
    
    # Target node data
    node_data = data[:, node_idx]
    node_card = cardinalities[node_idx]
    
    if len(parent_indices) == 0:
        counts = jnp.bincount(node_data, length=node_card)
        # Avoid log(0)
        counts = jnp.where(counts == 0, 1.0, counts.astype(jnp.float32))
        return jnp.sum(counts * jnp.log(counts)) - n_obs * jnp.log(n_obs)
    
    # Parent data mapping
    # idx = sum(p_i * mult_i)
    p_idx = jnp.zeros(n_obs, dtype=jnp.int32)
    mult = 1
    for p_i in parent_indices:
        p_idx += data[:, p_i] * mult
        mult *= cardinalities[p_i]
    
    # Joint index: parents * node_card + node
    joint_idx = p_idx * node_card + node_data
    
    joint_counts = jnp.bincount(joint_idx, length=mult * node_card)
    parent_counts = jnp.bincount(p_idx, length=mult)
    
    # Mask out zeros for log
    jc_mask = joint_counts > 0
    pc_mask = parent_counts > 0
    
    ll = jnp.sum(jnp.where(jc_mask, joint_counts * jnp.log(jnp.where(jc_mask, joint_counts, 1.0)), 0.0)) - \
         jnp.sum(jnp.where(pc_mask, parent_counts * jnp.log(jnp.where(pc_mask, parent_counts, 1.0)), 0.0))
    
    return ll

def benchmark():
    n_obs = 50000
    n_cols = 10
    n_levels = 4
    df, data_np, cols = generate_data(n_obs, n_cols, n_levels)
    data_jax = jnp.array(data_np)
    cards = jnp.array([n_levels] * n_cols)
    
    node = 0
    parents = [1, 2]
    parent_indices = jnp.array([1, 2])
    
    print(f"Benchmarking Score Calculation (N_obs={n_obs}, N_levels={n_levels})")
    
    # Pandas warm up
    s0 = pandas_score(df, "X0", ["X1", "X2"])
    
    start = time.time()
    for _ in range(100):
        _ = pandas_score(df, "X0", ["X1", "X2"])
    pd_time = (time.time() - start) / 100
    print(f"Pandas average time: {pd_time*1000:.4f} ms")
    
    # JAX warm up
    _ = jax_loglik(data_jax, 0, parent_indices, cards)
    
    start = time.time()
    for _ in range(100):
        _ = jax_loglik(data_jax, 0, parent_indices, cards).block_until_ready()
    jax_time = (time.time() - start) / 100
    print(f"JAX average time: {jax_time*1000:.4f} ms")
    
    print(f"Speedup: {pd_time / jax_time:.2f}x")
    
    # Check correctness
    s_jax = jax_loglik(data_jax, 0, parent_indices, cards)
    print(f"Correctness check: PD={s0:.6f}, JAX={s_jax:.6f}, Diff={abs(s0-s_jax):.2e}")

    # Vectorized check
    # Let's say we want to evaluate adding any other node as a parent
    def eval_addition(new_p):
        return jax_loglik(data_jax, 0, jnp.append(parent_indices, new_p), cards)

    v_eval = jit(vmap(eval_addition))
    candidates = jnp.array([3, 4, 5, 6, 7, 8, 9])
    
    # Warm up
    _ = v_eval(candidates)
    
    start = time.time()
    for _ in range(100):
        _ = v_eval(candidates).block_until_ready()
    v_jax_time = (time.time() - start) / 100
    print(f"JAX Vectorized (7 candidates) average time: {v_jax_time*1000:.4f} ms")
    print(f"Per-candidate time in vectorized mode: {(v_jax_time/len(candidates))*1000:.4f} ms")

if __name__ == "__main__":
    benchmark()
