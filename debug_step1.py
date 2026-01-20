
import pandas as pd
import numpy as np
import os
from bnlearn.network import BayesianNetwork
from bnlearn.score import score_node
import jax
import jax.numpy as jnp
from bnlearn.score.jax_discrete import jax_bic_discrete

# Setup
data_path = "tests/data/small_data.csv"
df = pd.read_csv(data_path)
for col in df.columns:
    df[col] = df[col].astype('category')

nodes = list(df.columns)
n_obs = len(df)
k_bic = np.log(n_obs) / 2.0

# JAX data
jax_data = jnp.array([df[col].cat.codes.values for col in nodes]).T
jax_cardinalities = tuple([len(df[col].cat.categories) for col in nodes])
node_to_idx = {node: i for i, node in enumerate(nodes)}

def get_bic(node, parents):
    n_idx = node_to_idx[node]
    p_indices = tuple(sorted([node_to_idx[p] for p in parents]))
    return float(jax_bic_discrete(jax_data, n_idx, p_indices, jax_cardinalities, k_bic))

# Initial scores (empty network)
scores = {node: get_bic(node, []) for node in nodes}
print(f"Initial Scores: {scores}")

# Deltas for Add
deltas = {}
for u in nodes:
    for v in nodes:
        if u == v: continue
        delta = get_bic(v, [u]) - scores[v]
        deltas[(u, v)] = delta

sorted_deltas = sorted(deltas.items(), key=lambda x: x[1], reverse=True)
print("\nTop 10 Additions:")
for op, d in sorted_deltas[:10]:
    print(f"Add {op}: {d:.6f}")
