
import pandas as pd
import numpy as np
import os
from bnlearn.network import BayesianNetwork
from bnlearn.score import score_network
from bnlearn.learning import hc

# Setup
data_path = "tests/data/small_data.csv"
df = pd.read_csv(data_path)
for col in df.columns:
    df[col] = df[col].astype('category')

nodes = list(df.columns)

# R Structure
r_arcs = [('LVV', 'PCWP'), ('HYP', 'STKV'), ('LVV', 'STKV'), ('STKV', 'CO'), ('HYP', 'LVV')]
r_arcs_df = pd.DataFrame(r_arcs, columns=['from', 'to'])
bn_r = BayesianNetwork(nodes, r_arcs_df)
score_r = score_network(bn_r, df, score_type='bic')

# Python HC
bn_py = hc(df, score='bic', max_iter=100)
score_py = score_network(bn_py, df, score_type='bic')

print(f"R Score: {score_r:.6f}")
print(f"Py Score: {score_py:.6f}")
print(f"Difference: {score_py - score_r:.6e}")
print(f"Py Arcs: {set(zip(bn_py.arcs['from'], bn_py.arcs['to']))}")
