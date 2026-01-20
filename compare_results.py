
import pandas as pd
import numpy as np
import os
import time
from bnlearn.learning import hc
from bnlearn.score import score_network
from bnlearn.network import BayesianNetwork

def load_r_results(data_dir, prefix):
    # Load Data
    data_path = os.path.join(data_dir, f'{prefix}_data.csv')
    df = pd.read_csv(data_path)
    for col in df.columns:
        df[col] = df[col].astype('category')
        
    # Load R Arcs
    arcs_path = os.path.join(data_dir, f'{prefix}_arcs_R.csv')
    r_arcs_df = pd.read_csv(arcs_path)
    r_arcs = set(zip(r_arcs_df['from'], r_arcs_df['to']))
    
    # Load R Score
    score_path = os.path.join(data_dir, f'{prefix}_score_R.txt')
    with open(score_path, 'r') as f:
        r_score = float(f.read().strip())
        
    return df, r_arcs, r_score

def compare(name, data_dir, prefix):
    print(f"\n--- Comparing Results for: {name} ---")
    
    # Load
    df, r_arcs, r_score = load_r_results(data_dir, prefix)
    print(f"Data shape: {df.shape}")
    print(f"R Score: {r_score:.6f}")
    print(f"R Arcs count: {len(r_arcs)}")
    
    # Run Python
    start_time = time.time()
    bn = hc(df, score='bic', max_iter=200)
    py_time = time.time() - start_time
    
    py_arcs = set(zip(bn.arcs['from'], bn.arcs['to']))
    
    # Calculate Py Score
    py_score = score_network(bn, df, score_type='bic')
    
    print(f"Python Score: {py_score:.6f}")
    print(f"Python Arcs count: {len(py_arcs)}")
    print(f"Python Execution Time: {py_time:.4f}s")
    
    # Comparisons
    score_diff = abs(r_score - py_score)
    print(f"Score Difference: {score_diff:.6e}")
    if score_diff < 1e-4:
        print(">> Scores MATCH ✅")
    else:
        print(">> Scores MISMATCH ❌")
        
    # Arcs
    common = r_arcs.intersection(py_arcs)
    only_r = r_arcs - py_arcs
    only_py = py_arcs - r_arcs
    
    if len(only_r) == 0 and len(only_py) == 0:
        print(">> Structures MATCH EXACTLY ✅")
    else:
        print(">> Structures MISMATCH ❌")
        print(f"   Common: {len(common)}")
        print(f"   Only in R: {only_r}")
        print(f"   Only in Py: {only_py}")
        
        # Jaccard
        intersection = len(common)
        union = len(r_arcs.union(py_arcs))
        jaccard = intersection / union if union > 0 else 1.0
        print(f"   Jaccard Similarity: {jaccard:.4f}")

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), 'tests', 'data')
    
    if not os.path.exists(data_dir):
        print("Data directory not found. Please run tests/generate_reference.R first.")
        exit(1)
        
    compare("Small Network (5 Nodes)", data_dir, "small")
    compare("Large Network (37 Nodes, Subset)", data_dir, "large")
