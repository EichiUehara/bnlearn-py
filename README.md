
# bnlearn-py

A Python port of the popular R package `bnlearn` for Bayesian Network structure learning, optimized with JAX.

This project provides a **functionally identical** implementation of the `bnlearn` Hill Climbing (`hc`) algorithm, ensuring exactly the same results as the R implementation on the same datasets.

## Implementation Details

The implementation mirrors the logic of `bnlearn`'s C backend:
-   **Algorithm**: Hill Climbing with restarts and perturbation.
-   **Score**: Discrete BIC, AIC, and Log-Likelihood.
-   **Optimization**: JAX-accelerated batched scoring using `vmap` and JIT.
-   **Search Strategy**: 
    -   Greedy search with specific operation ordering: `Add` -> `Delete` -> `Reverse`.
    -   Floating point tolerance matching R's machine epsilon behavior.
    -   Exact matching of R's loop order to ensure identical results in case of ties.

## Validation Results

We have conducted intensive verification by comparing the outputs of this Python implementation against the original R `bnlearn` package (v4.9) using identical datasets.

### Summary

| Dataset | Type | Nodes | Observations | Structure Match | Score Match |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Small Test** | Discrete | 5 | 1000 | ✅ **Exact** | ✅ **Exact** |
| **Alarm (Subset)** | Discrete | 37 | 2000 | ✅ **Exact** | ✅ **Exact** |

*(Note: "Exact" structure match means the sets of directed edges are identical. Score match is verified to double precision limits).*

### Detailed Verification Logs (JAX Optimized)

```text
--- Comparing Results for: Small Network (5 Nodes) ---
Data shape: (1000, 5)
R Score: -2460.827068
Python Score: -2460.827068
>> Scores MATCH ✅
>> Structures MATCH EXACTLY ✅

--- Comparing Results for: Large Network (37 Nodes, Subset) ---
Data shape: (2000, 37)
R Score: -23176.667311
Python Score: -23176.667311
Python Execution Time: ~57s (including JIT compilation)
>> Scores MATCH ✅
>> Structures MATCH EXACTLY ✅
```

## Usage

```python
import pandas as pd
from bnlearn.learning import hc
from bnlearn.score import score_network

# Load your data (pandas DataFrame with categorical columns)
df = pd.read_csv("data.csv")
for col in df.columns:
    df[col] = df[col].astype('category')

# Learn structure
bn = hc(df, score='bic')

# Inspect learned arcs
print(bn.arcs)

# Calculate score
print(score_network(bn, df, score_type='bic'))
```

## Running Tests

To verify the equivalence:

1.  **Install Dependencies**:
    ```bash
    pip install .
    ```

2.  **Run Equivalence Tests**:
    ```bash
    PYTHONPATH=src python3 -m unittest tests/test_equivalence.py
    ```

## Performance

This package leverages **JAX** for high-performance score calculation:
- **Vectorized Scoring**: Using `jax.vmap`, we evaluate all possible edge candidates in batches.
- **XLA Compilation**: Scoring kernels are JIT-compiled for machine-code execution.
- **Precision**: 64-bit precision is enabled (`jax_enable_x64`) for numerical parity with R.
- **Packaging**: Available on PyPI as `bnlearn-py`.
