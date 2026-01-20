
# bnlearn-py

A Python port of the popular R package `bnlearn` for Bayesian Network structure learning.

This project aims to provide a **functionally identical** implementation of the `bnlearn` Hill Climbing (`hc`) algorithm, ensuring exactly the same results as the R implementation on the same datasets.

## Implementation Details

The implementation mirrors the logic of `bnlearn`'s C backend:
-   **Algorithm**: Hill Climbing with restarts and perturbation.
-   **Score**: Discrete BIC (Bayesian Information Criterion), AIC, and Log-Likelihood.
-   **Search Strategy**: 
    -   Greedy search with specific operation ordering: `Add` -> `Delete` -> `Reverse`.
    -   Floating point tolerance matching R's machine epsilon behavior.
    -   cycle detection and invalid operation filtering.

## Validation Results

We have conducting intensive verification by comparing the outputs of this Python implementation against the original R `bnlearn` package (v4.9) using identical datasets.

### Method
1.  **Reference Generation**: Use R to generate synthetic data (Gaussian/Discrete) and learn a network using `hc(data, score="bic")`. Export the learned arcs and the final score.
2.  **Reproduction**: Load the same data in Python, run the ported `hc` function.
3.  **Comparison**: Assert exact equality of the arc set and floating-point equality of the score.

### Summary

| Dataset | Type | Nodes | Observations | Structure Match | Score Match |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Small Test** | Discrete | 5 | 1000 | ✅ **Exact** | ✅ **Exact** |
| **Alarm (Subset)** | Discrete | 37 | 2000 | ✅ **Exact** | ✅ **Exact** |

*(Note: "Exact" structure match means the sets of directed edges are identical. Score match is verified to 4+ decimal places).*

### Detailed Verification Logs

```text
--- Comparing Results for: Small Network (5 Nodes) ---
Data shape: (1000, 5)
R Score: -2460.827068
R Arcs count: 5
Python Score: -2460.827068
Python Arcs count: 5
Python Execution Time: 0.0605s
Score Difference: 9.549694e-12
>> Scores MATCH ✅
>> Structures MATCH EXACTLY ✅

--- Comparing Results for: Large Network (37 Nodes, Subset) ---
Data shape: (2000, 37)
R Score: -23176.667311
R Arcs count: 47
Python Score: -23176.667311
Python Arcs count: 47
Python Execution Time: 19.7716s
Score Difference: 3.637979e-11
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

To verify the equivalence yourself:

1.  **Install Dependencies**:
    ```bash
    pip install pandas numpy scipy
    ```
    *(You need R installed to regenerate reference data, but pre-generated data is included)*.

2.  **Generate Reference Data (Optional)**:
    ```bash
    Rscript tests/generate_reference.R
    ```

3.  **Run Equivalence Tests**:
    ```bash
    python3 -m unittest tests/test_equivalence.py
    ```

## Performance
While strictly equivalent, this pure Python implementation is currently slower than the highly optimized C backend of `bnlearn`. It is intended for research, education, and environments where R is not available, or where Python-native integration is prioritized over raw speed for massive datasets.

