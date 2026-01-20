
import numpy as np
import pandas as pd

def _get_cardinality(data, col):
    """
    Get cardinality of a column. 
    If categorical, use number of categories.
    Otherwise use number of unique values.
    """
    if isinstance(data[col].dtype, pd.CategoricalDtype):
        return len(data[col].cat.categories)
    else:
        return data[col].nunique()

def loglik_dnode_root(data, node, compute_params=False):
    """
    Compute log-likelihood for a discrete root node.
    
    Args:
        data (pd.DataFrame): The dataset.
        node (str): The node name.
        compute_params (bool): Whether to return parameter count.
        
    Returns:
        float: Log-likelihood.
        int (optional): Number of parameters.
    """
    counts = data[node].value_counts()
    n_obs = len(data)
    n_i = counts.values
    
    # sum(n_i * log(n_i / N))
    # = sum(n_i * log(n_i)) - sum(n_i * log(N))
    # = sum(n_i * log(n_i)) - N * log(N)
    
    # Handle 0 counts? value_counts doesn't return 0s usually.
    # But if we had 0s, 0*log(0) -> 0.
    
    term1 = np.sum(n_i * np.log(n_i))
    ll = term1 - n_obs * np.log(n_obs)
    
    if compute_params:
        n_levels = _get_cardinality(data, node)
        n_params = n_levels - 1
        return ll, n_params
        
    return ll

def loglik_dnode_parents(data, node, parents, compute_params=False):
    """
    Compute log-likelihood for a discrete node with parents.
    
    Args:
        data (pd.DataFrame): The dataset.
        node (str): The node name.
        parents (list): List of parent names.
        compute_params (bool): Whether to return parameter count.
        
    Returns:
        float: Log-likelihood.
        int (optional): Number of parameters.
    """
    # LL = sum(n_ij * log(n_ij / n_j))
    #    = sum(n_ij * log(n_ij)) - sum(n_j * log(n_j))
    
    # We use crosstab to get joint counts
    # Rows: Node, Cols: Parents
    
    if not parents:
        return loglik_dnode_root(data, node, compute_params)

    # To group by multiple parents correctly and handle unobserved configurations automatically,
    # pandas crosstab is useful.
    
    # Optimization: using groupby + value_counts might be faster or similar.
    # crosstab returns a dataframe/series.
    
    # We essentially need the counts of (node, *parents)
    # and counts of (*parents)
    
    cols = [node] + parents
    
    # Count frequencies of all observed configurations (x, p1, p2...)
    # This corresponds to n_ij
    joint_counts = data.value_counts(subset=cols, sort=False)
    n_ij = joint_counts.values
    
    # Count frequencies of parents configurations (p1, p2...)
    # This corresponds to n_j
    # We can aggregate n_ij by summing over node?
    
    # But data.value_counts(subset=parents) is safer/direct
    parent_counts = data.value_counts(subset=parents, sort=False)
    n_j = parent_counts.values
    
    # Note: We must ensure alignment isn't an issue.
    # The formula sum(n_ij * log(n_ij)) - sum(n_j * log(n_j)) works 
    # because sum(n_ij * log(n_j)) over i is n_j * log(n_j).
    # So we don't need to align i with j specifically, just sum over all i,j and all j.
    
    term1 = np.sum(n_ij * np.log(n_ij))
    term2 = np.sum(n_j * np.log(n_j))
    
    ll = term1 - term2
    
    if compute_params:
        node_levels = _get_cardinality(data, node)
        parent_levels = 1
        for p in parents:
            parent_levels *= _get_cardinality(data, p)
            
        n_params = (node_levels - 1) * parent_levels
        return ll, n_params
        
    return ll
