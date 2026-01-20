
import numpy as np
from .discrete import loglik_dnode_parents

def score_node(node, parents, data, score_type='bic', **kwargs):
    """
    Compute score for a single node given its parents.
    
    Args:
        node (str): The node name.
        parents (list): List of parent names.
        data (pd.DataFrame): The data.
        score_type (str): Score type (bic, aic, loglik).
        **kwargs: Extra arguments (k, etc).
        
    Returns:
        float: The score.
    """
    n_obs = len(data)
    
    # Dispatch based on data type (Discrete vs Gaussian)
    # For now assuming discrete if not specified
    # TODO: Add check for Gaussian
    
    if score_type in ['loglik', 'aic', 'bic']:
        ll, nparams = loglik_dnode_parents(data, node, parents, compute_params=True)
        
        if score_type == 'loglik':
            return ll
        elif score_type == 'aic':
            k = kwargs.get('k', 1.0)
            return ll - k * nparams
        elif score_type == 'bic':
            k = kwargs.get('k', np.log(n_obs) / 2.0)
            return ll - k * nparams
            
    raise ValueError(f"Unknown score type: {score_type}")

def score_network(network, data, score_type='bic', **kwargs):
    """
    Compute total score of the network.
    """
    total_score = 0
    
    # Iterate over all nodes
    for node in network.nodes:
        # Get parents
        # Check if network has structure info
        if hasattr(network, 'nodes_data') and node in network.nodes_data:
            parents = network.nodes_data[node]['parents']
        else:
            # Fallback or error?
            # If network is just nodes list? No, it should be BayesianNetwork object
            # Maybe recreate partial structure?
            parents = []
            
        total_score += score_node(node, parents, data, score_type, **kwargs)
        
    return total_score
