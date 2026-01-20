
from .hill_climbing import HillClimbing

def hc(data, score='bic', start=None, whitelist=None, blacklist=None, max_iter=float('inf'), restart=0, perturb=1, debug=False, **kwargs):
    """
    Hill Climbing structure learning algorithm.
    
    Args:
        data (pd.DataFrame): The dataset.
        score (str): Score type ('bic', 'aic', 'loglik').
        start (BayesianNetwork, optional): Starting network.
        whitelist (pd.DataFrame, optional): Whitelisted arcs.
        blacklist (pd.DataFrame, optional): Blacklisted arcs.
        max_iter (int): Maximum iterations.
        restart (int): Number of restarts.
        perturb (int): Number of perturbation steps.
        debug (bool): Debug mode.
        **kwargs: Additional arguments for scoring.
    
    Returns:
        BayesianNetwork: Learned network.
    """
    algorithm = HillClimbing(data, score_type=score, **kwargs)
    return algorithm.learn(start, whitelist, blacklist, max_iter, restart, perturb, debug)
