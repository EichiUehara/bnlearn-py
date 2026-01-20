
import numpy as np
import pandas as pd

def arcs2amat(arcs, nodes):
    """
    Convert an arc set to an adjacency matrix.
    
    Args:
        arcs (pd.DataFrame): DataFrame with columns 'from' and 'to'.
        nodes (list): List of node names.
        
    Returns:
        pd.DataFrame: Adjacency matrix (rows=from, cols=to) with 0/1.
    """
    n_nodes = len(nodes)
    amat = pd.DataFrame(0, index=nodes, columns=nodes, dtype=int)
    
    if arcs is None or arcs.empty:
        return amat

    for _, row in arcs.iterrows():
        u = row['from']
        v = row['to']
        if u in nodes and v in nodes:
            amat.loc[u, v] = 1
            
    return amat

def amat2arcs(amat, nodes=None):
    """
    Convert an adjacency matrix to an arc set.
    
    Args:
        amat (pd.DataFrame or np.ndarray): Adjacency matrix.
        nodes (list, optional): Node names if amat is ndarray.
        
    Returns:
        pd.DataFrame: DataFrame with columns 'from' and 'to'.
    """
    if isinstance(amat, pd.DataFrame):
        nodes = amat.columns.tolist()
        mat = amat.values
    else:
        if nodes is None:
            raise ValueError("nodes must be provided if amat is numpy array")
        mat = amat
        
    arcs = []
    rows, cols = mat.shape
    for i in range(rows):
        for j in range(cols):
            if mat[i, j] == 1:
                arcs.append({'from': nodes[i], 'to': nodes[j]})
                
    if not arcs:
        return pd.DataFrame(columns=['from', 'to'])
        
    return pd.DataFrame(arcs)
