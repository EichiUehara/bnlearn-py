
import pandas as pd
import numpy as np
from .utils.graph import arcs2amat

class BayesianNetwork:
    def __init__(self, nodes, arcs=None):
        self.nodes = list(nodes)
        if arcs is None:
            self.arcs = pd.DataFrame(columns=['from', 'to'])
        else:
            self.arcs = arcs
        self.nodes_data = {}
        self.learning = {}
        self._cache_structure()

    def _cache_structure(self):
        amat = arcs2amat(self.arcs, self.nodes)
        amat_values = amat.values
        n_nodes = len(self.nodes)
        
        # Mapping from index to name
        idx_to_name = {i: name for i, name in enumerate(self.nodes)}
        name_to_idx = {name: i for i, name in enumerate(self.nodes)}
        
        for cur in range(n_nodes):
            status = np.zeros(n_nodes, dtype=int)
            # 0: None, 1: Blanket, 2: Neighbour, 3: Parent, 4: Child
            
            # Identify parents, children, neighbors
            for i in range(n_nodes):
                if i == cur:
                    continue
                
                # Check connection cur -> i
                is_out = amat_values[cur, i] == 1
                # Check connection i -> cur
                is_in = amat_values[i, cur] == 1
                
                if is_out and not is_in: # cur -> i (Child)
                    status[i] = 4 # Child
                    
                    # Check spouses (parents of child i that are not cur)
                    for j in range(n_nodes):
                        if j == cur:
                            continue
                        # j -> i
                        if amat_values[j, i] == 1 and amat_values[i, j] == 0:
                            # If j is not already a neighbor/parent/child of cur?
                            # The C code says: don't mark a neighbour as in the markov blanket (it already is)
                            # But status check: if status[j] <= 1, set to 1.
                            if status[j] <= 1:
                                status[j] = 1 # Blanket (Spouse)
                
                elif is_out and is_in: # cur -- i (Neighbour / Undirected)
                     status[i] = 2 # Neighbour
                
                elif not is_out and is_in: # i -> cur (Parent)
                    status[i] = 3 # Parent
            
            # Collect lists
            parents = []
            children = []
            nbr = []
            mb = []
            
            for i in range(n_nodes):
                s = status[i]
                name = idx_to_name[i]
                
                if s == 4: # Child
                    children.append(name)
                    nbr.append(name)
                    mb.append(name)
                elif s == 3: # Parent
                    parents.append(name)
                    nbr.append(name)
                    mb.append(name)
                elif s == 2: # Neighbour
                    nbr.append(name)
                    mb.append(name)
                elif s == 1: # Blanket
                    mb.append(name)
            
            self.nodes_data[self.nodes[cur]] = {
                'mb': mb,
                'nbr': nbr,
                'parents': parents,
                'children': children
            }
            
    def __repr__(self):
        return f"BayesianNetwork({len(self.nodes)} nodes, {len(self.arcs)} arcs)"
