
import numpy as np
import pandas as pd
import random
import copy
from ..network import BayesianNetwork
from ..score.api import score_network, score_node
from ..utils.graph import arcs2amat, amat2arcs

try:
    import jax
    import jax.numpy as jnp
    from ..score.api import score_node # Add this if needed by other logic
    from ..score.jax_discrete import (
        jax_loglik_discrete, 
        jax_bic_discrete, 
        jax_bic_add_batch,
        BATCH_BUFFER_SIZE
    )
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

class HillClimbing:
    def __init__(self, data, score_type='bic', **kwargs):
        self.data = data
        self.score_type = score_type
        self.kwargs = kwargs
        self.nodes = list(data.columns)
        self.n_nodes = len(self.nodes)
        self.score_cache = {} # (node, tuple(sorted(parents))) -> score
        
        # Prepare JAX data if available
        self.use_jax = HAS_JAX and score_type in ['bic', 'loglik']
        if self.use_jax:
            # Ensure categorical and get codes
            self.jax_data = jnp.array([data[col].cat.codes.values if hasattr(data[col], 'cat') else data[col].values for col in self.nodes]).T
            self.jax_cardinalities = tuple([
                len(data[col].cat.categories) if hasattr(data[col], 'cat') else data[col].nunique()
                for col in self.nodes
            ])
            self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
            self.n_obs = len(data)
            self.k_bic = np.log(self.n_obs) / 2.0 if score_type == 'bic' else 0.0

    def learn(self, start=None, whitelist=None, blacklist=None, max_iter=float('inf'), restart=0, perturb=1, debug=False):
        
        # Initialize
        if start is None:
            current_nw = BayesianNetwork(self.nodes)
        else:
            current_nw = copy.deepcopy(start)
            
        # Apply whitelist
        if whitelist is not None and not whitelist.empty:
            current_nw.arcs = pd.concat([current_nw.arcs, whitelist], ignore_index=True).drop_duplicates()
            current_nw._cache_structure()

        # Initial score
        current_score = self._score_network(current_nw)
        
        best_global_nw = copy.deepcopy(current_nw)
        best_global_score = current_score
        
        restart_counter = restart
        iter_count = 0
        tol = np.sqrt(np.finfo(float).eps) 

        while True: 
            while iter_count < max_iter:
                if debug:
                    print(f"Iteration {iter_count}, Score: {current_score}")
                    
                best_op = None
                best_delta = 0 
                
                # Precompute/cache current node scores for efficiency?
                # Actually, individual score deltas are:
                # delta = new_score - old_score.
                # old_score is part of current_score. 
                # But calculating delta requires score(node | old_parents).
                # This should be cached.
                
                # Phase 1: Additions
                all_add_deltas = {}
                if self.use_jax and self.score_type == 'bic':
                    # Batch by v to use JAX efficiently
                    for v in self.nodes:
                        candidates = [u for u in self.nodes if u != v and \
                                     not self._is_blacklisted(u, v, blacklist) and \
                                     not self._has_arc(current_nw, u, v)]
                        if candidates:
                            deltas = self._get_batch_score_add(v, current_nw, candidates)
                            for i, u in enumerate(candidates):
                                all_add_deltas[(u, v)] = float(deltas[i])
                
                # Search in R order: u then v
                for u in self.nodes:
                    for v in self.nodes:
                        if u == v: continue
                        d = 0
                        if (u, v) in all_add_deltas:
                            d = all_add_deltas[(u, v)]
                        elif not self._is_blacklisted(u, v, blacklist) and not self._has_arc(current_nw, u, v):
                            # Fallback if not batched (e.g. non-BIC score)
                            d = self._score_delta(current_nw, u, v, 'add')
                        else:
                            continue
                            
                        if d - best_delta > tol:
                            if not self._creates_cycle(current_nw, u, v):
                                best_delta = d
                                best_op = ('add', u, v)

                # Phase 2: Deletions
                for u in self.nodes:
                    for v in self.nodes:
                        if u == v: continue
                        if self._is_whitelisted(u, v, whitelist): continue
                        if not self._has_arc(current_nw, u, v): continue
                            
                        delta = self._score_delta(current_nw, u, v, 'delete')
                        
                        if delta - best_delta > tol:
                            best_delta = delta
                            best_op = ('delete', u, v)

                # Phase 3: Reversals
                for u in self.nodes:
                    for v in self.nodes:
                        if u == v: continue
                        if self._is_whitelisted(u, v, whitelist): continue
                        if self._is_whitelisted(v, u, whitelist): continue
                        if not self._has_arc(current_nw, u, v): continue
                        if self._is_blacklisted(v, u, blacklist): continue

                        delta = self._score_delta(current_nw, u, v, 'reverse')
                        
                        if delta - best_delta > tol:
                            if not self._creates_cycle(current_nw, v, u, ignore_arc=(u, v)):
                                best_delta = delta
                                best_op = ('reverse', u, v)

                # Phase 2: Deletions
                for u in self.nodes:
                    for v in self.nodes:
                        if u == v: continue
                        if self._is_whitelisted(u, v, whitelist): continue
                        if not self._has_arc(current_nw, u, v): continue
                            
                        delta = self._score_delta(current_nw, u, v, 'delete')
                        
                        if delta - best_delta > tol:
                            best_delta = delta
                            best_op = ('delete', u, v)

                # Phase 3: Reversals
                for u in self.nodes:
                    for v in self.nodes:
                        if u == v: continue
                        if self._is_whitelisted(u, v, whitelist): continue
                        if self._is_whitelisted(v, u, whitelist): continue
                        if not self._has_arc(current_nw, u, v): continue
                        if self._is_blacklisted(v, u, blacklist): continue

                        delta = self._score_delta(current_nw, u, v, 'reverse')
                        
                        if delta - best_delta > tol:
                            if not self._creates_cycle(current_nw, v, u, ignore_arc=(u, v)):
                                best_delta = delta
                                best_op = ('reverse', u, v)

                # Apply best op
                if best_op: 
                    op, u, v = best_op
                    self._apply_op(current_nw, op, u, v)
                    current_score += best_delta
                    iter_count += 1
                else:
                    break
            
            if current_score > best_global_score:
                best_global_nw = copy.deepcopy(current_nw)
                best_global_score = current_score
            
            if restart_counter > 0:
                restart_counter -= 1
                if debug:
                    print(f"Restarting... {restart_counter} left. Best score: {best_global_score}")
                current_nw = self._perturb(best_global_nw, perturb, whitelist, blacklist)
                current_score = self._score_network(current_nw)
                if iter_count >= max_iter: break
            else:
                break
                
        return best_global_nw

    def _get_batch_score_add(self, v_name, network, candidates):
        if not self.use_jax or self.score_type != 'bic':
             # Fallback: manual calculation (should not be called if use_jax is False)
             return [self._score_delta(network, u, v_name, 'add') for u in candidates]

        v_idx = self.node_to_idx[v_name]
        v_parents = network.nodes_data[v_name]['parents']
        current_parent_indices = tuple(sorted([self.node_to_idx[p] for p in v_parents]))
        
        # Check if stride is too large for batch buffer
        current_stride = 1
        for idx in current_parent_indices:
            current_stride *= self.jax_cardinalities[idx]
        
        node_card = self.jax_cardinalities[v_idx]
        
        # Safety check for buffer size
        max_candidate_card = max([self.jax_cardinalities[self.node_to_idx[u]] for u in candidates])
        if current_stride * max_candidate_card * node_card > BATCH_BUFFER_SIZE:
             return [self._score_delta(network, u, v_name, 'add') for u in candidates]
        
        candidate_indices = jnp.array([self.node_to_idx[u] for u in candidates])
        current_node_score = self._get_score_node(v_name, v_parents)
        
        batch_scores = jax_bic_add_batch(
            self.jax_data, 
            v_idx, 
            current_parent_indices, 
            self.jax_cardinalities, 
            candidate_indices, 
            self.k_bic
        )
        
        if batch_scores is None:
             return [self._score_delta(network, u, v_name, 'add') for u in candidates]
             
        return np.array(batch_scores) - current_node_score

    def _get_score_node(self, node, parents):
        # Sort parents tuple for caching
        parents_tuple = tuple(sorted(parents))
        key = (node, parents_tuple)
        if key in self.score_cache:
            return self.score_cache[key]
        
        if self.use_jax:
            node_idx = self.node_to_idx[node]
            parent_indices = tuple(sorted([self.node_to_idx[p] for p in parents_tuple]))
            
            if self.score_type == 'bic':
                val = float(jax_bic_discrete(self.jax_data, node_idx, parent_indices, self.jax_cardinalities, self.k_bic))
            elif self.score_type == 'loglik':
                val = float(jax_loglik_discrete(self.jax_data, node_idx, parent_indices, self.jax_cardinalities))
            else:
                # Fallback for other scores not yet in JAX
                val = score_node(node, list(parents_tuple), self.data, self.score_type, **self.kwargs)
        else:
            val = score_node(node, list(parents_tuple), self.data, self.score_type, **self.kwargs)
            
        self.score_cache[key] = val
        return val

    def _is_blacklisted(self, u, v, blacklist):
        if blacklist is None or blacklist.empty: return False
        return ((blacklist['from'] == u) & (blacklist['to'] == v)).any()

    def _is_whitelisted(self, u, v, whitelist):
        if whitelist is None or whitelist.empty: return False
        return ((whitelist['from'] == u) & (whitelist['to'] == v)).any()

    def _has_arc(self, network, u, v):
        return ((network.arcs['from'] == u) & (network.arcs['to'] == v)).any()

    def _creates_cycle(self, network, u, v, ignore_arc=None):
        adj = {n: [] for n in self.nodes}
        for _, row in network.arcs.iterrows():
            if ignore_arc and row['from'] == ignore_arc[0] and row['to'] == ignore_arc[1]:
                continue
            adj[row['from']].append(row['to'])
        
        # DFS from v to u
        stack = [v]
        visited = set()
        while stack:
            curr = stack.pop()
            if curr == u: return True
            if curr not in visited:
                visited.add(curr)
                stack.extend(adj[curr])
        return False

    def _score_delta(self, network, u, v, op):
        if op == 'add':
            parents = network.nodes_data[v]['parents']
            parents_new = parents + [u]
            s_old = self._get_score_node(v, parents)
            s_new = self._get_score_node(v, parents_new)
            return s_new - s_old
            
        elif op == 'delete':
            parents = network.nodes_data[v]['parents']
            parents_new = [p for p in parents if p != u]
            s_old = self._get_score_node(v, parents)
            s_new = self._get_score_node(v, parents_new)
            return s_new - s_old
            
        elif op == 'reverse':
            parents_v = network.nodes_data[v]['parents']
            parents_v_new = [p for p in parents_v if p != u]
            parents_u = network.nodes_data[u]['parents']
            parents_u_new = parents_u + [v]
            
            sv_old = self._get_score_node(v, parents_v)
            sv_new = self._get_score_node(v, parents_v_new)
            
            su_old = self._get_score_node(u, parents_u)
            su_new = self._get_score_node(u, parents_u_new)
            
            return (sv_new - sv_old) + (su_new - su_old)
        return 0

    def _apply_op(self, network, op, u, v):
        if op == 'add':
            new_arc = pd.DataFrame({'from': [u], 'to': [v]})
            network.arcs = pd.concat([network.arcs, new_arc], ignore_index=True)
        elif op == 'delete':
            network.arcs = network.arcs[~((network.arcs['from'] == u) & (network.arcs['to'] == v))]
        elif op == 'reverse':
            network.arcs = network.arcs[~((network.arcs['from'] == u) & (network.arcs['to'] == v))]
            new_arc = pd.DataFrame({'from': [v], 'to': [u]})
            network.arcs = pd.concat([network.arcs, new_arc], ignore_index=True)
            
        network._cache_structure()

    def _score_network(self, network):
        # Use cached node scores if available
        # But for full network score, just sum cached nodes?
        # score_network function might not use self.score_cache.
        # Let's use our helper
        total = 0
        for node in self.nodes:
            if node in network.nodes_data:
                parents = network.nodes_data[node]['parents']
            else:
                parents = []
            total += self._get_score_node(node, parents)
        return total

    def _perturb(self, network, iter, whitelist, blacklist):
        new_nw = copy.deepcopy(network)
        ops_performed = 0
        attempts = 0
        max_attempts = iter * 100
        
        while ops_performed < iter and attempts < max_attempts:
            attempts += 1
            op = random.choice(['add', 'delete', 'reverse'])
            u, v = random.sample(self.nodes, 2)
            has_arc = self._has_arc(new_nw, u, v)
            
            if op == 'add':
                if not has_arc and not self._is_blacklisted(u, v, blacklist):
                    if not self._creates_cycle(new_nw, u, v):
                        self._apply_op(new_nw, 'add', u, v)
                        ops_performed += 1
                        
            elif op == 'delete':
                if has_arc and not self._is_whitelisted(u, v, whitelist):
                     self._apply_op(new_nw, 'delete', u, v)
                     ops_performed += 1
            
            elif op == 'reverse':
                if has_arc and not self._is_whitelisted(u, v, whitelist) and not self._is_whitelisted(v, u, whitelist):
                    if not self._is_blacklisted(v, u, blacklist):
                        if not self._creates_cycle(new_nw, v, u, ignore_arc=(u, v)):
                            self._apply_op(new_nw, 'reverse', u, v)
                            ops_performed += 1
                            
        return new_nw
