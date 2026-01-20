
import unittest
import pandas as pd
import numpy as np
import os
from bnlearn.learning import hc
from bnlearn.score import score_network
from bnlearn.network import BayesianNetwork

class TestEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Paths
        cls.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        # Load Small Data
        cls.small_data_path = os.path.join(cls.data_dir, 'small_data.csv')
        if os.path.exists(cls.small_data_path):
            cls.small_data = pd.read_csv(cls.small_data_path)
            # Ensure categorical
            for col in cls.small_data.columns:
                cls.small_data[col] = cls.small_data[col].astype('category')
        else:
            cls.small_data = None
            
        # Load Large Data
        cls.large_data_path = os.path.join(cls.data_dir, 'large_data.csv')
        if os.path.exists(cls.large_data_path):
            cls.large_data = pd.read_csv(cls.large_data_path)
            for col in cls.large_data.columns:
                cls.large_data[col] = cls.large_data[col].astype('category')
        else:
            cls.large_data = None
            
    def _read_r_arcs(self, filename):
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        arcs = set()
        for _, row in df.iterrows():
            arcs.add((row['from'], row['to']))
        return arcs

    def _read_r_score(self, filename):
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return float(f.read().strip())

    def test_small_network_structure(self):
        if self.small_data is None:
            self.skipTest("Small data not generated")
            
        # R result
        r_arcs = self._read_r_arcs("small_arcs_R.csv")
        if r_arcs is None:
            self.skipTest("R arcs not generated")
            
        # Python HC
        bn = hc(self.small_data, score='bic', max_iter=100)
        py_arcs = set(zip(bn.arcs['from'], bn.arcs['to']))
        
        # Compare
        self.assertEqual(r_arcs, py_arcs, f"Structure mismatch.\nR: {r_arcs}\nPy: {py_arcs}")

    def test_small_network_score(self):
        if self.small_data is None:
            self.skipTest("Small data not generated")
            
        r_score = self._read_r_score("small_score_R.txt")
        if r_score is None:
            self.skipTest("R score not generated")
            
        # Reconstruct network from R arcs to check score calculation first
        r_arcs_df = pd.read_csv(os.path.join(self.data_dir, "small_arcs_R.csv"))
        nodes = list(self.small_data.columns)
        
        # Build BN manually
        from bnlearn.network import BayesianNetwork
        bn = BayesianNetwork(nodes, r_arcs_df)
        
        # Calculate score using Python implementation
        py_score = score_network(bn, self.small_data, score_type='bic')
        
        # Compare scores
        # Allow small floating point difference
        self.assertAlmostEqual(r_score, py_score, places=4, msg="Score calculation mismatch on R structure")

    def test_large_network_structure(self):
        if self.large_data is None:
            self.skipTest("Large data not generated")
            
        r_arcs = self._read_r_arcs("large_arcs_R.csv")
        if r_arcs is None:
            self.skipTest("R arcs not generated")
            
        bn = hc(self.large_data, score='bic', max_iter=200)
        py_arcs = set(zip(bn.arcs['from'], bn.arcs['to']))
        
        self.assertEqual(r_arcs, py_arcs, f"Structure mismatch (Large).\nR: {r_arcs}\nPy: {py_arcs}")

if __name__ == '__main__':
    unittest.main()
