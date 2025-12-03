import torch
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional

class SCMGenerator:
    """
    Generates synthetic Structural Causal Models (SCMs) with random DAGs 
    and mixed linear/non-linear mechanisms.
    
    This class is the foundation of the Causal Data Engineer (CDE) component.
    It simulates a causal system where variables depend on each other according 
    to a Directed Acyclic Graph (DAG).
    
    Attributes:
        num_vars (int): Total number of variables (nodes) in the causal graph.
        edge_prob (float): Probability of an edge existing between any two nodes (sparsity control).
        mechanism_type (str): Type of causal mechanism ('linear', 'nonlinear', or 'mixed').
        seed (int): Random seed for reproducibility.
        dag (nx.DiGraph): The generated Directed Acyclic Graph structure.
        topological_order (List[int]): A valid topological sort of the DAG, used for data generation order.
        mechanisms (Dict[int, callable]): Mapping from node index to its causal function.
        noise_std (Dict[int, float]): Mapping from node index to the standard deviation of its additive noise.
    """
    def __init__(
        self, 
        num_vars: int, 
        edge_prob: float = 0.1, 
        mechanism_type: str = 'mixed',
        seed: int = 42
    ):
        """
        Initializes the SCM Generator.
        
        Args:
            num_vars: Number of variables in the SCM.
            edge_prob: Probability of an edge existing (sparsity).
            mechanism_type: 'linear', 'nonlinear', or 'mixed'.
            seed: Random seed for reproducibility.
        """
        self.num_vars = num_vars
        self.edge_prob = edge_prob
        self.mechanism_type = mechanism_type
        self.seed = seed
        
        self.dag = None
        self.topological_order = None
        self.mechanisms = {} # Map node -> callable mechanism
        self.noise_std = {} # Map node -> noise std dev
        
        self._setup()

    def _setup(self):
        """
        Internal setup method to generate the DAG and assign mechanisms.
        
        1. Sets random seeds.
        2. Generates a random DAG using Erdos-Renyi model (enforcing acyclicity).
        3. Computes topological order for sequential data generation.
        4. Assigns a causal mechanism (function) and noise term to each node.
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # 1. Generate Random DAG
        # We use a lower triangular matrix approach or standard random graph + acyclic check.
        # Here we use networkx's gnp_random_graph and enforce direction u < v to guarantee acyclicity.
        G = nx.gnp_random_graph(self.num_vars, self.edge_prob, directed=True, seed=self.seed)
        
        # Create DAG explicitly with all nodes to ensure isolated nodes are included
        self.dag = nx.DiGraph()
        self.dag.add_nodes_from(range(self.num_vars))
        self.dag.add_edges_from([(u, v) for (u, v) in G.edges() if u < v])
        
        # Ensure the graph is valid and we have a topological sort
        self.topological_order = list(nx.topological_sort(self.dag))
        
        # 2. Assign Mechanisms
        # For each node, we look at its parents and create a function f(parents) -> node
        for node in self.dag.nodes():
            parents = list(self.dag.predecessors(node))
            self.mechanisms[node] = self._create_mechanism(parents)
            # Assign random noise standard deviation to simulate variable variance
            self.noise_std[node] = np.random.uniform(0.1, 1.0)

    def _create_mechanism(self, parents: List[int]):
        """
        Creates a causal mechanism for a node given its parents.
        
        Args:
            parents: List of parent node indices.
            
        Returns:
            A callable function that takes parent values and returns the node's expected value.
        """
        if not parents:
            # Root node: Has no parents, so its value is just noise (mean 0).
            return lambda x_parents: 0.0
        
        mech_type = self.mechanism_type
        if mech_type == 'mixed':
            # Randomly choose between linear and nonlinear for this specific node
            mech_type = np.random.choice(['linear', 'nonlinear'])
            
        if mech_type == 'linear':
            # Linear mechanism: Y = w1*X1 + w2*X2 + ... + noise
            weights = np.random.uniform(-1, 1, size=len(parents))
            return lambda x_parents: np.dot(x_parents, weights)
        
        elif mech_type == 'nonlinear':
            # Nonlinear mechanism: Modeled as a small neural network (MLP).
            # Y = w2 * tanh(w1 * X)
            # This captures complex interactions between parents.
            
            # Hidden layer weights (input_dim -> 4)
            weights1 = np.random.uniform(-1, 1, size=(len(parents), 4))
            # Output layer weights (4 -> 1)
            weights2 = np.random.uniform(-1, 1, size=4)
            
            def nonlinear_mech(x_parents):
                # x_parents shape: (batch, num_parents) or (num_parents,)
                # Apply tanh activation for non-linearity
                h = np.tanh(np.dot(x_parents, weights1))
                return np.dot(h, weights2)
            
            return nonlinear_mech
        
        return lambda x: 0.0

    def generate_data(self, n_samples: int, intervention: Optional[Dict[int, float]] = None) -> torch.Tensor:
        """
        Generates data from the SCM by sampling sequentially in topological order.
        
        This method supports both observational data generation and interventional data generation.
        
        Args:
            n_samples: Number of samples to generate.
            intervention: Dictionary mapping node_idx -> value for do-interventions.
                          e.g., {3: 5.0} means do(X3 = 5.0).
                          If a node is intervened on, its mechanism is replaced by the constant value.
        
        Returns:
            Tensor of shape (n_samples, num_vars) containing the generated data.
        """
        # Initialize data matrix
        data = np.zeros((n_samples, self.num_vars))
        
        # Iterate through nodes in topological order to ensure parents are generated before children
        for node in self.topological_order:
            if intervention and node in intervention:
                # Hard intervention: do(X = x)
                # The value is fixed, and the causal mechanism from parents is ignored.
                data[:, node] = intervention[node]
            else:
                # Observational generation
                parents = list(self.dag.predecessors(node))
                if not parents:
                    # Root node logic (just mean 0)
                    mean = 0.0
                else:
                    # Compute effect of parents
                    parent_vals = data[:, parents]
                    mean = self.mechanisms[node](parent_vals)
                
                # Add Gaussian noise
                noise = np.random.normal(0, self.noise_std[node], size=n_samples)
                data[:, node] = mean + noise
                
        return torch.tensor(data, dtype=torch.float32)

if __name__ == "__main__":
    # Quick test
    scm = SCMGenerator(num_vars=5, edge_prob=0.5, seed=42)
    print("Topological order:", scm.topological_order)
    print("Edges:", scm.dag.edges())
    
    # Baseline
    data_base = scm.generate_data(5)
    print("\nBaseline data (first 5 rows):\n", data_base)
    
    # Intervention do(X0 = 10)
    data_int = scm.generate_data(5, intervention={0: 10.0})
    print("\nIntervention do(X0=10):\n", data_int)
