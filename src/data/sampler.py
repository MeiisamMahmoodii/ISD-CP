import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from .scm_generator import SCMGenerator

class DataSampler:
    """
    Handles sampling of baseline and interventional data from an SCM.
    
    This class acts as the interface between the SCMGenerator and the dataset creation process.
    It abstracts away the details of calling generate_data with specific intervention dictionaries.
    
    Attributes:
        scm (SCMGenerator): The SCM instance to sample from.
    """
    def __init__(self, scm: SCMGenerator):
        """
        Args:
            scm: An initialized SCMGenerator instance.
        """
        self.scm = scm

    def sample_baseline(self, n_samples: int) -> torch.Tensor:
        """
        Generates baseline (observational) data.
        
        This data represents the system in its natural state, without any external interventions.
        It is used to calculate the mean and standard deviation for the Fixed Reference Frame.
        
        Args:
            n_samples: Number of samples to generate.
            
        Returns:
            Tensor of shape (n_samples, num_vars).
        """
        return self.scm.generate_data(n_samples)

    def sample_interventions(
        self, 
        n_samples: int, 
        intervention_nodes: List[int], 
        intervention_values: List[float]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generates interventional data for specified nodes and values.
        
        This method systematically performs interventions on the specified nodes with the specified values.
        For each combination of (node, value), it generates a separate dataset.
        
        Args:
            n_samples: Number of samples per intervention.
            intervention_nodes: List of node indices to intervene on.
            intervention_values: List of values to set the intervened nodes to (e.g., [-5, 0, 5]).
            
        Returns:
            List of dictionaries, where each dictionary represents a specific intervention scenario:
                - 'data': The intervened data tensor (n_samples, num_vars).
                - 'mask': Binary mask tensor (num_vars,) where 1.0 indicates the intervened node.
                - 'value': The raw intervention value (float).
                - 'node_idx': The index of the intervened node (int).
        """
        results = []
        
        for node_idx in intervention_nodes:
            # Create a binary mask vector for this intervention
            # This mask tells the model WHICH variable was intervened on.
            mask = torch.zeros(self.scm.num_vars)
            mask[node_idx] = 1.0
            
            for val in intervention_values:
                # Generate data for this specific intervention: do(X_node = val)
                data = self.scm.generate_data(n_samples, intervention={node_idx: val})
                
                results.append({
                    'data': data,
                    'mask': mask,
                    'value': val,
                    'node_idx': node_idx
                })
                
        return results

if __name__ == "__main__":
    # Test
    scm = SCMGenerator(num_vars=5, seed=42)
    sampler = DataSampler(scm)
    
    # Baseline
    base_data = sampler.sample_baseline(10)
    print("Baseline shape:", base_data.shape)
    
    # Interventions
    int_nodes = [0, 2]
    int_vals = [-2.0, 2.0]
    int_data_list = sampler.sample_interventions(10, int_nodes, int_vals)
    
    print(f"Generated {len(int_data_list)} interventional datasets.")
    for item in int_data_list:
        print(f"Node: {item['node_idx']}, Val: {item['value']}, Mean: {item['data'][:, item['node_idx']].mean():.2f}")
