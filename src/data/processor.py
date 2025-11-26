import torch
import numpy as np
from typing import Dict, Tuple, Optional

class DataProcessor:
    """
    Handles data standardization and input preparation for the ISD-CP model.
    
    This class implements the "Fixed Reference Frame" philosophy.
    Crucially, it computes statistics (mean, std) ONLY from the baseline (unintervened) data.
    These statistics are then used to standardize both baseline and intervened data.
    This ensures that the model perceives the magnitude of an intervention relative to the 
    system's natural variability.
    
    Attributes:
        mean (torch.Tensor): Mean of each variable in the baseline dataset.
        std (torch.Tensor): Standard deviation of each variable in the baseline dataset.
        epsilon (float): Small constant to prevent division by zero.
    """
    def __init__(self):
        self.mean = None
        self.std = None
        self.epsilon = 1e-8

    def fit(self, baseline_data: torch.Tensor):
        """
        Computes mean and std from baseline data.
        
        Args:
            baseline_data: Tensor of shape (n_samples, num_vars) containing observational data.
        """
        self.mean = baseline_data.mean(dim=0)
        self.std = baseline_data.std(dim=0) + self.epsilon

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Standardizes data using the fitted mean and std.
        
        z = (x - mu) / sigma
        
        Args:
            data: Tensor of shape (n_samples, num_vars).
            
        Returns:
            Standardized tensor.
            
        Raises:
            ValueError: If the processor has not been fitted yet.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Processor must be fitted before transform.")
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Reverts standardization to get back original units.
        
        x = z * sigma + mu
        
        Args:
            data: Standardized tensor.
            
        Returns:
            Tensor in original units.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Processor must be fitted before inverse_transform.")
        return data * self.std + self.mean

    def prepare_input(
        self, 
        data: torch.Tensor, 
        mask: torch.Tensor, 
        value: float,
        node_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Prepares the input dictionary for the model, including standardization.
        
        This method constructs the full input package required by the CausalTransformer.
        It standardizes the input features (baseline state) and the intervention value.
        
        Args:
            data: The observed baseline data tensor (n_samples, num_vars).
            mask: Binary mask (num_vars,) indicating the intervened node.
            value: The raw intervention value (float).
            node_idx: Index of intervened node.
            
        Returns:
            Dictionary containing:
                - 'x': Standardized features (n_samples, num_vars).
                - 'mask': Binary mask expanded to batch size (n_samples, num_vars).
                - 'value': Standardized intervention value tensor (n_samples, num_vars). 
                           This tensor is all zeros EXCEPT at the intervened node index, 
                           where it holds the standardized intervention magnitude (z-score).
        """
        # Standardize features (S_baseline)
        x_std = self.transform(data)
        
        # Standardize intervention value
        # We need to standardize the value using the stats of the intervened variable
        # z_val = (val - mu_i) / sigma_i
        val_std = (value - self.mean[node_idx]) / self.std[node_idx]
        
        n_samples = data.shape[0]
        num_vars = data.shape[1]
        
        # Expand mask to batch dimension
        mask_batch = mask.unsqueeze(0).expand(n_samples, -1)
        
        # Create value tensor (zeros everywhere except intervened node)
        # This represents the "Shock Magnitude" vector M_value
        value_batch = torch.zeros_like(x_std)
        value_batch[:, node_idx] = val_std
        
        return {
            'x': x_std,
            'mask': mask_batch,
            'value': value_batch
        }

if __name__ == "__main__":
    # Test
    processor = DataProcessor()
    
    # Fake baseline
    baseline = torch.randn(100, 5) * 2 + 10 # Mean ~10, Std ~2
    processor.fit(baseline)
    print(f"Mean: {processor.mean}")
    print(f"Std: {processor.std}")
    
    # Transform
    sample = torch.tensor([[10.0, 10.0, 10.0, 10.0, 10.0]])
    std_sample = processor.transform(sample)
    print(f"Standardized sample: {std_sample}")
    
    # Prepare input
    mask = torch.zeros(5)
    mask[0] = 1.0
    inp = processor.prepare_input(sample, mask, value=12.0, node_idx=0)
    print("Input keys:", inp.keys())
    print("Value tensor:", inp['value'])
