import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np

class CausalDataset(Dataset):
    """
    PyTorch Dataset for loading pre-generated causal data.
    
    This dataset is designed to handle large-scale data stored in multiple .pt files.
    Each file represents a chunk of data (e.g., all data from one SCM).
    
    The dataset returns a LIST of batches (dictionaries) because each file contains
    multiple intervention scenarios.
    
    Attributes:
        files (List[str]): List of paths to the .pt data files.
    """
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Directory containing the .pt files.
        """
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        if not self.files:
            print(f"Warning: No .pt files found in {data_dir}")

    def __len__(self):
        """Returns the number of files (chunks) in the dataset."""
        return len(self.files)

    def __getitem__(self, idx):
        """
        Loads a single file from disk.
        
        Args:
            idx: Index of the file to load.
            
        Returns:
            A list of dictionaries. Each dictionary contains a batch of data for a specific intervention.
            Keys in each dict: 'x', 'mask', 'value', 'target'.
        """
        # Load the file
        # Each file contains a list of batches (one batch per intervention type)
        data = torch.load(self.files[idx])
        return data

def collate_fn(batch):
    """
    Custom collate function.
    
    Since __getitem__ returns a list of batches (not a single sample), 
    the default collate would try to stack these lists, which is not what we want.
    We want to keep the structure as is, or flatten it.
    
    For now, we just return the batch as is (a list of lists), and the Trainer 
    handles the iteration.
    """
    return torch.utils.data.dataloader.default_collate(batch)

class OnlineCausalDataset(Dataset):
    """
    Dataset that generates SCM data on-the-fly.
    
    Eliminates the need for storing terabytes of data.
    Each 'item' in this dataset is a full SCM's worth of data (a list of batches).
    
    Attributes:
        num_samples (int): Virtual length of the dataset (number of SCMs to generate per epoch).
        min_vars (int): Minimum number of variables per SCM.
        max_vars (int): Maximum number of variables per SCM.
        n_base (int): Number of baseline samples.
        n_int_samples (int): Number of samples per intervention.
        intervention_fraction (float): Fraction of nodes to intervene on.
    """
    def __init__(
        self, 
        num_samples: int,
        min_vars: int = 10,
        max_vars: int = 100,
        n_base: int = 1000,
        n_int_samples: int = 100,
        intervention_fraction: float = 0.2,
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.n_base = n_base
        self.n_int_samples = n_int_samples
        self.intervention_fraction = intervention_fraction
        self.intervention_fraction = intervention_fraction
        self.seed = seed
        self.epoch_offset = 0

    def set_epoch(self, epoch: int):
        """Updates the epoch offset to ensure new data generation."""
        self.epoch_offset = epoch * self.num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generates a fresh SCM and its data.
        """
        # Use idx to seed so it's deterministic per epoch if needed, 
        # or use random seed if we want infinite variation.
        # For reproducibility, we combine base seed + idx.
        # Note: In multi-worker setup, we might need worker_init_fn to avoid duplicates.
        
        # Import here to avoid circular imports if any (though unlikely given structure)
        from src.data.scm_generator import SCMGenerator
        from src.data.sampler import DataSampler
        from src.data.processor import DataProcessor
        import networkx as nx
        
        # 1. Determine num_vars for this SCM
        # We use the seed to deterministically pick num_vars
        # Combine base seed + epoch_offset + idx
        current_seed = self.seed + self.epoch_offset + idx
        rng = np.random.RandomState(current_seed)
        current_num_vars = rng.randint(self.min_vars, self.max_vars + 1)
        
        # 2. Create SCM
        # We add a large offset to seed to avoid overlap with validation seeds if any
        scm = SCMGenerator(num_vars=current_num_vars, seed=current_seed)
        sampler = DataSampler(scm)
        processor = DataProcessor()
        
        # 3. Generate Baseline
        baseline_data = sampler.sample_baseline(self.n_base)
        processor.fit(baseline_data)
        
        # 4. Generate Interventions
        intervention_vals = [-20.0, -10.0, -5.0, 5.0, 10.0, 20.0]
        num_int_nodes = max(1, int(current_num_vars * self.intervention_fraction))
        int_nodes = torch.randperm(current_num_vars)[:num_int_nodes].tolist()
        
        interventions = sampler.sample_interventions(
            n_samples=self.n_int_samples,
            intervention_nodes=int_nodes,
            intervention_values=intervention_vals
        )
        
        # Get Adjacency Matrix (Ground Truth)
        # nx.to_numpy_array returns float, convert to int binary
        adj = nx.to_numpy_array(scm.dag, weight=None)
        adj = torch.tensor(adj, dtype=torch.float32)
        
        # 4. Process
        processed_data = []
        for item in interventions:
            int_data = item['data']
            mask = item['mask']
            value = item['value']
            node_idx = item['node_idx']
            
            # Pair with random baseline samples
            base_indices = torch.randint(0, len(baseline_data), (len(int_data),))
            base_batch = baseline_data[base_indices]
            
            inp = processor.prepare_input(base_batch, mask, value, node_idx)
            target = processor.transform(int_data)
            
            batch_data = {
                'x': inp['x'],
                'mask': inp['mask'],
                'value': inp['value'],
                'target': target,
                'adj': adj # Add ground truth adjacency
            }
            processed_data.append(batch_data)
            
        return processed_data
