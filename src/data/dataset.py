import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os

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
