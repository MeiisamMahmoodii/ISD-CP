from src.data.dataset import CausalDataset
import torch

def test_dataset():
    ds = CausalDataset("data/test")
    print(f"Found {len(ds)} files.")
    
    if len(ds) > 0:
        item = ds[0]
        print(f"Item type: {type(item)}")
        # item is a list of dicts (batches)
        print(f"Number of batches in file: {len(item)}")
        
        first_batch = item[0]
        print("Keys:", first_batch.keys())
        print("X shape:", first_batch['x'].shape)
        print("Target shape:", first_batch['target'].shape)

if __name__ == "__main__":
    test_dataset()
