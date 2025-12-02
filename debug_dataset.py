from src.data.dataset import OnlineCausalDataset
import numpy as np

ds = OnlineCausalDataset(num_samples=20, min_vars=5, max_vars=20)

print("Checking SCM sizes and batch counts...")
for i in range(10):
    data = ds[i]
    # data is a list of batches
    num_batches = len(data)
    # We can peek at the adjacency matrix to see the true num_vars
    # adj is in the first batch, key 'adj'
    adj = data[0]['adj']
    num_vars = adj.shape[0]
    
    print(f"Sample {i}: Num Vars = {num_vars}, Num Batches = {num_batches}")
