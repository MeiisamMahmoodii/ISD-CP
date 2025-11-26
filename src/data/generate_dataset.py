import torch
import os
import argparse
from tqdm import tqdm
from src.data.scm_generator import SCMGenerator
from src.data.sampler import DataSampler
from src.data.processor import DataProcessor

def generate_dataset(
    output_dir: str,
    num_scms: int = 10,
    num_vars: int = 16,
    n_base: int = 1000,
    n_int_samples: int = 100,
    intervention_fraction: float = 0.2,
    seed: int = 42
):
    """
    Generates the full training dataset for ISD-CP.
    
    Process:
    1. Iterates `num_scms` times.
    2. For each iteration, creates a random SCM.
    3. Generates baseline data and fits the DataProcessor (standardization).
    4. Selects a subset of nodes to intervene on.
    5. Generates interventional data for multiple values.
    6. Pairs each interventional sample with a random baseline sample to create the input-output pair.
    7. Standardizes everything and saves to disk.
    
    Args:
        output_dir: Directory to save .pt files.
        num_scms: Total number of distinct SCMs to generate.
        num_vars: Number of variables per SCM.
        n_base: Number of baseline samples to generate (for stats).
        n_int_samples: Number of samples per intervention value.
        intervention_fraction: Fraction of nodes to intervene on.
        seed: Random seed.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Intervention values to sweep (z-score like magnitudes)
    intervention_vals = [-5.0, -2.0, 0.0, 2.0, 5.0]
    
    for scm_idx in tqdm(range(num_scms), desc="Generating SCMs"):
        # 1. Create SCM
        scm = SCMGenerator(num_vars=num_vars, seed=seed + scm_idx)
        sampler = DataSampler(scm)
        processor = DataProcessor()
        
        # 2. Generate Baseline & Fit Processor
        # We need baseline data to establish the "Fixed Reference Frame"
        baseline_data = sampler.sample_baseline(n_base)
        processor.fit(baseline_data)
        
        # 3. Generate Interventions
        # Select random nodes to intervene
        num_int_nodes = max(1, int(num_vars * intervention_fraction))
        int_nodes = torch.randperm(num_vars)[:num_int_nodes].tolist()
        
        interventions = sampler.sample_interventions(
            n_samples=n_int_samples,
            intervention_nodes=int_nodes,
            intervention_values=intervention_vals
        )
        
        # 4. Process and Save Data
        processed_data = []
        
        for item in interventions:
            int_data = item['data'] # (n_samples, num_vars)
            mask = item['mask']     # (num_vars,)
            value = item['value']   # float
            node_idx = item['node_idx']
            
            # Pair with random baseline samples
            # The model input requires a baseline state (S_baseline) to predict the effect.
            # We randomly sample from our generated baseline data.
            base_indices = torch.randint(0, len(baseline_data), (len(int_data),))
            base_batch = baseline_data[base_indices]
            
            # Prepare inputs
            # Input X is the BASELINE data (standardized).
            # Output Y is the INTERVENED data (standardized).
            
            inp = processor.prepare_input(base_batch, mask, value, node_idx)
            
            # Target: Standardized Intervened Data
            target = processor.transform(int_data)
            
            # Store as a dict of tensors
            batch_data = {
                'x': inp['x'],          # (n_samples, num_vars)
                'mask': inp['mask'],    # (n_samples, num_vars)
                'value': inp['value'],  # (n_samples, num_vars)
                'target': target        # (n_samples, num_vars)
            }
            processed_data.append(batch_data)
            
        # Save this SCM's data chunk
        torch.save(processed_data, os.path.join(output_dir, f"scm_{scm_idx}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/train")
    parser.add_argument("--num_scms", type=int, default=5)
    parser.add_argument("--num_vars", type=int, default=16)
    args = parser.parse_args()
    
    generate_dataset(
        output_dir=args.output_dir,
        num_scms=args.num_scms,
        num_vars=args.num_vars
    )
