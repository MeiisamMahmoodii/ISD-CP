import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model.transformer import CausalTransformer
from src.data.scm_generator import SCMGenerator
from src.data.processor import DataProcessor
from src.data.sampler import DataSampler
import networkx as nx

def test_intervention():
    # 1. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "checkpoints_test_4/model_best.pt"
    
    # Initialize model with same config as training (max_vars=20 was used for this run?)
    # Wait, the checkpoint was trained with max_vars=20? 
    # The user said "the model trained on 20 varables". 
    # In step 137 I ran: --max_vars 20
    # But the model init uses max_vars + 1.
    # Let's assume max_vars=1000 was default in code but I passed 20.
    # Actually, I should check the args used.
    # The command was: --max_vars 20 --d_model 256 --num_layers 8
    
    model = CausalTransformer(
        num_vars=21, # 20 + 1
        d_model=256,
        num_layers=8
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    model.eval()
    
    # 2. Generate Synthetic SCM (20 vars)
    seed = 42
    num_vars = 20
    scm = SCMGenerator(num_vars=num_vars, edge_prob=0.1, seed=seed)
    sampler = DataSampler(scm)
    processor = DataProcessor()
    
    # 3. Prepare Data
    # Generate baseline to fit processor
    baseline_data = sampler.sample_baseline(n_samples=1000)
    processor.fit(baseline_data)
    
    # Pick a node to intervene on (one that has children ideally)
    # Let's find a node with out-degree > 0
    out_degrees = [d for n, d in scm.dag.out_degree()]
    intervention_node = np.argmax(out_degrees)
    print(f"Intervening on Node {intervention_node} (Out-degree: {out_degrees[intervention_node]})")
    
    # Intervention Value
    intervention_val = 5.0 # Shift by 5 sigma roughly? No, raw value.
    # Baseline mean for this node
    mean_val = processor.mean[intervention_node].item()
    print(f"Baseline Mean for Node {intervention_node}: {mean_val:.4f}")
    print(f"Intervention Value: {intervention_val}")
    
    # Generate Ground Truth Intervention Data
    # We need "True Delta".
    # True Delta = E[Y | do(X=x)] - E[Y | do(X=baseline_mean)] ? 
    # Or just E[Y | do(X=x)] - x_baseline?
    # The model input is (x_baseline, mask, value).
    # Target is x_intervention.
    # So True Delta = x_intervention - x_baseline.
    
    # Let's take a single baseline sample
    base_sample_raw = baseline_data[0:1] # (1, 20)
    
    # Generate intervention sample corresponding to this baseline?
    # No, SCM generates new samples.
    # But we want to see the effect of changing X from base_val to int_val CETERIS PARIBUS?
    # Our SCM generator generates samples from scratch.
    # To measure "effect" properly in this setup (Counterfactual), we need to control noise.
    # The current SCM generator doesn't expose noise control easily for counterfactuals.
    # BUT, we can just compare E[Y|do(X=v)] vs E[Y].
    
    # Let's generate a batch of intervention samples and take the mean
    int_data_raw = scm.generate_data(n_samples=100, intervention={intervention_node: intervention_val})
    int_mean = int_data_raw.mean(dim=0)
    
    # For the input to the model, we use the baseline sample.
    # The model predicts: What would happen if we force X to v, given this specific baseline state?
    # Since we don't have counterfactual generation, we can't get the EXACT ground truth for THAT sample.
    # But we can check if the predicted mean matches the population intervention mean?
    # Or better: The model is trained to predict `target - x`.
    # `target` is a sample from P(Y|do(X)). `x` is a sample from P(Y).
    # They are independent samples in training!
    # So the model learns to predict E[Y|do(X)] - x.
    
    # So: Pred_Delta + x_baseline should approx E[Y|do(X)].
    
    # Prepare Input
    mask = torch.zeros(num_vars)
    mask[intervention_node] = 1.0
    
    inp = processor.prepare_input(base_sample_raw, mask, value=intervention_val, node_idx=intervention_node)
    
    x = inp['x'].to(device) # (1, 20)
    mask = inp['mask'].to(device) # (1, 20)
    value = inp['value'].to(device) # (1, 20)
    
    with torch.no_grad():
        pred_delta, adj = model(x, mask, value)
        
    # Convert Pred Delta back to original units?
    # pred_delta is in standardized units?
    # The target in training was `target_std - x_std`.
    # So `pred_delta` is in std units.
    
    # Reconstruct Predicted State
    # x_new_std = x_std + pred_delta
    pred_state_std = x + pred_delta
    pred_state_raw = processor.inverse_transform(pred_state_std.cpu())
    
    # Compare
    print("\n--- Results ---")
    print(f"{'Node':<5} {'Base':<10} {'Int_True':<10} {'Int_Pred':<10} {'Diff':<10}")
    
    # Get True Intervention Mean from 100 samples
    true_int_mean = int_mean.numpy()
    base_val = base_sample_raw[0].numpy()
    pred_val = pred_state_raw[0].numpy()
    
    for i in range(num_vars):
        is_int = "*" if i == intervention_node else ""
        diff = abs(true_int_mean[i] - pred_val[i])
        print(f"{i:<5}{is_int} {base_val[i]:<10.4f} {true_int_mean[i]:<10.4f} {pred_val[i]:<10.4f} {diff:<10.4f}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(true_int_mean, label='True Intervention Mean', marker='o')
    plt.plot(pred_val, label='Predicted State', marker='x')
    plt.axvline(intervention_node, color='r', linestyle='--', label='Intervention')
    plt.title(f"Intervention on Node {intervention_node} = {intervention_val}")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_intervention_result.png")
    print("\nSaved plot to test_intervention_result.png")

if __name__ == "__main__":
    test_intervention()
