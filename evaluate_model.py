import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model.transformer import CausalTransformer
from src.data.scm_generator import SCMGenerator
from src.data.processor import DataProcessor
from src.data.sampler import DataSampler
import os

def evaluate_model():
    # 1. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "checkpoints_prod/model_best.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    # Initialize model with same config as training
    # Training command: --max_vars 20 --d_model 256 --num_layers 8
    model = CausalTransformer(
        num_vars=101, # Match checkpoint (101 embeddings)
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
    
    # 2. Setup Data Generation
    seed = 42
    num_vars = 20
    # Generate a fixed SCM for consistent testing
    scm = SCMGenerator(num_vars=num_vars, edge_prob=0.1, seed=seed)
    sampler = DataSampler(scm)
    processor = DataProcessor()
    
    # Fit processor on baseline data
    baseline_data = sampler.sample_baseline(n_samples=1000)
    processor.fit(baseline_data)
    
    # 3. Define Test Cases
    # We will test interventions on different nodes with different values
    # Find nodes with children to see propagation effects
    out_degrees = [d for n, d in scm.dag.out_degree()]
    nodes_with_children = [i for i, d in enumerate(out_degrees) if d > 0]
    
    if not nodes_with_children:
        print("No nodes with children found in the SCM. Cannot test propagation.")
        nodes_with_children = [0, 1, 2] # Fallback

    test_cases = []
    # Case 1: Intervene on a root/parent node (high impact)
    test_cases.append({
        "node": nodes_with_children[0],
        "value": 5.0,
        "desc": "High positive shift on parent node"
    })
    
    # Case 2: Intervene on same node, negative shift
    test_cases.append({
        "node": nodes_with_children[0],
        "value": -5.0,
        "desc": "High negative shift on parent node"
    })
    
    # Case 3: Intervene on another node
    if len(nodes_with_children) > 1:
        test_cases.append({
            "node": nodes_with_children[1],
            "value": 3.0,
            "desc": "Moderate shift on another parent node"
        })
        
    # Case 4: Intervene on a leaf node (should have little propagation)
    leaf_nodes = [i for i, d in enumerate(out_degrees) if d == 0]
    if leaf_nodes:
        test_cases.append({
            "node": leaf_nodes[0],
            "value": 10.0,
            "desc": "Large shift on leaf node (should not propagate)"
        })

    print(f"\nRunning {len(test_cases)} test cases...")
    
    for i, case in enumerate(test_cases):
        run_test_case(model, scm, processor, case, device, case_id=i+1)

def run_test_case(model, scm, processor, case, device, case_id):
    intervention_node = case["node"]
    intervention_val = case["value"]
    desc = case["desc"]
    
    print(f"\n=== Case {case_id}: {desc} ===")
    print(f"Intervention: Node {intervention_node} = {intervention_val}")
    
    # Generate Ground Truth
    # We compare E[Y|do(X=v)] vs Baseline Sample
    # Note: Ideally we want counterfactuals, but we'll use population means for "True Effect"
    
    # 1. Get a baseline sample (just one for input)
    # We use a fixed seed for reproducibility of the sample
    torch.manual_seed(42 + case_id)
    base_sample_raw = scm.generate_data(n_samples=1) # (1, 20)
    
    # 2. Get True Intervention Mean (Population level)
    int_data_raw = scm.generate_data(n_samples=500, intervention={intervention_node: intervention_val})
    true_int_mean = int_data_raw.mean(dim=0).numpy()
    
    # 3. Model Prediction
    mask = torch.zeros(scm.num_vars)
    mask[intervention_node] = 1.0
    
    inp = processor.prepare_input(base_sample_raw, mask, value=intervention_val, node_idx=intervention_node)
    
    x = inp['x'].to(device)
    mask_t = inp['mask'].to(device)
    value_t = inp['value'].to(device)
    
    with torch.no_grad():
        pred_delta, adj = model(x, mask_t, value_t)
        
    # Reconstruct
    # pred_delta is standardized difference
    # We want: pred_state = x_raw + inverse_transform(pred_delta) ?
    # No, model predicts delta in standardized space.
    # So: pred_state_std = x_std + pred_delta
    # pred_state_raw = inverse_transform(pred_state_std)
    
    # Note: prepare_input normalizes x.
    # x is already standardized.
    pred_state_std = x + pred_delta
    pred_state_raw = processor.inverse_transform(pred_state_std.cpu()).numpy()[0]
    base_val_raw = base_sample_raw[0].numpy()
    
    # 4. Analysis
    # We want to see if the model predicts changes in OTHER nodes.
    # Calculate "Effect": Value - Baseline
    true_effect = true_int_mean - base_val_raw
    pred_effect = pred_state_raw - base_val_raw
    
    print(f"{'Node':<5} {'Base':<10} {'True_Int':<10} {'Pred_Int':<10} {'True_Eff':<10} {'Pred_Eff':<10} {'Error':<10}")
    
    mae = 0
    count = 0
    
    for n in range(scm.num_vars):
        is_int = "*" if n == intervention_node else ""
        err = abs(pred_state_raw[n] - true_int_mean[n])
        
        # Highlight significant effects
        eff_str = f"{true_effect[n]:.4f}"
        if abs(true_effect[n]) > 0.5 and n != intervention_node:
            eff_str += " (!)"
            
        print(f"{n:<5}{is_int} {base_val_raw[n]:<10.4f} {true_int_mean[n]:<10.4f} {pred_state_raw[n]:<10.4f} {eff_str:<10} {pred_effect[n]:<10.4f} {err:<10.4f}")
        
        if n != intervention_node:
            mae += err
            count += 1
            
    print(f"MAE on non-intervened nodes: {mae/count:.4f}")

if __name__ == "__main__":
    evaluate_model()
