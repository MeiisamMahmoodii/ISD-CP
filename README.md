# Structure-Agnostic Causal Transformer (ISD-CP)

## Overview
**ISD-CP** (Implicit Structure Discovery - Causal Prediction) is a scalable, structure-agnostic Transformer model designed to predict the consequences of interventions ($\text{do}(X=x)$) on tabular data.

Unlike traditional Causal Bayesian Networks that require explicit DAG discovery, ISD-CP treats causal inference as an end-to-end regression problem. It leverages the Transformer's self-attention mechanism to implicitly learn the causal structure and dependencies between variables.

## Core Philosophy
1.  **Fixed Reference Frame**: All data is standardized using statistics ($\mu, \sigma$) derived exclusively from the **baseline (unintervened)** state. This ensures the model perceives the magnitude of interventions relative to the system's natural variability.
2.  **Structure-Agnostic**: The model does not take a graph as input. It learns the graph structure implicitly from the data.
3.  **End-to-End Prediction**: The model predicts the post-intervention state of the entire system given a baseline state and an intervention token.
4.  **Online Data Generation**: Data is generated on-the-fly during training, enabling infinite variety and zero storage overhead.

## Key Features
-   **Scalable Architecture**: Handles variable numbers of columns (10 to 1000+) dynamically.
-   **Online Data Generation**: Eliminates the need for terabytes of disk storage by generating SCMs in memory.
-   **Implicit Structure Learning**: Extracts causal graphs (DAGs) from attention weights for interpretability.
-   **Production-Ready Monitoring**: Integrated TensorBoard logging for Loss, SHD (Structural Hamming Distance), and F1 Score.
-   **GPU Optimized**: Fully utilizes CUDA for training while CPU handles data generation.

## Project Structure
```
ISD-CP/
├── src/
│   ├── data/           # Data generation pipeline
│   │   ├── scm_generator.py    # Generates synthetic SCMs (Linear/Non-linear)
│   │   ├── sampler.py          # Samples baseline and interventional data
│   │   ├── processor.py        # Standardization logic
│   │   └── dataset.py          # OnlineCausalDataset (On-the-fly generation)
│   ├── model/          # Model architecture
│   │   └── transformer.py      # Causal Transformer with Attention Extraction
│   ├── train/          # Training logic
│   │   ├── trainer.py          # Training loop with TensorBoard & Metrics
│   │   └── train.py            # Main entry point
│   └── utils/          # Utilities
│       ├── metrics.py          # SHD, F1, Attention Analysis
│       └── monitor.py          # GPU monitoring
├── configs/            # Configuration files
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## Installation
1.  Clone the repository.
2.  Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training
Run the training script. It will automatically generate data and train the model.

```bash
# Train with default settings (100 SCMs/epoch, variable cols 10-1000)
python -m src.train.train --output_dir checkpoints_prod
```

**Key Arguments:**
-   `--num_scms`: Number of unique SCMs to generate per epoch (default: 100).
-   `--batch_size`: Number of samples per update (default: 100).
-   `--min_vars` / `--max_vars`: Range of variables per SCM (default: 10-1000).
-   `--epochs`: Number of epochs (default: 100).

### Monitoring
Visualize training progress with TensorBoard:
```bash
tensorboard --logdir checkpoints_prod/logs
```

## Project History & Changelog

### Phase I: Causal Data Engineer (CDE)
-   Implemented `SCMGenerator` for synthetic SCMs with random DAGs and mixed mechanisms.
-   Implemented `DataSampler` for baseline and interventional data.
-   Implemented `DataProcessor` for fixed reference frame standardization.

### Phase II: ML Engineer (MLE)
-   Implemented `CausalTransformer` with feature, variable ID, mask, and value embeddings.
-   Implemented `Trainer` class for robust training loops.
-   Verified model forward pass and loss convergence.

### Phase III: ML Architect (MLA) & Scaling
-   **Online Data Generation**: Replaced disk-based dataset with `OnlineCausalDataset` to solve storage issues (1.9TB -> 0GB).
-   **Variable Columns**: Updated pipeline to support dynamic number of variables (10-1000) per SCM.
-   **Attention Extraction**: Added capability to extract implicit DAGs from Transformer attention weights.
-   **Metrics & Monitoring**: Integrated SHD and F1 score computation into the validation loop and added TensorBoard logging.
-   **GPU Optimization**: Parallelized data generation (CPU) and training (GPU) using `num_workers`.

## License
[MIT License](LICENSE)
