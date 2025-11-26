# Structure-Agnostic Causal Transformer (ISD-CP)

## Overview
**ISD-CP** (Implicit Structure Discovery - Causal Prediction) is a scalable, structure-agnostic Transformer model designed to predict the consequences of interventions ($\text{do}(X=x)$) on tabular data.

Unlike traditional Causal Bayesian Networks that require explicit DAG discovery, ISD-CP treats causal inference as an end-to-end regression problem. It leverages the Transformer's self-attention mechanism to implicitly learn the causal structure and dependencies between variables.

## Core Philosophy
1.  **Fixed Reference Frame**: All data is standardized using statistics ($\mu, \sigma$) derived exclusively from the **baseline (unintervened)** state. This ensures the model perceives the magnitude of interventions relative to the system's natural variability.
2.  **Structure-Agnostic**: The model does not take a graph as input. It learns the graph structure implicitly from the data.
3.  **End-to-End Prediction**: The model predicts the post-intervention state of the entire system given a baseline state and an intervention token.

## Project Structure
```
ISD-CP/
├── src/
│   ├── data/           # Data generation, sampling, and processing
│   │   ├── scm_generator.py    # Generates synthetic SCMs
│   │   ├── sampler.py          # Samples baseline and interventional data
│   │   ├── processor.py        # Standardization logic
│   │   ├── dataset.py          # PyTorch Dataset
│   │   └── generate_dataset.py # Script to generate full datasets
│   ├── model/          # Model architecture
│   │   └── transformer.py      # Causal Transformer implementation
│   ├── train/          # Training logic
│   │   ├── trainer.py          # Training loop
│   │   └── train.py            # Main training script
│   └── utils/          # Utilities
│       ├── metrics.py          # SHD/F1 metrics
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

### 1. Generate Data
Generate synthetic SCM data for training.
```bash
python -m src.data.generate_dataset --output_dir data/train --num_scms 500 --num_vars 128
```

### 2. Train Model
Train the Causal Transformer.
```bash
python -m src.train.train --data_dir data/train --num_vars 128 --epochs 100 --output_dir checkpoints
```

## License
[MIT License](LICENSE)
