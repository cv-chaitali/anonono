# FineScope: Sparse Autoencoder-Guided Domain-Specific Dataset Curation

This supplement provides the complete implementation of the FineScope framework for domain-specific LLM adaptation through sparse autoencoder-guided dataset curation.

## Overview

FineScope consists of two main stages:
1. **Sparse Autoencoder Training**: Train SAEs on LLM activations to learn compressed representations
2. **Dataset Curation**: Use SAE embeddings and cosine similarity to select domain-relevant samples from large datasets

## Features

- ✅ **SAE Training**: Complete training pipeline with L1 sparsity and reconstruction loss
- ✅ **Top-K Activation Selection**: Efficient processing of large activation tensors
- ✅ **Embedding Extraction**: Extract domain-specific features using trained SAEs
- ✅ **Cosine Similarity Curation**: Select samples based on embedding similarity
- ✅ **Clustering-Based Selection**: Advanced selection using K-means clustering
- ✅ **Quality Evaluation**: Comprehensive metrics for curation quality
- ✅ **Visualization**: PCA plots and similarity heatmaps
- ✅ **Dummy Dataset**: Complete demonstration with synthetic data

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Make setup script executable
chmod +x setup_env.sh

# Run automated setup
./setup_env.sh

# Activate environment
source finescope_env/bin/activate
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv finescope_env
source finescope_env/bin/activate  # On Windows: finescope_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the complete FineScope demonstration
python main.py
```

This will:
1. Create dummy LLM activations with domain-specific patterns
2. Train a sparse autoencoder on the activations
3. Extract embeddings using the trained SAE
4. Curate a domain-specific dataset using cosine similarity
5. Evaluate and visualize the results

## File Structure

```
aaai_supplement/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup_env.sh             # Environment setup script
├── config.py                # Configuration classes
├── sae.py                   # Sparse Autoencoder implementation
├── curation.py              # Dataset curation functions
├── data_generator.py        # Dummy data generation
├── main.py                  # Complete demonstration script
└── dummy_data/              # Generated dummy datasets
```

## Usage Examples

### Basic Usage

```python
from config import SaeConfig, TrainingConfig, CurationConfig
from sae import train_sae, extract_embeddings
from curation import run_curation_pipeline
from data_generator import create_activation_dataset

# Create dummy data
target_acts, large_acts, labels = create_activation_dataset(
    target_domain="stem",
    target_samples=50,
    large_dataset_samples=1000
)

# Train SAE
sae_cfg = SaeConfig(expansion_factor=32, k=32)
train_cfg = TrainingConfig(num_epochs=10, device="cpu")
sae = train_sae(large_acts, sae_cfg, train_cfg)

# Run curation
curation_cfg = CurationConfig(top_k_selection=100, use_clustering=True)
results = run_curation_pipeline(target_acts, large_acts, sae, curation_cfg)
```

### Advanced Usage

```python
# Custom SAE configuration
sae_cfg = SaeConfig(
    expansion_factor=64,      # Larger SAE
    k=64,                    # More active features
    multi_topk=True,         # Use Multi-TopK loss
    normalize_decoder=True    # Normalize decoder weights
)

# Custom training configuration
train_cfg = TrainingConfig(
    batch_size=64,
    num_epochs=20,
    learning_rate=5e-5,
    l1_weight=2e-3,         # Stronger sparsity
    device="cuda"            # Use GPU if available
)

# Custom curation configuration
curation_cfg = CurationConfig(
    top_k_activations=200,   # More activations
    top_k_selection=200,     # More selected samples
    use_clustering=True,      # Enable clustering
    num_clusters=10,         # More clusters
    similarity_threshold=0.6  # Higher threshold
)
```

## Configuration

### SAE Configuration (`SaeConfig`)

- `expansion_factor`: Multiple of input dimension for SAE size
- `k`: Number of active features (top-k)
- `normalize_decoder`: Whether to normalize decoder weights
- `multi_topk`: Whether to use Multi-TopK loss

### Training Configuration (`TrainingConfig`)

- `batch_size`: Training batch size
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimization
- `l1_weight`: Weight for L1 sparsity loss
- `device`: Device to train on ("cpu" or "cuda")

### Curation Configuration (`CurationConfig`)

- `top_k_activations`: Number of top activations to select
- `top_k_selection`: Number of samples to select from large dataset
- `use_clustering`: Whether to use clustering-based selection
- `num_clusters`: Number of clusters for K-means
- `similarity_threshold`: Minimum cosine similarity threshold

## Output and Results

The demonstration produces:

1. **Trained SAE Model**: Saved to `trained_sae/`
2. **Dummy Dataset**: Saved to `dummy_data/`
3. **Visualizations**: PCA plots and similarity heatmaps
4. **Quality Metrics**: Similarity scores and diversity measures

### Example Output

```
=== FineScope Experiment Setup ===
Creating new dummy dataset...
Target activations: torch.Size([50, 128, 512])
Large dataset activations: torch.Size([1000, 128, 512])

=== SAE Training ===
Training on 1000 samples with shape torch.Size([1000, 128, 512])
Epoch 1/5: Reconstruction Loss: 0.123456, L1 Loss: 0.045678

=== Dataset Curation Experiment ===
Extracting embeddings from target domain...
Extracting embeddings from large dataset...
Curating dataset using cosine similarity...

=== Curation Results ===
Selected 100 samples
Average similarity to target: 0.8234
Diversity of selected samples: 0.6543

=== Detailed Analysis ===
Domain distribution in selected samples:
  stem: 45 samples (45.0%)
  humanities: 20 samples (20.0%)
  business: 15 samples (15.0%)
  arts: 12 samples (12.0%)
  social_sciences: 8 samples (8.0%)

Target domain accuracy: 45.0%
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy, scikit-learn
- Matplotlib, Seaborn
- tqdm, safetensors, einops

## Anonymization

This code has been anonymized for submission. All identifying information, author names, and institution-specific details have been removed.

## Citation

If you use this code in your research, please cite the original FineScope paper.

## License

This code is provided for research purposes only. 