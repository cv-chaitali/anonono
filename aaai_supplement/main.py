#!/usr/bin/env python3
"""
FineScope: Sparse Autoencoder-Guided Domain-Specific Dataset Curation

This script demonstrates the complete FineScope pipeline including:
1. SAE training on LLM activations
2. Embedding extraction using trained SAE
3. Dataset curation using cosine similarity
4. Quality evaluation and visualization

Example usage:
    python main.py
"""

import torch
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any

from config import SaeConfig, TrainingConfig, CurationConfig
from sae import Sae, train_sae, extract_embeddings
from curation import run_curation_pipeline
from data_generator import create_activation_dataset, save_dummy_dataset, load_dummy_dataset


def setup_experiment(
    target_domain: str = "stem",
    target_samples: int = 50,
    large_dataset_samples: int = 1000,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Setup the FineScope experiment."""
    
    print("=== FineScope Experiment Setup ===")
    
    # Create or load dummy dataset
    data_dir = "dummy_data"
    if os.path.exists(data_dir):
        print("Loading existing dummy dataset...")
        target_activations, large_activations, domain_labels = load_dummy_dataset(data_dir)
    else:
        print("Creating new dummy dataset...")
        target_activations, large_activations, domain_labels = create_activation_dataset(
            target_domain=target_domain,
            target_samples=target_samples,
            large_dataset_samples=large_dataset_samples,
            device=device
        )
        save_dummy_dataset(target_activations, large_activations, domain_labels, data_dir)
    
    # Configuration
    sae_cfg = SaeConfig(
        expansion_factor=32,
        k=32,
        normalize_decoder=True,
        multi_topk=True
    )
    
    train_cfg = TrainingConfig(
        batch_size=32,
        num_epochs=5,  # Reduced for demo
        learning_rate=1e-4,
        l1_weight=1e-3,
        device=device
    )
    
    curation_cfg = CurationConfig(
        top_k_activations=100,
        top_k_selection=100,
        use_clustering=True,
        num_clusters=5
    )
    
    return {
        'target_activations': target_activations,
        'large_activations': large_activations,
        'domain_labels': domain_labels,
        'sae_cfg': sae_cfg,
        'train_cfg': train_cfg,
        'curation_cfg': curation_cfg,
        'target_domain': target_domain
    }


def train_sae_model(
    activations: torch.Tensor,
    sae_cfg: SaeConfig,
    train_cfg: TrainingConfig,
    save_path: str = "trained_sae"
) -> Sae:
    """Train a sparse autoencoder on activations."""
    
    print("\n=== SAE Training ===")
    print(f"Training on {len(activations)} samples with shape {activations.shape}")
    print(f"SAE configuration: {sae_cfg}")
    print(f"Training configuration: {train_cfg}")
    
    # Train SAE
    sae = train_sae(activations, sae_cfg, train_cfg, save_path)
    
    print(f"SAE training completed. Model saved to {save_path}")
    return sae


def run_curation_experiment(
    target_activations: torch.Tensor,
    large_activations: torch.Tensor,
    sae: Sae,
    curation_cfg: CurationConfig,
    target_domain: str,
    visualize: bool = True
) -> Dict[str, Any]:
    """Run the dataset curation experiment."""
    
    print(f"\n=== Dataset Curation Experiment ===")
    print(f"Target domain: {target_domain}")
    print(f"Curation configuration: {curation_cfg}")
    
    # Run curation pipeline
    target_embeddings, large_embeddings, selected_indices, metrics = run_curation_pipeline(
        target_activations=target_activations,
        large_dataset_activations=large_activations,
        sae=sae,
        cfg=curation_cfg,
        visualize=visualize
    )
    
    print(f"\n=== Curation Results ===")
    print(f"Target embeddings shape: {target_embeddings.shape}")
    print(f"Large dataset embeddings shape: {large_embeddings.shape}")
    print(f"Selected {len(selected_indices)} samples")
    
    print(f"\n=== Quality Metrics ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return {
        'target_embeddings': target_embeddings,
        'large_embeddings': large_embeddings,
        'selected_indices': selected_indices,
        'metrics': metrics
    }


def analyze_results(
    results: Dict[str, Any],
    domain_labels: list,
    target_domain: str
):
    """Analyze and print detailed results."""
    
    selected_indices = results['selected_indices']
    selected_domains = [domain_labels[i] for i in selected_indices]
    
    print(f"\n=== Detailed Analysis ===")
    print(f"Target domain: {target_domain}")
    print(f"Total selected samples: {len(selected_indices)}")
    
    # Domain distribution analysis
    from collections import Counter
    domain_counts = Counter(selected_domains)
    
    print(f"\nDomain distribution in selected samples:")
    for domain, count in domain_counts.most_common():
        percentage = (count / len(selected_indices)) * 100
        print(f"  {domain}: {count} samples ({percentage:.1f}%)")
    
    # Target domain accuracy
    target_domain_count = domain_counts.get(target_domain, 0)
    target_accuracy = (target_domain_count / len(selected_indices)) * 100
    print(f"\nTarget domain accuracy: {target_accuracy:.1f}%")
    
    # Similarity analysis
    metrics = results['metrics']
    print(f"Average similarity to target: {metrics['avg_similarity_to_target']:.4f}")
    print(f"Diversity of selected samples: {metrics['diversity_of_selected']:.4f}")


def main():
    """Main function demonstrating the complete FineScope pipeline."""
    
    print("FineScope: Sparse Autoencoder-Guided Domain-Specific Dataset Curation")
    print("=" * 80)
    
    # Setup experiment
    experiment_config = setup_experiment(
        target_domain="stem",
        target_samples=50,
        large_dataset_samples=1000,
        device="cpu"
    )
    
    # Extract components
    target_activations = experiment_config['target_activations']
    large_activations = experiment_config['large_activations']
    domain_labels = experiment_config['domain_labels']
    sae_cfg = experiment_config['sae_cfg']
    train_cfg = experiment_config['train_cfg']
    curation_cfg = experiment_config['curation_cfg']
    target_domain = experiment_config['target_domain']
    
    # Train SAE
    sae = train_sae_model(
        large_activations,  # Train on large dataset
        sae_cfg,
        train_cfg,
        save_path="trained_sae"
    )
    
    # Run curation experiment
    results = run_curation_experiment(
        target_activations,
        large_activations,
        sae,
        curation_cfg,
        target_domain,
        visualize=True
    )
    
    # Analyze results
    analyze_results(results, domain_labels, target_domain)
    
    print(f"\n=== Pipeline Complete ===")
    print("FineScope successfully demonstrated:")
    print("1. ✅ SAE training on LLM activations")
    print("2. ✅ Top-k activation selection for efficiency")
    print("3. ✅ Embedding extraction using trained SAE")
    print("4. ✅ Cosine similarity-based dataset curation")
    print("5. ✅ Clustering-based sample selection")
    print("6. ✅ Quality evaluation and visualization")
    
    print(f"\nResults saved to:")
    print(f"  - Trained SAE: trained_sae/")
    print(f"  - Dummy dataset: dummy_data/")
    print(f"  - Visualizations: (displayed above)")


if __name__ == "__main__":
    main() 