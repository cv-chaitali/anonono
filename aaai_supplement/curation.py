import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from sae import Sae, extract_embeddings
from config import CurationConfig


def compute_cosine_similarity_matrix(
    embeddings1: np.ndarray, 
    embeddings2: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity between two sets of embeddings."""
    return cosine_similarity(embeddings1, embeddings2)


def select_top_k_similar(
    similarities: np.ndarray, 
    k: int
) -> np.ndarray:
    """Select top-k most similar samples based on similarity matrix."""
    # For each target sample, get top-k indices from large dataset
    top_indices = np.argsort(-similarities, axis=1)[:, :k]
    return np.unique(top_indices.flatten())


def select_with_clustering(
    target_embeddings: np.ndarray,
    large_dataset_embeddings: np.ndarray,
    similarities: np.ndarray,
    num_clusters: int = 5,
    top_k_per_cluster: int = 20
) -> np.ndarray:
    """Select samples using clustering approach."""
    # Cluster the large dataset embeddings
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(large_dataset_embeddings)
    
    selected_indices = []
    
    # For each target sample, find the most similar cluster
    for target_idx in range(len(target_embeddings)):
        # Find the cluster that contains the most similar samples
        cluster_similarities = []
        for cluster_id in range(num_clusters):
            cluster_mask = (cluster_labels == cluster_id)
            if cluster_mask.sum() > 0:
                cluster_sim = similarities[target_idx, cluster_mask].mean()
                cluster_similarities.append((cluster_id, cluster_sim))
        
        # Sort clusters by average similarity
        cluster_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Select top samples from the best clusters
        for cluster_id, _ in cluster_similarities[:2]:  # Top 2 clusters
            cluster_mask = (cluster_labels == cluster_id)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 0:
                # Get similarities for this cluster
                cluster_sims = similarities[target_idx, cluster_indices]
                # Select top-k from this cluster
                top_cluster_indices = cluster_indices[np.argsort(-cluster_sims)[:top_k_per_cluster]]
                selected_indices.extend(top_cluster_indices)
    
    return np.unique(selected_indices)


def curate_dataset(
    target_embeddings: np.ndarray,
    large_dataset_embeddings: np.ndarray,
    cfg: CurationConfig
) -> np.ndarray:
    """
    Curate dataset by selecting samples most similar to target domain.
    
    Args:
        target_embeddings: Embeddings of target domain samples
        large_dataset_embeddings: Embeddings of large dataset samples
        cfg: Curation configuration
        
    Returns:
        Indices of selected samples from large dataset
    """
    print("Computing cosine similarities...")
    similarities = compute_cosine_similarity_matrix(target_embeddings, large_dataset_embeddings)
    
    if cfg.use_clustering:
        print("Using clustering-based selection...")
        selected_indices = select_with_clustering(
            target_embeddings, 
            large_dataset_embeddings, 
            similarities,
            num_clusters=cfg.num_clusters,
            top_k_per_cluster=cfg.top_k_selection // cfg.num_clusters
        )
    else:
        print("Using direct top-k selection...")
        selected_indices = select_top_k_similar(similarities, cfg.top_k_selection)
    
    return selected_indices


def evaluate_curation_quality(
    target_embeddings: np.ndarray,
    large_dataset_embeddings: np.ndarray,
    selected_indices: np.ndarray
) -> Dict[str, float]:
    """Evaluate the quality of dataset curation."""
    selected_embeddings = large_dataset_embeddings[selected_indices]
    
    # Compute similarities between target and selected samples
    similarities = compute_cosine_similarity_matrix(target_embeddings, selected_embeddings)
    
    avg_similarity = np.mean(similarities)
    min_similarity = np.min(similarities)
    max_similarity = np.max(similarities)
    
    # Compute diversity of selected samples
    selected_similarities = compute_cosine_similarity_matrix(selected_embeddings, selected_embeddings)
    np.fill_diagonal(selected_similarities, 0)  # Remove self-similarities
    diversity = 1 - np.mean(selected_similarities)
    
    return {
        'avg_similarity_to_target': avg_similarity,
        'min_similarity_to_target': min_similarity,
        'max_similarity_to_target': max_similarity,
        'diversity_of_selected': diversity,
        'num_selected': len(selected_indices)
    }


def visualize_curation_results(
    target_embeddings: np.ndarray,
    large_dataset_embeddings: np.ndarray,
    selected_indices: np.ndarray,
    save_path: Optional[str] = None
):
    """Visualize curation results using PCA and similarity heatmaps."""
    
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    all_embeddings = np.vstack([target_embeddings, large_dataset_embeddings])
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    target_2d = embeddings_2d[:len(target_embeddings)]
    large_2d = embeddings_2d[len(target_embeddings):]
    selected_2d = large_2d[selected_indices]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    ax1.scatter(large_2d[:, 0], large_2d[:, 1], alpha=0.3, label='Large Dataset', color='lightblue')
    ax1.scatter(selected_2d[:, 0], selected_2d[:, 1], alpha=0.8, label='Selected', color='red')
    ax1.scatter(target_2d[:, 0], target_2d[:, 1], alpha=0.8, label='Target Domain', color='green', s=100)
    ax1.set_title('PCA Visualization of Embeddings')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Similarity heatmap
    similarities = compute_cosine_similarity_matrix(target_embeddings, large_dataset_embeddings[selected_indices])
    sns.heatmap(similarities, ax=ax2, cmap='viridis', cbar_kws={'label': 'Cosine Similarity'})
    ax2.set_title('Similarity Matrix: Target vs Selected')
    ax2.set_xlabel('Selected Samples')
    ax2.set_ylabel('Target Samples')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def run_curation_pipeline(
    target_activations: torch.Tensor,
    large_dataset_activations: torch.Tensor,
    sae: Sae,
    cfg: CurationConfig,
    visualize: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Run the complete dataset curation pipeline.
    
    Args:
        target_activations: Activations from target domain samples
        large_dataset_activations: Activations from large mixed dataset
        sae: Trained sparse autoencoder
        cfg: Curation configuration
        visualize: Whether to create visualizations
        
    Returns:
        Tuple of (target_embeddings, large_dataset_embeddings, selected_indices, metrics)
    """
    print("=== FineScope Dataset Curation Pipeline ===")
    
    # Extract embeddings
    print("Extracting embeddings from target domain...")
    target_embeddings = extract_embeddings(
        sae, target_activations, cfg.top_k_activations
    )
    
    print("Extracting embeddings from large dataset...")
    large_dataset_embeddings = extract_embeddings(
        sae, large_dataset_activations, cfg.top_k_activations
    )
    
    # Curate dataset
    print("Curating dataset using cosine similarity...")
    selected_indices = curate_dataset(
        target_embeddings, large_dataset_embeddings, cfg
    )
    
    # Evaluate quality
    print("Evaluating curation quality...")
    metrics = evaluate_curation_quality(
        target_embeddings, large_dataset_embeddings, selected_indices
    )
    
    # Visualize results
    if visualize:
        print("Creating visualizations...")
        visualize_curation_results(
            target_embeddings, large_dataset_embeddings, selected_indices
        )
    
    return target_embeddings, large_dataset_embeddings, selected_indices, metrics 