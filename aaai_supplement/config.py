from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class SaeConfig:
    """Configuration for sparse autoencoder."""
    
    expansion_factor: int = 32
    """Multiple of the input dimension to use as the SAE dimension."""
    
    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""
    
    num_latents: int = 0
    """Number of latents to use. If 0, use `expansion_factor`."""
    
    k: int = 32
    """Number of nonzero features."""
    
    multi_topk: bool = True
    """Use Multi-TopK loss."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'expansion_factor': self.expansion_factor,
            'normalize_decoder': self.normalize_decoder,
            'num_latents': self.num_latents,
            'k': self.k,
            'multi_topk': self.multi_topk
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], drop_extra_fields: bool = False) -> 'SaeConfig':
        """Create config from dictionary."""
        if drop_extra_fields:
            valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
            config_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """Configuration for SAE training."""
    
    batch_size: int = 32
    """Batch size for training."""
    
    num_epochs: int = 10
    """Number of training epochs."""
    
    learning_rate: float = 1e-4
    """Learning rate for training."""
    
    l1_weight: float = 1e-3
    """Weight for L1 sparsity loss."""
    
    device: str = "cpu"
    """Device to train on."""
    
    save_every: int = 5
    """Save model every N epochs."""
    
    log_every: int = 100
    """Log metrics every N steps."""


@dataclass
class CurationConfig:
    """Configuration for dataset curation."""
    
    top_k_activations: int = 100
    """Number of top activations to select."""
    
    top_k_selection: int = 100
    """Number of samples to select from large dataset."""
    
    similarity_threshold: float = 0.5
    """Minimum cosine similarity threshold."""
    
    use_clustering: bool = True
    """Whether to use clustering for selection."""
    
    num_clusters: int = 5
    """Number of clusters for K-means.""" 