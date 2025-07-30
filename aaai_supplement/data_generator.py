import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from datasets import Dataset
import json


def create_dummy_activations(
    num_samples: int = 1000,
    seq_len: int = 128,
    hidden_dim: int = 512,
    device: str = "cpu"
) -> torch.Tensor:
    """Create dummy LLM activations for demonstration."""
    return torch.randn(num_samples, seq_len, hidden_dim, device=device)


def create_domain_specific_activations(
    domain: str,
    num_samples: int = 100,
    seq_len: int = 128,
    hidden_dim: int = 512,
    device: str = "cpu"
) -> torch.Tensor:
    """Create domain-specific activations with controlled patterns."""
    
    # Create base activations
    activations = torch.randn(num_samples, seq_len, hidden_dim, device=device)
    
    # Add domain-specific patterns
    if domain == "stem":
        # STEM domain: emphasize certain dimensions
        stem_dims = torch.randint(0, hidden_dim, (hidden_dim // 4,))
        activations[:, :, stem_dims] *= 2.0
    elif domain == "humanities":
        # Humanities domain: different emphasis
        humanities_dims = torch.randint(hidden_dim // 2, hidden_dim, (hidden_dim // 4,))
        activations[:, :, humanities_dims] *= 1.5
    elif domain == "business":
        # Business domain: moderate emphasis
        business_dims = torch.randint(hidden_dim // 4, 3 * hidden_dim // 4, (hidden_dim // 4,))
        activations[:, :, business_dims] *= 1.8
    
    return activations


def create_mixed_domain_activations(
    domains: List[str],
    samples_per_domain: int = 200,
    seq_len: int = 128,
    hidden_dim: int = 512,
    device: str = "cpu"
) -> Tuple[torch.Tensor, List[str]]:
    """Create mixed domain activations with domain labels."""
    
    all_activations = []
    domain_labels = []
    
    for domain in domains:
        domain_acts = create_domain_specific_activations(
            domain, samples_per_domain, seq_len, hidden_dim, device
        )
        all_activations.append(domain_acts)
        domain_labels.extend([domain] * samples_per_domain)
    
    return torch.cat(all_activations, dim=0), domain_labels


def create_dummy_text_dataset(
    num_samples: int = 1000,
    domains: List[str] = None
) -> Dataset:
    """Create a dummy text dataset with domain-specific samples."""
    
    if domains is None:
        domains = ["stem", "humanities", "business", "arts", "social_sciences"]
    
    samples_per_domain = num_samples // len(domains)
    
    # Domain-specific text templates
    domain_templates = {
        "stem": [
            "The mathematical equation demonstrates the relationship between variables.",
            "Chemical reactions involve the rearrangement of atomic bonds.",
            "Physical forces govern the motion of celestial bodies.",
            "Biological processes regulate cellular functions.",
            "Engineering principles apply to structural design."
        ],
        "humanities": [
            "Literary analysis reveals underlying themes in the text.",
            "Historical events shaped cultural development.",
            "Philosophical inquiry explores fundamental questions.",
            "Artistic expression communicates human experience.",
            "Linguistic patterns reflect cognitive processes."
        ],
        "business": [
            "Market analysis indicates economic trends.",
            "Strategic planning optimizes resource allocation.",
            "Financial modeling predicts investment returns.",
            "Organizational behavior influences productivity.",
            "Supply chain management ensures efficiency."
        ],
        "arts": [
            "Creative expression manifests through various media.",
            "Aesthetic principles guide artistic composition.",
            "Cultural heritage preserves traditional values.",
            "Performance art engages audience participation.",
            "Visual design communicates complex ideas."
        ],
        "social_sciences": [
            "Social dynamics influence group behavior.",
            "Psychological factors affect decision-making.",
            "Economic policies impact societal outcomes.",
            "Political systems determine governance structures.",
            "Cultural norms shape individual identity."
        ]
    }
    
    texts = []
    labels = []
    
    for domain in domains:
        templates = domain_templates.get(domain, ["Sample text from " + domain])
        
        for i in range(samples_per_domain):
            # Select random template and add variation
            template = np.random.choice(templates)
            variation = f" (Sample {i+1})"
            text = template + variation
            
            texts.append(text)
            labels.append(domain)
    
    return Dataset.from_dict({
        "text": texts,
        "domain": labels
    })


def create_activation_dataset(
    target_domain: str = "stem",
    large_dataset_domains: List[str] = None,
    target_samples: int = 50,
    large_dataset_samples: int = 1000,
    seq_len: int = 128,
    hidden_dim: int = 512,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Create a complete dataset for FineScope demonstration."""
    
    if large_dataset_domains is None:
        large_dataset_domains = ["stem", "humanities", "business", "arts", "social_sciences"]
    
    print(f"Creating target domain activations for '{target_domain}'...")
    target_activations = create_domain_specific_activations(
        target_domain, target_samples, seq_len, hidden_dim, device
    )
    
    print(f"Creating large mixed dataset activations...")
    large_activations, domain_labels = create_mixed_domain_activations(
        large_dataset_domains, large_dataset_samples // len(large_dataset_domains),
        seq_len, hidden_dim, device
    )
    
    return target_activations, large_activations, domain_labels


def save_dummy_dataset(
    target_activations: torch.Tensor,
    large_activations: torch.Tensor,
    domain_labels: List[str],
    save_dir: str = "dummy_data"
):
    """Save dummy dataset to disk."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Save activations
    torch.save(target_activations, os.path.join(save_dir, "target_activations.pt"))
    torch.save(large_activations, os.path.join(save_dir, "large_activations.pt"))
    
    # Save domain labels
    with open(os.path.join(save_dir, "domain_labels.json"), "w") as f:
        json.dump(domain_labels, f)
    
    print(f"Dataset saved to {save_dir}/")
    print(f"Target activations: {target_activations.shape}")
    print(f"Large dataset activations: {large_activations.shape}")
    print(f"Domain labels: {len(domain_labels)} samples")


def load_dummy_dataset(load_dir: str = "dummy_data") -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Load dummy dataset from disk."""
    import os
    
    target_activations = torch.load(os.path.join(load_dir, "target_activations.pt"))
    large_activations = torch.load(os.path.join(load_dir, "large_activations.pt"))
    
    with open(os.path.join(load_dir, "domain_labels.json"), "r") as f:
        domain_labels = json.load(f)
    
    return target_activations, large_activations, domain_labels 