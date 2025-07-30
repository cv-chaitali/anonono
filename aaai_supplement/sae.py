import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from pathlib import Path
from typing import NamedTuple, Optional, Dict, Any
from tqdm import tqdm
import numpy as np
from safetensors.torch import load_model, save_model

from config import SaeConfig, TrainingConfig


class EncoderOutput(NamedTuple):
    """Output from SAE encoder."""
    top_acts: Tensor
    top_indices: Tensor


class ForwardOutput(NamedTuple):
    """Output from SAE forward pass."""
    sae_out: Tensor
    latent_acts: Tensor
    latent_indices: Tensor
    fvu: Tensor
    auxk_loss: Tensor
    l1_loss: Tensor
    reconstruction_loss: Tensor
    multi_topk_fvu: Tensor


class Sae(nn.Module):
    """Sparse Autoencoder for LLM activation compression."""
    
    def __init__(
        self,
        d_in: int,
        cfg: SaeConfig,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
        decoder: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor
        
        # Initialize encoder
        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()
        
        # Initialize decoder
        self.W_dec = nn.Parameter(self.encoder.weight.data.clone()) if decoder else None
        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))
        
        if decoder and self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()
    
    @staticmethod
    def load_from_disk(
        path: Path | str,
        device: str = "cpu",
        decoder: bool = True,
    ) -> "Sae":
        """Load a pre-trained SAE from disk."""
        path = Path(path)
        
        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SaeConfig.from_dict(cfg_dict, drop_extra_fields=True)
        
        sae = Sae(d_in, cfg, device=device, decoder=decoder)
        load_model(
            model=sae,
            filename=str(path / "sae.safetensors"),
            device=str(device),
            strict=decoder,
        )
        return sae
    
    def save_to_disk(self, path: Path | str):
        """Save SAE to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(
                {
                    **self.cfg.to_dict(),
                    "d_in": self.d_in,
                },
                f,
            )
    
    @property
    def device(self):
        return self.encoder.weight.device
    
    @property
    def dtype(self):
        return self.encoder.weight.dtype
    
    def pre_acts(self, x: Tensor) -> Tensor:
        """Compute pre-activations."""
        sae_in = x.to(self.dtype) - self.b_dec
        out = self.encoder(sae_in)
        return nn.functional.relu(out)
    
    def select_topk(self, latents: Tensor) -> EncoderOutput:
        """Select the top-k latents."""
        return EncoderOutput(*latents.topk(self.cfg.k, sorted=False))
    
    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        return self.select_topk(self.pre_acts(x))
    
    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        """Decode top-k activations back to original space."""
        assert self.W_dec is not None, "Decoder weight was not initialized."
        
        # Eager implementation
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.W_dec.shape[-1],))
        acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
        y = acts @ self.W_dec.mT
        return y + self.b_dec
    
    def forward(self, x: Tensor, dead_mask: Optional[Tensor] = None) -> ForwardOutput:
        """Forward pass through the SAE."""
        pre_acts = self.pre_acts(x)
        
        # Decode and compute residual
        top_acts, top_indices = self.select_topk(pre_acts)
        sae_out = self.decode(top_acts, top_indices)
        e = sae_out - x
        
        # Compute losses
        total_variance = (x - x.mean(0)).pow(2).sum()
        reconstruction_loss = (
            ((e) ** 2).mean(dim=1) / (x**2).mean(dim=1)
        ).mean()
        l1_loss = (
            pre_acts.abs().sum(dim=1) / x.norm(dim=1)
        ).mean()
        
        # AuxK loss (if applicable)
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = x.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)
        
        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance
        
        # Multi-TopK FVU
        if self.cfg.multi_topk:
            top_acts, top_indices = pre_acts.topk(4 * self.cfg.k, sorted=False)
            sae_out = self.decode(top_acts, top_indices)
            multi_topk_fvu = (sae_out - x).pow(2).sum() / total_variance
        else:
            multi_topk_fvu = sae_out.new_tensor(0.0)
        
        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
            multi_topk_fvu,
            l1_loss,
            reconstruction_loss
        )
    
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        """Normalize decoder weights to unit norm."""
        assert self.W_dec is not None, "Decoder weight was not initialized."
        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps
    
    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """Remove gradient components parallel to decoder directions."""
        assert self.W_dec is not None, "Decoder weight was not initialized."
        assert self.W_dec.grad is not None
        
        import einops
        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )


def train_sae(
    activations: Tensor,
    cfg: SaeConfig,
    train_cfg: TrainingConfig,
    save_path: Optional[str] = None
) -> Sae:
    """Train a sparse autoencoder on activations."""
    
    # Initialize SAE
    d_in = activations.shape[-1]
    sae = Sae(d_in, cfg, device=train_cfg.device)
    
    # Setup optimizer
    optimizer = optim.Adam(sae.parameters(), lr=train_cfg.learning_rate)
    
    # Training loop
    num_batches = (len(activations) + train_cfg.batch_size - 1) // train_cfg.batch_size
    
    print(f"Training SAE for {train_cfg.num_epochs} epochs...")
    
    for epoch in range(train_cfg.num_epochs):
        sae.train()
        total_loss = 0.0
        total_reconstruction_loss = 0.0
        total_l1_loss = 0.0
        
        # Shuffle data
        indices = torch.randperm(len(activations))
        
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{train_cfg.num_epochs}"):
            start_idx = batch_idx * train_cfg.batch_size
            end_idx = min(start_idx + train_cfg.batch_size, len(activations))
            batch_indices = indices[start_idx:end_idx]
            batch_data = activations[batch_indices].to(train_cfg.device)
            
            # Forward pass
            output = sae(batch_data)
            
            # Compute loss
            reconstruction_loss = output.reconstruction_loss
            l1_loss = output.l1_loss
            total_loss = reconstruction_loss + train_cfg.l1_weight * l1_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Remove gradient components parallel to decoder directions
            sae.remove_gradient_parallel_to_decoder_directions()
            
            optimizer.step()
            
            # Update metrics
            total_reconstruction_loss += reconstruction_loss.item()
            total_l1_loss += l1_loss.item()
        
        # Log metrics
        avg_reconstruction_loss = total_reconstruction_loss / num_batches
        avg_l1_loss = total_l1_loss / num_batches
        
        if (epoch + 1) % train_cfg.log_every == 0:
            print(f"Epoch {epoch+1}: Reconstruction Loss: {avg_reconstruction_loss:.6f}, L1 Loss: {avg_l1_loss:.6f}")
        
        # Save model
        if save_path and (epoch + 1) % train_cfg.save_every == 0:
            sae.save_to_disk(f"{save_path}_epoch_{epoch+1}")
    
    # Save final model
    if save_path:
        sae.save_to_disk(save_path)
    
    return sae


def extract_embeddings(sae: Sae, activations: Tensor, top_k: int = 100) -> np.ndarray:
    """Extract embeddings from activations using trained SAE."""
    sae.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(len(activations)), desc="Extracting embeddings"):
            # Get activations for current sample
            sample_acts = activations[i]  # (seq_len, hidden_dim)
            
            # Select top-k activations
            top_vals, _ = torch.topk(torch.abs(sample_acts), top_k, dim=-1, sorted=False)
            avg_acts = top_vals.mean(dim=0).to(torch.float32)
            
            # Get SAE embeddings
            sae_output = sae(avg_acts.unsqueeze(0))
            embedding = sae_output.latent_acts.cpu().numpy().flatten()
            embeddings.append(embedding)
    
    return np.array(embeddings) 