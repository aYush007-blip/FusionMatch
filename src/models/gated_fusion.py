"""Quality-Aware Gated Fusion Module for Cross-Modal Representation Learning."""

from typing import Tuple
import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """Dynamically projects and fuses visual and textual representations.
    
    The fusion gate weights (g_v, g_t with g_v + g_t = 1) are computed via a learned
    gating network conditioned jointly on the projected multimodal features and
    the heuristic quality proxy scalars (blur variance q_v and token density q_t).
    """

    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 768,
        shared_dim: int = 768,
        hidden_gate_dim: int = 128,
    ) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.shared_dim = shared_dim

        # Modality projection layers
        self.vision_proj = nn.Linear(vision_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)

        # Gate network: conditioned on [v_proj, t_proj, q_v, q_t]
        gate_input_dim = shared_dim * 2 + 2
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_gate_dim),
            nn.ReLU(),
            nn.Linear(hidden_gate_dim, 2),  # Raw logits for [g_v, g_t]
        )

    def forward(
        self,
        v_pool: torch.Tensor,
        t_pool: torch.Tensor,
        q_v: torch.Tensor,
        q_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuses vision and text vectors guided by quality signals.
        
        Args:
            v_pool: Visual embedding tensor of shape (B, vision_dim)
            t_pool: Text embedding tensor of shape (B, text_dim)
            q_v: Image quality proxy tensor of shape (B,) or (B, 1)
            q_t: Text quality proxy tensor of shape (B,) or (B, 1)
            
        Returns:
            fused: Multimodal fused representation of shape (B, shared_dim)
            gates: Modality weights tensor of shape (B, 2) where g_v + g_t = 1
        """
        # Ensure 2D column tensors for quality proxies
        if q_v.ndim == 1:
            q_v = q_v.unsqueeze(-1)
        if q_t.ndim == 1:
            q_t = q_t.unsqueeze(-1)

        # Align devices/dtypes if needed
        q_v = q_v.to(dtype=v_pool.dtype, device=v_pool.device)
        q_t = q_t.to(dtype=t_pool.dtype, device=t_pool.device)

        # Project modalities into shared dimension
        v_proj = self.vision_proj(v_pool)  # (B, shared_dim)
        t_proj = self.text_proj(t_pool)    # (B, shared_dim)

        # Gate network inputs
        gate_input = torch.cat([v_proj, t_proj, q_v, q_t], dim=-1)  # (B, 2*shared_dim + 2)
        gate_logits = self.gate_net(gate_input)                     # (B, 2)
        gates = torch.softmax(gate_logits, dim=-1)                   # (B, 2)

        g_v = gates[:, 0:1]  # (B, 1)
        g_t = gates[:, 1:2]  # (B, 1)

        # Convex combination of projected features
        fused = g_v * v_proj + g_t * t_proj  # (B, shared_dim)
        return fused, gates
