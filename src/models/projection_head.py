"""Contrastive Projection Head with L2 Hypersphere Normalization."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Multi-layer perceptron projection head mapping fused representations
    to a compact, L2-normalized embedding space (unit hypersphere).
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 512,
        out_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input representation to unit-norm embedding.
        
        Args:
            x: Input tensor of shape (B, in_dim)
            
        Returns:
            embedding: L2-normalized tensor of shape (B, out_dim) where ||z||_2 = 1.0
        """
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)
