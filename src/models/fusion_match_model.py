"""Top-Level FusionMatch Multi-Modal Model Architecture."""

from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
from .siglip_encoder import SiglipDualEncoder
from .gated_fusion import GatedFusion
from .projection_head import ProjectionHead
from .quality_proxies import image_quality_score, text_quality_score


class FusionMatchModel(nn.Module):
    """FusionMatch Cross-Modal & Multi-View Product Representation Model.
    
    Composes:
    1. SiglipDualEncoder: Vision & Text transformer towers with multi-view pooling.
    2. GatedFusion: Quality-aware learnable softmax gating network.
    3. ProjectionHead: MLP with L2 hypersphere normalization.
    """

    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        freeze_vision: bool = True,
        freeze_text: bool = True,
        unfreeze_last_n_blocks: int = 0,
        embed_dim: int = 256,
        shared_dim: int = 768,
        hidden_gate_dim: int = 128,
        use_mock: bool = False,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.embed_dim = embed_dim
        self.shared_dim = shared_dim

        # 1. Dual Encoder Backbone
        self.encoder = SiglipDualEncoder(
            model_id=model_id,
            freeze_vision=freeze_vision,
            freeze_text=freeze_text,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            use_mock=use_mock,
        )

        # 2. Quality-Aware Gated Fusion Module
        self.fusion = GatedFusion(
            vision_dim=self.encoder.vision_dim,
            text_dim=self.encoder.text_dim,
            shared_dim=shared_dim,
            hidden_gate_dim=hidden_gate_dim,
        )

        # 3. Contrastive Projection Head (Unit-Norm 256-d Embedding)
        self.proj_head = ProjectionHead(
            in_dim=shared_dim,
            hidden_dim=512,
            out_dim=embed_dim,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        q_v: Optional[torch.Tensor] = None,
        q_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs end-to-end forward pass producing unit-norm embeddings and gating weights.
        
        Args:
            pixel_values: Image tensor of shape (B, 3, H, W) or multi-view (B, K, 3, H, W).
            input_ids: Text token tensor of shape (B, L).
            attention_mask: Text attention mask of shape (B, L).
            q_v: Optional visual quality proxy scores of shape (B,) or (B, 1).
            q_t: Optional textual quality proxy scores of shape (B,) or (B, 1).
            
        Returns:
            embedding: L2-normalized representation of shape (B, embed_dim) with ||z||_2 = 1.0.
            gates: Modality weights tensor of shape (B, 2) where g_v + g_t = 1.0.
        """
        B = pixel_values.size(0)

        # Default quality proxies to 1.0 if not provided
        if q_v is None:
            q_v = torch.ones(B, dtype=torch.float32, device=pixel_values.device)
        if q_t is None:
            q_t = torch.ones(B, dtype=torch.float32, device=input_ids.device)

        # 1. Dual-tower encoding
        v_pool, t_pool = self.encoder(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # 2. Quality-weighted gated fusion
        fused, gates = self.fusion(v_pool, t_pool, q_v, q_t)

        # 3. Contrastive projection & hypersphere normalization
        embedding = self.proj_head(fused)

        return embedding, gates

    def encode_multimodal(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        q_v: Optional[torch.Tensor] = None,
        q_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference wrapper returning normalized embedding and gate values."""
        return self.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            q_v=q_v,
            q_t=q_t,
        )

    def num_trainable_params(self) -> int:
        """Returns total count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        """Returns total count of all model parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_param_budget_summary(self) -> Dict[str, Any]:
        """Provides component-level breakdown of parameter counts."""
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        fusion_trainable = sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)

        proj_params = sum(p.numel() for p in self.proj_head.parameters())
        proj_trainable = sum(p.numel() for p in self.proj_head.parameters() if p.requires_grad)

        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

        return {
            "total_params": self.num_total_params(),
            "trainable_params": self.num_trainable_params(),
            "encoder": {
                "total": encoder_params,
                "trainable": encoder_trainable,
            },
            "fusion": {
                "total": fusion_params,
                "trainable": fusion_trainable,
            },
            "projection_head": {
                "total": proj_params,
                "trainable": proj_trainable,
            },
        }
