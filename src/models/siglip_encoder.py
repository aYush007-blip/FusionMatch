"""SigLIP Dual-Tower Encoder Wrapper with Multi-View Pooling and Freezing Controls."""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class SiglipDualEncoder(nn.Module):
    """Dual-tower vision and text encoder wrapping Google's SigLIP architecture.
    
    Exposes pooled vision and text representations separately and provides
    multi-view visual aggregation (mean-pooling over angle perspectives) and
    fine-grained parameter freezing / selective unfreezing.
    """

    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        freeze_vision: bool = True,
        freeze_text: bool = True,
        unfreeze_last_n_blocks: int = 0,
        use_mock: bool = False,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.use_mock = use_mock

        if not use_mock:
            try:
                self.backbone = AutoModel.from_pretrained(model_id)
                self.vision_dim = getattr(
                    self.backbone.config.vision_config, "hidden_size", 768
                )
                self.text_dim = getattr(
                    self.backbone.config.text_config, "hidden_size", 768
                )
            except Exception as e:
                print(f"Warning: Could not load pretrained weights ({e}). Initializing mock backbone.")
                self.use_mock = True

        if self.use_mock:
            self.vision_dim = 768
            self.text_dim = 768
            self.vision_mock = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=16, stride=16),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 768),
            )
            self.text_mock = nn.Sequential(
                nn.Embedding(32000, 128),
                nn.Linear(128, 768),
            )

        self._set_trainable(freeze_vision, freeze_text, unfreeze_last_n_blocks)

    def _set_trainable(
        self, freeze_vision: bool, freeze_text: bool, unfreeze_last_n_blocks: int
    ) -> None:
        """Configures parameter gradient requirements for vision and text towers."""
        if self.use_mock:
            for p in self.vision_mock.parameters():
                p.requires_grad = not freeze_vision
            for p in self.text_mock.parameters():
                p.requires_grad = not freeze_text
            return

        # Configure vision tower
        if hasattr(self.backbone, "vision_model"):
            for p in self.backbone.vision_model.parameters():
                p.requires_grad = not freeze_vision

            if unfreeze_last_n_blocks > 0 and hasattr(self.backbone.vision_model, "encoder"):
                v_layers = self.backbone.vision_model.encoder.layers
                for layer in list(v_layers)[-unfreeze_last_n_blocks:]:
                    for p in layer.parameters():
                        p.requires_grad = True

        # Configure text tower
        if hasattr(self.backbone, "text_model"):
            for p in self.backbone.text_model.parameters():
                p.requires_grad = not freeze_text

            if unfreeze_last_n_blocks > 0 and hasattr(self.backbone.text_model, "encoder"):
                t_layers = self.backbone.text_model.encoder.layers
                for layer in list(t_layers)[-unfreeze_last_n_blocks:]:
                    for p in layer.parameters():
                        p.requires_grad = True

    def forward_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encodes single or multi-view image tensors.
        
        Args:
            pixel_values: Tensor of shape (B, 3, H, W) or (B, K, 3, H, W)
            
        Returns:
            v_pool: Pooled visual representation of shape (B, vision_dim)
        """
        if pixel_values.ndim == 5:
            # Multi-view input (B, K, 3, H, W)
            B, K, C, H, W = pixel_values.shape
            flat_pixels = pixel_values.view(B * K, C, H, W)
            
            if self.use_mock:
                v_feat = self.vision_mock(flat_pixels)  # (B*K, 768)
            else:
                out = self.backbone.vision_model(pixel_values=flat_pixels)
                v_feat = (
                    out.pooler_output
                    if getattr(out, "pooler_output", None) is not None
                    else out.last_hidden_state.mean(dim=1)
                )
            
            # Mean-pool across the K angle perspectives: (B, K, 768) -> (B, 768)
            v_pool = v_feat.view(B, K, -1).mean(dim=1)
        else:
            # Single-view input (B, 3, H, W)
            if self.use_mock:
                v_pool = self.vision_mock(pixel_values)
            else:
                out = self.backbone.vision_model(pixel_values=pixel_values)
                v_pool = (
                    out.pooler_output
                    if getattr(out, "pooler_output", None) is not None
                    else out.last_hidden_state.mean(dim=1)
                )

        return v_pool

    def forward_text(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encodes text token sequences.
        
        Args:
            input_ids: Tensor of shape (B, L)
            attention_mask: Tensor of shape (B, L)
            
        Returns:
            t_pool: Pooled text representation of shape (B, text_dim)
        """
        if self.use_mock:
            emb = self.text_mock[0](input_ids)  # (B, L, 128)
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                sum_emb = (emb * mask).sum(dim=1)
                lens = mask.sum(dim=1).clamp(min=1.0)
                mean_emb = sum_emb / lens
            else:
                mean_emb = emb.mean(dim=1)
            t_pool = self.text_mock[1](mean_emb)  # (B, 768)
        else:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            out = self.backbone.text_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            t_pool = (
                out.pooler_output
                if getattr(out, "pooler_output", None) is not None
                else out.last_hidden_state.mean(dim=1)
            )

        return t_pool

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs joint forward pass over vision and text towers.
        
        Returns:
            Tuple of (v_pool, t_pool), each of shape (B, 768).
        """
        v_pool = self.forward_vision(pixel_values)
        t_pool = self.forward_text(input_ids, attention_mask)
        return v_pool, t_pool
