"""FusionMatch Model Architecture Package."""

from .siglip_encoder import SiglipDualEncoder
from .quality_proxies import image_quality_score, text_quality_score
from .gated_fusion import GatedFusion
from .projection_head import ProjectionHead
from .fusion_match_model import FusionMatchModel

__all__ = [
    "SiglipDualEncoder",
    "image_quality_score",
    "text_quality_score",
    "GatedFusion",
    "ProjectionHead",
    "FusionMatchModel",
]
