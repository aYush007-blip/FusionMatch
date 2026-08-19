import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel
from src.config import settings

class MultiViewAttentionPooling(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.attn_vector = nn.Parameter(torch.randn(embed_dim, 1))

    def forward(self, view_tensors):
        if view_tensors.size(0) == 1:
            return F.normalize(view_tensors, p=2, dim=-1)
        scores = torch.matmul(view_tensors, self.attn_vector)
        weights = torch.softmax(scores, dim=0)
        pooled = torch.sum(weights * view_tensors, dim=0, keepdim=True)
        return F.normalize(pooled, p=2, dim=-1)

class DedupModelEngine:
    def __init__(self):
        self.device = torch.device("cpu")
        self.processor = AutoProcessor.from_pretrained(settings.MODEL_NAME)
        self.model = AutoModel.from_pretrained(settings.MODEL_NAME).to(self.device).eval()
        self.attention_pooler = MultiViewAttentionPooling(embed_dim=settings.EMBEDDING_DIM).to(self.device).eval()

    def compute_dynamic_alpha(self, title: str) -> float:
        word_count = len(title.strip().split())
        if word_count <= 3:
            return 0.85
        elif word_count >= 10:
            return 0.50
        return 0.70

    @torch.no_grad()
    def extract_fused_vector(self, images: list[Image.Image], title: str) -> torch.Tensor:
        # Multi-Angle Vision Encoding
        img_inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        img_out = self.model.get_image_features(**img_inputs)
        img_tensors = img_out.pooler_output if hasattr(img_out, "pooler_output") else (img_out[0] if not isinstance(img_out, torch.Tensor) else img_out)
        pooled_img = self.attention_pooler(img_tensors)

        # Text Encoding
        txt_inputs = self.processor(text=[title], padding="max_length", return_tensors="pt").to(self.device)
        txt_out = self.model.get_text_features(**txt_inputs)
        txt_tensors = txt_out.pooler_output if hasattr(txt_out, "pooler_output") else (txt_out[0] if not isinstance(txt_out, torch.Tensor) else txt_out)
        txt_vec = F.normalize(txt_tensors, p=2, dim=-1)

        # Dynamic Gated Fusion
        alpha = self.compute_dynamic_alpha(title)
        fused = F.normalize((alpha * pooled_img) + ((1.0 - alpha) * txt_vec), p=2, dim=-1)
        return fused