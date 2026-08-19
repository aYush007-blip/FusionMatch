"""Contrastive Loss and Hard Negative Mining Modules for FusionMatch."""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE (Noise Contrastive Estimation) Contrastive Loss.
    
    Supports both symmetric in-batch negative pairs and optional mined hard negative
    representations from the HardNegativeMiner bank.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temp: bool = False,
        symmetric: bool = False,
    ) -> None:
        super().__init__()
        self.symmetric = symmetric
        import math
        init_val = float(math.log(max(temperature, 1e-4)))
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
        else:
            self.register_buffer("log_temp", torch.tensor(init_val, dtype=torch.float32))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp).clamp(min=0.01, max=1.0)

    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        hard_negative_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes InfoNCE contrastive loss.
        
        Args:
            anchor_emb: Tensor of shape (B, D), L2-normalized.
            positive_emb: Tensor of shape (B, D), L2-normalized.
            hard_negative_emb: Optional mined hard negative tensor of shape (B, K, D).
            
        Returns:
            Scalar contrastive cross-entropy loss tensor.
        """
        B = anchor_emb.size(0)
        temp = self.temperature
        device = anchor_emb.device

        # In-batch cross similarity logits (B, B)
        logits_in_batch = (anchor_emb @ positive_emb.T) / temp
        labels = torch.arange(B, device=device, dtype=torch.long)

        if hard_negative_emb is not None and hard_negative_emb.size(1) > 0:
            # Similarity of each anchor to its K mined hard negatives: (B, K)
            hard_logits = torch.einsum("bd,bkd->bk", anchor_emb, hard_negative_emb) / temp
            logits = torch.cat([logits_in_batch, hard_logits], dim=1)  # (B, B + K)
        else:
            logits = logits_in_batch

        loss = F.cross_entropy(logits, labels)

        if self.symmetric and hard_negative_emb is None:
            logits_rev = (positive_emb @ anchor_emb.T) / temp
            loss_rev = F.cross_entropy(logits_rev, labels)
            loss = 0.5 * (loss + loss_rev)

        return loss


class HardNegativeMiner:
    """FIFO Embedding Bank for Mining In-Category Hard Negatives.
    
    Stores a rolling buffer of catalog embeddings and associated SKU identifiers.
    When querying hard negatives for an anchor batch, same-SKU entries are masked
    with a large negative penalty (-10^4) to prevent false-negative contamination.
    """

    def __init__(
        self,
        bank_size: int = 4096,
        embed_dim: int = 256,
        k: int = 4,
        device: str = "cpu",
    ) -> None:
        self.bank_size = bank_size
        self.embed_dim = embed_dim
        self.k = k
        self.device = device

        self.bank = torch.zeros(bank_size, embed_dim, device=device)
        self.sku_ids: List[Optional[str]] = [None] * bank_size
        self.ptr = 0
        self.full = False

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, sku_ids: List[str]) -> None:
        """Appends new batch embeddings and SKU IDs to the FIFO buffer."""
        n = embeddings.size(0)
        if n == 0:
            return

        emb_cpu = embeddings.detach().to(self.device)
        end = self.ptr + n

        if end <= self.bank_size:
            self.bank[self.ptr:end] = emb_cpu
            self.sku_ids[self.ptr:end] = sku_ids
        else:
            first = self.bank_size - self.ptr
            self.bank[self.ptr:] = emb_cpu[:first]
            self.sku_ids[self.ptr:] = sku_ids[:first]

            remaining = n - first
            self.bank[:remaining] = emb_cpu[first:]
            self.sku_ids[:remaining] = sku_ids[first:]
            self.full = True

        self.ptr = end % self.bank_size

    @property
    def current_size(self) -> int:
        """Returns the number of valid items currently in the bank."""
        return self.bank_size if self.full else self.ptr

    @torch.no_grad()
    def mine(
        self,
        anchor_embeddings: torch.Tensor,
        anchor_sku_ids: List[str],
    ) -> torch.Tensor:
        """Retrieves top-K most similar hard negatives for each anchor SKU.
        
        Args:
            anchor_embeddings: Tensor of shape (B, embed_dim)
            anchor_sku_ids: List of string SKU IDs of length B
            
        Returns:
            hard_negs: Tensor of shape (B, K, embed_dim)
        """
        B = anchor_embeddings.size(0)
        num_avail = self.current_size

        if num_avail < self.k:
            # Not enough items in bank yet
            return torch.empty((B, 0, self.embed_dim), device=anchor_embeddings.device)

        anchors_dev = anchor_embeddings.to(self.device)
        active_bank = self.bank[:num_avail]  # (N, D)
        active_skus = self.sku_ids[:num_avail]

        # Compute cosine similarity between anchors and bank: (B, N)
        sims = anchors_dev @ active_bank.T

        hard_negs = []
        actual_k = min(self.k, num_avail)

        for i, sku in enumerate(anchor_sku_ids):
            row = sims[i].clone()
            # Mask out any entry that belongs to the same SKU (false negative prevention)
            mask = torch.tensor(
                [1.0 if active_skus[j] == sku else 0.0 for j in range(num_avail)],
                device=self.device,
                dtype=torch.float32,
            )
            row = row - mask * 1e4
            topk_idx = torch.topk(row, k=actual_k).indices
            hard_negs.append(active_bank[topk_idx])

        stacked = torch.stack(hard_negs, dim=0)  # (B, K, D)
        return stacked.to(anchor_embeddings.device)
