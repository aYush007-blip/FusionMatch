"""High-Performance Inference Engine combining ONNX Runtime and FAISS Vector Search."""

from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import io
import base64
import json
import numpy as np
from PIL import Image
import faiss
import onnxruntime as ort
from transformers import AutoTokenizer

from .schemas import CheckResponse, Candidate, BatchCheckItem, BatchCheckResult
from ..indexing.threshold_calibration import BayesianThresholdCalibrator
from ..models.quality_proxies import compute_single_image_quality, compute_single_text_quality


class FusionMatchInferenceEngine:
    """Orchestrates end-to-end multi-modal inference:
    
    1. Preprocessing & Quality Proxy extraction on submitted Image & Text.
    2. ONNX Runtime forward pass (INT8 optimized).
    3. IVF-PQ FAISS sub-millisecond retrieval.
    4. Dynamic Bayesian threshold evaluation per category.
    """

    def __init__(
        self,
        onnx_path: Union[str, Path],
        index_path: Union[str, Path],
        id_map_path: Union[str, Path],
        thresholds_path: Optional[Union[str, Path]] = None,
        model_id: str = "google/siglip-base-patch16-224",
        nprobe: int = 16,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.index_path = Path(index_path)
        self.id_map_path = Path(id_map_path)
        self.nprobe = nprobe

        # 1. Initialize ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # 2. Initialize FAISS Vector Index
        self.index = faiss.read_index(str(self.index_path))
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

        # 3. Load ID Map
        with open(self.id_map_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)

        # 4. Load Bayesian Calibrator
        if thresholds_path and Path(thresholds_path).exists():
            self.calibrator = BayesianThresholdCalibrator.load(thresholds_path)
        else:
            self.calibrator = BayesianThresholdCalibrator()

        # 5. Initialize Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            self.tokenizer = None

    def _decode_image(self, image_base64: str) -> Tuple[np.ndarray, float]:
        """Decodes base64 string to normalized image tensor array and quality score."""
        try:
            # Strip data URI prefix if present
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            raw_bytes = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception as e:
            # Fallback for blank/corrupted images
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        q_v = compute_single_image_quality(img)
        img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)

        # Convert to float32 (3, 224, 224) with SigLIP normalization
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))  # (3, 224, 224)
        return arr, float(q_v)

    def _tokenize_text(self, text: Optional[str]) -> Tuple[np.ndarray, np.ndarray, float]:
        """Tokenizes text string and calculates text density quality proxy."""
        text_str = (text or "").strip()
        q_t = compute_single_text_quality(text_str)

        if self.tokenizer is not None:
            enc = self.tokenizer(
                text_str if text_str else "<empty>",
                padding="max_length",
                max_length=32,
                truncation=True,
                return_tensors="np",
            )
            input_ids = enc["input_ids"].astype(np.int64)
            attention_mask = enc["attention_mask"].astype(np.int64)
        else:
            # Tokenizer fallback
            input_ids = np.zeros((1, 32), dtype=np.int64)
            attention_mask = np.zeros((1, 32), dtype=np.int64)
            words = text_str.split()[:32]
            for i, w in enumerate(words):
                input_ids[0, i] = abs(hash(w)) % 10000 + 1
                attention_mask[0, i] = 1

        return input_ids, attention_mask, float(q_t)

    def _embed_multimodal(
        self,
        pixel_array: np.ndarray,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        q_v: float,
        q_t: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Executes ONNX forward pass returning unit-norm embedding and gate weights."""
        # Ensure batch dimension (1, ...)
        if pixel_array.ndim == 3:
            pixel_array = np.expand_dims(pixel_array, axis=0)

        inputs = {
            "pixel_values": pixel_array.astype(np.float32),
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
            "q_v": np.array([q_v], dtype=np.float32),
            "q_t": np.array([q_t], dtype=np.float32),
        }

        outputs = self.session.run(None, inputs)
        emb = outputs[0]  # (1, 256)
        gates = outputs[1][0]  # [g_v, g_t]

        # L2 normalize
        norm = np.linalg.norm(emb, axis=-1, keepdims=True)
        if norm > 0:
            emb = emb / norm

        gate_dict = {
            "visual": float(gates[0]),
            "textual": float(gates[1]),
        }
        return emb, gate_dict

    def check_single(
        self,
        image_base64: str,
        title: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 5,
    ) -> CheckResponse:
        """Processes a single product listing for duplicate detection."""
        pixel_arr, q_v = self._decode_image(image_base64)
        input_ids, attn_mask, q_t = self._tokenize_text(title)

        emb, gate_weights = self._embed_multimodal(pixel_arr, input_ids, attn_mask, q_v, q_t)

        # FAISS search
        sims, indices = self.index.search(emb.astype(np.float32), top_k)
        candidates: List[Candidate] = []

        for sim, idx in zip(sims[0], indices[0]):
            if idx >= 0:
                sku = self.id_map.get(str(idx), f"SKU_{idx}")
                candidates.append(Candidate(sku_id=sku, similarity=float(np.clip(sim, 0.0, 1.0))))

        threshold = self.calibrator.get_threshold(category)
        top_sim = candidates[0].similarity if candidates else 0.0
        is_dup = bool(top_sim >= threshold)

        return CheckResponse(
            is_duplicate=is_dup,
            threshold_used=threshold,
            candidates=candidates,
            gate_weights=gate_weights,
        )

    def check_batch(
        self,
        items: List[BatchCheckItem],
        top_k: int = 5,
    ) -> List[BatchCheckResult]:
        """Processes a batch of items sequentially or vectorized."""
        results: List[BatchCheckResult] = []
        for item in items:
            single_resp = self.check_single(
                image_base64=item.image_base64,
                title=item.title,
                category=item.category,
                top_k=top_k,
            )
            results.append(
                BatchCheckResult(
                    item_id=item.item_id,
                    is_duplicate=single_resp.is_duplicate,
                    threshold_used=single_resp.threshold_used,
                    candidates=single_resp.candidates,
                    gate_weights=single_resp.gate_weights,
                )
            )
        return results
