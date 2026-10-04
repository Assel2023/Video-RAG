# src/videorag/ingestion/encoder.py — Multimodal Encoder
from __future__ import annotations
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from videorag.config import (
    SIGLIP2_MODEL,
    TEXT_MODEL,
)
from videorag.logger import get_logger

log = get_logger(__name__)


def _pooled_features(output, modality: str):
    """Return pooled embeddings across Transformers ModelOutput/tuple APIs."""
    features = getattr(output, "pooler_output", None)
    if features is None and isinstance(output, (tuple, list)) and len(output) > 1:
        features = output[1]
    if features is None:
        raise RuntimeError(f"SigLIP2 returned no pooled {modality} features")
    return features


class MultimodalEncoder:
    """
    Encodes video frames with SigLIP 2 and transcripts with multilingual MiniLM.

    Visual and transcript vectors stay in separate Qdrant named-vector spaces:
    SigLIP 2 emits 768 dimensions and multilingual MiniLM emits 384 dimensions.
    The searcher combines their ranked results at query time.
    """

    def __init__(
        self,
        text_model_name:       str = TEXT_MODEL,
    ):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.visual_model = "siglip2"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Loading SigLIP 2 model: {SIGLIP2_MODEL} on {self.device}")
        self.siglip_processor = AutoProcessor.from_pretrained(SIGLIP2_MODEL)
        self.siglip_model = AutoModel.from_pretrained(SIGLIP2_MODEL).to(self.device).eval()
        # SigLIP2's pooled vision features use the vision hidden size (768).
        self.visual_dim = int(self.siglip_model.config.vision_config.hidden_size)

        log.info(f"Loading Audio Text model: {text_model_name}")
        self.text_audio = SentenceTransformer(text_model_name)
        log.info(f"MultimodalEncoder ready — visual={self.visual_model}/{self.visual_dim}, audio=384")

    # ── Visual Track ─────────────────────────────────────────────
    def encode_visual(self, frame_path: str) -> np.ndarray:
        """Encode one keyframe with SigLIP 2."""
        try:
            img = Image.open(frame_path).convert("RGB")
            return self.encode_visual_batch([img])[0]
        except Exception as exc:
            log.warning(f"Visual encoding failed ({frame_path}): {exc}")
            return np.zeros(self.visual_dim, dtype=np.float32)

    def encode_visual_batch(
        self, images: list[Image.Image], batch_size: int = 32
    ) -> np.ndarray:
        """Encode images into normalized SigLIP 2 vectors."""
        import torch
        vectors = []
        for start in range(0, len(images), batch_size):
            inputs = self.siglip_processor(
                images=images[start:start + batch_size], return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                output = self.siglip_model.get_image_features(**inputs)
                batch = _pooled_features(output, "image")
                batch = torch.nn.functional.normalize(batch, p=2, dim=-1)
            vectors.append(batch.cpu().numpy().astype(np.float32))
        return np.concatenate(vectors, axis=0)

    def encode_visual_text(self, texts: str | list[str]) -> np.ndarray:
        """Encode natural-language visual queries in SigLIP 2's text space."""
        import torch
        text_batch = [texts] if isinstance(texts, str) else texts
        inputs = self.siglip_processor(
            text=text_batch,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            output = self.siglip_model.get_text_features(**inputs)
            vectors = _pooled_features(output, "text")
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=-1)
        return vectors.cpu().numpy().astype(np.float32)

    # ── Audio Track ──────────────────────────────────────────────
    def encode_audio(self, transcript: str) -> np.ndarray:
        """Encode transcript → normalized 384-dim multilingual vector."""
        if not transcript or len(transcript.strip()) < 3:
            return np.zeros(384, dtype=np.float32)
        vec = self.text_audio.encode(
            transcript, convert_to_numpy=True, normalize_embeddings=True
        )
        return vec.astype(np.float32)
