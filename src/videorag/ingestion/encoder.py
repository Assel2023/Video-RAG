# src/videorag/ingestion/encoder.py — Multimodal Encoder
from __future__ import annotations
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from videorag.config import CLIP_IMAGE_MODEL, CLIP_TEXT_MODEL, TEXT_MODEL, EMBED_DIM, VISUAL_ALPHA
from videorag.logger import get_logger

log = get_logger(__name__)


class MultimodalEncoder:
    """
    Encodes video chunks into unified multimodal embeddings.

    Implements the Joint Understanding fusion formula:

        V = f(Visual, Audio, Temporal)
        fused = α · visual_norm + (1-α) · zero_pad(audio_norm, 512)
        fused = fused / ‖fused‖₂

    where:
        visual_norm:  CLIP ViT-B/32 image embedding  (512-dim)
        audio_norm:   Multilingual MiniLM text emb.   (384-dim → 512-dim padded)
        α:            VISUAL_ALPHA (default 0.5)
    """

    def __init__(
        self,
        clip_image_model_name: str = CLIP_IMAGE_MODEL,
        clip_text_model_name:  str = CLIP_TEXT_MODEL,
        text_model_name:       str = TEXT_MODEL,
        alpha: float = VISUAL_ALPHA,
    ):
        log.info(f"Loading CLIP Image model: {clip_image_model_name}")
        self.clip_image = SentenceTransformer(clip_image_model_name)
        
        log.info(f"Loading CLIP Text (Multilingual) model: {clip_text_model_name}")
        self.clip_text  = SentenceTransformer(clip_text_model_name)
        
        log.info(f"Loading Audio Text model: {text_model_name}")
        self.text_audio = SentenceTransformer(text_model_name)
        
        self.alpha = alpha
        log.info(
            f"MultimodalEncoder ready — "
            f"α(visual)={alpha}, α(audio)={1-alpha}, dim={EMBED_DIM}"
        )

    # ── Visual Track ─────────────────────────────────────────────
    def encode_visual(self, frame_path: str) -> np.ndarray:
        """Encode a keyframe image → normalized 512-dim CLIP vector."""
        try:
            img = Image.open(frame_path).convert("RGB")
            vec = self.clip_image.encode(
                img, convert_to_numpy=True, normalize_embeddings=True
            )
            return vec.astype(np.float32)
        except Exception as exc:
            log.warning(f"Visual encoding failed ({frame_path}): {exc}")
            return np.zeros(EMBED_DIM, dtype=np.float32)

    # ── Audio Track ──────────────────────────────────────────────
    def encode_audio(self, transcript: str) -> np.ndarray:
        """Encode transcript → normalized 384-dim multilingual vector."""
        if not transcript or len(transcript.strip()) < 3:
            return np.zeros(384, dtype=np.float32)
        vec = self.text_audio.encode(
            transcript, convert_to_numpy=True, normalize_embeddings=True
        )
        return vec.astype(np.float32)

    # ── Fusion ───────────────────────────────────────────────────
    def fuse(
        self, visual_vec: np.ndarray, audio_vec: np.ndarray
    ) -> list[float]:
        """
        Joint fusion of visual and audio embeddings.

        Formula:
            audio_projected = zero_pad(audio_vec, EMBED_DIM)
            fused = α·visual + (1-α)·audio_projected
            fused = fused / ‖fused‖₂
        """
        audio_projected = np.zeros(EMBED_DIM, dtype=np.float32)
        audio_projected[: len(audio_vec)] = audio_vec

        fused = self.alpha * visual_vec + (1.0 - self.alpha) * audio_projected
        norm  = np.linalg.norm(fused)
        if norm > 1e-9:
            fused /= norm
        return fused.tolist()

    # ── Query Encoding ───────────────────────────────────────────
    def encode_query(self, query: str) -> list[float]:
        """
        Encode a text query into the shared 512-dim embedding space.

        Uses the multilingual CLIP text encoder + multilingual MiniLM,
        then fuses them identically to how chunk embeddings are stored.
        """
        clip_vec = self.clip_text.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        text_vec = self.text_audio.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        return self.fuse(clip_vec, text_vec)
