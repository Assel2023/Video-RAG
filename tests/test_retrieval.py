# tests/test_retrieval.py — Integration and Unit Tests
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    from videorag.config import EMBED_DIM, CHUNK_SIZE
    from videorag.ingestion.encoder import MultimodalEncoder
    from videorag.database.vector_store import VectorStore
    assert EMBED_DIM == 512
    assert CHUNK_SIZE == 5

def test_fusion_math():
    import numpy as np
    from videorag.ingestion.encoder import MultimodalEncoder
    encoder = MultimodalEncoder()
    dummy_visual = np.random.randn(512).astype(np.float32)
    dummy_audio = np.random.randn(384).astype(np.float32)
    fused = encoder.fuse(dummy_visual, dummy_audio)
    assert len(fused) == 512
    # Ensure it is normalized (unit vector)
    norm = np.linalg.norm(fused)
    assert np.isclose(norm, 1.0, atol=1e-3)

if __name__ == "__main__":
    print("Running tests...")
    test_imports()
    print("✓ test_imports passed")
    test_fusion_math()
    print("✓ test_fusion_math passed")
    print("All tests passed successfully!")
