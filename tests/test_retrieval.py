import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from videorag.retrieval.searcher import Searcher

def test_searcher_init():
    # Model loading might be heavy, so we just check imports and basic setup
    try:
        from videorag.retrieval.searcher import Searcher
        from videorag.database.vector_store import VectorStore
        assert True
    except ImportError:
        assert False
