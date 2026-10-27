"""User-run setup checks; does not process PDFs or download/encode E5."""
from __future__ import annotations

import importlib
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main() -> int:
    result = subprocess.run([sys.executable, "-m", "pip", "check"], check=False)
    if result.returncode:
        return result.returncode
    for name in ("pypdf", "pytesseract", "pdf2image", "transformers", "sentence_transformers",
                 "torch", "numpy", "faiss", "yaml", "pytest", "langchain",
                 "langchain_core", "langchain_community", "langchain_experimental"):
        importlib.import_module(name)
        print(f"Import OK: {name}")
    if version("langchain") != "0.3.27" or version("langchain-experimental") != "0.3.4":
        raise RuntimeError("Reinstall requirements.lock: incompatible SemanticChunker version.")
    import numpy as np
    from langchain_experimental.text_splitter import SemanticChunker, calculate_cosine_distances
    from chunkeval.chunking import semantic_breakpoints
    from chunkeval.config import load_config
    from chunkeval.embedding import embedding_identity

    units = [{"unit_id": str(i), "sentence_id": str(i), "char_start": i,
              "char_end": i + 1, "text": "x"} for i in range(3)]
    vectors = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    distances, _ = calculate_cosine_distances([
        {"combined_sentence_embedding": row.tolist()} for row in vectors])
    # This calls only the pinned threshold helper; no external embedding service.
    adapter = SemanticChunker(embeddings=None, buffer_size=0,
                              breakpoint_threshold_type="percentile",
                              breakpoint_threshold_amount=80)
    threshold, _ = adapter._calculate_breakpoint_threshold(distances)
    expected = [i for i, distance in enumerate(distances) if distance > threshold]
    if semantic_breakpoints(units, vectors, 80) != expected:
        raise RuntimeError("Semantic adapter differs from the pinned library.")
    embedding_identity(load_config())
    print("Dependency, import, revision, and SemanticChunker adapter checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Setup check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
