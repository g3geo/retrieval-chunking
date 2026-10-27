"""Synthetic corpus/vector fixtures with real FAISS; no downloaded E5 model."""

from copy import deepcopy

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from chunkeval.config import PipelineError, file_sha256, write_json
from chunkeval.embedding import embedding_identity
from chunkeval.indexing import build_index, load_index


class SyntheticEncoder:
    def __init__(self, config):
        self.config = config
        self.tokenizer = self
        self.identity = embedding_identity(config)
        self.embedding_config_hash = self.identity["embedding_config_hash"]

    def count(self, text, prefix):
        return len(text + prefix) + 2

    def encode_passages(self, texts):
        rows = []
        for text in texts:
            row = np.zeros(384, dtype=np.float32)
            row[sum(text.encode()) % 384] = 1
            rows.append(row)
        return np.stack(rows)


def config_fixture(tmp_path):
    return {"_app_root": tmp_path, "embedding": {"revision": "a" * 40}, "seed": 42,
            "semantic": {"sentence_prefix": "query: ", "buffer_size": 0},
            "chunking": {condition: {"strategy": "synthetic", "max_input_tokens": 256 if condition == "C1" else 512}
                         for condition in ("C1", "C2", "C3", "C4", "C5")}}


def chunk_fixture(condition, doc_id, start, text, number):
    return {"chunk_id": f"{condition}-{doc_id}-{number}", "condition": condition,
            "doc_id": doc_id, "text": text, "char_start": start, "char_end": start + len(text),
            "page_start": 1, "page_end": 1, "page_numbers": [1],
            "n_tokens_with_prefix": len("passage: " + text) + 2,
            "sentence_ids": [f"{doc_id}-S{number}"], "split_from": [], "structure_refs": []}


def test_roundtrip_deterministic_mapping_and_duplicate_self_search(tmp_path):
    config = config_fixture(tmp_path)
    encoder = SyntheticEncoder(config)
    chunks = [chunk_fixture("C1", "D2", 0, "duplikat", 0),
              chunk_fixture("C1", "D1", 12, "lain", 1),
              chunk_fixture("C1", "D1", 0, "duplikat", 0)]
    output = tmp_path / "indexes/C1"
    loaded = build_index("C1", chunks, encoder, config, "a" * 64, output)
    assert loaded.index.d == 384 and loaded.index.ntotal == 3
    assert [row["chunk_id"] for row in loaded.chunks] == ["C1-D1-0", "C1-D1-1", "C1-D2-0"]
    vectors = encoder.encode_passages([row["text"] for row in loaded.chunks])
    scores, row_ids = loaded.index.search(vectors[:1], loaded.index.ntotal)
    assert scores[0, 0] == pytest.approx(1)
    assert {int(row) for row, score in zip(row_ids[0], scores[0]) if score == 1} == {0, 2}
    original_hash = loaded.manifest_hash
    repeat = build_index("C1", chunks, encoder, config, "a" * 64, output)
    assert repeat.manifest_hash == original_hash
    assert load_index(output, config, "a" * 64).chunks == loaded.chunks


def test_checksum_corruption_and_stale_corpus_are_rejected(tmp_path):
    config = config_fixture(tmp_path)
    output = tmp_path / "indexes/C1"
    build_index("C1", [chunk_fixture("C1", "D1", 0, "isi", 0)], SyntheticEncoder(config), config, "a" * 64, output)
    with pytest.raises(PipelineError, match="corpus"):
        load_index(output, config, "b" * 64)
    mapping = output / "chunks.jsonl"
    mapping.write_bytes(mapping.read_bytes() + b"\n")
    with pytest.raises(PipelineError, match="checksum"):
        load_index(output, config, "a" * 64)


@pytest.mark.parametrize("change", ["embedding", "chunking", "semantic"])
def test_current_configuration_mismatch_rejected(tmp_path, change):
    config = config_fixture(tmp_path)
    output = tmp_path / "indexes/C1"
    build_index("C1", [chunk_fixture("C1", "D1", 0, "isi", 0)], SyntheticEncoder(config), config, "a" * 64, output)
    changed = deepcopy(config)
    if change == "embedding":
        changed["embedding"]["revision"] = "b" * 40
    elif change == "chunking":
        changed["chunking"]["C1"]["max_input_tokens"] = 250
    else:
        changed["semantic"]["buffer_size"] = 1
    with pytest.raises(PipelineError, match="configuration"):
        load_index(output, changed, "a" * 64)


def test_empty_mapping_and_wrong_metadata_token_count_rejected(tmp_path):
    config = config_fixture(tmp_path)
    encoder = SyntheticEncoder(config)
    with pytest.raises(PipelineError, match="empty"):
        build_index("C1", [], encoder, config, "a" * 64, tmp_path / "empty")
    row = chunk_fixture("C1", "D1", 0, "isi", 0)
    row["n_tokens_with_prefix"] += 1
    with pytest.raises(PipelineError, match="tokens"):
        build_index("C1", [row], encoder, config, "a" * 64, tmp_path / "bad")


def test_loaded_non_normalized_vectors_rejected_even_with_updated_checksum(tmp_path):
    config = config_fixture(tmp_path)
    output = tmp_path / "indexes/C1"
    loaded = build_index("C1", [chunk_fixture("C1", "D1", 0, "isi", 0)], SyntheticEncoder(config), config, "a" * 64, output)
    corrupted = faiss.IndexFlatIP(384)
    corrupted.add(np.ones((1, 384), dtype=np.float32))
    faiss.write_index(corrupted, str(output / "index.faiss"))
    loaded.manifest["index_sha256"] = file_sha256(output / "index.faiss")
    write_json(output / "manifest.json", loaded.manifest)
    with pytest.raises(PipelineError, match="normalized"):
        load_index(output, config, "a" * 64)


def test_output_cannot_escape_app_root(tmp_path):
    config = config_fixture(tmp_path)
    with pytest.raises(PipelineError, match="application root"):
        build_index("C1", [chunk_fixture("C1", "D1", 0, "isi", 0)], SyntheticEncoder(config), config, "a" * 64, tmp_path.parent / "escaped")


def test_cache_toggle_is_runtime_only_but_chunking_source_change_invalidates_index(tmp_path, monkeypatch):
    import chunkeval.indexing as module

    config = config_fixture(tmp_path)
    config["semantic"]["cache_enabled"] = True
    output = tmp_path / "indexes/C1"
    build_index("C1", [chunk_fixture("C1", "D1", 0, "isi", 0)], SyntheticEncoder(config), config, "a" * 64, output)
    config["semantic"]["cache_enabled"] = False
    assert load_index(output, config, "a" * 64).index.ntotal == 1
    original = module.file_sha256
    monkeypatch.setattr(module, "file_sha256", lambda path: "b" * 64 if path.name == "chunking.py" else original(path))
    with pytest.raises(PipelineError, match="chunking configuration"):
        load_index(output, config, "a" * 64)
