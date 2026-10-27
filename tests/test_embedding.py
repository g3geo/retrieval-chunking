"""Synthetic tests: these fixtures are not research inputs and never load E5."""

from copy import deepcopy
import hashlib

import numpy as np
import pytest

from chunkeval.config import PipelineError
from chunkeval.embedding import E5Encoder, _NoTruncationTokenizer, embedding_identity, validate_vectors


class SyntheticTokenizer:
    """One Unicode character is one token, plus two synthetic special tokens."""

    def count(self, text, prefix):
        return len(prefix + text) + 2


class SyntheticModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
            vector = np.random.default_rng(seed).normal(size=384).astype(np.float32)
            vector /= np.linalg.norm(vector)
            rows.append(vector)
        return np.stack(rows)


def synthetic_config(tmp_path):
    return {
        "_app_root": tmp_path, "seed": 42,
        "embedding": {"revision": "a" * 40},
        "semantic": {"sentence_prefix": "query: ", "cache_enabled": True},
    }


def synthetic_encoder(config):
    # Bypass only the downloaded-model constructor; exercise the real wrapper/cache.
    encoder = E5Encoder.__new__(E5Encoder)
    encoder.config = config
    encoder.identity = embedding_identity(config)
    encoder.resolved_config = encoder.identity["resolved_config"]
    encoder.embedding_config_hash = encoder.identity["embedding_config_hash"]
    encoder.dependency_versions = encoder.identity["dependency_versions"]
    encoder.cache_dir = config["_app_root"] / "data/cache/semantic"
    encoder.cache_enabled = config["semantic"]["cache_enabled"]
    encoder.tokenizer = SyntheticTokenizer()
    encoder.model = SyntheticModel()
    return encoder


@pytest.mark.parametrize("revision", [None, "main", "v1", "a" * 39])
def test_revision_must_be_concrete_without_model_loading(tmp_path, revision):
    config = synthetic_config(tmp_path)
    config["embedding"]["revision"] = revision
    with pytest.raises(PipelineError, match="concrete"):
        embedding_identity(config)


def test_full_input_prefix_and_special_tokens_checked_before_encode(tmp_path):
    encoder = synthetic_encoder(synthetic_config(tmp_path))
    valid_text = "漢" * (512 - len("passage: ") - 2)
    vector = encoder.encode_passages([valid_text])
    assert vector.shape == (1, 384) and vector.dtype == np.float32
    assert encoder.model.calls[0][0] == ["passage: " + valid_text]
    assert encoder.model.calls[0][1]["normalize_embeddings"] is True
    assert encoder.model.calls[0][1]["prompt"] == ""
    with pytest.raises(PipelineError, match="not truncated"):
        encoder.encode_passages([valid_text + "x"])
    with pytest.raises(PipelineError, match="not truncated"):
        encoder.encode_queries(["x" * 504])
    with pytest.raises(PipelineError, match="split"):
        encoder.encode_units([{"text": "x" * 504}])
    assert len(encoder.model.calls) == 1


def test_semantic_cache_reuses_vectors_without_dropping_unit_provenance(tmp_path):
    encoder = synthetic_encoder(synthetic_config(tmp_path))
    units = [
        {"unit_id": "S1", "sentence_id": "S1", "char_start": 0, "char_end": 4, "text": "sama"},
        {"unit_id": "S2", "sentence_id": "S2", "char_start": 5, "char_end": 9, "text": "sama"},
        {"unit_id": "S3", "sentence_id": "S3", "char_start": 10, "char_end": 15, "text": "beda!"},
    ]
    before = deepcopy(units)
    first = encoder.encode_units(units)
    second = encoder.encode_units(units)
    assert units == before and first.shape == (3, 384)
    assert np.array_equal(first, second) and np.array_equal(first[0], first[1])
    assert len(encoder.model.calls) == 1
    assert encoder.model.calls[0][0] == ["query: sama", "query: beda!"]
    uncached_config = deepcopy(encoder.config)
    uncached_config["semantic"]["cache_enabled"] = False
    uncached = synthetic_encoder(uncached_config)
    assert np.array_equal(first, uncached.encode_units(units))


@pytest.mark.parametrize("change", ["text", "prefix", "revision", "seed"])
def test_semantic_cache_invalidates_changed_identity(tmp_path, change):
    config = synthetic_config(tmp_path)
    encoder = synthetic_encoder(config)
    encoder.encode_units([{"text": "unit"}])
    changed = deepcopy(config)
    text = "unit"
    if change == "text":
        text = "unit!"
    elif change == "prefix":
        changed["semantic"]["sentence_prefix"] = "passage: "
    elif change == "revision":
        changed["embedding"]["revision"] = "b" * 40
    else:
        changed["seed"] = 43
    other = synthetic_encoder(changed)
    other.encode_units([{"text": text}])
    assert len(other.model.calls) == 1
    assert len(list(encoder.cache_dir.glob("*.npy"))) == 2


def test_invalid_cached_vector_is_recomputed(tmp_path):
    encoder = synthetic_encoder(synthetic_config(tmp_path))
    encoder.encode_units([{"text": "unit"}])
    path = next(encoder.cache_dir.glob("*.npy"))
    np.save(path, np.full((1, 384), np.nan, dtype=np.float32))
    vectors = encoder.encode_units([{"text": "unit"}])
    assert len(encoder.model.calls) == 2
    validate_vectors(vectors, count=1)


@pytest.mark.parametrize("bad", [
    np.zeros((1, 384), dtype=np.float32),
    np.ones((1, 383), dtype=np.float32),
    np.ones((1, 384), dtype=np.float64),
    np.full((1, 384), np.inf, dtype=np.float32),
])
def test_invalid_outputs_are_rejected(bad):
    with pytest.raises(PipelineError):
        validate_vectors(bad, count=1)


def test_dependency_and_embedding_code_change_identity(tmp_path, monkeypatch):
    import chunkeval.embedding as module

    config = synthetic_config(tmp_path)
    first = embedding_identity(config)["embedding_config_hash"]
    original = module.importlib.metadata.version
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "different" if name == "torch" else original(name))
    second = embedding_identity(config)["embedding_config_hash"]
    assert second != first
    monkeypatch.setattr(module, "EMBEDDING_CODE_VERSION", "changed")
    assert embedding_identity(config)["embedding_config_hash"] != second


def test_internal_sentence_transformer_truncation_is_disabled_and_overflow_fails():
    import torch

    class SyntheticBatchFastTokenizer:
        is_fast = True

        def __init__(self):
            self.calls = []

        def __call__(self, texts, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("return_tensors") == "pt":
                lengths = [len(text) + 2 for text in texts]
                mask = torch.zeros((len(lengths), max(lengths)), dtype=torch.int64)
                for row, length in enumerate(lengths):
                    mask[row, :length] = 1
                return {"input_ids": mask.clone(), "attention_mask": mask}
            return {"input_ids": list(range(len(texts) + 2))}

    original = SyntheticBatchFastTokenizer()
    facade = _NoTruncationTokenizer(original, 512)
    assert facade.is_fast is True
    encoded = facade(["x" * 510], truncation="longest_first", max_length=7, return_tensors="pt", padding=True)
    assert encoded["input_ids"].shape == (1, 512)
    assert original.calls[-1]["truncation"] is False and "max_length" not in original.calls[-1]
    with pytest.raises(PipelineError, match="No input was truncated"):
        facade(["x" * 511], truncation="longest_first", max_length=512, return_tensors="pt")
    # Source counting/offset mapping must still inspect whole long documents.
    assert len(facade("x" * 2000, truncation=False)["input_ids"]) == 2002
