"""Pinned E5 encoding and a provenance-sensitive cache of semantic unit vectors."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from .config import PipelineError, canonical_hash, data_path


EMBEDDING_CODE_VERSION = "e5-source-slice-v1"
EMBEDDING_DEPENDENCIES = (
    "numpy", "torch", "transformers", "sentence-transformers", "tokenizers",
    "huggingface-hub", "safetensors", "scipy", "scikit-learn",
)
DEFAULT_EMBEDDING = {
    "model": "intfloat/multilingual-e5-small", "revision": None,
    "dimension": 384, "max_input_tokens": 512,
    "passage_prefix": "passage: ", "query_prefix": "query: ",
    "normalize": True, "device": "cpu", "batch_size": 32,
}


def embedding_identity(config: dict) -> dict:
    """Resolve the complete encoder identity without loading/downloading a model."""
    resolved = {**DEFAULT_EMBEDDING, **config.get("embedding", {})}
    revision = resolved["revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise PipelineError("Embedding revision must be a concrete 40-character model commit; run setup before encoding.")
    if resolved["model"] != DEFAULT_EMBEDDING["model"]:
        raise PipelineError("This pipeline requires intfloat/multilingual-e5-small.")
    if resolved["dimension"] != 384 or resolved["max_input_tokens"] != 512:
        raise PipelineError("E5 requires dimension=384 and max_input_tokens=512.")
    if resolved["normalize"] is not True or resolved.get("dtype", "float32") != "float32":
        raise PipelineError("E5 embeddings must use L2 normalization.")
    if resolved["passage_prefix"] != "passage: " or resolved["query_prefix"] != "query: ":
        raise PipelineError("E5 passage/query prefixes must be 'passage: ' and 'query: '.")
    if not isinstance(resolved["batch_size"], int) or isinstance(resolved["batch_size"], bool) or resolved["batch_size"] < 1:
        raise PipelineError("Embedding batch_size must be a positive integer.")
    semantic_prefix = config.get("semantic", {}).get("sentence_prefix", "query: ")
    if not isinstance(semantic_prefix, str) or not semantic_prefix:
        raise PipelineError("semantic.sentence_prefix must be a nonempty string.")
    resolved.update({
        "revision": revision.lower(), "tokenizer_revision": revision.lower(),
        "dtype": "float32", "semantic_prefix": semantic_prefix,
        "pooling": "sentence-transformers-model-configuration",
        "truncation": False, "special_tokens": True,
        "seed": config.get("seed", 42),
    })
    dependencies = {}
    for package in EMBEDDING_DEPENDENCIES:
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "not-installed"
    code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    tokenizer_code_sha256 = hashlib.sha256(Path(__file__).with_name("tokenization.py").read_bytes()).hexdigest()
    identity = {
        "resolved_config": resolved,
        "dependency_versions": dependencies,
        "code_version": EMBEDDING_CODE_VERSION,
        "code_sha256": code_sha256,
        "tokenizer_code_sha256": tokenizer_code_sha256,
    }
    return {**identity, "embedding_config_hash": canonical_hash(identity)}


def validate_vectors(vectors, *, count: int | None = None, dimension: int = 384) -> np.ndarray:
    """Validate vectors without silently fixing malformed cached/indexed values."""
    matrix = np.asarray(vectors)
    if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise PipelineError(f"Embeddings must be a float32 matrix with {dimension} columns.")
    if count is not None and matrix.shape[0] != count:
        raise PipelineError("Embedding count does not match input count.")
    if not np.isfinite(matrix).all():
        raise PipelineError("Embeddings contain nonfinite values.")
    if len(matrix) and not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise PipelineError("Embedding vectors are not L2-normalized.")
    return np.ascontiguousarray(matrix)


def _within_app(path: Path, config: dict) -> Path:
    root = Path(config["_app_root"]).resolve()
    path = path.resolve()
    if not path.is_relative_to(root):
        raise PipelineError(f"Output must stay within application root: {path}")
    return path


class _NoTruncationTokenizer:
    """Keep ST's batch tokenization uncapped, with an explicit overflow check.

    SentenceTransformer 3.4.1's Transformer.tokenize supplies longest_first
    truncation unconditionally. The facade changes that argument for the pinned
    HF tokenizer, while source counting/offset calls may still inspect long docs.
    """

    def __init__(self, tokenizer, limit: int):
        self._tokenizer = tokenizer
        self._limit = limit

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def __call__(self, *args, **kwargs):
        kwargs["truncation"] = False
        kwargs.pop("max_length", None)
        encoded = self._tokenizer(*args, **kwargs)
        if kwargs.get("return_tensors") is not None:
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                counts = attention_mask.sum(-1)
                actual_max = int(counts.max().item()) if counts.numel() else 0
            else:
                actual_max = int(encoded["input_ids"].shape[-1])
            if actual_max > self._limit:
                raise PipelineError(f"Encoder batch contains {actual_max} input tokens; maximum is {self._limit}. No input was truncated.")
        return encoded


class E5Encoder:
    """Encode raw source text; this wrapper adds the selected prefix exactly once."""

    def __init__(self, config: dict, cache_dir: Path | str | None = None):
        self.config = config
        self.identity = embedding_identity(config)
        self.resolved_config = self.identity["resolved_config"]
        self.embedding_config_hash = self.identity["embedding_config_hash"]
        self.dependency_versions = self.identity["dependency_versions"]
        self.cache_dir = _within_app(Path(cache_dir) if cache_dir is not None else data_path(config, "cache", "semantic"), config)
        self.cache_enabled = config.get("semantic", {}).get("cache_enabled", True)
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            from .tokenization import SourceTokenizer

            np.random.seed(self.resolved_config["seed"])
            torch.manual_seed(self.resolved_config["seed"])
            self.model = SentenceTransformer(
                self.resolved_config["model"], revision=self.resolved_config["revision"],
                device=self.resolved_config["device"], trust_remote_code=False,
                cache_folder=str(data_path(config, "cache", "model")),
            )
            if not getattr(self.model.tokenizer, "is_fast", False):
                raise PipelineError("Pinned E5 tokenizer must be a fast tokenizer with source offsets.")
            self.model.max_seq_length = self.resolved_config["max_input_tokens"]
            self.model.tokenizer.model_max_length = self.resolved_config["max_input_tokens"]
            if self.model.max_seq_length < self.resolved_config["max_input_tokens"]:
                raise PipelineError("Encoder internal token limit is smaller than validated input limit.")
            self.model.tokenizer = _NoTruncationTokenizer(self.model.tokenizer, self.resolved_config["max_input_tokens"])
            self.tokenizer = SourceTokenizer(self.model.tokenizer)
            self.model.eval()
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(f"Unable to load pinned E5 model/tokenizer: {exc}") from exc

    def _encode(self, texts: Iterable[str], prefix: str) -> np.ndarray:
        texts = list(texts)
        # Count each complete prefix + source text with special tokens and no truncation.
        # Validation precedes model.encode, including cached semantic-unit requests.
        for position, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise PipelineError(f"Embedding input {position} must contain non-whitespace text.")
            count = self.tokenizer.count(text, prefix)
            if count > self.resolved_config["max_input_tokens"]:
                raise PipelineError(f"Embedding input {position} has {count} tokens; maximum is 512. Input was not truncated.")
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        try:
            # No configured ST prompt can add another prefix. All texts here are raw
            # source strings; adding a prefix here never changes their saved offsets.
            matrix = self.model.encode(
                [prefix + text for text in texts],
                batch_size=self.resolved_config["batch_size"],
                show_progress_bar=False, convert_to_numpy=True,
                normalize_embeddings=True, prompt="",
            )
        except Exception as exc:
            raise PipelineError(f"E5 encoding failed: {exc}") from exc
        return validate_vectors(matrix, count=len(texts))

    def encode_passages(self, texts: Iterable[str]) -> np.ndarray:
        return self._encode(texts, self.resolved_config["passage_prefix"])

    def encode_queries(self, texts: Iterable[str]) -> np.ndarray:
        return self._encode(texts, self.resolved_config["query_prefix"])

    def _cache_key(self, text: str) -> str:
        return canonical_hash({
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "semantic_prefix": self.resolved_config["semantic_prefix"],
            "embedding_identity": self.identity,
        })

    def encode_units(self, units: Iterable[dict]) -> np.ndarray:
        units = list(units)
        texts = [unit["text"] for unit in units]
        prefix = self.resolved_config["semantic_prefix"]
        # A cache hit cannot bypass the same full-input validation as a miss.
        for position, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise PipelineError(f"Semantic unit {position} must contain non-whitespace text.")
            count = self.tokenizer.count(text, prefix)
            if count > self.resolved_config["max_input_tokens"]:
                raise PipelineError(f"Semantic unit {position} has {count} tokens; split it before encoding (maximum 512).")
        if not self.cache_enabled:
            return self._encode(texts, prefix)
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        keys = [self._cache_key(text) for text in texts]
        by_key: dict[str, np.ndarray] = {}
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts):
            if key in by_key or key in missing:
                continue
            path = self.cache_dir / f"{key}.npy"
            try:
                cached = np.load(path, allow_pickle=False)
                by_key[key] = validate_vectors(cached, count=1)[0]
            except (OSError, ValueError, EOFError, PipelineError):
                missing[key] = text
        if missing:
            encoded = self._encode(missing.values(), prefix)
            for key, vector in zip(missing, encoded):
                by_key[key] = vector
                buffer = io.BytesIO()
                np.save(buffer, vector.reshape(1, 384), allow_pickle=False)
                target = self.cache_dir / f"{key}.npy"
                temporary = target.with_suffix(".npy.tmp")
                temporary.write_bytes(buffer.getvalue())
                temporary.replace(target)
        return validate_vectors(np.stack([by_key[key] for key in keys]), count=len(units))
