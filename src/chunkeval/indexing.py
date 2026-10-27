"""Deterministic, independently verifiable FAISS indexes for C1 through C5."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import tempfile

from .config import PipelineError, canonical_hash, file_sha256, read_jsonl, write_json, write_jsonl
from .embedding import _within_app, embedding_identity, validate_vectors


CONDITIONS = ("C1", "C2", "C3", "C4", "C5")
MANIFEST_SCHEMA_VERSION = 1
REQUIRED_CHUNK_FIELDS = (
    "chunk_id", "condition", "doc_id", "text", "char_start", "char_end",
    "page_start", "page_end", "page_numbers", "n_tokens_with_prefix",
    "sentence_ids", "split_from", "structure_refs",
)


@dataclass
class LoadedIndex:
    index: object
    chunks: list[dict]
    manifest: dict
    manifest_hash: str


def _faiss():
    try:
        import faiss
        return faiss
    except ImportError as exc:
        raise PipelineError("FAISS is unavailable; install the locked faiss-cpu dependency.") from exc


def _faiss_version() -> str:
    try:
        return importlib.metadata.version("faiss-cpu")
    except importlib.metadata.PackageNotFoundError as exc:
        raise PipelineError("faiss-cpu dependency metadata is unavailable; reinstall requirements.lock.") from exc


def _chunking_config(condition: str, config: dict) -> dict:
    if condition not in CONDITIONS:
        raise PipelineError(f"Unknown chunk condition: {condition}")
    try:
        chunking = config["chunking"][condition]
    except KeyError as exc:
        raise PipelineError(f"Missing chunking configuration for {condition}.") from exc
    return {
        "condition": condition, "chunking": chunking,
        "semantic": {key: value for key, value in config.get("semantic", {}).items()
                     if key != "cache_enabled"},
        "code_sha256": {name: file_sha256(Path(__file__).with_name(name + ".py"))
                        for name in ("chunking", "segmentation")},
    }


def _validate_mapping(chunks: list[dict], condition: str) -> None:
    if not chunks:
        raise PipelineError("Cannot build/load an empty research index.")
    ids = set()
    previous = None
    for row_id, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or any(field not in chunk for field in REQUIRED_CHUNK_FIELDS):
            raise PipelineError(f"Chunk row {row_id} is missing required mapping metadata.")
        if chunk["condition"] != condition:
            raise PipelineError(f"Chunk row {row_id} belongs to another condition.")
        if not isinstance(chunk["chunk_id"], str) or not chunk["chunk_id"] or chunk["chunk_id"] in ids:
            raise PipelineError(f"Invalid or duplicate chunk_id in row {row_id}.")
        ids.add(chunk["chunk_id"])
        if not isinstance(chunk["doc_id"], str) or not chunk["doc_id"]:
            raise PipelineError(f"Missing document identity in row {row_id}.")
        start, end = chunk["char_start"], chunk["char_end"]
        if any(not isinstance(value, int) or isinstance(value, bool) for value in (start, end)) or start < 0 or end <= start:
            raise PipelineError(f"Invalid character span in chunk row {row_id}.")
        if not isinstance(chunk["text"], str) or not chunk["text"].strip() or len(chunk["text"]) != end - start:
            raise PipelineError(f"Invalid source-slice text in chunk row {row_id}.")
        tokens = chunk["n_tokens_with_prefix"]
        limit = 256 if condition == "C1" else 512
        if not isinstance(tokens, int) or isinstance(tokens, bool) or not 1 <= tokens <= limit:
            raise PipelineError(f"Invalid input token count in chunk row {row_id}.")
        pages = chunk["page_numbers"]
        if not isinstance(pages, list) or not pages or any(not isinstance(page, int) or isinstance(page, bool) or page < 1 for page in pages) or pages != sorted(set(pages)):
            raise PipelineError(f"Invalid page mapping in chunk row {row_id}.")
        if chunk["page_start"] != pages[0] or chunk["page_end"] != pages[-1]:
            raise PipelineError(f"Page range and page_numbers disagree in chunk row {row_id}.")
        if any(not isinstance(chunk[field], list) for field in ("sentence_ids", "split_from", "structure_refs")):
            raise PipelineError(f"Invalid provenance metadata in chunk row {row_id}.")
        previous_structure_start = -1
        for structure in chunk["structure_refs"]:
            if not isinstance(structure, dict) or any(field not in structure for field in ("char_start", "char_end", "bab", "pasal", "ayat", "butir")):
                raise PipelineError(f"Incomplete structure reference in chunk row {row_id}.")
            structure_start, structure_end = structure["char_start"], structure["char_end"]
            if (any(not isinstance(value, int) or isinstance(value, bool) for value in (structure_start, structure_end))
                    or structure_start < 0 or structure_end <= structure_start):
                raise PipelineError(f"Invalid structure-reference character span in chunk row {row_id}.")
            if structure_start < previous_structure_start:
                raise PipelineError(f"Structure references are not ordered by source span in chunk row {row_id}.")
            previous_structure_start = structure_start
            if max(start, structure_start) >= min(end, structure_end):
                raise PipelineError(f"Nonintersecting structure reference in chunk row {row_id}.")
        ordering = (chunk["doc_id"], start, end, chunk["chunk_id"])
        if previous is not None and ordering < previous:
            raise PipelineError("Chunk rows must be ordered deterministically by doc_id and source span.")
        previous = ordering


def build_index(condition: str, chunks, encoder, config: dict, corpus_hash: str, output_dir: Path | str) -> LoadedIndex:
    """Build one index atomically after all mapping/vector checks have passed."""
    faiss = _faiss()
    output = _within_app(Path(output_dir), config)
    rows = list(chunks)
    if any(not isinstance(row, dict) or any(field not in row for field in REQUIRED_CHUNK_FIELDS) for row in rows):
        raise PipelineError("Chunk rows must contain all required mapping metadata.")
    try:
        rows.sort(key=lambda row: (row["doc_id"], row["char_start"], row["char_end"], row["chunk_id"]))
    except TypeError as exc:
        raise PipelineError("Chunk row document/span identities have invalid types.") from exc
    _validate_mapping(rows, condition)
    if not isinstance(corpus_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", corpus_hash):
        raise PipelineError("A deterministic SHA256 corpus_hash is required to build an index.")
    identity = embedding_identity(config)
    if encoder.embedding_config_hash != identity["embedding_config_hash"]:
        raise PipelineError("Encoder configuration differs from index configuration; rebuild with the current encoder.")
    for row in rows:
        actual = encoder.tokenizer.count(row["text"], identity["resolved_config"]["passage_prefix"])
        limit = config["chunking"][condition].get("max_input_tokens", 512)
        if actual != row["n_tokens_with_prefix"] or actual > min(limit, 512):
            raise PipelineError(f"Chunk {row['chunk_id']} has inconsistent/over-limit input tokens; rebuild chunks.")
    vectors = validate_vectors(encoder.encode_passages([row["text"] for row in rows]), count=len(rows))
    index = faiss.IndexFlatIP(384)
    index.add(vectors)
    resolved_chunking = _chunking_config(condition, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{condition}-", dir=output.parent))
    try:
        faiss.write_index(index, str(staging / "index.faiss"))
        write_jsonl(staging / "chunks.jsonl", rows)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "condition": condition, "corpus_hash": corpus_hash,
            "configuration": resolved_chunking,
            "chunking_config_hash": canonical_hash(resolved_chunking),
            "embedding_config_hash": identity["embedding_config_hash"],
            "embedding_identity": identity,
            "model_revision": identity["resolved_config"]["revision"],
            "tokenizer_revision": identity["resolved_config"]["tokenizer_revision"],
            "dependency_versions": {**identity["dependency_versions"], "faiss-cpu": _faiss_version()},
            "chunks_sha256": file_sha256(staging / "chunks.jsonl"),
            "index_sha256": file_sha256(staging / "index.faiss"),
            "vector_count": len(rows), "dimension": 384, "dtype": "float32",
            "normalized": True, "index_type": "IndexFlatIP", "metric": "inner_product",
        }
        write_json(staging / "manifest.json", manifest)
        # Only validated complete files are published. A partial replacement is
        # detectable on load because manifest checksums must match both artifacts.
        output.mkdir(parents=True, exist_ok=True)
        for name in ("index.faiss", "chunks.jsonl", "manifest.json"):
            (staging / name).replace(output / name)
    finally:
        shutil.rmtree(staging)
    return load_index(output, config, corpus_hash)


def load_index(index_dir: Path | str, config: dict, corpus_hash: str | None = None) -> LoadedIndex:
    """Reject stale, partial, mismatched or corrupted index artifacts."""
    faiss = _faiss()
    directory = _within_app(Path(index_dir), config)
    rebuild = " Rebuild chunks and all affected indexes."
    try:
        with (directory / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise PipelineError("Unsupported index manifest schema." + rebuild)
        condition = manifest["condition"]
        expected_identity = embedding_identity(config)
        expected_chunking = _chunking_config(condition, config)
        if manifest["embedding_identity"] != expected_identity or manifest["embedding_config_hash"] != expected_identity["embedding_config_hash"]:
            raise PipelineError("Index embedding/model/tokenizer/dependency identity differs from the current configuration." + rebuild)
        if manifest["configuration"] != expected_chunking or manifest["chunking_config_hash"] != canonical_hash(expected_chunking):
            raise PipelineError("Index chunking configuration differs from current configuration." + rebuild)
        dependencies = {**expected_identity["dependency_versions"], "faiss-cpu": _faiss_version()}
        if manifest["dependency_versions"] != dependencies:
            raise PipelineError("Index dependency versions differ from this locked environment." + rebuild)
        resolved = expected_identity["resolved_config"]
        if manifest["model_revision"] != resolved["revision"] or manifest["tokenizer_revision"] != resolved["tokenizer_revision"]:
            raise PipelineError("Index model/tokenizer revision differs from configuration." + rebuild)
        if not isinstance(manifest["corpus_hash"], str) or not re.fullmatch(r"[0-9a-f]{64}", manifest["corpus_hash"]):
            raise PipelineError("Invalid corpus identity in index manifest." + rebuild)
        if corpus_hash is not None and manifest["corpus_hash"] != corpus_hash:
            raise PipelineError("Index corpus hash differs from the current cleaned corpus." + rebuild)
        for name, checksum in (("chunks.jsonl", "chunks_sha256"), ("index.faiss", "index_sha256")):
            if file_sha256(directory / name) != manifest[checksum]:
                raise PipelineError(f"Index {name} checksum mismatch." + rebuild)
        chunks = list(read_jsonl(directory / "chunks.jsonl"))
        _validate_mapping(chunks, condition)
        index = faiss.read_index(str(directory / "index.faiss"))
        if manifest["dimension"] != 384 or index.d != 384 or manifest["dtype"] != "float32" or manifest["normalized"] is not True:
            raise PipelineError("Index vector dimension/dtype/normalization declaration is invalid." + rebuild)
        if manifest["index_type"] != "IndexFlatIP" or manifest["metric"] != "inner_product" or not isinstance(index, faiss.IndexFlatIP) or index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise PipelineError("Index must be a single FAISS IndexFlatIP(384)." + rebuild)
        if not index.ntotal or index.ntotal != len(chunks) or manifest["vector_count"] != len(chunks):
            raise PipelineError("Index ntotal and mapping/vector count disagree or index is empty." + rebuild)
        validate_vectors(index.reconstruct_n(0, index.ntotal), count=index.ntotal)
        return LoadedIndex(index=index, chunks=chunks, manifest=manifest, manifest_hash=file_sha256(directory / "manifest.json"))
    except PipelineError:
        raise
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        raise PipelineError(f"Unable to load valid index from {directory}: {exc}." + rebuild) from exc
