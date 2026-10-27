"""Exact cosine retrieval with deterministic ordering across all FAISS rows."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np

from .config import PipelineError, write_jsonl
from .embedding import _within_app, validate_vectors
from .indexing import CONDITIONS


def validate_bundle(indexes) -> dict:
    """Require five complete indexes of the same corpus and encoder identity."""
    if isinstance(indexes, Mapping):
        by_condition = dict(indexes)
    else:
        by_condition = {}
        for loaded in indexes:
            condition = loaded.manifest.get("condition")
            if condition in by_condition:
                raise PipelineError(f"Duplicate index condition in bundle: {condition}.")
            by_condition[condition] = loaded
    if set(by_condition) != set(CONDITIONS):
        raise PipelineError("Batch/search --all requires exactly the complete C1-C5 index bundle; rebuild missing conditions.")
    reference = None
    for condition in CONDITIONS:
        loaded = by_condition[condition]
        manifest = loaded.manifest
        if manifest.get("condition") != condition:
            raise PipelineError(f"Index mapping key {condition} disagrees with its manifest.")
        if loaded.index.ntotal < 1 or loaded.index.ntotal != len(loaded.chunks):
            raise PipelineError(f"Index {condition} is empty or its mapping count differs from ntotal.")
        identity = tuple(manifest.get(key) for key in (
            "corpus_hash", "model_revision", "tokenizer_revision", "embedding_config_hash",
        ))
        if any(not value for value in identity):
            raise PipelineError(f"Index {condition} lacks required corpus/encoder identity.")
        if reference is None:
            reference = identity
        elif identity != reference:
            raise PipelineError("Index bundle mixes corpus hashes or model/tokenizer/embedding configurations; rebuild all five conditions.")
    return by_condition


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 10:
        raise PipelineError("Retrieval k must be an integer from 1 through 10.")


def search_index(loaded_index, query_vector, k: int = 10) -> list[dict]:
    """Request all rows so ties crossing K are resolved by smallest row_id."""
    _validate_k(k)
    vector = np.asarray(query_vector)
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    vector = validate_vectors(vector, count=1)
    total = loaded_index.index.ntotal
    if total < 1 or total != len(loaded_index.chunks):
        raise PipelineError("Search requires a nonempty index with aligned chunk mapping.")
    scores, row_ids = loaded_index.index.search(vector, total)
    candidates = list(zip(scores[0], row_ids[0]))
    if len(candidates) != total or any(row < 0 or row >= total for _, row in candidates) or len({int(row) for _, row in candidates}) != total:
        raise PipelineError("FAISS search returned an incomplete/invalid full-row mapping.")
    if any(not np.isfinite(score) for score, _ in candidates):
        raise PipelineError("FAISS search returned a nonfinite score.")
    candidates.sort(key=lambda item: (-float(item[0]), int(item[1])))
    return [
        {**loaded_index.chunks[int(row_id)], "row_id": int(row_id), "rank": rank, "score": float(score)}
        for rank, (score, row_id) in enumerate(candidates[:min(k, total)], start=1)
    ]


def _validate_queries(queries, tokenizer, config: dict) -> list[dict]:
    if not isinstance(queries, list):
        raise PipelineError("Query input must be a JSON array of {qid, text} objects.")
    ids = set()
    prefix = config.get("embedding", {}).get("query_prefix", "query: ")
    limit = config.get("embedding", {}).get("max_input_tokens", 512)
    validated = []
    for position, query in enumerate(queries):
        if not isinstance(query, dict) or not isinstance(query.get("qid"), str) or not query["qid"].strip():
            raise PipelineError(f"Query {position} requires a nonempty string qid.")
        if query["qid"] in ids:
            raise PipelineError(f"Duplicate query qid: {query['qid']}.")
        ids.add(query["qid"])
        text = query.get("text")
        if not isinstance(text, str) or not text.strip():
            raise PipelineError(f"Query {query['qid']} requires nonempty text.")
        count = tokenizer.count(text, prefix)
        if count > limit:
            raise PipelineError(f"Query {query['qid']} has {count} tokens; maximum is {limit}. Query was not truncated.")
        validated.append({"qid": query["qid"], "text": text})
    return validated


def load_queries(path: Path | str, tokenizer, config: dict) -> list[dict]:
    """Read only researcher-supplied queries; never generate research inputs."""
    try:
        with Path(path).open(encoding="utf-8") as handle:
            queries = json.load(handle)
    except (OSError, ValueError) as exc:
        raise PipelineError(f"Cannot read query JSON {path}: {exc}") from exc
    return _validate_queries(queries, tokenizer, config)


def retrieve_batch(queries, indexes, encoder, output_path: Path | str, top_k: int = 10) -> list[dict]:
    """Write complete query-condition records using one shared vector per text."""
    _validate_k(top_k)
    bundle = validate_bundle(indexes)
    output_path = _within_app(Path(output_path), encoder.config)
    for condition, loaded in bundle.items():
        if loaded.index.ntotal < top_k:
            raise PipelineError(f"Batch top-{top_k} requires at least {top_k} chunks in {condition}; found {loaded.index.ntotal}.")
        if loaded.manifest["embedding_config_hash"] != encoder.embedding_config_hash:
            raise PipelineError("Query encoder identity differs from the index bundle; rebuild indexes with the current configuration.")
    queries = _validate_queries(queries, encoder.tokenizer, encoder.config)
    texts = list(dict.fromkeys(query["text"] for query in queries))
    vectors = validate_vectors(encoder.encode_queries(texts), count=len(texts))
    by_text = dict(zip(texts, vectors))
    rows = []
    for query in queries:
        for condition in CONDITIONS:
            loaded = bundle[condition]
            rows.append({
                "qid": query["qid"], "query_text": query["text"], "condition": condition,
                "index_manifest_hash": loaded.manifest_hash,
                "results": search_index(loaded, by_text[query["text"]], top_k),
            })
    write_jsonl(output_path, rows)
    return rows
