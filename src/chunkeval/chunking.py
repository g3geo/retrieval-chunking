"""Five deterministic chunk conditions on one canonical Unicode source."""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Iterable

import numpy as np

from .config import PipelineError
from .tokenization import SourceTokenizer


CONDITIONS = ("C1", "C2", "C3", "C4", "C5")


def _partition_spans(text: str, start: int, end: int, boundaries: Iterable[int]) -> list[tuple[int, int]]:
    cuts = sorted({start, end, *(int(boundary) for boundary in boundaries if start < boundary < end)})
    spans: list[tuple[int, int]] = []
    pending_start = start
    for left, right in zip(cuts, cuts[1:]):
        if text[left:right].strip():
            spans.append((pending_start, right))
            pending_start = right
        elif spans:
            spans[-1] = (spans[-1][0], right)
            pending_start = right
    return spans


def semantic_units(doc: dict, tokenizer: SourceTokenizer, config: dict) -> list[dict]:
    """Use the shared sentence annotations, absorbing gaps without losing text."""
    text = doc["text"]
    sentences = doc.get("sentences", [])
    if not text.strip():
        return []
    if not sentences:
        raise PipelineError(f"{doc['doc_id']}: substantive text has no sentence annotations.")
    prefix = config["semantic"]["sentence_prefix"]
    limit = int(config["embedding"]["max_input_tokens"])
    result = []
    previous_end = 0
    for index, sentence in enumerate(sentences):
        sentence_start, sentence_end = sentence["char_start"], sentence["char_end"]
        if not 0 <= sentence_start < sentence_end <= len(text) or sentence_start < previous_end:
            raise PipelineError("Shared sentence annotations are not ordered source spans.")
        start = 0 if index == 0 else sentence_start
        end = sentences[index + 1]["char_start"] if index + 1 < len(sentences) else len(text)
        if end < sentence_end:
            raise PipelineError("Shared sentence annotations overlap.")
        fragments = tokenizer.split_span(text, start, end, limit, prefix)
        source_unit_id = f"{doc['doc_id']}:{sentence['sentence_id']}:U"
        for fragment_index, (left, right) in enumerate(fragments, 1):
            result.append({
                "unit_id": f"{doc['doc_id']}:{sentence['sentence_id']}:U{fragment_index:04d}",
                "sentence_id": sentence["sentence_id"],
                "char_start": left, "char_end": right, "text": text[left:right],
                "split_from": [source_unit_id] if len(fragments) > 1 else [],
            })
        previous_end = sentence_end
    return result


@lru_cache(maxsize=1)
def _semantic_distance_helper():
    try:
        installed = version("langchain-experimental")
    except PackageNotFoundError as exc:
        raise PipelineError("Semantic chunking requires locked langchain-experimental==0.3.4.") from exc
    if installed != "0.3.4":
        raise PipelineError(f"Semantic adapter requires langchain-experimental==0.3.4; found {installed}.")
    try:
        from langchain_experimental.text_splitter import calculate_cosine_distances
    except ImportError as exc:
        raise PipelineError("The pinned SemanticChunker cosine-distance helper is unavailable.") from exc
    return calculate_cosine_distances


def semantic_breakpoints(units: list[dict], unit_embeddings, percentile: float) -> list[int]:
    """Return source-unit indices after which the pinned percentile rule splits.

    SemanticChunker 0.3.4's calculate_cosine_distances helper computes the
    adjacent cosine distances. Its percentile threshold uses NumPy's linear
    interpolation and split_text applies strict `distance > threshold`.
    Buffer size is zero because each supplied vector represents exactly one unit.
    """
    if not isinstance(percentile, (int, float)) or not np.isfinite(percentile) or not 0 <= percentile <= 100:
        raise PipelineError("Semantic percentile must be between 0 and 100.")
    if not units:
        if np.asarray(unit_embeddings).size:
            raise PipelineError("Semantic vectors must align with their source units.")
        return []
    if any(unit for unit in units):
        required = {"unit_id", "sentence_id", "char_start", "char_end", "text"}
        previous_end = None
        unit_ids = set()
        for unit in units:
            if not isinstance(unit, dict) or not required.issubset(unit):
                raise PipelineError("Semantic source units must include their IDs, text, and source spans.")
            start, end, text = unit["char_start"], unit["char_end"], unit["text"]
            if (not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end
                    or not isinstance(text, str) or len(text) != end - start or not text.strip()
                    or (previous_end is not None and start != previous_end)):
                raise PipelineError("Semantic source units must have contiguous, ordered Unicode source spans.")
            if (not isinstance(unit["unit_id"], str) or not unit["unit_id"]
                    or unit["unit_id"] in unit_ids or not isinstance(unit["sentence_id"], str)
                    or not unit["sentence_id"]):
                raise PipelineError("Semantic source units must retain unique unit IDs and original sentence IDs.")
            unit_ids.add(unit["unit_id"])
            previous_end = end
    try:
        vectors = np.asarray(unit_embeddings, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PipelineError("Semantic vectors must be a numeric matrix aligned with source units.") from exc
    if vectors.ndim != 2 or vectors.shape[0] != len(units) or vectors.shape[1] == 0:
        raise PipelineError("Semantic vectors must align with their source units.")
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4, rtol=1e-4):
        raise PipelineError("Semantic vectors must be finite and L2-normalized.")
    if len(units) == 1:
        return []
    helper = _semantic_distance_helper()
    distances, _ = helper([
        {"combined_sentence_embedding": vector.tolist()} for vector in vectors
    ])
    threshold = float(np.percentile(distances, percentile, method="linear"))
    return [index for index, distance in enumerate(distances) if distance > threshold]


def _oversize_spans(doc: dict, start: int, end: int, tokenizer: SourceTokenizer, config: dict) -> list[tuple[int, int]]:
    text = doc["text"]
    prefix = config["embedding"]["passage_prefix"]
    limit = int(config["embedding"]["max_input_tokens"])
    if tokenizer.count(text[start:end], prefix) <= limit:
        return [(start, end)]
    atoms = _partition_spans(text, start, end, (sentence["char_start"] for sentence in doc.get("sentences", [])))
    fragments = [fragment for left, right in atoms
                 for fragment in tokenizer.split_span(text, left, right, limit, prefix)]
    output: list[tuple[int, int]] = []
    for left, right in fragments:
        if output and tokenizer.count(text[output[-1][0]:right], prefix) <= limit:
            output[-1] = (output[-1][0], right)
        else:
            output.append((left, right))
    return output


def _metadata(doc: dict, condition: str, start: int, end: int, number: int,
              tokenizer: SourceTokenizer, config: dict, split_from=None) -> dict:
    pages = [page["page_number"] for page in doc.get("pages", [])
             if max(start, page["char_start"]) < min(end, page["char_end"])]
    sentence_ids = [sentence["sentence_id"] for sentence in doc.get("sentences", [])
                    if max(start, sentence["char_start"]) < min(end, sentence["char_end"])]
    references = []
    for structure in sorted(doc.get("structures", []), key=lambda row: row["char_start"]):
        if max(start, structure["char_start"]) < min(end, structure["char_end"]):
            reference = {key: structure.get(key) for key in ("bab", "pasal", "ayat", "butir")}
            reference.update(char_start=structure["char_start"], char_end=structure["char_end"])
            if "structure_id" in structure:
                reference["structure_id"] = structure["structure_id"]
            references.append(reference)
    return {
        "chunk_id": f"{doc['doc_id']}_{condition}_{number:06d}",
        "condition": condition, "doc_id": doc["doc_id"], "text": doc["text"][start:end],
        "char_start": start, "char_end": end,
        "page_start": min(pages) if pages else None, "page_end": max(pages) if pages else None,
        "page_numbers": pages,
        "n_tokens_with_prefix": tokenizer.count(doc["text"][start:end], config["embedding"]["passage_prefix"]),
        "sentence_ids": sentence_ids, "split_from": [] if split_from is None else list(split_from),
        "structure_refs": references,
    }


def _fixed_chunks(doc: dict, condition: str, tokenizer: SourceTokenizer, config: dict) -> list[dict]:
    text = doc["text"]
    offsets = tokenizer.offsets(text)
    if not offsets:
        return []
    settings = config["chunking"][condition]
    limit, overlap = int(settings["max_input_tokens"]), int(settings["overlap_tokens"])
    prefix = config["embedding"]["passage_prefix"]
    output = []
    token_start = 0
    while token_start < len(offsets):
        char_start = 0 if token_start == 0 else offsets[token_start][0]
        token_end = min(token_start + limit, len(offsets))
        while token_end > token_start:
            char_end = len(text) if token_end == len(offsets) else offsets[token_end][0]
            if char_end > char_start and tokenizer.count(text[char_start:char_end], prefix) <= limit:
                break
            token_end -= 1
        if token_end <= token_start:
            raise PipelineError("A source token cannot fit the fixed chunk's complete input budget.")
        while token_end < len(offsets):
            next_end = len(text) if token_end + 1 == len(offsets) else offsets[token_end + 1][0]
            if tokenizer.count(text[char_start:next_end], prefix) > limit:
                break
            token_end += 1
            char_end = next_end
        chunk = _metadata(doc, condition, char_start, char_end, len(output) + 1, tokenizer, config)
        chunk.update(token_start=token_start, token_end=token_end)
        output.append(chunk)
        if token_end == len(offsets):
            break
        next_start = token_end - overlap
        if next_start <= token_start:
            raise PipelineError("Fixed overlap leaves no forward progress within the token budget.")
        token_start = next_start
    return output


def build_chunks(doc: dict, condition: str, tokenizer: SourceTokenizer, config: dict, unit_embeddings=None) -> list[dict]:
    if condition not in CONDITIONS:
        raise PipelineError(f"Unknown chunk condition: {condition}")
    if condition in ("C1", "C2"):
        output = _fixed_chunks(doc, condition, tokenizer, config)
    else:
        text = doc["text"]
        units = []
        if condition == "C3":
            # Hierarchical ranges are provenance; only their ordered starts are
            # boundaries. Nested ranges are never emitted as overlapping chunks.
            boundaries = [paragraph["char_start"] for paragraph in doc.get("paragraphs", [])]
            boundaries.extend(structure["char_start"] for structure in doc.get("structures", [])
                              if any(structure.get(label) is not None for label in ("pasal", "ayat", "butir")))
            parents = _partition_spans(text, 0, len(text), boundaries)
        else:
            if config["semantic"]["buffer_size"] != 0 or config["semantic"]["threshold_type"] != "percentile":
                raise PipelineError("Semantic conditions require buffer_size=0 and a percentile threshold.")
            units = semantic_units(doc, tokenizer, config)
            if unit_embeddings is None and units:
                raise PipelineError("Supply semantic unit embeddings shared by C4 and C5.")
            breakpoints = semantic_breakpoints(units, [] if unit_embeddings is None else unit_embeddings,
                                               config["chunking"][condition]["percentile"])
            parents = []
            first_unit = 0
            for last_unit in [*breakpoints, len(units) - 1] if units else []:
                parents.append((units[first_unit]["char_start"], units[last_unit]["char_end"]))
                first_unit = last_unit + 1
        output = []
        for parent_number, (start, end) in enumerate(parents, 1):
            fragments = _oversize_spans(doc, start, end, tokenizer, config)
            split_from = [f"{doc['doc_id']}:{condition}:P{parent_number:06d}"] if len(fragments) > 1 else []
            for left, right in fragments:
                provenance = list(split_from)
                for unit in units:
                    if max(left, unit["char_start"]) < min(right, unit["char_end"]):
                        for source_parent in unit.get("split_from", []):
                            if source_parent not in provenance:
                                provenance.append(source_parent)
                output.append(_metadata(doc, condition, left, right, len(output) + 1, tokenizer, config, provenance))
    validate_chunks(doc, output, condition, tokenizer, config)
    return output


def validate_chunks(doc: dict, chunks: list[dict], condition: str, tokenizer: SourceTokenizer, config: dict) -> None:
    """Reject source loss, overlap errors, stale metadata, or oversized inputs."""
    if condition not in CONDITIONS:
        raise PipelineError(f"Unknown chunk condition: {condition}")
    text = doc["text"]
    coverage = np.zeros(len(text), dtype=np.int32)
    limit = int(config["chunking"][condition]["max_input_tokens"]) if condition in ("C1", "C2") else int(config["embedding"]["max_input_tokens"])
    offsets = tokenizer.offsets(text) if condition in ("C1", "C2") else []
    seen_ids = set()
    previous = None
    for number, chunk in enumerate(chunks, 1):
        start, end = chunk.get("char_start"), chunk.get("char_end")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
            raise PipelineError("Chunk has invalid Unicode source offsets.")
        if chunk.get("text") != text[start:end] or not chunk["text"].strip():
            raise PipelineError("Chunk text must be a substantive exact source slice.")
        if chunk.get("doc_id") != doc["doc_id"] or chunk.get("condition") != condition:
            raise PipelineError("Chunk document/condition metadata does not match its source.")
        if not isinstance(chunk.get("chunk_id"), str) or chunk["chunk_id"] in seen_ids:
            raise PipelineError("Chunk IDs must be present and unique.")
        seen_ids.add(chunk["chunk_id"])
        if not isinstance(chunk.get("split_from"), list) or any(
            not isinstance(parent, str) or not parent for parent in chunk["split_from"]
        ):
            raise PipelineError("Chunk split_from must list its source parent IDs (empty when unsplit).")
        expected = _metadata(doc, condition, start, end, number, tokenizer, config, chunk.get("split_from"))
        for key in ("n_tokens_with_prefix", "page_start", "page_end", "page_numbers", "sentence_ids", "structure_refs"):
            if key not in chunk or chunk[key] != expected[key]:
                raise PipelineError(f"Chunk has stale or invalid {key} metadata.")
        if expected["n_tokens_with_prefix"] > limit or expected["n_tokens_with_prefix"] > int(config["embedding"]["max_input_tokens"]):
            raise PipelineError(f"{condition} chunk exceeds its complete input token budget.")
        if not expected["page_numbers"] and text[start:end].strip():
            raise PipelineError("Substantive chunk has no source page mapping.")
        if not expected["sentence_ids"]:
            raise PipelineError("Substantive chunk has no source sentence mapping.")
        if previous is not None and start <= previous["char_start"]:
            raise PipelineError("Chunks must follow source order with forward progress.")
        if condition in ("C1", "C2"):
            token_start, token_end = chunk.get("token_start"), chunk.get("token_end")
            if not isinstance(token_start, int) or not isinstance(token_end, int) or not 0 <= token_start < token_end <= len(offsets):
                raise PipelineError("Fixed chunk lacks valid source-token coordinates.")
            expected_start = 0 if token_start == 0 else offsets[token_start][0]
            expected_end = len(text) if token_end == len(offsets) else offsets[token_end][0]
            if (start, end) != (expected_start, expected_end):
                raise PipelineError("Fixed source-token coordinates do not match source character spans.")
            if previous is None and token_start != 0:
                raise PipelineError("Fixed chunking must begin at the first source token.")
            if previous is not None and token_start != previous["token_end"] - int(config["chunking"][condition]["overlap_tokens"]):
                raise PipelineError("Fixed chunks do not have the required source-token overlap.")
        elif previous is not None and start < previous["char_end"]:
            raise PipelineError("C3-C5 chunks must have no overlap.")
        coverage[start:end] += 1
        previous = chunk
    nonspace = np.array([not character.isspace() for character in text], dtype=bool)
    if np.any(coverage[nonspace] == 0):
        raise PipelineError("Chunking lost non-whitespace source characters.")
    if condition not in ("C1", "C2") and np.any(coverage[nonspace] != 1):
        raise PipelineError("C3-C5 must cover non-whitespace source characters exactly once.")
    if condition in ("C1", "C2") and chunks and chunks[-1]["token_end"] != len(offsets):
        raise PipelineError("Fixed chunking did not reach the final source token.")
