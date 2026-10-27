"""Meaningful source-coverage and adapter tests with synthetic vectors/tokens."""

from copy import deepcopy
from importlib.metadata import version

import numpy as np
import pytest

from chunkeval.chunking import build_chunks, semantic_breakpoints, semantic_units, validate_chunks
from chunkeval.config import PipelineError
from chunkeval.tokenization import SourceTokenizer


class SyntheticCharacterFastTokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        offsets = [(index, index + 1) for index, character in enumerate(text) if not character.isspace()]
        ids = list(range(len(offsets)))
        if kwargs["add_special_tokens"]:
            ids = [-1, *ids, -2]
        result = {"input_ids": ids}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = offsets
        return result


@pytest.fixture
def tokenizer():
    return SourceTokenizer(SyntheticCharacterFastTokenizer())


@pytest.fixture
def config():
    return {
        "embedding": {"max_input_tokens": 512, "passage_prefix": "passage: ", "query_prefix": "query: "},
        "semantic": {"sentence_prefix": "query: ", "buffer_size": 0, "threshold_type": "percentile"},
        "chunking": {
            "C1": {"strategy": "fixed", "max_input_tokens": 256, "overlap_tokens": 50},
            "C2": {"strategy": "fixed", "max_input_tokens": 512, "overlap_tokens": 50},
            "C3": {"strategy": "paragraph", "overlap_tokens": 0, "merge_short": False},
            "C4": {"strategy": "semantic", "percentile": 95, "overlap_tokens": 0},
            "C5": {"strategy": "semantic", "percentile": 80, "overlap_tokens": 0},
        },
    }


def synthetic_doc(text, cuts=(), page_cut=None):
    """Explicit synthetic source annotations, independent of PDF/model downloads."""
    boundaries = [0, *cuts, len(text)] if text else [0]
    sentences = [{"sentence_id": f"S{index}", "char_start": left, "char_end": right,
                  "text": text[left:right]}
                 for index, (left, right) in enumerate(zip(boundaries, boundaries[1:]), 1)
                 if text[left:right].strip()]
    pages = ([{"page_number": 1, "char_start": 0, "char_end": len(text), "method": "synthetic"}]
             if page_cut is None else [
                 {"page_number": 1, "char_start": 0, "char_end": page_cut, "method": "synthetic"},
                 {"page_number": 2, "char_start": page_cut, "char_end": len(text), "method": "synthetic"},
             ])
    return {"doc_id": "SYNTHETIC", "text": text, "pages": pages, "sentences": sentences,
            "paragraphs": [{"paragraph_id": "P1", "char_start": 0, "char_end": len(text), "indent_level": 0}] if text else [],
            "structures": []}


@pytest.mark.parametrize("condition,limit", [("C1", 256), ("C2", 512)])
def test_fixed_full_input_budget_corrected_endpoint_and_exact_50_source_token_overlap(tokenizer, config, condition, limit):
    doc = synthetic_doc("é😀 " * 850, page_cut=800)
    chunks = build_chunks(doc, condition, tokenizer, config)
    assert len(chunks) > 2
    assert all(chunk["n_tokens_with_prefix"] <= limit for chunk in chunks)
    # passage has eight synthetic source tokens, plus two special tokens.
    assert chunks[0]["token_end"] == limit - 10
    assert all(previous["token_end"] - following["token_start"] == 50
               for previous, following in zip(chunks, chunks[1:]))
    assert chunks[-1]["token_end"] == len(tokenizer.offsets(doc["text"]))
    for chunk in chunks:
        assert chunk["text"] == doc["text"][chunk["char_start"]:chunk["char_end"]]
    assert any(chunk["page_numbers"] == [1, 2] for chunk in chunks)


def test_paragraph_structural_starts_partition_nested_ranges_without_overlap(tokenizer, config):
    text = "BAB I\nPasal 1\n(1) sesuai Pasal 5, Rp. 1.000.\n a. Bantuan.\n  1. Rincian.\nPasal 2\nPenutup."
    doc = synthetic_doc(text)
    marker_starts = [text.index("Pasal 1"), text.index("(1)"), text.index("a. Bantuan"),
                     text.index("1. Rincian"), text.index("Pasal 2")]
    doc["structures"] = [
        {"structure_id": "BAB", "char_start": 0, "char_end": len(text), "bab": "I", "pasal": None, "ayat": None, "butir": None},
        {"structure_id": "PASAL1", "char_start": marker_starts[0], "char_end": marker_starts[-1], "bab": "I", "pasal": "1", "ayat": None, "butir": None},
        {"structure_id": "AYAT", "char_start": marker_starts[1], "char_end": marker_starts[-1], "bab": "I", "pasal": "1", "ayat": "1", "butir": None},
        {"structure_id": "BUTIRa", "char_start": marker_starts[2], "char_end": marker_starts[-1], "bab": "I", "pasal": "1", "ayat": "1", "butir": "a"},
        {"structure_id": "BUTIR1", "char_start": marker_starts[3], "char_end": marker_starts[-1], "bab": "I", "pasal": "1", "ayat": "1", "butir": "1"},
        {"structure_id": "PASAL2", "char_start": marker_starts[-1], "char_end": len(text), "bab": "I", "pasal": "2", "ayat": None, "butir": None},
    ]
    chunks = build_chunks(doc, "C3", tokenizer, config)
    assert [chunk["char_start"] for chunk in chunks] == [0, *marker_starts]
    assert "".join(chunk["text"] for chunk in chunks) == text
    assert len(chunks) == 6  # The inline Pasal 5 reference is not a boundary.
    assert chunks[4]["structure_refs"][-1]["butir"] == "1"
    crossing = build_chunks(doc, "C2", tokenizer, config)[0]
    assert {reference["pasal"] for reference in crossing["structure_refs"]} >= {"1", "2"}


@pytest.mark.parametrize("condition", ["C3", "C4", "C5"])
def test_one_oversized_sentence_keeps_every_character_and_original_sentence_id(tokenizer, config, condition):
    doc = synthetic_doc("\n" + "é😀 " * 950 + "Fin.\n")
    units = semantic_units(doc, tokenizer, config)
    vectors = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(units), 1))
    chunks = build_chunks(doc, condition, tokenizer, config, vectors)
    assert len(chunks) > 1
    assert "".join(chunk["text"] for chunk in chunks) == doc["text"]
    assert all(chunk["sentence_ids"] == ["S1"] for chunk in chunks)
    assert all(chunk["n_tokens_with_prefix"] <= 512 for chunk in chunks)
    assert len({tuple(chunk["split_from"]) for chunk in chunks}) == 1
    assert chunks[0]["split_from"]


def test_semantic_unit_gaps_repeat_text_and_prefix_overhead_keep_provenance(tokenizer, config):
    text = "\nAwal.\n\nAwal.\n" + "😀" * 900 + "\n"
    second = text.index("Awal.", text.index("Awal.") + 1)
    third = text.index("😀")
    doc = synthetic_doc(text, cuts=[second, third])
    units = semantic_units(doc, tokenizer, config)
    assert "".join(unit["text"] for unit in units) == text
    assert len({unit["unit_id"] for unit in units}) == len(units)
    assert all(unit["text"] == text[unit["char_start"]:unit["char_end"]] for unit in units)
    assert all(tokenizer.count(unit["text"], "query: ") <= 512 for unit in units)
    assert {unit["sentence_id"] for unit in units if unit["char_start"] >= third} == {"S3"}
    assert all(unit["split_from"] for unit in units if unit["char_start"] >= third)
    # passage overhead is larger than semantic query overhead. An initially
    # legal semantic fragment must still be checked against the passage budget.
    vectors = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(units), 1))
    chunks = build_chunks(doc, "C4", tokenizer, config, vectors)
    assert "".join(chunk["text"] for chunk in chunks) == text
    assert all(chunk["n_tokens_with_prefix"] <= 512 for chunk in chunks)


def test_semantic_fragment_provenance_survives_breakpoints(tokenizer, config):
    doc = synthetic_doc("x" * 1800)
    units = semantic_units(doc, tokenizer, config)
    assert len(units) == 4
    # Only the distance before the last fragment exceeds the percentile. That
    # short final parent fits passage input but still originated in a split unit.
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    chunks = build_chunks(doc, "C4", tokenizer, config, vectors)
    assert chunks[-1]["char_start"] == units[-1]["char_start"]
    assert chunks[-1]["split_from"] == units[-1]["split_from"]
    assert chunks[-1]["sentence_ids"] == ["S1"]
    assert "".join(chunk["text"] for chunk in chunks) == doc["text"]


def test_oversize_splitter_does_not_merge_different_paragraph_parents(tokenizer, config):
    text = "a" * 700 + "\n\n" + "b" * 700
    second_start = 702
    doc = synthetic_doc(text, cuts=[second_start])
    doc["paragraphs"] = [
        {"paragraph_id": "P1", "char_start": 0, "char_end": second_start, "indent_level": 0},
        {"paragraph_id": "P2", "char_start": second_start, "char_end": len(text), "indent_level": 0},
    ]
    chunks = build_chunks(doc, "C3", tokenizer, config)
    assert "".join(chunk["text"] for chunk in chunks) == text
    assert len({tuple(chunk["split_from"]) for chunk in chunks}) == 2
    assert all(chunk["char_end"] <= second_start or chunk["char_start"] >= second_start for chunk in chunks)


def test_pinned_adapter_matches_semanticchunker_zero_buffer_and_linear_percentile():
    from langchain_experimental.text_splitter import SemanticChunker, calculate_cosine_distances

    assert version("langchain-experimental") == "0.3.4"
    angles = np.array([0.0, 0.01, 0.03, 0.23, 0.73, 2.3])
    vectors = np.column_stack([np.cos(angles), np.sin(angles)])
    units = [{"unit_id": f"U{index}", "sentence_id": f"S{index}", "char_start": index,
              "char_end": index + 1, "text": "x"} for index in range(len(vectors))]
    distances, _ = calculate_cosine_distances([
        {"combined_sentence_embedding": vector.tolist()} for vector in vectors
    ])
    for percentile in (80, 95):
        pinned = SemanticChunker(embeddings=None, buffer_size=0, breakpoint_threshold_type="percentile",
                                 breakpoint_threshold_amount=percentile)
        threshold, threshold_array = pinned._calculate_breakpoint_threshold(distances)
        assert threshold == np.percentile(distances, percentile, method="linear")
        expected = [index for index, distance in enumerate(threshold_array) if distance > threshold]
        assert semantic_breakpoints(units, vectors, percentile) == expected


def test_semantic_zero_one_two_and_equal_distances_do_not_break():
    assert semantic_breakpoints([], [], 95) == []
    assert semantic_breakpoints([{}], [[1.0, 0.0]], 95) == []
    assert semantic_breakpoints([{}, {}], [[1.0, 0.0], [0.0, 1.0]], 95) == []
    assert semantic_breakpoints([{}] * 4, [[1.0, 0.0]] * 4, 80) == []
    with pytest.raises(PipelineError, match="align"):
        semantic_breakpoints([{}, {}], [[1.0, 0.0]], 95)
    with pytest.raises(PipelineError, match="L2-normalized"):
        semantic_breakpoints([{}, {}], [[2.0, 0.0], [1.0, 0.0]], 95)


def test_semantic_adapter_rejects_stale_source_unit_spans():
    units = [{"unit_id": "U1", "sentence_id": "S1", "char_start": 0, "char_end": 2, "text": "ab"},
             {"unit_id": "U2", "sentence_id": "S2", "char_start": 3, "char_end": 5, "text": "cd"}]
    with pytest.raises(PipelineError, match="contiguous"):
        semantic_breakpoints(units, [[1.0, 0.0], [0.0, 1.0]], 95)


def test_semantic_percentiles_are_document_local(tokenizer, config):
    doc = synthetic_doc("A. B. C. D. E.", cuts=[3, 6, 9, 12])
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, -1.0]])
    for condition in ("C4", "C5"):
        chunks = build_chunks(doc, condition, tokenizer, config, vectors)
        assert len(chunks) == 2
        assert chunks[0]["text"] == "A. B. C. D. "
        assert chunks[1]["text"] == "E."
    # A second document has only one distance, so that document cannot split
    # even though the distance is above thresholds from the first document.
    other = synthetic_doc("X. Y.", cuts=[3])
    assert len(build_chunks(other, "C4", tokenizer, config, [[1.0, 0.0], [-1.0, 0.0]])) == 1


def test_p80_can_split_more_often_than_p95_on_same_ordered_unit_vectors():
    angles = np.concatenate([[0.0], np.cumsum(np.linspace(0.1, 1.0, 10))])
    vectors = np.column_stack([np.cos(angles), np.sin(angles)])
    units = [{"unit_id": f"U{index}", "sentence_id": f"S{index}", "char_start": index,
              "char_end": index + 1, "text": "x"} for index in range(len(vectors))]
    assert semantic_breakpoints(units, vectors, 95) == [9]
    assert semantic_breakpoints(units, vectors, 80) == [8, 9]


@pytest.mark.parametrize("condition", ["C1", "C2", "C3", "C4", "C5"])
def test_empty_fixture_has_zero_chunks(tokenizer, config, condition):
    assert build_chunks(synthetic_doc(""), condition, tokenizer, config, []) == []


def test_validator_rejects_stale_mapping_lost_content_and_wrong_overlap(tokenizer, config):
    doc = synthetic_doc("é😀 " * 500)
    fixed = build_chunks(doc, "C1", tokenizer, config)
    damaged = deepcopy(fixed)
    damaged[1]["token_start"] += 1
    with pytest.raises(PipelineError, match="coordinates|overlap"):
        validate_chunks(doc, damaged, "C1", tokenizer, config)
    damaged = deepcopy(fixed)
    damaged[0]["page_numbers"] = [99]
    with pytest.raises(PipelineError, match="page_numbers"):
        validate_chunks(doc, damaged, "C1", tokenizer, config)
    paragraph = build_chunks(doc, "C3", tokenizer, config)
    with pytest.raises(PipelineError, match="lost"):
        validate_chunks(doc, paragraph[:-1], "C3", tokenizer, config)
