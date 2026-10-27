"""Synthetic query/chunk fixtures with real FAISS; no research queries generated."""

import json

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from chunkeval.config import PipelineError, read_jsonl
from chunkeval.embedding import embedding_identity
from chunkeval.indexing import LoadedIndex
from chunkeval.retrieval import load_queries, retrieve_batch, search_index, validate_bundle


class SyntheticEncoder:
    def __init__(self, tmp_path):
        self.config = {"_app_root": tmp_path, "embedding": {"revision": "a" * 40}, "seed": 42}
        self.tokenizer = self
        self.embedding_config_hash = embedding_identity(self.config)["embedding_config_hash"]
        self.calls = []

    def count(self, text, prefix):
        return len(prefix + text) + 2

    def encode_queries(self, texts):
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        vectors[:, 0] = 1
        return vectors


class SearchSpy:
    def __init__(self, index):
        self.index = index
        self.requested = []

    @property
    def ntotal(self):
        return self.index.ntotal

    def search(self, vectors, k):
        self.requested.append(k)
        return self.index.search(vectors, k)


def synthetic_loaded(condition, encoder, count=15):
    index = faiss.IndexFlatIP(384)
    vectors = np.zeros((count, 384), dtype=np.float32)
    vectors[:, 0] = 1
    index.add(vectors)
    chunks = []
    for row_id in range(count):
        text = "duplikat Unicode 漢"
        chunks.append({
            "chunk_id": f"{condition}-{row_id}", "condition": condition, "doc_id": "D-fixture",
            "text": text, "char_start": row_id * 30, "char_end": row_id * 30 + len(text),
            "page_start": 1, "page_end": 2, "page_numbers": [1, 2],
            "n_tokens_with_prefix": len("passage: " + text) + 2,
            "sentence_ids": [f"S{row_id}"], "split_from": [],
            "structure_refs": [{"char_start": 0, "char_end": count * 30,
                                "bab": "I", "pasal": "5", "ayat": None, "butir": None}],
        })
    manifest = {"condition": condition, "corpus_hash": "a" * 64,
                "model_revision": "a" * 40, "tokenizer_revision": "a" * 40,
                "embedding_config_hash": encoder.embedding_config_hash}
    return LoadedIndex(SearchSpy(index), chunks, manifest, "b" * 64)


def test_ties_crossing_k_use_all_rows_and_smallest_row_ids(tmp_path):
    encoder = SyntheticEncoder(tmp_path)
    loaded = synthetic_loaded("C1", encoder, 15)
    vector = encoder.encode_queries(["fixture"])[0]
    results = search_index(loaded, vector, 10)
    assert loaded.index.requested == [15]
    assert [result["row_id"] for result in results] == list(range(10))
    assert [result["rank"] for result in results] == list(range(1, 11))
    assert all(result["score"] == 1 for result in results)
    assert results[0]["text"] == loaded.chunks[0]["text"]
    assert results[0]["structure_refs"] == loaded.chunks[0]["structure_refs"]


def test_search_small_index_has_actual_count_and_no_padding(tmp_path):
    encoder = SyntheticEncoder(tmp_path)
    vector = encoder.encode_queries(["fixture"])[0]
    results = search_index(synthetic_loaded("C3", encoder, 2), vector, 10)
    assert len(results) == 2
    assert all(result["row_id"] >= 0 for result in results)
    with pytest.raises(PipelineError, match="nonempty"):
        search_index(synthetic_loaded("C3", encoder, 0), vector)


def test_batch_shares_one_embedding_per_unique_text_and_writes_complete_rows(tmp_path):
    encoder = SyntheticEncoder(tmp_path)
    indexes = {condition: synthetic_loaded(condition, encoder) for condition in ("C1", "C2", "C3", "C4", "C5")}
    queries = [{"qid": "synthetic-Q1", "text": "same fixture"},
               {"qid": "synthetic-Q2", "text": "same fixture"},
               {"qid": "synthetic-Q3", "text": "other fixture"}]
    output = tmp_path / "results/results.jsonl"
    rows = retrieve_batch(queries, indexes, encoder, output)
    assert encoder.calls == [["same fixture", "other fixture"]]
    assert len(rows) == 15 and rows == read_jsonl(output)
    assert all(len(row["results"]) == 10 for row in rows)
    assert all(row["index_manifest_hash"] == "b" * 64 for row in rows)
    assert {row["condition"] for row in rows} == set(indexes)
    assert all(loaded.index.requested == [15, 15, 15] for loaded in indexes.values())


@pytest.mark.parametrize("field", ["corpus_hash", "model_revision", "tokenizer_revision", "embedding_config_hash"])
def test_bundle_rejects_each_identity_mismatch(tmp_path, field):
    encoder = SyntheticEncoder(tmp_path)
    indexes = {condition: synthetic_loaded(condition, encoder) for condition in ("C1", "C2", "C3", "C4", "C5")}
    indexes["C5"].manifest[field] = "different"
    with pytest.raises(PipelineError, match="mixes"):
        validate_bundle(indexes)


def test_incomplete_bundle_and_batch_less_than_k_rejected_before_encode(tmp_path):
    encoder = SyntheticEncoder(tmp_path)
    indexes = {condition: synthetic_loaded(condition, encoder) for condition in ("C1", "C2", "C3", "C4", "C5")}
    partial = {key: value for key, value in indexes.items() if key != "C5"}
    with pytest.raises(PipelineError, match="complete"):
        validate_bundle(partial)
    indexes["C5"] = synthetic_loaded("C5", encoder, 9)
    output = tmp_path / "not-written.jsonl"
    with pytest.raises(PipelineError, match="at least 10"):
        retrieve_batch([{"qid": "synthetic", "text": "fixture"}], indexes, encoder, output)
    assert encoder.calls == [] and not output.exists()


@pytest.mark.parametrize("queries, expected", [
    ([{"qid": "synthetic", "text": ""}], "nonempty text"),
    ([{"qid": "synthetic", "text": "x" * 504}], "not truncated"),
    ([{"qid": "synthetic", "text": "a"}, {"qid": "synthetic", "text": "b"}], "Duplicate"),
    ({"qid": "synthetic", "text": "a"}, "JSON array"),
])
def test_invalid_query_input_rejected(tmp_path, queries, expected):
    encoder = SyntheticEncoder(tmp_path)
    path = tmp_path / "synthetic-queries.json"
    path.write_text(json.dumps(queries), encoding="utf-8")
    with pytest.raises(PipelineError, match=expected):
        load_queries(path, encoder.tokenizer, encoder.config)


def test_query_text_is_preserved_exactly(tmp_path):
    encoder = SyntheticEncoder(tmp_path)
    queries = [{"qid": "synthetic", "text": "  Rp. 1.000,00 untuk 漢\nPasal 5?  "}]
    path = tmp_path / "synthetic-queries.json"
    path.write_text(json.dumps(queries, ensure_ascii=False), encoding="utf-8")
    assert load_queries(path, encoder.tokenizer, encoder.config) == queries
