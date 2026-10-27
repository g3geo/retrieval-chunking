"""Synthetic end-to-end corpus test; never uses the research PDFs or E5 weights."""
from copy import deepcopy

import numpy as np
import pytest
import yaml
from pypdf import PdfWriter

from chunkeval.chunking import build_chunks, semantic_units, validate_chunks
from chunkeval.config import CONDITIONS, PipelineError, corpus_hash, file_sha256, load_config, read_jsonl
from chunkeval.embedding import E5Encoder, embedding_identity
from chunkeval.indexing import build_index, load_index
from chunkeval.preprocess import load_clean_corpus, prepare_corpus
from chunkeval.retrieval import retrieve_batch, search_index, validate_bundle
from chunkeval.tokenization import SourceTokenizer


class SyntheticFastTokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        offsets = [(i, i + 1) for i, char in enumerate(text) if not char.isspace()]
        ids = list(range(len(offsets)))
        if kwargs.get("add_special_tokens", True):
            ids = [-1, *ids, -2]
        result = {"input_ids": ids}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = offsets
        return result


class SyntheticLocalModel:
    """Deliberate identical vectors exercise ties and duplicate source provenance."""
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        result = np.zeros((len(texts), 384), dtype=np.float32)
        result[:, 0] = 1
        return result


def fixture_config(tmp_path):
    config = deepcopy(load_config())
    config["_app_root"] = tmp_path
    config["_config_path"] = tmp_path / "configs/config.yaml"
    configs = tmp_path / "configs"
    configs.mkdir()
    sources = tmp_path / "sources"
    sources.mkdir()
    records, rules = [], {}
    for number in range(1, 3):
        doc_id = f"SYNTHETIC{number}"
        path = sources / f"{doc_id}.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        writer.add_blank_page(width=612, height=792)
        with path.open("wb") as handle:
            writer.write(handle)
        records.append({"doc_id": doc_id, "path": path.name})
        pages = {}
        for page in (1, 2):
            text = "\n\n".join(
                f"Pasal {i}\n\n(1) Fixture sintetis café 漢 Rp. 1.000,00 sesuai Pasal 5. "
                "Bantuan pengujian ini bukan data penelitian."
                for i in range(1 + (page - 1) * 55, 56 + (page - 1) * 55))
            replacement = configs / f"{doc_id}-p{page}.txt"
            replacement.write_text(text, encoding="utf-8")
            pages[page] = {"replacement_text_file": str(replacement),
                           "reason": "Synthetic fixture with deterministic reviewed text"}
        rules[doc_id] = {"source_sha256": file_sha256(path), "pages": pages}
    (configs / "corpus.yaml").write_text(yaml.safe_dump(
        {"source_root": "../sources", "expected_total_pages": 4, "documents": records}), encoding="utf-8")
    (configs / "cleaning.yaml").write_text(yaml.safe_dump(
        {"defaults": {"dpi": 300, "ocr_min_nonspace_chars": 40}, "documents": rules}), encoding="utf-8")
    return config


def fixture_encoder(config):
    encoder = E5Encoder.__new__(E5Encoder)
    encoder.config = config
    encoder.identity = embedding_identity(config)
    encoder.resolved_config = encoder.identity["resolved_config"]
    encoder.embedding_config_hash = encoder.identity["embedding_config_hash"]
    encoder.dependency_versions = encoder.identity["dependency_versions"]
    encoder.cache_enabled = True
    encoder.cache_dir = config["_app_root"] / "data/cache/semantic"
    encoder.tokenizer = SourceTokenizer(SyntheticFastTokenizer())
    encoder.model = SyntheticLocalModel()
    return encoder


def test_small_corpus_five_conditions_indices_retrieval_reload_and_stale_replacement(tmp_path):
    pytest.importorskip("faiss")
    config = fixture_config(tmp_path)
    documents = prepare_corpus(config)
    assert documents == load_clean_corpus(config)
    initial_hash = corpus_hash(documents)
    encoder = fixture_encoder(config)
    bundle = {condition: [] for condition in CONDITIONS}
    for document in documents:
        units = semantic_units(document, encoder.tokenizer, config)
        vectors = encoder.encode_units(units)
        # Duplicate unit text may use the same cache vector but IDs and spans persist.
        assert len(units) == len(vectors)
        for condition in CONDITIONS:
            chunks = build_chunks(document, condition, encoder.tokenizer, config,
                                  vectors if condition in ("C4", "C5") else None)
            validate_chunks(document, chunks, condition, encoder.tokenizer, config)
            bundle[condition].extend(chunks)
    indexes = {condition: build_index(condition, chunks, encoder, config, initial_hash,
                                     tmp_path / "data/indexes" / condition)
               for condition, chunks in bundle.items()}
    validate_bundle(indexes)
    assert all(loaded.index.ntotal >= 10 for loaded in indexes.values())
    output = tmp_path / "data/results/results.jsonl"
    queries = [{"qid": "SYNTHETIC-Q1", "text": "Pertanyaan sintetis untuk pengujian?"}]
    rows = retrieve_batch(queries, indexes, encoder, output)
    assert len(rows) == 5 and rows == read_jsonl(output)
    assert all([hit["row_id"] for hit in row["results"]] == list(range(10)) for row in rows)
    reloaded = {condition: load_index(tmp_path / "data/indexes" / condition, config, initial_hash)
                for condition in CONDITIONS}
    vector = encoder.encode_queries([queries[0]["text"]])[0]
    for condition in CONDITIONS:
        assert search_index(reloaded[condition], vector) == rows[CONDITIONS.index(condition)]["results"]
    # A length-changing reviewed correction invalidates corpus and every index.
    replacement = tmp_path / "configs/SYNTHETIC1-p1.txt"
    replacement.write_text(replacement.read_text(encoding="utf-8") + "\n\nPerubahan panjang teks.", encoding="utf-8")
    with pytest.raises(PipelineError, match="rules changed"):
        load_clean_corpus(config)
    updated = prepare_corpus(config)
    assert corpus_hash(updated) != initial_hash
    with pytest.raises(PipelineError, match="corpus hash"):
        load_index(tmp_path / "data/indexes/C1", config, corpus_hash(updated))

