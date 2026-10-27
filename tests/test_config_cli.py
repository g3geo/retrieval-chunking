"""Configuration provenance and saved-result access without model encoding."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from chunkeval import cli
from chunkeval.config import PipelineError, corpus_hash, data_path, load_config, validate_config, write_jsonl


def test_corpus_hash_is_deterministic_and_excludes_runtime_paths():
    document = {"doc_id": "SYNTHETIC", "source_sha256": "a" * 64, "text": "café 漢",
                "pages": [{"page_number": 1, "char_start": 0, "char_end": 6}],
                "paragraphs": [], "sentences": [], "structures": [], "cleaning_rules": {}}
    other = {**document, "timestamp": "tomorrow", "output_path": "other", "corpus_hash": "irrelevant"}
    assert corpus_hash([document]) == corpus_hash([other])
    assert corpus_hash([document]) != corpus_hash([{**document, "text": "café 漢!"}])


def test_config_preserves_experiment_and_output_is_contained(tmp_path):
    config = deepcopy(load_config())
    config["_app_root"] = tmp_path
    config["chunking"]["C4"]["percentile"] = 80
    with pytest.raises(PipelineError, match="C4 percentile"):
        validate_config(config)
    config["data_dir"] = "../outside"
    with pytest.raises(PipelineError, match="inside"):
        data_path(config)


def test_every_stage_accepts_its_own_config_argument():
    for command in ("prepare", "chunk", "index", "retrieve", "search", "validate"):
        arguments = [command, "--config", "custom.yaml"]
        if command == "retrieve":
            arguments.extend(["--queries", "synthetic-queries.json"])
        assert cli.parser().parse_args(arguments).config == "custom.yaml"


@pytest.mark.parametrize("condition_args", [[], ["--condition", "C3"]])
def test_saved_jsonl_search_does_not_load_corpus_or_encoder(tmp_path, monkeypatch, capsys, condition_args):
    path = tmp_path / "results.jsonl"
    write_jsonl(path, [{"qid": "SYNTHETIC-Q1", "condition": "C3", "query_text": "synthetic",
                       "results": [{"rank": 1, "score": 0.123456789, "doc_id": "SYNTHETIC",
                                    "page_numbers": [1], "n_tokens_with_prefix": 6,
                                    "structure_refs": [], "text": "Synthetic saved result"}]}])
    # The early branch must not access missing PDF/index/model files.
    args = cli.parser().parse_args(["search", "--results", str(path), "--qid", "SYNTHETIC-Q1",
                                   *condition_args, "--full-text"])
    cli.run(args, {})
    assert "Synthetic saved result" in capsys.readouterr().out


def test_terminal_search_scope_and_existing_single_query_defaults():
    assert cli._search_conditions(cli.parser().parse_args(["search"])) == cli.CONDITIONS
    assert cli._search_conditions(cli.parser().parse_args(["search", "--condition", "C2"])) == ("C2",)
    assert cli._search_conditions(cli.parser().parse_args(["search", "--query", "synthetic"])) == ("C3",)
    assert cli._search_conditions(cli.parser().parse_args(["search", "--query", "synthetic", "--all"])) == cli.CONDITIONS


def test_terminal_search_reuses_query_vector_across_conditions_and_recovers_input(monkeypatch, capsys):
    from chunkeval import retrieval
    inputs = iter(["   ", "  Pertanyaan satu?  ", "terlalu panjang", "Pertanyaan dua?", "/Keluar"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    encoded, searched = [], []

    class Encoder:
        def encode_queries(self, texts):
            assert len(texts) == 1
            encoded.extend(texts)
            if texts[0] == "terlalu panjang":
                raise PipelineError("maximum is 512; input was not truncated")
            return [object()]

    indexes = {condition: SimpleNamespace(manifest_hash="synthetic", condition=condition)
               for condition in cli.CONDITIONS}

    def fake_search(index, vector, k):
        searched.append((index.condition, vector, k))
        return [{"rank": 1, "score": 0.25, "doc_id": "SYNTHETIC", "page_numbers": [2],
                 "n_tokens_with_prefix": 8, "structure_refs": [], "text": "Synthetic terminal result"}]

    monkeypatch.setattr(retrieval, "search_index", fake_search)
    args = cli.parser().parse_args(["search", "--k", "5", "--full-text"])
    cli._interactive_search(args, cli.CONDITIONS, indexes, Encoder())
    assert encoded == ["Pertanyaan satu?", "terlalu panjang", "Pertanyaan dua?"]
    assert [condition for condition, _, _ in searched] == list(cli.CONDITIONS) * 2
    assert all(k == 5 for _, _, k in searched)
    assert all(vector is searched[0][1] for _, vector, _ in searched[:5])
    assert all(vector is searched[5][1] for _, vector, _ in searched[5:])
    assert searched[0][1] is not searched[5][1]
    output = capsys.readouterr()
    assert output.out.count("Synthetic terminal result") == 10
    assert "Pertanyaan tidak boleh kosong" in output.out and "Pencarian selesai" in output.out
    assert "maximum is 512" in output.err


@pytest.mark.parametrize("end", [EOFError, KeyboardInterrupt])
def test_terminal_search_closes_cleanly_on_eof_or_ctrl_c(monkeypatch, capsys, end):
    def stop(prompt):
        raise end()

    monkeypatch.setattr("builtins.input", stop)
    args = cli.parser().parse_args(["search"])
    cli._interactive_search(args, cli.CONDITIONS, {}, None)
    assert "Pencarian selesai" in capsys.readouterr().out


def test_terminal_search_loads_corpus_index_and_encoder_once_per_session(monkeypatch):
    from chunkeval import embedding, preprocess
    calls = {"corpus": 0, "indexes": 0, "encoder": 0, "queries": []}
    encoder = object()
    indexes = {"C3": object()}
    inputs = iter(["satu", "dua", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    def load_corpus(config):
        calls["corpus"] += 1
        return []

    def load_indexes(config, docs, conditions):
        calls["indexes"] += 1
        assert conditions == ("C3",)
        return indexes

    def load_encoder(config):
        calls["encoder"] += 1
        return encoder

    def search(query, args, conditions, loaded, model):
        assert loaded is indexes and model is encoder
        calls["queries"].append(query)

    monkeypatch.setattr(preprocess, "load_clean_corpus", load_corpus)
    monkeypatch.setattr(cli, "_indexes", load_indexes)
    monkeypatch.setattr(embedding, "E5Encoder", load_encoder)
    monkeypatch.setattr(cli, "_search_query", search)
    cli.run(cli.parser().parse_args(["search", "--condition", "C3"]), {})
    assert calls == {"corpus": 1, "indexes": 1, "encoder": 1, "queries": ["satu", "dua"]}
