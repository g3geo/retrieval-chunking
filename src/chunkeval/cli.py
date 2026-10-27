"""Independent commands for preparing, chunking, indexing and retrieval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from time import perf_counter
from uuid import uuid4

from .config import (CONDITIONS, PipelineError, canonical_hash, corpus_hash, data_path,
                     file_sha256, load_config, read_jsonl, write_json, write_jsonl)
from .terminal_ui import TerminalUI


def _stage(config: dict, name: str) -> Path:
    parent = data_path(config)
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{name}-", dir=parent))


def _remove_generated(config: dict, path: Path) -> None:
    root = data_path(config).resolve()
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise PipelineError(f"Refusing to remove a path outside generated stage output: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _publish(config: dict, stage: Path, destination: Path) -> None:
    # Publish a whole condition bundle so interrupted builds cannot appear complete.
    backup = destination.with_name(f".{destination.name}-previous-{uuid4().hex}")
    had_previous = destination.exists()
    if had_previous:
        destination.replace(backup)
    try:
        stage.replace(destination)
    except OSError:
        if had_previous:
            backup.replace(destination)
        raise
    if had_previous:
        _remove_generated(config, backup)


def _chunk_identity(config: dict) -> dict:
    from .embedding import embedding_identity
    from . import chunking, segmentation, tokenization
    return {"chunking": config["chunking"],
            "semantic": {key: value for key, value in config["semantic"].items() if key != "cache_enabled"},
            "encoder": embedding_identity(config),
            "code": {module.__name__: file_sha256(module.__file__)
                     for module in (chunking, segmentation, tokenization)}}


def _load_chunks(config: dict, docs: list[dict], tokenizer) -> dict[str, list[dict]]:
    from .chunking import validate_chunks
    root = data_path(config, "chunks")
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineError(f"Missing/invalid chunks manifest. Run chunk first: {exc}") from exc
    if manifest.get("corpus_hash") != corpus_hash(docs) or manifest.get("identity") != _chunk_identity(config):
        raise PipelineError("Chunks were built with another corpus/configuration/code. Run chunk and index again.")
    bundle = {}
    for condition in CONDITIONS:
        path = root / condition / "chunks.jsonl"
        if file_sha256(path) != manifest.get("checksums", {}).get(condition):
            raise PipelineError(f"Chunk checksum mismatch for {condition}; run chunk again.")
        rows = read_jsonl(path)
        known_ids = {doc["doc_id"] for doc in docs}
        if any(row.get("doc_id") not in known_ids for row in rows):
            raise PipelineError(f"Unknown document in {condition} chunks.")
        if len({row.get("chunk_id") for row in rows}) != len(rows):
            raise PipelineError(f"Duplicate chunk IDs in {condition}.")
        for doc in docs:
            validate_chunks(doc, [row for row in rows if row["doc_id"] == doc["doc_id"]],
                            condition, tokenizer, config)
        bundle[condition] = rows
    return bundle


def _indexes(config: dict, docs: list[dict], conditions=CONDITIONS) -> dict:
    from .indexing import load_index
    current_hash = corpus_hash(docs)
    result = {}
    for condition in conditions:
        loaded = load_index(data_path(config, "indexes", condition), config, current_hash)
        if loaded.manifest["condition"] != condition:
            raise PipelineError(f"Index folder {condition} contains a different condition.")
        result[condition] = loaded
    return result


def _display(record: dict, full_text: bool, ui: TerminalUI | None = None) -> None:
    (ui or TerminalUI()).result(record, full_text)


def _saved_search(args) -> None:
    if args.query:
        raise PipelineError("Use either --results or --query for search.")
    records = read_jsonl(args.results)
    selected = [row for row in records if (not args.qid or row.get("qid") == args.qid)
                and (args.all or row.get("condition") == (args.condition or "C3"))]
    if not selected:
        raise PipelineError("No saved results match the requested qid/condition.")
    ui = TerminalUI(no_color=args.no_color)
    for record in selected:
        if not isinstance(record.get("results"), list) or record.get("condition") not in CONDITIONS:
            raise PipelineError("Malformed saved retrieval record.")
        for hit in record["results"]:
            required = ("rank", "score", "doc_id", "page_numbers", "n_tokens_with_prefix",
                        "structure_refs", "text")
            if not isinstance(hit, dict) or any(key not in hit for key in required):
                raise PipelineError("Saved retrieval hit lacks display metadata.")
            if (not isinstance(hit["text"], str) or not isinstance(hit["page_numbers"], list)
                    or not isinstance(hit["structure_refs"], list)
                    or any(not isinstance(ref, dict) for ref in hit["structure_refs"])
                    or not isinstance(hit["score"], (int, float))):
                raise PipelineError("Saved retrieval hit contains invalid text/page/structure/score metadata.")
        ui.query(record.get("query_text", record.get("qid", "Hasil tersimpan")))
        _display({**record, "results": record["results"][:args.k]}, args.full_text, ui)


def _search_conditions(args) -> tuple[str, ...]:
    # Bare terminal search compares all conditions. Existing one-shot and saved
    # search keep their C3 default; an explicit --condition selects just one.
    if args.all or (args.query is None and args.condition is None):
        return CONDITIONS
    return (args.condition or "C3",)


def _search_query(query: str, args, conditions, indexes, encoder) -> list[dict]:
    from .retrieval import search_index
    started = perf_counter()
    vector = encoder.encode_queries([query])[0]
    records = [{"qid": "search", "query_text": query, "condition": condition,
                "index_manifest_hash": indexes[condition].manifest_hash,
                "results": search_index(indexes[condition], vector, args.k)} for condition in conditions]
    ui = TerminalUI(no_color=args.no_color, interactive=args.query is None)
    ui.query(query, perf_counter() - started)
    ui.summary(records)
    for record in records:
        _display(record, args.full_text, ui)
    return records


def _terminal_command(query: str, args, conditions, indexes, last_records, ui):
    """Handle display controls without re-encoding or altering stored artifacts."""
    words = query.split()
    command = words[0].casefold()
    if command in {"/bantuan", "/help"} and len(words) == 1:
        ui.help(indexes)
    elif command == "/mode" and len(words) == 2:
        mode = words[1].upper()
        if mode in {"SEMUA", "ALL"}:
            conditions = tuple(condition for condition in CONDITIONS if condition in indexes)
        elif mode in indexes:
            conditions = (mode,)
        else:
            raise PipelineError("Pilih kondisi yang sudah dimuat: " + ", ".join(indexes))
        ui.banner(conditions, args.k, args.full_text)
    elif command == "/k" and len(words) == 2:
        try:
            k = int(words[1])
        except ValueError as exc:
            raise PipelineError("Gunakan /k dengan angka 1 sampai 10, misalnya /k 5.") from exc
        if not 1 <= k <= 10:
            raise PipelineError("Jumlah hasil harus antara 1 dan 10.")
        args.k = k
        ui.message(f"Jumlah hasil per kondisi: {k}")
    elif command == "/teks" and len(words) == 2 and words[1].casefold() in {"lengkap", "ringkas"}:
        args.full_text = words[1].casefold() == "lengkap"
        ui.message("Tampilan teks: " + ("lengkap" if args.full_text else "ringkas"))
    elif command == "/lihat" and len(words) == 3:
        try:
            rank = int(words[2])
        except ValueError as exc:
            raise PipelineError("Gunakan /lihat KONDISI NOMOR, misalnya /lihat C3 1.") from exc
        condition = words[1].upper()
        hit = next((hit for record in last_records if record["condition"] == condition
                    for hit in record["results"] if hit["rank"] == rank), None)
        if hit is None:
            raise PipelineError("Hasil tersebut tidak ada pada pencarian terakhir. Ketik pertanyaan dahulu atau periksa kondisi dan nomor hasil.")
        ui.hit(condition, hit, True)
    else:
        raise PipelineError("Perintah tidak dikenal atau formatnya salah. Ketik /bantuan untuk melihat pilihan.")
    return conditions


def _interactive_search(args, conditions, indexes, encoder) -> None:
    ui = TerminalUI(no_color=args.no_color, interactive=True)
    errors = TerminalUI(no_color=args.no_color, stream=sys.stderr)
    last_records = []
    ui.banner(conditions, args.k, args.full_text)
    try:
        while True:
            query = input(ui.prompt()).strip()
            if query.casefold() in {"/keluar", "keluar", "/exit", "exit", "/quit", "quit"}:
                break
            if not query:
                ui.message("Pertanyaan tidak boleh kosong. Silakan ketik pertanyaan.")
                continue
            try:
                if query.startswith("/"):
                    conditions = _terminal_command(query, args, conditions, indexes, last_records, ui)
                    continue
                ui.message("Mencari dokumen yang sesuai...")
                last_records = _search_query(query, args, conditions, indexes, encoder) or []
            except PipelineError as exc:
                # Invalid/oversize input must not close a reusable search session.
                errors.error(str(exc))
    except (EOFError, KeyboardInterrupt):
        ui.write()
    ui.message("Pencarian selesai.")


def run(args, config: dict) -> None:
    if args.command == "search" and args.results:
        _saved_search(args)
        return
    if args.command == "search" and args.query is None:
        TerminalUI(no_color=args.no_color).message("Memuat korpus, indeks, dan model E5 untuk pencarian terminal...")
    from .preprocess import load_clean_corpus, prepare_corpus
    if args.command == "prepare":
        docs = prepare_corpus(config)
        print(f"Korpus bersih: {len(docs)} dokumen. Hash: {corpus_hash(docs)}")
        return
    docs = load_clean_corpus(config)
    if args.command == "chunk":
        from .chunking import build_chunks, semantic_units, validate_chunks
        from .embedding import E5Encoder
        encoder = E5Encoder(config, data_path(config, "cache", "semantic"))
        bundle = {condition: [] for condition in CONDITIONS}
        for doc in docs:
            units = semantic_units(doc, encoder.tokenizer, config)
            unit_vectors = encoder.encode_units(units)
            for condition in CONDITIONS:
                chunks = build_chunks(doc, condition, encoder.tokenizer, config,
                                      unit_vectors if condition in ("C4", "C5") else None)
                validate_chunks(doc, chunks, condition, encoder.tokenizer, config)
                bundle[condition].extend(chunks)
        stage = _stage(config, "chunks")
        try:
            checksums = {}
            for condition, rows in bundle.items():
                rows.sort(key=lambda row: (row["doc_id"], row["char_start"], row["char_end"]))
                path = stage / condition / "chunks.jsonl"
                write_jsonl(path, rows)
                checksums[condition] = file_sha256(path)
            write_json(stage / "manifest.json", {"corpus_hash": corpus_hash(docs),
                        "identity": _chunk_identity(config), "checksums": checksums})
            _publish(config, stage, data_path(config, "chunks"))
        finally:
            _remove_generated(config, stage)
        print("Chunk tersimpan: " + ", ".join(f"{key}={len(rows)}" for key, rows in bundle.items()))
        return
    if args.command == "index":
        from .embedding import E5Encoder
        from .indexing import build_index
        from .retrieval import validate_bundle
        encoder = E5Encoder(config, data_path(config, "cache", "semantic"))
        chunks = _load_chunks(config, docs, encoder.tokenizer)
        stage = _stage(config, "indexes")
        try:
            indexes = {condition: build_index(condition, chunks[condition], encoder, config,
                                             corpus_hash(docs), stage / condition)
                       for condition in CONDITIONS}
            validate_bundle(indexes)
            _publish(config, stage, data_path(config, "indexes"))
        finally:
            _remove_generated(config, stage)
        print("Lima indeks FAISS tersimpan di " + str(data_path(config, "indexes")))
        return
    if args.command == "validate":
        from .chunking import validate_chunks
        from .retrieval import validate_bundle
        from .tokenization import load_tokenizer
        import numpy as np
        tokenizer = load_tokenizer(config)
        chunks = _load_chunks(config, docs, tokenizer)
        indexes = validate_bundle(_indexes(config, docs))
        for condition, loaded in indexes.items():
            if canonical_hash(chunks[condition]) != canonical_hash(loaded.chunks):
                raise PipelineError(f"{condition} index mapping differs from current chunks; rebuild index.")
            for row_id in range(loaded.index.ntotal):
                vector = np.asarray(loaded.index.reconstruct(row_id), dtype=np.float32)
                scores, ids = loaded.index.search(vector.reshape(1, -1), loaded.index.ntotal)
                if not np.isclose(scores[0, 0], 1.0, atol=1e-5):
                    raise PipelineError(f"{condition} self-search maximum score differs from 1.")
                own_score = scores[0, np.flatnonzero(ids[0] == row_id)[0]]
                if not np.isclose(own_score, 1.0, atol=1e-5):
                    raise PipelineError(f"{condition} passage self-search mapping is invalid.")
        print("Validasi span, token, coverage, metadata, overlap, mapping, indeks, dan self-search selesai.")
        return
    from .embedding import E5Encoder
    from .retrieval import load_queries, retrieve_batch, validate_bundle
    conditions = CONDITIONS if args.command == "retrieve" else _search_conditions(args)
    indexes = _indexes(config, docs, conditions)
    if len(conditions) == 5:
        validate_bundle(indexes)
    encoder = E5Encoder(config)
    if args.command == "retrieve":
        queries = load_queries(args.queries, encoder.tokenizer, config)
        rows = retrieve_batch(queries, indexes, encoder, data_path(config, "results", "results.jsonl"),
                              config["retrieval"]["top_k"])
        print(f"Tersimpan {len(rows)} pasangan kueri-kondisi: " + str(data_path(config, "results", "results.jsonl")))
        return
    if args.query is None:
        _interactive_search(args, conditions, indexes, encoder)
    elif not args.query.strip():
        raise PipelineError("search --query must contain non-whitespace text.")
    else:
        _search_query(args.query, args, conditions, indexes, encoder)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Pipeline retrieval E5 lokal C1-C5")
    root.add_argument("--config", default=None)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("prepare", "chunk", "index", "retrieve", "search", "validate"):
        item = commands.add_parser(name)
        item.add_argument("--config", default=argparse.SUPPRESS)
        if name == "retrieve":
            item.add_argument("--queries", required=True, type=Path)
        elif name == "search":
            item.add_argument("--query", help="Search once; omit to enter questions interactively in the terminal")
            mode = item.add_mutually_exclusive_group()
            mode.add_argument("--condition", choices=CONDITIONS,
                              help="Search one condition (default: all in interactive mode; C3 with --query/--results)")
            mode.add_argument("--all", action="store_true")
            item.add_argument("--k", type=int, choices=range(1, 11), default=10)
            item.add_argument("--full-text", action="store_true")
            item.add_argument("--no-color", action="store_true", help="Disable terminal colors")
            item.add_argument("--results", type=Path, help="Read saved JSONL without loading/encoding E5")
            item.add_argument("--qid", help="Filter saved JSONL by qid")
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        run(args, load_config(args.config))
        return 0
    except (PipelineError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
