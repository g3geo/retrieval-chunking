"""Configuration, deterministic provenance hashes and atomic file helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

CONDITIONS = ("C1", "C2", "C3", "C4", "C5")
APP_ROOT = Path(__file__).resolve().parents[2]


class PipelineError(ValueError):
    """Actionable pipeline failure; no partially valid result is accepted."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def canonical_hash(value: Any) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_config(config: dict) -> dict:
    return {k: _jsonable(v) for k, v in config.items() if not k.startswith("_")}


def read_yaml(path: str | Path) -> dict:
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PipelineError(f"Cannot read YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"YAML must contain a mapping: {path}")
    return value


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path).resolve() if path else APP_ROOT / "configs/config.yaml"
    config = read_yaml(config_path)
    config["_app_root"] = APP_ROOT
    config["_config_path"] = config_path
    config.setdefault("corpus", "configs/corpus.yaml")
    config.setdefault("cleaning", "configs/cleaning.yaml")
    config.setdefault("data_dir", "data")
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Reject changes that would silently change the specified experiment."""
    try:
        emb, sem, chunks = config["embedding"], config["semantic"], config["chunking"]
        if emb["model"] != "intfloat/multilingual-e5-small":
            raise PipelineError("This experiment requires multilingual-e5-small.")
        if emb["dimension"] != 384 or emb["max_input_tokens"] != 512:
            raise PipelineError("E5 dimension must be 384 and input limit 512.")
        if emb.get("normalize") is not True or emb.get("dtype", "float32") != "float32":
            raise PipelineError("Embedding requires normalized float32 vectors.")
        if emb["passage_prefix"] != "passage: " or emb["query_prefix"] != "query: ":
            raise PipelineError("Use the specified E5 passage/query prefixes.")
        if not isinstance(emb["batch_size"], int) or emb["batch_size"] < 1:
            raise PipelineError("embedding.batch_size must be a positive integer.")
        if sem["buffer_size"] != 0 or sem["threshold_type"] != "percentile":
            raise PipelineError("Semantic chunking requires buffer_size=0 and percentile.")
        if not isinstance(sem["sentence_prefix"], str) or not sem["sentence_prefix"]:
            raise PipelineError("semantic.sentence_prefix must be nonempty.")
        if set(chunks) != set(CONDITIONS):
            raise PipelineError("Exactly C1..C5 must be configured.")
        expected = {"C1": ("fixed", 256, 50), "C2": ("fixed", 512, 50),
                    "C3": ("paragraph", 512, 0), "C4": ("semantic", 512, 0),
                    "C5": ("semantic", 512, 0)}
        for name, (strategy, limit, overlap) in expected.items():
            item = chunks[name]
            if (item["strategy"] != strategy or item.get("max_input_tokens", 512) != limit
                    or item["overlap_tokens"] != overlap):
                raise PipelineError(f"Invalid strategy/token/overlap settings for {name}.")
        if chunks["C3"].get("merge_short", False):
            raise PipelineError("C3 short paragraphs must not be merged.")
        for name, percentile in (("C4", 95), ("C5", 80)):
            if chunks[name]["percentile"] != percentile:
                raise PipelineError(f"{name} percentile must be {percentile}.")
        if config["retrieval"]["top_k"] != 10:
            raise PipelineError("Batch retrieval top_k must be 10.")
        if not isinstance(config["seed"], int):
            raise PipelineError("seed must be an integer.")
        data_path(config)
    except (KeyError, TypeError) as exc:
        raise PipelineError(f"Missing/invalid configuration field: {exc}") from exc


def data_path(config: dict, *parts: str) -> Path:
    root = Path(config["_app_root"]).resolve()
    target = (root / config.get("data_dir", "data") / Path(*parts)).resolve()
    if not target.is_relative_to(root):
        raise PipelineError("All output paths must stay inside the application directory.")
    return target


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def read_jsonl(path: str | Path) -> list[dict]:
    try:
        records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read JSONL {path}: {exc}") from exc
    if any(not isinstance(row, dict) for row in records):
        raise PipelineError(f"JSONL rows must be objects: {path}")
    return records


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(destination)


def corpus_hash(documents: list[dict]) -> str:
    # Select semantic inputs explicitly: paths, timing, logs and derived hash fields
    # do not participate. Cleaning rules include replacement content hashes.
    fields = ("doc_id", "source_sha256", "text", "pages", "paragraphs", "sentences",
              "structures", "cleaning_rules")
    volatile = {"timestamp", "runtime", "runtime_seconds", "output_path", "corpus_hash",
                "text_sha256", "created_at", "source_path", "replacement_text_file"}

    def stable(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: stable(v) for k, v in value.items()
                    if k not in volatile and not k.startswith("_")}
        if isinstance(value, (list, tuple)):
            return [stable(v) for v in value]
        return value

    ordered = sorted(documents, key=lambda d: d["doc_id"])
    if len({d["doc_id"] for d in ordered}) != len(ordered):
        raise PipelineError("Duplicate doc_id in clean corpus.")
    return canonical_hash([stable({field: doc.get(field) for field in fields}) for doc in ordered])
