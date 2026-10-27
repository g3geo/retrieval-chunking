"""Local PDF/OCR selection, reviewed cleaning, and canonical corpus provenance.

The source PDFs are read only. Ambiguous raster/table content cannot become a
valid research corpus until an explicit page override resolves it.
"""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import yaml

from .config import PipelineError, canonical_hash, file_sha256, write_json
from .segmentation import AYAT, BAB, LETTER, PASAL, annotate_document, number_enumeration

PREPROCESS_VERSION = "canonical-geometry-v1"


def normalize_lines(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    return [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise PipelineError(f"Configuration file not found: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise PipelineError(f"Cannot read YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"Expected a YAML mapping: {path}")
    return value


def _resolve_app_path(config: dict, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(config["_app_root"]) / path).resolve()


def _inputs(config: dict) -> tuple[list[dict], dict]:
    corpus_path = _resolve_app_path(config, config.get("corpus", "configs/corpus.yaml"))
    cleaning_path = _resolve_app_path(config, config.get("cleaning", "configs/cleaning.yaml"))
    corpus, cleaning = _load_yaml(corpus_path), _load_yaml(cleaning_path)
    source_root = (corpus_path.parent / corpus.get("source_root", "../..")).resolve()
    entries = corpus.get("documents")
    if not isinstance(entries, list) or not entries:
        raise PipelineError("corpus.yaml documents must be a non-empty list")
    documents, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or not re.fullmatch(r"[A-Za-z0-9_-]+", str(entry.get("doc_id", ""))):
            raise PipelineError("Each source needs a safe non-empty doc_id and PDF path")
        doc_id = entry["doc_id"]
        if doc_id in seen:
            raise PipelineError(f"Duplicate source doc_id: {doc_id}")
        seen.add(doc_id)
        value = entry.get("path")
        if not isinstance(value, str) or not value:
            raise PipelineError(f"{doc_id}: missing source path")
        path = Path(value)
        path = path.resolve() if path.is_absolute() else (source_root / path).resolve()
        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise PipelineError(f"{doc_id}: source PDF not found: {path}")
        documents.append({"doc_id": doc_id, "path": path})
    defaults = cleaning.get("defaults", {})
    rules = cleaning.get("documents", {})
    if not isinstance(defaults, dict) or not isinstance(rules, dict):
        raise PipelineError("cleaning.yaml defaults/documents must be mappings")
    unknown = set(rules) - seen
    if unknown:
        raise PipelineError(f"Cleaning rules name unknown documents: {sorted(unknown)}")
    expected = corpus.get("expected_total_pages")
    if expected is not None and (isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0):
        raise PipelineError("expected_total_pages must be a positive integer")
    return sorted(documents, key=lambda row: row["doc_id"]), {**cleaning, "_expected_total_pages": expected}


def _rules_for(config: dict, cleaning: dict, entry: dict, source_hash: str) -> tuple[dict, dict]:
    supplied_defaults = cleaning.get("defaults", {})
    defaults = {"ocr_min_nonspace_chars": 40, "dpi": supplied_defaults.get("ocr_dpi", 300), **supplied_defaults}
    rule = cleaning.get("documents", {}).get(entry["doc_id"], {})
    if not isinstance(rule, dict):
        raise PipelineError(f"{entry['doc_id']}: document cleaning rule must be a mapping")
    if rule and rule.get("source_sha256") != source_hash:
        raise PipelineError(f"{entry['doc_id']}: source_sha256 differs from the actual PDF; review rules and rebuild")
    method = rule.get("method", defaults.get("method", "auto"))
    if method not in {"auto", "pypdf", "ocr"}:
        raise PipelineError(f"{entry['doc_id']}: invalid method {method!r}")
    for key in ("dpi", "ocr_min_nonspace_chars"):
        if isinstance(defaults[key], bool) or not isinstance(defaults[key], int) or defaults[key] <= 0:
            raise PipelineError(f"cleaning defaults {key} must be a positive integer")
    return defaults, rule


def _page_rules(rule: dict, page_count: int, doc_id: str) -> dict[int, dict]:
    pages = rule.get("pages", {})
    if not isinstance(pages, dict):
        raise PipelineError(f"{doc_id}: pages overrides must be a mapping")
    result = {}
    for number, override in pages.items():
        if isinstance(number, bool) or not re.fullmatch(r"[1-9]\d*", str(number)):
            raise PipelineError(f"{doc_id}: page overrides use PDF page numbers starting at 1")
        number = int(number)
        if number > page_count or number in result:
            raise PipelineError(f"{doc_id}: invalid or duplicate PDF page {number}")
        if not isinstance(override, dict):
            raise PipelineError(f"{doc_id} page {number}: override must be a mapping")
        if override.get("method", rule.get("method", "auto")) not in {"auto", "pypdf", "ocr"}:
            raise PipelineError(f"{doc_id} page {number}: invalid extraction method")
        if "include" in override and not isinstance(override["include"], bool):
            raise PipelineError(f"{doc_id} page {number}: include must be true or false")
        if override.get("include") is False and not str(override.get("reason", "")).strip():
            raise PipelineError(f"{doc_id} page {number}: excluded pages require a reason")
        result[number] = override
    return result


def _replacement(config: dict, override: dict) -> tuple[str | None, str | None]:
    value = override.get("replacement_text_file")
    if value is None:
        return None, None
    if not isinstance(value, str) or not value:
        raise PipelineError("replacement_text_file must be a non-empty path")
    path = _resolve_app_path(config, value)
    if not path.is_file():
        raise PipelineError(f"Replacement text file not found: {path}")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise PipelineError(f"Replacement must be readable UTF-8: {path}") from exc
    return text, hashlib.sha256(raw).hexdigest()


def _rules_identity(config: dict, defaults: dict, rule: dict) -> dict:
    import json
    from importlib.metadata import PackageNotFoundError, version
    from . import segmentation
    dependencies = {}
    for package in ("pypdf", "pytesseract", "pdf2image", "Pillow"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "unavailable"
    replacements = {}
    for number, override in rule.get("pages", {}).items():
        if isinstance(override, dict) and override.get("include", True) and override.get("replacement_text_file"):
            _, replacements[str(number)] = _replacement(config, override)
    # YAML permits integer page keys; JSON persists them as strings. Normalize
    # before both hashing and comparisons so an unchanged corpus reloads cleanly.
    return json.loads(json.dumps({"preprocess_version": PREPROCESS_VERSION,
                                 "code_sha256": {"preprocess": file_sha256(Path(__file__)),
                                                 "segmentation": file_sha256(Path(segmentation.__file__))},
                                 "dependency_versions": dependencies, "defaults": defaults,
                                 "document_rules": rule, "replacement_sha256": replacements}))


def _transform(cm: list, x: float, y: float) -> tuple[float, float]:
    return cm[0] * x + cm[2] * y + cm[4], cm[1] * x + cm[3] * y + cm[5]


def _group_fragments(fragments: list[dict], tolerance: float = 3.0) -> list[dict]:
    groups: list[list[dict]] = []
    for fragment in sorted(fragments, key=lambda row: (row["y0"], row["x0"])):
        # OCR words on the SAME baseline have different top coordinates because
        # capitals, accents and descenders have different bounding-box heights.
        # A fixed 3-pixel top comparison breaks prose into apparent columns.
        center = (fragment["y0"] + fragment["y1"]) / 2
        candidates = []
        for group in groups:
            centers = sorted((word["y0"] + word["y1"]) / 2 for word in group)
            row_center = centers[len(centers) // 2]
            heights = sorted(max(1, word["y1"] - word["y0"]) for word in group)
            allowance = max(tolerance, min(heights[len(heights) // 2],
                                            max(1, fragment["y1"] - fragment["y0"])) * .5)
            if abs(center - row_center) <= allowance:
                candidates.append((abs(center - row_center), group))
        if candidates:
            min(candidates, key=lambda item: item[0])[1].append(fragment)
        else:
            groups.append([fragment])
    lines = []
    for group in groups:
        group.sort(key=lambda word: word["x0"])
        lines.append({"text": " ".join(row["text"] for row in group), "words": group,
                      "x0": min(row["x0"] for row in group), "y0": min(row["y0"] for row in group),
                      "x1": max(row["x1"] for row in group), "y1": max(row["y1"] for row in group)})
    return lines


def _source_ordered_lines(raw: str, fragments: list[dict]) -> tuple[list[dict], bool]:
    """Attach visitor geometry to plain-source text without sorting its words.

    Pypdf visitor coordinates can reflect the end of a text block. Sorting those
    coordinates must NEVER replace the actual extraction order. Whitespace-free
    source/visitor equality allows split visitor runs to map to whole source
    words; otherwise text is retained and the geometry is explicitly unavailable.
    """
    raw = unicodedata.normalize("NFC", raw.replace("\r\n", "\n").replace("\r", "\n"))
    normalized = [{**fragment, "text": unicodedata.normalize("NFC", fragment["text"])}
                  for fragment in fragments]
    source_words = [match.group() for match in re.finditer(r"\S+", raw)]
    aligned = bool(normalized) and "".join(source_words) == "".join(f["text"] for f in normalized)
    if not aligned:
        return [{"text": line, "words": []} for line in raw.split("\n")], False
    positions, cursor = [], 0
    for fragment in normalized:
        positions.append((cursor, cursor + len(fragment["text"]), fragment))
        cursor += len(fragment["text"])
    lines, source_cursor, fragment_cursor = [], 0, 0
    for source_line in raw.split("\n"):
        words = []
        for match in re.finditer(r"\S+", source_line):
            start, end = source_cursor, source_cursor + len(match.group())
            while fragment_cursor < len(positions) and positions[fragment_cursor][1] <= start:
                fragment_cursor += 1
            boxes, candidate = [], fragment_cursor
            while candidate < len(positions) and positions[candidate][0] < end:
                left, right, fragment = positions[candidate]
                width = fragment["x1"] - fragment["x0"]
                boxes.append({"x0": fragment["x0"] + width * (max(start, left) - left) / (right - left),
                              "x1": fragment["x0"] + width * (min(end, right) - left) / (right - left),
                              "y0": fragment["y0"], "y1": fragment["y1"]})
                candidate += 1
            words.append({"text": match.group(), "geometry": "estimated_glyph_width",
                          "x0": min(box["x0"] for box in boxes), "x1": max(box["x1"] for box in boxes),
                          "y0": min(box["y0"] for box in boxes), "y1": max(box["y1"] for box in boxes)})
            source_cursor = end
        line = {"text": source_line, "words": words}
        if words:
            line.update(x0=min(w["x0"] for w in words), x1=max(w["x1"] for w in words),
                        y0=min(w["y0"] for w in words), y1=max(w["y1"] for w in words))
        lines.append(line)
    return lines, True


def extract_pdf_page(page) -> dict:
    """Extract once in plain mode; visitors supply geometry, never a second corpus."""
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    fragments, image_regions = [], []

    def visitor_text(text, cm, tm, font_dict, font_size):
        if not text.strip():
            return
        x, baseline = _transform(cm, float(tm[4]), float(tm[5]))
        size = max(1.0, abs(float(font_size)))
        for line_number, line in enumerate(text.replace("\r", "").split("\n")):
            top = height - baseline - size + line_number * size * 1.2
            for match in re.finditer(r"\S+", line):
                # PDF positioning supplies the origin; estimated glyph widths are
                # retained as such. Ambiguous column intersections require review.
                start = x + match.start() * size * .5
                end = start + len(match.group()) * size * .5
                fragments.append({"text": match.group(), "x0": start, "y0": top,
                                  "x1": end, "y1": top + size, "geometry": "estimated_glyph_width"})

    def visitor_operand(operator, operands, cm, tm):
        if operator != b"Do" or not operands:
            return
        try:
            resources = page.get("/Resources", {}).get_object()
            xobjects = resources.get("/XObject", {}).get_object()
            obj = xobjects.get(operands[0]).get_object()
            if obj.get("/Subtype") == "/Form":
                image_regions.append({"bounds": None, "area_ratio": None, "ambiguous": True})
                return
            if obj.get("/Subtype") != "/Image":
                return
            corners = [_transform(cm, a, b) for a, b in ((0, 0), (1, 0), (0, 1), (1, 1))]
            xs, ys = [p[0] for p in corners], [p[1] for p in corners]
            image_regions.append({"bounds": [min(xs), height - max(ys), max(xs), height - min(ys)],
                                  "area_ratio": abs((max(xs) - min(xs)) * (max(ys) - min(ys))) / max(width * height, 1)})
        except (AttributeError, KeyError, TypeError, ValueError):
            # Nested form XObjects cannot be classified confidently as decoration.
            image_regions.append({"bounds": None, "area_ratio": None, "ambiguous": True})

    raw = page.extract_text(extraction_mode="plain", visitor_text=visitor_text,
                            visitor_operand_before=visitor_operand) or ""
    lines, aligned = _source_ordered_lines(raw, fragments)
    return {"text": raw, "lines": lines, "units": "pdf_points", "width": width,
            "geometry_status": "source_aligned" if aligned else "unavailable",
            "geometry_review_required": bool(fragments) and not aligned,
            "height": height, "image_regions": image_regions, "method": "pypdf"}


def raster_status(extracted: dict) -> str:
    regions = extracted.get("image_regions", [])
    if any(region.get("area_ratio", 0) is not None and region.get("area_ratio", 0) >= .12 for region in regions):
        return "substantive"
    if any(region.get("ambiguous") or region.get("area_ratio") is None or .025 < region.get("area_ratio", 0) < .12 for region in regions):
        return "ambiguous"
    return "decorative" if regions else "none"


def ocr_pdf_page(path: Path, page_number: int, defaults: dict) -> dict:
    """Render one PDF page and retain Tesseract words, confidence, and line IDs."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
        from pytesseract import Output
        if defaults.get("tesseract_cmd"):
            pytesseract.pytesseract.tesseract_cmd = defaults["tesseract_cmd"]
        if "ind" not in pytesseract.get_languages(config=""):
            raise PipelineError("OCR requires the Tesseract 'ind' language data")
        kwargs = {"dpi": defaults["dpi"], "first_page": page_number, "last_page": page_number}
        if defaults.get("poppler_path"):
            kwargs["poppler_path"] = defaults["poppler_path"]
        images = convert_from_path(str(path), **kwargs)
        if len(images) != 1:
            raise PipelineError(f"OCR did not render exactly one page: {page_number}")
        image = images[0]
        data = pytesseract.image_to_data(image, lang="ind", output_type=Output.DICT)
        grouped = defaultdict(list)
        for i, value in enumerate(data["text"]):
            if not value.strip():
                continue
            x, y, w, h = (float(data[key][i]) for key in ("left", "top", "width", "height"))
            identity = tuple(int(data[key][i]) for key in ("block_num", "par_num", "line_num"))
            grouped[identity].append({"text": value, "x0": x, "y0": y, "x1": x + w, "y1": y + h,
                                      "confidence": float(data["conf"][i]), "line_identity": list(identity)})
        lines = []
        for identity, words in grouped.items():
            words.sort(key=lambda row: row["x0"])
            lines.append({"text": " ".join(word["text"] for word in words), "words": words,
                          "x0": min(word["x0"] for word in words), "y0": min(word["y0"] for word in words),
                          "x1": max(word["x1"] for word in words), "y1": max(word["y1"] for word in words),
                          "line_identity": list(identity)})
        lines.sort(key=lambda row: (row["y0"], row["x0"]))
        return {"text": "\n".join(line["text"] for line in lines), "lines": lines,
                "units": "ocr_pixels", "dpi": defaults["dpi"], "width": image.width,
                "height": image.height, "method": "ocr"}
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(f"OCR failed for PDF page {page_number}; check Tesseract ind and Poppler: {exc}") from exc


def _bounds(value, length: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise PipelineError(f"{label} needs {length} coordinates")
    if any(isinstance(item, bool) or not isinstance(item, (float, int)) or not math.isfinite(item) for item in value):
        raise PipelineError(f"{label} coordinates must be finite numbers")
    result = list(map(float, value))
    if (length == 2 and result[0] >= result[1]) or (length == 4 and (result[0] >= result[2] or result[1] >= result[3])):
        raise PipelineError(f"{label} has invalid coordinate order")
    return result


def _inside(word: dict, bounds: list[float]) -> bool:
    x, y = (word["x0"] + word["x1"]) / 2, (word["y0"] + word["y1"]) / 2
    return bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]


def linearize_tables(selected: dict, regions: list[dict]) -> tuple[list[dict], list[str]]:
    """Replace each region once, preserving explicit header-column-value pairs."""
    lines = [{**line, "words": list(line.get("words", []))} for line in selected.get("lines", [])]
    if not isinstance(regions, list):
        raise PipelineError("table_regions must be a list")
    reviews, used_bounds = [], []
    for index, region in enumerate(regions):
        if not isinstance(region, dict):
            raise PipelineError("A table region must be a mapping")
        units = region.get("units")
        if units not in {"pdf_points", "ocr_pixels"}:
            raise PipelineError("Table region units must explicitly be pdf_points or ocr_pixels")
        if units != selected.get("units"):
            raise PipelineError("Table coordinates do not match the selected extraction method")
        if units == "ocr_pixels" and region.get("dpi") != selected.get("dpi"):
            raise PipelineError("OCR pixel table regions require the actual render DPI")
        bounds = _bounds(region.get("bounds"), 4, "table bounds")
        if any(max(bounds[0], other[0]) < min(bounds[2], other[2]) and
               max(bounds[1], other[1]) < min(bounds[3], other[3]) for other in used_bounds):
            raise PipelineError("Overlapping table regions would duplicate source content")
        used_bounds.append(bounds)
        columns = region.get("columns", [])
        if not isinstance(columns, list) or not columns:
            reviews.append(f"table {index + 1}: explicit verified column names and bounds are required")
            continue
        resolved_columns = []
        for column in columns:
            if not isinstance(column, dict) or not isinstance(column.get("name"), str) or not column["name"].strip():
                raise PipelineError("Each table column requires a verified non-empty name")
            cb = _bounds(column.get("bounds"), 2, "column bounds")
            if cb[0] < bounds[0] or cb[1] > bounds[2]:
                raise PipelineError("Table column lies outside the table region")
            if resolved_columns and cb[0] < resolved_columns[-1][1][1]:
                raise PipelineError("Table column bounds must be ordered and non-overlapping")
            resolved_columns.append((column["name"].strip(), cb))
        words = [word for line in lines for word in line.get("words", []) if _inside(word, bounds)]
        if not words:
            reviews.append(f"table {index + 1}: no positioned words inside the region")
            continue
        rows = _group_fragments(words, tolerance=max(2.0, sorted(w["y1"] - w["y0"] for w in words)[len(words) // 2] * .4))
        header_rows = region.get("header_rows", 1)
        if isinstance(header_rows, bool) or not isinstance(header_rows, int) or not 0 <= header_rows < len(rows):
            raise PipelineError("header_rows must leave at least one table data row")
        output = []
        if region.get("merged_header"):
            if not isinstance(region["merged_header"], str):
                raise PipelineError("merged_header must be verified text")
            output.append(region["merged_header"].strip())
        ambiguous = []
        for row in rows[header_rows:]:
            cells = [[] for _ in resolved_columns]
            for word in row["words"]:
                center = (word["x0"] + word["x1"]) / 2
                candidates = [i for i, (_, cb) in enumerate(resolved_columns) if cb[0] <= center <= cb[1]]
                intersections = [i for i, (_, cb) in enumerate(resolved_columns)
                                 if max(0, min(word["x1"], cb[1]) - max(word["x0"], cb[0])) > .2 * max(word["x1"] - word["x0"], 1)]
                if len(candidates) != 1 or len(intersections) > 1:
                    ambiguous.append(f"uncertain column assignment for {word['text']!r}")
                    continue
                if word.get("confidence", 100) < 40:
                    ambiguous.append(f"low-confidence table value {word['text']!r}")
                cells[candidates[0]].append(word["text"])
            if not region.get("allow_empty_cells", False) and any(not cell for cell in cells):
                ambiguous.append("missing/ambiguous table cell")
            output.append(" | ".join(f"{name}: {' '.join(cell)}" for (name, _), cell in zip(resolved_columns, cells)))
        if ambiguous:
            reviews.append(f"table {index + 1}: " + "; ".join(dict.fromkeys(ambiguous)))
            # Keep the original region intact for inspection; never silently flatten.
            continue
        first = next(i for i, line in enumerate(lines) if any(_inside(word, bounds) for word in line.get("words", [])))
        table_line = {"text": "\n".join(output), "words": [], "x0": bounds[0], "y0": bounds[1],
                      "x1": bounds[2], "y1": bounds[3], "table": True, "protect": True}
        remaining = []
        for i, line in enumerate(lines):
            if i == first:
                remaining.append(table_line)
            survivors = [word for word in line.get("words", []) if not _inside(word, bounds)]
            if line.get("words"):
                if survivors:
                    remaining.append({**line, "words": survivors, "text": " ".join(word["text"] for word in survivors)})
            else:
                remaining.append(line)
        lines = remaining
    return lines, reviews


def _table_candidate(lines: list[dict], width: float) -> bool:
    if width <= 0:
        return False
    gaps = []
    # Tesseract can assign two columns to separate blocks; align positioned words
    # across those blocks before checking gaps, rather than missing such tables.
    positioned_words = [word for line in lines for word in line.get("words", [])]
    aligned = _group_fragments(positioned_words) if positioned_words else lines
    for row_number, line in enumerate(aligned):
        words = sorted(line.get("words", []), key=lambda word: word["x0"])
        for left, right in zip(words, words[1:]):
            gap = right["x0"] - left["x1"]
            if gap > width * .12 and (len(words) >= 4 or any(char.isdigit() for char in line["text"])):
                gaps.append((left["x1"], right["x0"], row_number))
    # A table/multiple columns has a shared empty corridor between columns.
    # Two unrelated large gaps in justified prose are not enough evidence.
    for left, right, row in gaps:
        for other_left, other_right, other_row in gaps:
            if other_row != row and min(right, other_right) - max(left, other_left) > width * .06:
                return True
    return False


def _marker(line: str) -> bool:
    return bool(BAB.match(line) or PASAL.match(line) or AYAT.match(line) or LETTER.match(line) or number_enumeration(line))


def clean_page(selected: dict, document_rule: dict, override: dict) -> tuple[str, dict[int, int], list[str]]:
    """Apply explicit removals/corrections before unwrapping positioned lines."""
    source_lines = selected.get("lines", [])
    raw_lines = []
    for line in source_lines:
        pieces = normalize_lines(line["text"])
        raw_lines.extend({**line, "text": piece, "protect": line.get("protect", False)} for piece in pieces)
    if not raw_lines:
        raw_lines = [{"text": line} for line in normalize_lines(selected.get("text", ""))]
    breaks = override.get("paragraph_breaks", [])
    if not isinstance(breaks, list) or any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(raw_lines) for n in breaks):
        raise PipelineError("paragraph_breaks must be valid 1-based line numbers after table/replacement normalization")
    patterns = []
    for key in ("remove_line_patterns", "header_patterns", "footer_patterns"):
        values = override.get(key, document_rule.get(key, []))
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise PipelineError(f"{key} must be a list of explicit regular expressions")
        try:
            patterns.extend(re.compile(value) for value in values)
        except re.error as exc:
            raise PipelineError(f"Invalid cleaning regex: {exc}") from exc
    corrections = override.get("corrections", [])
    if not isinstance(corrections, list):
        raise PipelineError("corrections must be a list of verified old/new pairs")
    notes, retained, pending_break = [], [], False
    for i, line in enumerate(raw_lines, 1):
        pending_break = pending_break or i in breaks
        value = line["text"]
        remove_page_number = override.get("remove_page_numbers", document_rule.get("remove_page_numbers", False))
        edge = i <= 3 or i > len(raw_lines) - 3
        if any(pattern.fullmatch(value) for pattern in patterns) or (remove_page_number and edge and re.fullmatch(r"[-–]?\s*\d+\s*[-–]?", value)):
            notes.append(f"removed line {i}: {value}")
            continue
        retained.append({**line, "text": value, "force_break": pending_break})
        pending_break = False
    # Corrections are exact, explicit replacements and may alter line counts.
    # Their newline-bearing form is rejected so paragraph_breaks stays defined.
    for correction in corrections:
        if not isinstance(correction, dict) or not isinstance(correction.get("old"), str) or not correction["old"] or not isinstance(correction.get("new"), str):
            raise PipelineError("A correction requires non-empty old and string new")
        if "\n" in correction["old"] or "\n" in correction["new"]:
            raise PipelineError("Use replacement_text_file for corrections that alter lines")
        count = 0
        for line in retained:
            count += line["text"].count(correction["old"])
            line["text"] = line["text"].replace(correction["old"], correction["new"])
        if count == 0:
            raise PipelineError(f"Verified correction does not match selected text: {correction['old']!r}")
        notes.append(f"correction {correction['old']!r} -> {correction['new']!r}, occurrences={count}")
    origins = [line["x0"] for line in retained if "x0" in line and line["text"] and not line.get("table")]
    baseline = min(origins) if origins else 0
    levels = sorted({round(x - baseline, 1) for x in origins if x - baseline >= 8})
    paragraphs, current, current_indent = [], [], 0
    previous = None
    for line in retained:
        value = line["text"]
        if not value:
            if current:
                paragraphs.append((" ".join(current), current_indent))
                current = []
            previous = line
            continue
        geometric_indent = sum(1 for delta in levels if line.get("x0", baseline) - baseline >= delta - 3) if "x0" in line else 0
        indent = int(line.get("indent_level", geometric_indent))
        gap_boundary = False
        if previous and all(key in previous and key in line for key in ("y0", "y1")):
            line_height = max(1, previous["y1"] - previous["y0"])
            gap_boundary = line["y0"] - previous["y1"] > .75 * line_height
        ocr_paragraph_boundary = bool(previous and previous.get("line_identity") and line.get("line_identity") and
                                      previous["line_identity"][:2] != line["line_identity"][:2])
        heading = bool(BAB.match(value) or PASAL.match(value))
        boundary = (line.get("force_break") or line.get("protect") or _marker(value) or gap_boundary or
                    ocr_paragraph_boundary or (previous and previous.get("protect")))
        if boundary and current:
            paragraphs.append((" ".join(current), current_indent))
            current = []
        if not current:
            current_indent = indent
        current.append(value)
        previous = {**line, "protect": line.get("protect", False) or heading}
    if current:
        paragraphs.append((" ".join(current), current_indent))
    text, hints = "", {}
    for paragraph, indent in paragraphs:
        if text:
            text += "\n\n"
        hints[len(text)] = indent
        text += paragraph
    return text, hints, notes


def process_page(config: dict, path: Path, page, page_number: int, defaults: dict,
                 document_rule: dict, override: dict) -> dict:
    """Choose exactly one page source; exclusions and full replacement take priority."""
    if override.get("include", True) is False:
        return {"text": "", "indent_hints": {}, "method": "excluded", "included": False,
                "log": [f"excluded: {override['reason']}"], "review_required": []}
    replacement, replacement_hash = _replacement(config, override)
    reviews, notes = [], []
    if replacement is not None:
        selected = {"text": replacement, "lines": [], "method": "replacement"}
        notes.append(f"replacement_sha256={replacement_hash}")
        if override.get("table_regions"):
            notes.append("full replacement already represents the page; table reconstruction skipped")
    else:
        requested = override.get("method", document_rule.get("method", defaults.get("method", "auto")))
        digital = extract_pdf_page(page)
        status = raster_status(digital)
        char_count = len(re.sub(r"\s", "", digital["text"]))
        candidate = char_count < defaults["ocr_min_nonspace_chars"] or status == "substantive"
        if requested == "ocr" or (requested == "auto" and candidate):
            ocr = ocr_pdf_page(path, page_number, defaults)
            if requested == "ocr" or status == "substantive" or len(re.sub(r"\s", "", ocr["text"])) > char_count:
                selected = ocr
            else:
                selected = digital
            notes.append(f"OCR candidate: digital_nonspace={char_count}, raster={status}")
        else:
            selected = digital
        if requested == "auto" and status == "ambiguous":
            reviews.append("raster region is ambiguous; select method or verified full replacement explicitly")
        notes.append(f"selected {selected['method']}; requested={requested}; raster={status}")
        if selected.get("geometry_review_required"):
            reviews.append("pypdf visitor geometry does not align with complete source text; use OCR or verified replacement")
        regions = override.get("table_regions", [])
        if regions:
            selected["lines"], table_reviews = linearize_tables(selected, regions)
            selected["text"] = "\n".join(line["text"] for line in selected["lines"])
            reviews.extend(table_reviews)
            notes.append(f"table regions reconstructed once: {len(regions)}")
        elif _table_candidate(selected.get("lines", []), selected.get("width", 1)):
            reviews.append("possible table/multiple columns need verified region rules or full replacement")
    text, hints, cleaning_notes = clean_page(selected, {**defaults, **document_rule}, override)
    notes.extend(cleaning_notes)
    first_line = next((line for line in normalize_lines(text) if line), "")
    start_paragraph = (1 in override.get("paragraph_breaks", []) or _marker(first_line) or
                       bool(selected.get("lines") and selected["lines"][0].get("protect")))
    return {"text": text, "indent_hints": hints, "method": selected["method"], "included": True,
            "log": notes, "review_required": reviews, "start_paragraph": start_paragraph,
            "review_details": ({"units": selected.get("units"), "dpi": selected.get("dpi"),
                                "width": selected.get("width"), "height": selected.get("height"),
                                "geometry_status": selected.get("geometry_status"),
                                "selected_text": selected.get("text", ""),
                                "lines": selected.get("lines", [])} if reviews else None)}


def finalize_document(doc_id: str, source_hash: str, page_results: list[dict], rules_identity: dict) -> dict:
    text, pages, hints, log, reviews, review_details = "", [], {}, [], [], []
    for page_number, result in enumerate(page_results, 1):
        log.append({"page_number": page_number, "method": result["method"], "notes": result.get("log", [])})
        reviews.extend({"page_number": page_number, "reason": reason} for reason in result.get("review_required", []))
        if result.get("review_details"):
            review_details.append({"page_number": page_number, **result["review_details"]})
        if not result.get("included", True):
            continue
        page_text = result["text"]
        if text and page_text:
            # Physical page turns are not automatic paragraph/sentence breaks.
            # A protected legal/table start or explicit line-1 override is one.
            first_line = next((line for line in normalize_lines(page_text) if line), "")
            text += "\n\n" if result.get("start_paragraph", _marker(first_line)) else "\n"
        start = len(text)
        text += page_text
        pages.append({"page_number": page_number, "char_start": start, "char_end": len(text), "method": result["method"]})
        hints.update({start + int(offset): level for offset, level in result.get("indent_hints", {}).items()})
    if not text.strip():
        raise PipelineError(f"{doc_id}: preprocessing removed or failed to read all substantive text")
    doc = {"doc_id": doc_id, "source_sha256": source_hash, "text": text,
           "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "pages": pages,
           "cleaning_rules": rules_identity, "rules_sha256": canonical_hash(rules_identity),
           "log": log, "review_required": reviews, "review_details": review_details,
           "_paragraph_indent_hints": hints}
    return annotate_document(doc)


def _data_subdir(config: dict, name: str) -> Path:
    root = Path(config["_app_root"]).resolve()
    path = (root / config.get("data_dir", "data") / name).resolve()
    if not path.is_relative_to(root):
        raise PipelineError("All clean outputs must stay inside the application root")
    return path


def _clean_dir(config: dict) -> Path:
    return _data_subdir(config, "clean")


def prepare_corpus(config: dict) -> list[dict]:
    """Prepare every source, then persist only a complete valid corpus."""
    entries, cleaning = _inputs(config)
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PipelineError("PDF extraction requires pypdf 4.x") from exc
    documents, readers = [], []
    try:
        readers = [PdfReader(str(entry["path"])) for entry in entries]
    except Exception as exc:
        raise PipelineError(f"Cannot open all source PDFs: {exc}") from exc
    total_pages = sum(len(reader.pages) for reader in readers)
    expected = cleaning.get("_expected_total_pages")
    if expected is not None and expected != total_pages:
        raise PipelineError(f"Expected {expected} source PDF pages, found {total_pages}; verify the corpus")
    for entry, reader in zip(entries, readers):
        source_hash = file_sha256(entry["path"])
        defaults, rule = _rules_for(config, cleaning, entry, source_hash)
        identity = _rules_identity(config, defaults, rule)
        try:
            overrides = _page_rules(rule, len(reader.pages), entry["doc_id"])
            results = [process_page(config, entry["path"], page, number, defaults, rule, overrides.get(number, {}))
                       for number, page in enumerate(reader.pages, 1)]
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(f"{entry['doc_id']}: PDF processing failed: {exc}") from exc
        documents.append(finalize_document(entry["doc_id"], source_hash, results, identity))
    unresolved = [(doc["doc_id"], review) for doc in documents for review in doc["review_required"]]
    if unresolved:
        review_dir = _data_subdir(config, "review")
        review_dir.mkdir(parents=True, exist_ok=True)
        for doc in documents:
            write_json(review_dir / f"{doc['doc_id']}.json", {**doc, "draft": True})
        issues = [{"doc_id": doc_id, **review} for doc_id, review in unresolved]
        page_count = len({(row["doc_id"], row["page_number"]) for row in issues})
        write_json(review_dir / "summary.json", {"status": "review_required",
                   "page_count": page_count, "issue_count": len(issues), "issues": issues})
        detail = "; ".join(f"{row['doc_id']} page {row['page_number']}: {row['reason']}" for row in issues[:8])
        remaining = f"; plus {len(issues) - 8} more issue(s) in the summary" if len(issues) > 8 else ""
        raise PipelineError(f"review_required: {len(issues)} issue(s) on {page_count} page(s); canonical corpus was not saved. "
                            f"Inspect {review_dir / 'summary.json'} and document drafts; resolve cleaning overrides. {detail}{remaining}")
    target = _clean_dir(config)
    target.mkdir(parents=True, exist_ok=True)
    for doc in documents:
        write_json(target / f"{doc['doc_id']}.json", doc)
    # A successful rebuild supersedes the previous failure report. Keep old
    # drafts for inspection, but never leave their status as the latest result.
    review_dir = _data_subdir(config, "review")
    review_dir.mkdir(parents=True, exist_ok=True)
    write_json(review_dir / "summary.json", {"status": "ready", "page_count": 0,
               "issue_count": 0, "issues": [], "document_count": len(documents)})
    return documents


def load_clean_corpus(config: dict) -> list[dict]:
    """Refuse stale sources, cleaning rules, replacements, spans, or annotations."""
    import json
    entries, cleaning = _inputs(config)
    try:
        from pypdf import PdfReader
        page_counts = {entry["doc_id"]: len(PdfReader(str(entry["path"])).pages) for entry in entries}
    except Exception as exc:
        raise PipelineError(f"Cannot validate source PDF page counts: {exc}") from exc
    expected = cleaning.get("_expected_total_pages")
    if expected is not None and sum(page_counts.values()) != expected:
        raise PipelineError(f"Expected {expected} source PDF pages, found {sum(page_counts.values())}; verify the corpus")
    documents = []
    for entry in entries:
        source_hash = file_sha256(entry["path"])
        defaults, rule = _rules_for(config, cleaning, entry, source_hash)
        identity = _rules_identity(config, defaults, rule)
        overrides = _page_rules(rule, page_counts[entry["doc_id"]], entry["doc_id"])
        path = _clean_dir(config) / f"{entry['doc_id']}.json"
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PipelineError(f"Clean corpus missing/invalid for {entry['doc_id']}; run prepare: {exc}") from exc
        if not isinstance(doc, dict):
            raise PipelineError(f"{entry['doc_id']}: clean document must be a JSON object; rebuild with prepare")
        if doc.get("doc_id") != entry["doc_id"] or doc.get("source_sha256") != source_hash:
            raise PipelineError(f"{entry['doc_id']}: saved source provenance changed; rebuild with prepare")
        if doc.get("rules_sha256") != canonical_hash(identity) or doc.get("cleaning_rules") != identity:
            raise PipelineError(f"{entry['doc_id']}: cleaning/replacement rules changed; rebuild with prepare")
        text = doc.get("text")
        if not isinstance(text, str) or not text.strip() or doc.get("text_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
            raise PipelineError(f"{entry['doc_id']}: canonical text/hash invalid; rebuild with prepare")
        if not isinstance(doc.get("review_required"), list):
            raise PipelineError(f"{entry['doc_id']}: review status missing/invalid; rebuild with prepare")
        if doc.get("draft") or doc["review_required"]:
            raise PipelineError(f"{entry['doc_id']}: unresolved review_required cannot be used for research")
        for key in ("pages", "log", "paragraphs", "sentences", "structures"):
            if not isinstance(doc.get(key), list) or any(not isinstance(row, dict) for row in doc[key]):
                raise PipelineError(f"{entry['doc_id']}: {key} must be an array of objects; rebuild with prepare")
        for key in ("paragraphs", "sentences", "structures"):
            for row in doc[key]:
                a, b = row.get("char_start"), row.get("char_end")
                if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int) or not isinstance(b, int) or not 0 <= a < b <= len(text):
                    raise PipelineError(f"{entry['doc_id']}: invalid final {key} spans; rebuild with prepare")
                if key == "paragraphs" and (isinstance(row.get("indent_level"), bool) or not isinstance(row.get("indent_level"), int) or row["indent_level"] < 0):
                    raise PipelineError(f"{entry['doc_id']}: invalid paragraph geometry; rebuild with prepare")
        pages = doc["pages"]
        expected_pages = [number for number in range(1, page_counts[entry["doc_id"]] + 1)
                          if overrides.get(number, {}).get("include", True)]
        if [row.get("page_number") for row in pages] != expected_pages:
            raise PipelineError(f"{entry['doc_id']}: page mapping differs from reviewed source inclusions; rebuild with prepare")
        if [row.get("page_number") for row in doc.get("log", [])] != list(range(1, page_counts[entry["doc_id"]] + 1)):
            raise PipelineError(f"{entry['doc_id']}: extraction log does not cover every source page; rebuild with prepare")
        previous_end, previous_number = 0, 0
        for page in pages:
            if not (isinstance(page.get("page_number"), int) and not isinstance(page["page_number"], bool) and page["page_number"] > previous_number and
                    isinstance(page.get("char_start"), int) and not isinstance(page["char_start"], bool) and
                    isinstance(page.get("char_end"), int) and not isinstance(page["char_end"], bool) and
                    previous_end <= page["char_start"] <= page["char_end"] <= len(text) and
                    page.get("method") in {"pypdf", "ocr", "replacement"}):
                raise PipelineError(f"{entry['doc_id']}: invalid final page spans; rebuild with prepare")
            if text[previous_end:page["char_start"]].strip():
                raise PipelineError(f"{entry['doc_id']}: page map omits substantive text; rebuild with prepare")
            previous_end, previous_number = page["char_end"], page["page_number"]
        if not pages or text[previous_end:].strip():
            raise PipelineError(f"{entry['doc_id']}: incomplete final page map; rebuild with prepare")
        regenerated = annotate_document({**doc, "paragraphs": [dict(p) for p in doc.get("paragraphs", [])]})
        for key in ("paragraphs", "sentences", "structures"):
            if doc.get(key) != regenerated[key]:
                raise PipelineError(f"{entry['doc_id']}: final {key} annotations invalid; rebuild with prepare")
        documents.append(doc)
    return documents
