"""Offset-preserving sentence and legal hierarchy annotations.

All offsets index the final Python Unicode string. This module is deliberately
shared by paragraph chunking and semantic chunking.
"""
from __future__ import annotations

import re

BAB = re.compile(r"^BAB\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
PASAL = re.compile(r"^Pasal\s+(\d+[A-Za-z]?)\s*$", re.IGNORECASE)
AYAT = re.compile(r"^\((\d+)\)\s+")
LETTER = re.compile(r"^([a-z])\.\s+", re.IGNORECASE)
NUMBER = re.compile(r"^(\d{1,3})[.)]\s+")
NUMBERED_DATE = re.compile(r"^\d{1,2}\.\s+(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+\d{4}\b", re.I)
ABBREVIATIONS = {
    "no", "nom", "rp", "dr", "prof", "ir", "dll", "dsb", "dst", "yth",
    "kab", "kec", "kel", "jl", "sdr", "tn", "ny", "pt", "cv", "h", "hj",
    "s.d", "a.n", "u.p", "s.h", "m.h", "s.pd", "m.si", "s.e", "s.sos",
}
URL = re.compile(r"(?:https?://|www\.)\S+|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")


def number_enumeration(line: str):
    """An Indonesian calendar date cannot become a numbered legal item."""
    return None if NUMBERED_DATE.match(line) else NUMBER.match(line)


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split Indonesian prose without damaging abbreviations or numeric values.

    Blank paragraphs/headings are boundaries; ordinary wrapped lines are not.
    Sentence spans omit surrounding whitespace, which chunk construction keeps
    through source slices between adjacent sentences.
    """
    if not text.strip():
        return []
    protected = [(m.start(), m.end()) for m in URL.finditer(text)]
    boundaries = {0, len(text)}
    for m in re.finditer(r"\n[ \t]*\n+", text):
        boundaries.add(m.start())
        boundaries.add(m.end())
    for m in re.finditer(r"(?m)^(?:BAB\s+(?:[IVXLCDM]+|\d+)\b[^\n]*|Pasal\s+\d+[A-Za-z]?\s*)$", text, re.I):
        boundaries.update((m.start(), m.end()))
    for match in re.finditer(r"[.!?]+[\"'”’)]*", text):
        pos, end = match.start(), match.end()
        # A URL's final prose full stop is allowed; its internal dots are not.
        containing = next(((a, b) for a, b in protected if a <= pos < b), None)
        if containing and not (end == containing[1] and text[pos] in ".!?" and pos > containing[0]):
            continue
        if end < len(text) and not text[end].isspace():
            continue
        if text[pos] == ".":
            if pos and pos + 1 < len(text) and text[pos - 1].isdigit() and text[pos + 1].isdigit():
                continue
            before = re.search(r"([\w.]+)$", text[:pos], re.UNICODE)
            token = before.group(1).lower().strip(".") if before else ""
            if token in ABBREVIATIONS or (len(token) == 1 and token.isalpha()):
                continue
            line_prefix = text[text.rfind("\n", 0, pos) + 1:pos].strip()
            if re.fullmatch(r"\d{1,3}|[a-z]", line_prefix, re.I):
                continue
        boundaries.add(end)
    ordered = sorted(boundaries)
    result = []
    for a, b in zip(ordered, ordered[1:]):
        a, b = _trim_span(text, a, b)
        if a < b:
            result.append((a, b))
    return result


def _markers(text: str, indent_by_start: dict[int, int]) -> list[dict]:
    markers = []
    has_pasal = False
    letter_active = False
    current_letter_indent = 0
    for match in re.finditer(r"(?m)^[^\n]+", text):
        line = match.group(0).strip()
        start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
        indent = indent_by_start.get(start, 0)
        kind, value, depth = None, None, None
        if m := BAB.match(line):
            kind, value, depth = "bab", m.group(1).upper(), 0
            has_pasal = letter_active = False
        elif m := PASAL.match(line):
            kind, value, depth = "pasal", m.group(1), 1
            has_pasal, letter_active = True, False
        elif m := AYAT.match(line):
            kind, value, depth = "ayat", m.group(1), 2
            letter_active = False
        elif m := LETTER.match(line):
            kind, value, depth = "butir", m.group(1).lower() + ".", 3 + indent
            letter_active, current_letter_indent = True, indent
        elif m := number_enumeration(line):
            # A line-leading list marker is not a decimal, date, or prose reference.
            # Under a letter item, geometry decides whether the numeric item is
            # nested. Extraction whitespace is never used as that evidence.
            nested = letter_active and indent > current_letter_indent
            kind, value, depth = "butir", m.group(1) + ".", (4 + indent if nested else 3 + indent)
            if not nested:
                letter_active = False
        if kind:
            markers.append({"start": start, "kind": kind, "value": value, "depth": depth})
    return markers


def annotate_document(doc: dict) -> dict:
    """Recompute every annotation from final text, retaining geometric indent hints."""
    text = doc["text"]
    doc_id = doc["doc_id"]
    hints = dict(doc.pop("_paragraph_indent_hints", {}))
    for paragraph in doc.get("paragraphs", []):
        hints.setdefault(paragraph["char_start"], paragraph.get("indent_level", 0))
    # JSON roundtrips may stringify private hint keys.
    hints = {int(start): int(level) for start, level in hints.items()}
    markers = _markers(text, hints)
    boundaries = {0, len(text)}
    boundaries.update(marker["start"] for marker in markers)
    for m in re.finditer(r"\n[ \t]*\n+", text):
        boundaries.add(m.end())
    starts = sorted(boundaries)
    paragraphs = []
    for a, b in zip(starts, starts[1:]):
        a, b = _trim_span(text, a, b)
        if a < b:
            paragraphs.append({"paragraph_id": f"{doc_id}:P{len(paragraphs) + 1:04d}",
                               "char_start": a, "char_end": b,
                               "indent_level": hints.get(a, 0)})
    structures = []
    stack = []
    for marker in markers:
        while stack and stack[-1][0] >= marker["depth"]:
            _, prior = stack.pop()
            prior["char_end"] = marker["start"]
        labels = {key: None for key in ("bab", "pasal", "ayat", "butir")}
        for _, ancestor in stack:
            for key in labels:
                if ancestor[key] is not None:
                    labels[key] = ancestor[key]
        if marker["kind"] == "butir" and labels["butir"]:
            labels["butir"] += " / " + marker["value"]
        else:
            labels[marker["kind"]] = marker["value"]
        unit = {"structure_id": f"{doc_id}:U{len(structures) + 1:04d}",
                "char_start": marker["start"], "char_end": len(text), **labels}
        structures.append(unit)
        stack.append((marker["depth"], unit))
    if not structures and text.strip():
        structures = [{"structure_id": f"{doc_id}:U{i + 1:04d}",
                       "char_start": p["char_start"], "char_end": p["char_end"],
                       "bab": None, "pasal": None, "ayat": None, "butir": None}
                      for i, p in enumerate(paragraphs)]
    elif structures and structures[0]["char_start"] > 0:
        structures.insert(0, {"structure_id": f"{doc_id}:U0000", "char_start": 0,
                              "char_end": structures[0]["char_start"],
                              "bab": None, "pasal": None, "ayat": None, "butir": None})
    doc["paragraphs"] = paragraphs
    doc["structures"] = structures
    doc["sentences"] = [{"sentence_id": f"{doc_id}:S{i + 1:05d}",
                         "char_start": a, "char_end": b, "text": text[a:b]}
                        for i, (a, b) in enumerate(sentence_spans(text))]
    return doc
