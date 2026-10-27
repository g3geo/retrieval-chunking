"""Synthetic, researcher-independent PDF/OCR geometry and provenance fixtures."""
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from chunkeval.config import PipelineError, file_sha256, write_json
from chunkeval import preprocess as pp


def word(text, x, y, width=30, confidence=99):
    return {"text": text, "x0": x, "y0": y, "x1": x + width, "y1": y + 10, "confidence": confidence}


def selected(rows, method="ocr", units="ocr_pixels"):
    lines = []
    for words in rows:
        lines.append({"text": " ".join(item["text"] for item in words), "words": words,
                      "x0": min(item["x0"] for item in words), "y0": min(item["y0"] for item in words),
                      "x1": max(item["x1"] for item in words), "y1": max(item["y1"] for item in words)})
    return {"text": "\n".join(line["text"] for line in lines), "lines": lines,
            "method": method, "units": units, "dpi": 300, "width": 200, "height": 300, "image_regions": []}


def fixture_config(tmp_path, rule=None):
    from pypdf import PdfWriter
    app = tmp_path / "app"
    (app / "configs").mkdir(parents=True)
    source = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=300)
    with source.open("wb") as stream:
        writer.write(stream)
    config = {"_app_root": app, "corpus": "configs/corpus.yaml", "cleaning": "configs/cleaning.yaml", "data_dir": "data"}
    corpus = {"source_root": "../..", "documents": [{"doc_id": "TEST", "path": "source.pdf"}]}
    cleaning = {"defaults": {"dpi": 300, "ocr_min_nonspace_chars": 40}, "documents": {}}
    if rule is not None:
        cleaning["documents"]["TEST"] = {"source_sha256": file_sha256(source), **rule}
    (app / "configs" / "corpus.yaml").write_text(yaml.safe_dump(corpus), encoding="utf-8")
    (app / "configs" / "cleaning.yaml").write_text(yaml.safe_dump(cleaning), encoding="utf-8")
    return config, source, cleaning


def test_final_unicode_page_mapping_after_changed_length_replacement():
    pages = [{"text": "Pasal 1\n\ncafé", "method": "replacement", "indent_hints": {}},
             {"text": "Pasal 2\n\ncafé", "method": "ocr", "indent_hints": {}}]
    doc = pp.finalize_document("TEST", "f" * 64, pages, {})
    assert doc["pages"][1]["char_start"] == len(pages[0]["text"]) + 2
    assert doc["text"][doc["pages"][1]["char_start"]:] == pages[1]["text"]
    assert all(s["text"] == doc["text"][s["char_start"]:s["char_end"]] for s in doc["sentences"])
    assert doc["text_sha256"] == hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
    changed = pp.finalize_document("TEST", "f" * 64, [{**pages[0], "text": "Pasal 1\n\ncafé panjang"}, pages[1]], {})
    assert changed["text_sha256"] != doc["text_sha256"]
    assert changed["pages"][1]["char_start"] > doc["pages"][1]["char_start"]
    assert changed["sentences"][-1]["char_start"] > doc["sentences"][-1]["char_start"]


def test_same_sentence_and_paragraph_can_continue_across_pdf_page():
    pages = [{"text": "Kalimat café", "method": "pypdf", "indent_hints": {}},
             {"text": "yang berlanjut di halaman berikutnya.", "method": "pypdf", "indent_hints": {}}]
    doc = pp.finalize_document("TEST", "f" * 64, pages, {})
    assert doc["text"] == "Kalimat café\nyang berlanjut di halaman berikutnya."
    assert doc["pages"][1]["char_start"] == len(pages[0]["text"]) + 1
    assert len(doc["sentences"]) == len(doc["paragraphs"]) == 1
    forced = pp.finalize_document("TEST", "f" * 64, [pages[0], {**pages[1], "start_paragraph": True}], {})
    assert len(forced["paragraphs"]) == 2


def test_full_replacement_priority_and_no_duplicate_ocr(tmp_path, monkeypatch):
    config, source, _ = fixture_config(tmp_path)
    replacement = Path(config["_app_root"]) / "reviewed.txt"
    replacement.write_text("Pasal 5\r\nBantuan cafe\u0301 Rp. 1.000.", encoding="utf-8")
    monkeypatch.setattr(pp, "extract_pdf_page", lambda page: pytest.fail("replacement must bypass extraction"))
    monkeypatch.setattr(pp, "ocr_pdf_page", lambda *args: pytest.fail("replacement must bypass OCR"))
    result = pp.process_page(config, source, None, 1, {"dpi": 300, "ocr_min_nonspace_chars": 40}, {},
                             {"replacement_text_file": "reviewed.txt", "method": "ocr", "paragraph_breaks": [2]})
    assert result["text"] == "Pasal 5\n\nBantuan café Rp. 1.000."
    assert result["method"] == "replacement"
    assert result["text"].count("Rp.") == 1
    assert "replacement_sha256=" in result["log"][0]


def test_digital_header_with_substantive_raster_form_selects_ocr(tmp_path, monkeypatch):
    config, source, _ = fixture_config(tmp_path)
    digital = selected([[word("Formulir", 0, 0), word("Permohonan", 45, 0)]], method="pypdf", units="pdf_points")
    digital["image_regions"] = [{"area_ratio": .7, "bounds": [0, 20, 200, 280]}]
    ocr = {"text": "Formulir Permohonan\n\nNama penerima bantuan harus diisi lengkap.", "lines": [], "method": "ocr", "width": 200}
    monkeypatch.setattr(pp, "extract_pdf_page", lambda page: digital)
    monkeypatch.setattr(pp, "ocr_pdf_page", lambda *args: ocr)
    result = pp.process_page(config, source, None, 1, {"dpi": 300, "ocr_min_nonspace_chars": 40}, {}, {})
    assert result["method"] == "ocr"
    assert "Nama penerima" in result["text"]
    assert result["text"].count("Formulir") == 1


def test_logo_does_not_force_ocr_and_ambiguous_raster_needs_override(tmp_path, monkeypatch):
    config, source, _ = fixture_config(tmp_path)
    digital = {"text": "Isi digital substantif yang panjang dan lengkap tetap dipertahankan.", "lines": [],
               "method": "pypdf", "width": 200, "image_regions": [{"area_ratio": .01}]}
    monkeypatch.setattr(pp, "extract_pdf_page", lambda page: digital)
    monkeypatch.setattr(pp, "ocr_pdf_page", lambda *args: pytest.fail("logo should not force OCR"))
    defaults = {"dpi": 300, "ocr_min_nonspace_chars": 40}
    assert pp.process_page(config, source, None, 1, defaults, {}, {})["method"] == "pypdf"
    digital["image_regions"] = [{"area_ratio": .08}]
    assert pp.process_page(config, source, None, 1, defaults, {}, {})["review_required"]
    assert not pp.process_page(config, source, None, 1, defaults, {}, {"method": "pypdf"})["review_required"]


def test_short_title_not_discarded_as_empty(tmp_path, monkeypatch):
    config, source, _ = fixture_config(tmp_path)
    digital = {"text": "LAMPIRAN", "lines": [], "method": "pypdf", "width": 200, "image_regions": []}
    monkeypatch.setattr(pp, "extract_pdf_page", lambda page: digital)
    monkeypatch.setattr(pp, "ocr_pdf_page", lambda *args: {"text": "", "lines": [], "method": "ocr", "width": 200})
    result = pp.process_page(config, source, None, 1, {"dpi": 300, "ocr_min_nonspace_chars": 40}, {}, {})
    assert result["text"] == "LAMPIRAN"


def table_fixture():
    extraction = selected([
        [word("Bantuan Tunai", 20, 0, 150)],
        [word("Nama", 10, 20), word("Nominal", 110, 20)],
        [word("Ana", 10, 40), word("Rp.", 100, 40, 20), word("1.000", 130, 40)],
        [word("Budi", 10, 60), word("Rp.", 100, 60, 20), word("2.000", 130, 60)],
        [word("Catatan", 0, 100)],
    ])
    region = {"bounds": [0, 0, 200, 80], "units": "ocr_pixels", "dpi": 300, "header_rows": 2,
              "merged_header": "Bantuan Tunai", "columns": [{"name": "Nama", "bounds": [0, 80]},
                                                                {"name": "Nominal", "bounds": [90, 200]}]}
    return extraction, region


def test_table_header_pairs_amounts_and_source_once():
    extraction, region = table_fixture()
    lines, reviews = pp.linearize_tables(extraction, [region])
    assert not reviews
    text, _, _ = pp.clean_page({**extraction, "lines": lines}, {}, {})
    assert "Nama: Ana | Nominal: Rp. 1.000" in text
    assert "Nama: Budi | Nominal: Rp. 2.000" in text
    assert text.count("1.000") == text.count("2.000") == text.count("Bantuan Tunai") == 1
    assert text.count("Catatan") == 1


@pytest.mark.parametrize("change", [{"units": "unknown"}, {"units": "pdf_points"}, {"dpi": 150}])
def test_table_units_and_dpi_required(change):
    extraction, region = table_fixture()
    with pytest.raises(PipelineError):
        pp.linearize_tables(extraction, [{**region, **change}])


def test_ambiguous_table_retains_original_for_review():
    extraction, region = table_fixture()
    extraction["lines"][2]["words"][-1]["confidence"] = 10
    lines, reviews = pp.linearize_tables(extraction, [region])
    assert reviews and "low-confidence" in reviews[0]
    assert lines == extraction["lines"]
    assert all(not line.get("table") for line in lines)


def test_overlapping_table_regions_rejected():
    extraction, region = table_fixture()
    with pytest.raises(PipelineError, match="Overlapping"):
        pp.linearize_tables(extraction, [region, region])


def test_geometry_and_explicit_breaks_survive_unwrap():
    data = selected([[word("Pasal 5", 0, 0)], [word("a. Syarat", 0, 20)],
                     [word("lanjutan", 0, 32)], [word("1. Dokumen", 20, 44)],
                     [word("penjelasan", 20, 56)], [word("paragraf baru", 0, 68)]], method="pypdf", units="pdf_points")
    text, hints, _ = pp.clean_page(data, {}, {"paragraph_breaks": [6]})
    assert "a. Syarat lanjutan" in text
    assert "1. Dokumen penjelasan" in text
    assert hints[text.index("1. Dokumen")] > hints[text.index("a. Syarat")]
    assert "\n\nparagraf baru" in text
    doc = pp.finalize_document("TEST", "f" * 64, [{"text": text, "method": "pypdf", "indent_hints": hints}], {})
    assert any(row["butir"] == "a. / 1." for row in doc["structures"])


def test_exclusion_priority_invalid_page_and_hash(tmp_path, monkeypatch):
    config, source, cleaning = fixture_config(tmp_path, {"pages": {1: {"include": False, "reason": "reviewed cover"}}})
    monkeypatch.setattr(pp, "extract_pdf_page", lambda page: pytest.fail("excluded page must bypass extraction"))
    result = pp.process_page(config, source, None, 1, {}, {}, {"include": False, "reason": "reviewed cover", "replacement_text_file": "missing.txt"})
    assert not result["included"]
    with pytest.raises(PipelineError):
        pp._page_rules({"pages": {0: {}}}, 2, "TEST")
    with pytest.raises(PipelineError):
        pp._page_rules({"pages": {3: {}}}, 2, "TEST")
    with pytest.raises(PipelineError):
        pp._page_rules({"pages": {1: {"include": False}}}, 2, "TEST")
    with pytest.raises(PipelineError, match="source_sha256"):
        pp._rules_for(config, cleaning, {"doc_id": "TEST"}, "f" * 64)


def test_invalid_break_and_empty_research_document():
    with pytest.raises(PipelineError, match="paragraph_breaks"):
        pp.clean_page({"text": "isi", "lines": []}, {}, {"paragraph_breaks": [2]})
    with pytest.raises(PipelineError, match="substantive"):
        pp.finalize_document("TEST", "f" * 64, [{"text": "", "method": "ocr"}], {})


def test_explicit_boundary_survives_removing_its_header_line():
    text, _, _ = pp.clean_page({"text": "Paragraf pertama\nHEADER\nParagraf berikutnya", "lines": []},
                               {"header_patterns": ["HEADER"]}, {"paragraph_breaks": [2]})
    assert text == "Paragraf pertama\n\nParagraf berikutnya"


def save_fixture_corpus(config, source, cleaning):
    defaults, rule = pp._rules_for(config, cleaning, {"doc_id": "TEST"}, file_sha256(source))
    identity = pp._rules_identity(config, defaults, rule)
    doc = pp.finalize_document("TEST", file_sha256(source), [{"text": "Pasal 1\n\nIsi café substantif.", "method": "replacement"}], identity)
    path = Path(config["_app_root"]) / "data" / "clean" / "TEST.json"
    path.parent.mkdir(parents=True)
    write_json(path, doc)
    return doc, path


def test_clean_load_integer_page_keys_and_stale_source(tmp_path):
    config, source, cleaning = fixture_config(tmp_path, {"pages": {1: {"method": "ocr"}}})
    doc, _ = save_fixture_corpus(config, source, cleaning)
    assert pp.load_clean_corpus(config) == [doc]
    source.write_bytes(source.read_bytes() + b"\nchanged source provenance\n")
    with pytest.raises(PipelineError, match="source_sha256"):
        pp.load_clean_corpus(config)


def test_replacement_hash_or_rule_change_requires_rebuild(tmp_path):
    config, source, cleaning = fixture_config(tmp_path, {"pages": {1: {"replacement_text_file": "reviewed.txt"}}})
    replacement = Path(config["_app_root"]) / "reviewed.txt"
    replacement.write_text("reviewed original", encoding="utf-8")
    save_fixture_corpus(config, source, cleaning)
    assert pp.load_clean_corpus(config)
    replacement.write_text("reviewed altered", encoding="utf-8")
    with pytest.raises(PipelineError, match="rules changed"):
        pp.load_clean_corpus(config)


def test_saved_annotation_and_text_corruption_rejected(tmp_path):
    config, source, cleaning = fixture_config(tmp_path)
    doc, path = save_fixture_corpus(config, source, cleaning)
    doc["sentences"][0]["char_start"] += 1
    write_json(path, doc)
    with pytest.raises(PipelineError, match="annotations invalid"):
        pp.load_clean_corpus(config)


def test_prepare_review_failure_saves_no_partial_corpus(tmp_path, monkeypatch):
    import pypdf
    config, source, cleaning = fixture_config(tmp_path)
    monkeypatch.setattr(pypdf, "PdfReader", lambda path: type("Reader", (), {"pages": [object()]})())
    monkeypatch.setattr(pp, "process_page", lambda *args: {"text": "Table needing verified input.", "method": "ocr", "review_required": ["uncertain columns"]})
    with pytest.raises(PipelineError, match="review_required"):
        pp.prepare_corpus(config)
    assert not (Path(config["_app_root"]) / "data" / "clean").exists()
    draft = json.loads((Path(config["_app_root"]) / "data" / "review" / "TEST.json").read_text(encoding="utf-8"))
    assert draft["draft"] and draft["review_required"]


def test_output_escape_rejected(tmp_path):
    with pytest.raises(PipelineError, match="application root"):
        pp._clean_dir({"_app_root": tmp_path, "data_dir": "../outside"})


def test_successful_prepare_replaces_previous_review_failure_report(tmp_path, monkeypatch):
    import pypdf
    config, _, _ = fixture_config(tmp_path)
    review_dir = Path(config["_app_root"]) / "data" / "review"
    review_dir.mkdir(parents=True)
    write_json(review_dir / "summary.json", {"status": "review_required", "issue_count": 1,
                                             "page_count": 1, "issues": [{"reason": "old issue"}]})
    monkeypatch.setattr(pypdf, "PdfReader", lambda path: type("Reader", (), {"pages": [object()]})())
    monkeypatch.setattr(pp, "process_page", lambda *args: {
        "text": "Pasal 1\n\nIsi yang sudah diperiksa.", "method": "replacement", "review_required": []})
    documents = pp.prepare_corpus(config)
    report = json.loads((review_dir / "summary.json").read_text(encoding="utf-8"))
    assert report == {"status": "ready", "page_count": 0, "issue_count": 0,
                      "issues": [], "document_count": 1}
    saved = json.loads((Path(config["_app_root"]) / "data" / "clean" / "TEST.json").read_text(encoding="utf-8"))
    assert saved == documents[0] and saved["review_required"] == []


def test_default_removal_rules_and_ocr_dpi_alias(tmp_path, monkeypatch):
    config, source, cleaning = fixture_config(tmp_path)
    supplied = {**cleaning, "defaults": {"ocr_dpi": 250, "ocr_min_nonspace_chars": 40,
                                        "header_patterns": ["HEADER"], "remove_page_numbers": True}}
    defaults, rule = pp._rules_for(config, supplied, {"doc_id": "TEST"}, file_sha256(source))
    assert defaults["dpi"] == 250
    monkeypatch.setattr(pp, "extract_pdf_page", lambda page: {"text": "HEADER\nIsi substantif panjang yang harus tetap disimpan dengan lengkap.\n5",
                                                              "lines": [], "width": 200, "method": "pypdf", "image_regions": []})
    result = pp.process_page(config, source, None, 1, defaults, rule, {"method": "pypdf"})
    assert "HEADER" not in result["text"] and not result["text"].endswith("5")


def test_expected_source_page_count_validated_before_ocr(tmp_path, monkeypatch):
    config, source, _ = fixture_config(tmp_path)
    path = Path(config["_app_root"]) / "configs" / "corpus.yaml"
    corpus = yaml.safe_load(path.read_text(encoding="utf-8"))
    corpus["expected_total_pages"] = 2
    path.write_text(yaml.safe_dump(corpus), encoding="utf-8")
    monkeypatch.setattr(pp, "process_page", lambda *args: pytest.fail("wrong source page counts must fail before extraction"))
    with pytest.raises(PipelineError, match="Expected 2 source PDF pages, found 1"):
        pp.prepare_corpus(config)


def test_saved_source_page_mapping_inclusions_validated(tmp_path):
    config, source, cleaning = fixture_config(tmp_path)
    doc, path = save_fixture_corpus(config, source, cleaning)
    doc["pages"][0]["page_number"] = 2
    write_json(path, doc)
    with pytest.raises(PipelineError, match="mapping differs"):
        pp.load_clean_corpus(config)


def test_form_xobject_is_ambiguous_instead_of_silently_ignored():
    class Object(dict):
        def get_object(self):
            return self

    class Page(Object):
        mediabox = type("Box", (), {"width": 200, "height": 300})()

        def extract_text(self, **kwargs):
            kwargs["visitor_operand_before"](b"Do", ["/Form1"], [1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0])
            return "A digital title above an unknown nested form."

    page = Page({"/Resources": Object({"/XObject": Object({"/Form1": Object({"/Subtype": "/Form"})})})})
    extraction = pp.extract_pdf_page(page)
    assert pp.raster_status(extraction) == "ambiguous"


def test_ocr_table_detection_aligns_separate_column_blocks():
    lines = selected([[word("Ana", 0, 0)], [word("Rp. 1.000", 120, 0)],
                      [word("Budi", 0, 20)], [word("Rp. 2.000", 120, 20)]])["lines"]
    assert pp._table_candidate(lines, 200)


def test_ocr_paragraph_identity_preserved_before_unwrap():
    data = selected([[word("Paragraf pertama", 0, 0)], [word("lanjutan", 0, 12)],
                     [word("Paragraf kedua", 0, 24)]])
    for line, identity in zip(data["lines"], [[1, 1, 1], [1, 1, 2], [1, 2, 1]]):
        line["line_identity"] = identity
    text, _, _ = pp.clean_page(data, {}, {})
    assert text == "Paragraf pertama lanjutan\n\nParagraf kedua"


def test_ocr_word_heights_do_not_turn_prose_into_false_columns():
    # On each real line, uppercase/lowercase words have different top pixels.
    # Grouping only equal tops used to omit alternating words and create gaps.
    rows = []
    for y in (0, 40):
        rows.append([word("Satu", 0, y, 30), word("dua", 40, y + 4, 30),
                     word("Tiga", 80, y, 30), word("empat", 120, y + 4, 30)])
    assert not pp._table_candidate(selected(rows)["lines"], 200)
    grouped = pp._group_fragments([item for row in rows for item in row])
    assert len(grouped) == 2
    assert all(line["text"] == "Satu dua Tiga empat" for line in grouped)


def test_two_unrelated_large_gaps_do_not_prove_a_shared_table_column():
    data = selected([[word("Awal", 0, 0, 20), word("lanjut", 70, 0, 30)],
                     [word("Teks", 80, 40, 20), word("akhir", 160, 40, 30)]])
    assert not pp._table_candidate(data["lines"], 200)


def test_plain_source_order_and_complete_words_survive_misleading_coordinates():
    raw = "Pasal 5\nIsi café berulang.\nIsi café berulang."
    # Actual stream order is correct; y-coordinates misleadingly put body first.
    fragments = [word("Pa", 0, 100, 10), word("sal", 10, 100, 15), word("5", 35, 100, 10)]
    for y in (0, 20):
        fragments.extend([word("Isi", 0, y), word("café", 40, y), word("berulang.", 80, y)])
    lines, aligned = pp._source_ordered_lines(raw, fragments)
    assert aligned
    assert "\n".join(line["text"] for line in lines) == raw
    assert [item["text"] for item in lines[0]["words"]] == ["Pasal", "5"]
    cleaned, _, _ = pp.clean_page({"text": raw, "lines": lines}, {}, {})
    assert cleaned.startswith("Pasal 5\n\nIsi café berulang.")
    assert cleaned.count("Isi café berulang.") == 2


def test_missing_visitor_words_preserve_complete_source_and_reject_geometry():
    raw = "Isi lengkap café tidak boleh kehilangan kata."
    lines, aligned = pp._source_ordered_lines(raw, [word("Isi", 0, 0)])
    assert not aligned
    assert "\n".join(line["text"] for line in lines) == raw
    assert all(not line["words"] for line in lines)


def test_unaligned_digital_geometry_requires_review_and_saves_selected_details(tmp_path, monkeypatch):
    config, source, _ = fixture_config(tmp_path)
    raw = "Isi lengkap substantif yang tidak boleh kehilangan kata dalam hasil ekstraksi."
    monkeypatch.setattr(pp, "extract_pdf_page", lambda page: {
        "text": raw, "lines": [{"text": raw, "words": []}], "method": "pypdf",
        "units": "pdf_points", "width": 200, "height": 300,
        "image_regions": [], "geometry_review_required": True, "geometry_status": "unavailable"})
    result = pp.process_page(config, source, None, 1,
                             {"dpi": 300, "ocr_min_nonspace_chars": 40}, {}, {})
    assert result["text"] == raw
    assert any("geometry does not align" in reason for reason in result["review_required"])
    assert result["review_details"]["selected_text"] == raw
    assert result["review_details"]["units"] == "pdf_points"
