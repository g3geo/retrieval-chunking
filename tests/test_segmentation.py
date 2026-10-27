"""Synthetic legal fixtures; these are not research queries or corrections."""
from chunkeval.segmentation import annotate_document, sentence_spans


def document(text, hints=None):
    return annotate_document({"doc_id": "TEST", "text": text, "_paragraph_indent_hints": hints or {}})


def test_indonesian_abbreviations_amounts_urls_and_enumeration():
    text = ("No. 20 Tahun 2017 berlaku. Bantuan Rp. 1.000,50 diberikan kepada Dr. Andi. "
            "Lihat https://contoh.id/aturan.\n\na. Memenuhi syarat sesuai Pasal 5.\n\n"
            "1. Jumlah 2.5 dapat diperiksa.")
    slices = [text[a:b] for a, b in sentence_spans(text)]
    assert slices == ["No. 20 Tahun 2017 berlaku.",
                      "Bantuan Rp. 1.000,50 diberikan kepada Dr. Andi.",
                      "Lihat https://contoh.id/aturan.",
                      "a. Memenuhi syarat sesuai Pasal 5.",
                      "1. Jumlah 2.5 dapat diperiksa."]


def test_nested_structure_spans_parent_labels_and_prose_reference():
    text = ("BAB II\n\nPasal 5\n\n(1) Hak penerima mengikuti ketentuan.\n\n"
            "a. Persyaratan penerima.\n\n1. Dokumen identitas.\n\n2. Dokumen keluarga.\n\n"
            "b. Sesuai Pasal 9, penerima memenuhi syarat.\n\nPasal 6\n\n(1) Hak berikutnya.")
    starts = {line: text.index(line) for line in ["a. Persyaratan", "1. Dokumen", "2. Dokumen", "b. Sesuai"]}
    doc = document(text, {starts["a. Persyaratan"]: 0, starts["1. Dokumen"]: 1,
                          starts["2. Dokumen"]: 1, starts["b. Sesuai"]: 0})
    rows = doc["structures"]
    assert [row["pasal"] for row in rows if text[row["char_start"]:row["char_start"] + 5] == "Pasal"] == ["5", "6"]
    numeric = next(row for row in rows if row["butir"] == "a. / 1.")
    assert (numeric["bab"], numeric["pasal"], numeric["ayat"]) == ("II", "5", "1")
    assert text[numeric["char_start"]:numeric["char_end"]].startswith("1. Dokumen identitas.")
    assert numeric["char_end"] == text.index("2. Dokumen")
    pasal5 = next(row for row in rows if text[row["char_start"]:].startswith("Pasal 5"))
    assert pasal5["char_end"] == text.index("Pasal 6")
    assert [p["indent_level"] for p in doc["paragraphs"] if text[p["char_start"]:].startswith("1. Dokumen")] == [1]
    assert all(s["text"] == text[s["char_start"]:s["char_end"]] for s in doc["sentences"])


def test_whitespace_not_evidence_of_nested_geometry():
    text = "Pasal 5\n\na. Persyaratan.\n\n    1. Dokumen."
    doc = document(text)
    numeric = next(row for row in doc["structures"] if row["butir"] == "1.")
    assert numeric["butir"] == "1."


def test_reference_dates_and_decimal_are_not_legal_markers():
    text = "Ketentuan sesuai Pasal 5 berlaku pada 1. Januari 2026.\n\nNilai 12.5 tidak berubah.\n\n1. Januari 2026 merupakan tanggal yang dicantumkan."
    doc = document(text)
    assert all(row["pasal"] is None and row["butir"] is None for row in doc["structures"])


def test_unicode_repeated_text_and_reannotation_stability():
    text = "Pasal 1\n\nPenerima café berhak.\n\nPasal 2\n\nPenerima café berhak."
    doc = document(text)
    repeated = [row for row in doc["sentences"] if row["text"] == "Penerima café berhak."]
    assert len(repeated) == 2
    assert repeated[0]["char_start"] != repeated[1]["char_start"]
    assert annotate_document({**doc}) == doc
    assert sentence_spans("") == []


def test_abbreviations_degree_and_initials():
    text = "Dr. A. Ahmad, S.H. hadir bersama PT. Contoh. Selanjutnya dibayar Rp. 500.000."
    assert [text[a:b] for a, b in sentence_spans(text)] == [
        "Dr. A. Ahmad, S.H. hadir bersama PT. Contoh.", "Selanjutnya dibayar Rp. 500.000."]
