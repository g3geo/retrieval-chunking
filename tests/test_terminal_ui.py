"""Terminal width, literal corpus text, and interaction without model encoding."""
import io
from types import SimpleNamespace

from chunkeval import cli
from chunkeval.terminal_ui import TerminalUI, _columns


def hit(text="Isi sumber yang sudah tersimpan."):
    return {"rank": 1, "score": 0.123456789, "doc_id": "D1", "page_numbers": [7, 8],
            "n_tokens_with_prefix": 24, "structure_refs": [{"pasal": "5"}], "text": text}


def test_panels_fit_narrow_terminal_and_wide_unicode_without_changing_source():
    stream = io.StringIO()
    ui = TerminalUI(stream=stream, width=44, no_color=True)
    source = "漢字 " * 20 + "\n" + "DokumenTanpaSpasi" * 12
    result = hit(source)
    ui.hit("C3", result, True)
    lines = stream.getvalue().splitlines()
    assert all(_columns(line) <= ui.width for line in lines)
    assert result["text"] == source and result["score"] == 0.123456789
    assert "0.123457" in stream.getvalue()
    assert "7, 8" in stream.getvalue()


def test_redirected_output_contains_no_escape_sequences_from_corpus_or_ui():
    stream = io.StringIO()
    ui = TerminalUI(stream=stream)
    ui.hit("C3", hit("Isi \033[31mberwarna\033[0m dan \033]0;judul\007 dokumen."), True)
    output = stream.getvalue()
    assert "\033" not in output and "\007" not in output
    assert "Isi" in output and "dokumen." in output


def test_terminal_controls_and_result_detail_do_not_encode_or_search_again(monkeypatch, capsys):
    queries = iter(["pertama", "/lihat c3 1", "/k 0", "/mode C9", "/k 3", "/mode c1",
                    "/teks lengkap", "kedua", "/mode semua", "/bantuan", "ketiga", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(queries))
    calls = []
    indexes = {condition: SimpleNamespace(manifest_hash="synthetic") for condition in cli.CONDITIONS}

    def search(query, args, conditions, loaded, encoder):
        assert loaded is indexes
        calls.append((query, conditions, args.k, args.full_text))
        return [{"condition": condition, "results": [hit("Detail tersimpan tanpa encode ulang.")]}
                for condition in conditions]

    monkeypatch.setattr(cli, "_search_query", search)
    args = cli.parser().parse_args(["search", "--no-color"])
    cli._interactive_search(args, cli.CONDITIONS, indexes, object())
    assert calls == [("pertama", cli.CONDITIONS, 10, False),
                     ("kedua", ("C1",), 3, True),
                     ("ketiga", cli.CONDITIONS, 3, True)]
    output = capsys.readouterr()
    assert "Detail tersimpan tanpa encode ulang." in output.out
    assert "PILIHAN PENCARIAN" in output.out
    assert "antara 1 dan 10" in output.err and "sudah dimuat" in output.err


def test_legacy_console_encoding_uses_ascii_borders_and_readable_fallback():
    class LegacyStream(io.StringIO):
        encoding = "cp1252"

    stream = LegacyStream()
    ui = TerminalUI(stream=stream, width=60, no_color=True)
    ui.hit("C3", hit("Isi café dan 漢."), True)
    output = stream.getvalue()
    assert "+" in output and "|" in output
    assert "café" in output
    output.encode("cp1252")
