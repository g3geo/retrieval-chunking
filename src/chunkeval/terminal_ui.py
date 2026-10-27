"""Readable terminal panels for retrieval; only Python's standard library."""
from __future__ import annotations

import os
import shutil
import sys
import unicodedata


CONDITION_NAMES = {
    "C1": "Fixed 256 token",
    "C2": "Fixed 512 token",
    "C3": "Paragraf / struktur",
    "C4": "Semantic percentile 95",
    "C5": "Semantic percentile 80",
}


def _safe_text(value) -> str:
    # Corpus text is data, never a terminal control sequence. In particular,
    # PDF/OCR escape characters must not change colors or execute OSC commands.
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    return "".join(char for char in text if char == "\n" or
                   unicodedata.category(char) not in {"Cc", "Cf", "Cs"})


def _columns(text: str) -> int:
    return sum(0 if unicodedata.combining(char) else
               2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in text)


def _split_line(text: str, width: int) -> list[str]:
    """Wrap by display columns, including long tokens and full-width Unicode."""
    if not text:
        return [""]
    result = []
    while _columns(text) > width:
        columns, end = 0, 0
        for index, char in enumerate(text):
            size = _columns(char)
            if columns + size > width:
                break
            columns += size
            end = index + 1
        end = max(end, 1)
        space = text.rfind(" ", 0, end + 1)
        cut = space if space > 0 else end
        result.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        result.append(text)
    return result


def _color_supported(stream) -> bool:
    if os.environ.get("NO_COLOR") is not None or os.environ.get("TERM") == "dumb":
        return False
    if not getattr(stream, "isatty", lambda: False)():
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel.GetStdHandle.restype = wintypes.HANDLE
        kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        handle = kernel.GetStdHandle(wintypes.DWORD(-12 if stream is sys.stderr else -11))
        mode = wintypes.DWORD()
        return bool(kernel.GetConsoleMode(handle, ctypes.byref(mode)) and
                    kernel.SetConsoleMode(handle, mode.value | 0x0004))
    except (AttributeError, OSError, TypeError, ValueError):
        return False


class TerminalUI:
    def __init__(self, *, no_color: bool = False, stream=None, width: int | None = None,
                 interactive: bool = False):
        self.stream = sys.stdout if stream is None else stream
        self.fixed_width = width
        self.interactive = interactive
        self.color = not no_color and _color_supported(self.stream)
        try:
            "╭╮╰╯─│".encode(getattr(self.stream, "encoding", None) or "utf-8")
            self.border = ("╭", "╮", "╰", "╯", "─", "│")
        except (UnicodeError, LookupError):
            self.border = ("+", "+", "+", "+", "-", "|")

    @property
    def width(self) -> int:
        columns = self.fixed_width if self.fixed_width is not None else shutil.get_terminal_size((100, 24)).columns
        return max(12, min(columns - 1, 108))

    def style(self, text: str, kind: str = "accent") -> str:
        codes = {"accent": "1;36", "title": "1;37", "muted": "90", "good": "1;32", "error": "1;31"}
        return f"\033[{codes[kind]}m{text}\033[0m" if self.color else text

    def write(self, text: str = "") -> None:
        # Legacy Windows encodings retain readable panels even if source text
        # contains characters that the console cannot represent.
        encoding = getattr(self.stream, "encoding", None)
        if encoding:
            text = text.encode(encoding, errors="replace").decode(encoding)
        print(text, file=self.stream)

    def text(self, value) -> str:
        value = _safe_text(value)
        encoding = getattr(self.stream, "encoding", None)
        return value.encode(encoding, errors="replace").decode(encoding) if encoding else value

    def panel(self, title: str, lines: list[str], *, kind: str = "accent") -> None:
        width = self.width
        inner = width - 4
        tl, tr, bl, br, horizontal, vertical = self.border
        self.write(self.style(tl + horizontal * (width - 2) + tr, kind))
        for line, heading in [(title, True), *[(value, False) for value in lines]]:
            for piece in self.text(line).split("\n"):
                for wrapped in _split_line(piece, inner):
                    padding = " " * max(0, inner - _columns(wrapped))
                    body = self.style(wrapped, "title" if heading else kind) if heading else wrapped
                    self.write(self.style(vertical, kind) + " " + body + padding + " " + self.style(vertical, kind))
            if heading:
                self.write(self.style(vertical, kind) + " " + horizontal * inner + " " + self.style(vertical, kind))
        self.write(self.style(bl + horizontal * (width - 2) + br, kind))

    def banner(self, conditions, k: int, full_text: bool) -> None:
        self.write()
        self.panel("RETRIEVAL  /  PENCARIAN DOKUMEN", [
            f"Kondisi: {', '.join(conditions)}   |   Hasil per kondisi: {k}",
            "Tampilan: " + ("teks lengkap" if full_text else "cuplikan teks"),
            "Ketik pertanyaan lalu Enter. /bantuan untuk pilihan; /keluar untuk selesai.",
        ])

    def prompt(self) -> str:
        return self.style("\nPertanyaan > ")

    def message(self, text: str) -> None:
        self.write(self.style(_safe_text(text), "muted"))

    def error(self, text: str) -> None:
        self.write()
        self.panel("ERROR", [_safe_text(text)], kind="error")

    def help(self, available) -> None:
        self.write()
        self.panel("PILIHAN PENCARIAN", [
            "/mode C3       Cari pada satu kondisi yang sudah dimuat",
            "/mode semua    Cari pada seluruh kondisi yang sudah dimuat",
            "/k 5           Tampilkan 5 hasil per kondisi (1-10)",
            "/teks lengkap  Tampilkan isi chunk lengkap",
            "/teks ringkas  Tampilkan cuplikan teks",
            "/lihat C3 1    Buka teks lengkap hasil nomor 1 dari pencarian terakhir",
            "/bantuan       Tampilkan pilihan ini",
            "/keluar        Akhiri sesi (Ctrl+C juga bisa)",
            "Kondisi yang dimuat: " + ", ".join(available),
        ])

    def query(self, text: str, elapsed: float | None = None) -> None:
        self.write()
        lines = [_safe_text(text)]
        if elapsed is not None:
            lines.extend(["", f"Waktu pencarian: {elapsed:.2f} detik"])
        self.panel("PERTANYAAN", lines)

    def summary(self, records: list[dict]) -> None:
        if len(records) < 2:
            return
        lines = []
        for record in records:
            hits = record["results"]
            if hits:
                best = hits[0]
                pages = ", ".join(map(str, best["page_numbers"]))
                lines.append(f"{record['condition']}  |  {len(hits)} hasil  |  Skor teratas {float(best['score']):.6f}  |  {best['doc_id']} hal. {pages}")
            else:
                lines.append(f"{record['condition']}  |  Tidak ada hasil")
        self.write()
        self.panel("RINGKASAN C1-C5", lines)

    def result(self, record: dict, full_text: bool) -> None:
        condition = record["condition"]
        self.write()
        self.panel(f"{condition}  /  {CONDITION_NAMES.get(condition, condition)}", [
            f"{len(record['results'])} hasil, diurutkan menurut skor kemiripan",
        ])
        if not record["results"]:
            self.message("Tidak ada hasil.")
        for hit in record["results"]:
            self.hit(condition, hit, full_text)

    def hit(self, condition: str, hit: dict, full_text: bool) -> None:
        labels = []
        for ref in hit["structure_refs"]:
            label = " / ".join(f"{key.upper() if key == 'bab' else key.title()} {ref[key]}"
                               for key in ("bab", "pasal", "ayat", "butir") if ref.get(key) is not None)
            if label and label not in labels:
                labels.append(label)
        text = _safe_text(hit["text"])
        if not full_text and len(text) > 320:
            text = text[:320].rstrip() + "..."
        self.write()
        lines = [
            f"Dokumen: {hit['doc_id']}   |   Halaman: {', '.join(map(str, hit['page_numbers']))}",
            f"Token: {hit['n_tokens_with_prefix']}   |   Struktur: {'; '.join(labels) or '-'}",
            "", text,
        ]
        if not full_text and len(_safe_text(hit["text"])) > 320:
            detail_hint = (f"Baca lengkap: /lihat {condition} {hit['rank']}" if self.interactive else
                           "Gunakan --full-text untuk membaca teks lengkap.")
            lines.extend(["", detail_hint])
        self.panel(f"#{hit['rank']}  |  {condition}  |  Skor {float(hit['score']):.6f}", lines,
                   kind="good" if hit["rank"] == 1 else "accent")
