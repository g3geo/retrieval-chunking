# Retrieval dan Chunking C1–C5

Implementasi Python lokal untuk eksperimen perbandingan strategi chunking (C1 sampai C5)
pada dokumen kebijakan bantuan sosial berbahasa Indonesia, dengan model embedding
`intfloat/multilingual-e5-small` dan indeks FAISS. Rancangan eksperimen mengikuti proposal skripsi.

**Status (26 Oktober 2026):** korpus bersih (8 dokumen, 171 halaman), lima kelompok chunk
(C1: 238, C2: 107, C3: 3.108, C4: 235, C5: 735), dan lima indeks FAISS sudah dibangun.
`check_environment.py`, `validate.py`, dan 122 kasus uji lulus (tes memakai pengaturan
lingkungan pada bagian Setup Windows). `data/queries.json` masih berisi array kosong,
sehingga `retrieve.py` belum dijalankan dan belum ada hasil retrieval. Lock dependency
berasal dari resolver environment Windows Python 3.12.

## File yang dijalankan per tahap

| File | Pekerjaan | Prasyarat | Output |
|---|---|---|---|
| `prepare.py` | Ekstraksi PDF/OCR, cleaning, segmentasi final | PDF sumber dan aturan cleaning | `data/clean/*.json` |
| `chunk.py` | Membuat C1–C5; cache vektor unit semantic bersama | Korpus bersih valid | `data/chunks/`, `data/cache/semantic/` |
| `index.py` | Embedding passage dan lima indeks FAISS | Chunk valid | `data/indexes/C1` sampai `C5` |
| `retrieve.py` | Batch kueri penelitian pada lima kondisi | Lima indeks konsisten dan kueri peneliti | `data/results/results.jsonl` |
| `search.py` | Pencarian interaktif, satu kueri, atau membaca hasil tersimpan | Indeks, kecuali mode membaca JSONL | Output terminal |
| `validate.py` | Memeriksa span/token/coverage/overlap/metadata/indeks | Korpus, chunk, dan indeks | Hasil pemeriksaan terminal |

File tersebut merupakan entry point kecil. Implementasi tiap tahap berada di modul
berbeda pada `src/chunkeval/`: `preprocess.py`, `segmentation.py`,
`tokenization.py`, `chunking.py`, `embedding.py`, `indexing.py`, `retrieval.py`.
`cli.py` menghubungkan command tersebut. Tidak ada eksekusi tahap lain secara
otomatis ketika Anda memilih satu file. Tahap yang prasyaratnya belum tersedia
akan berhenti dengan pesan error.

## Setup Windows

Jalankan PowerShell dari folder aplikasi:

```powershell
Set-Location W:\Projek\ASKRIPSI\Retrieval_Chunking
```

Environment `.venv` sudah dibuat pada komputer ini. Untuk environment baru,
gunakan Python 3.12 agar cocok dengan lock Windows yang disertakan:

```powershell
python -m venv .venv
$env:TEMP = Join-Path (Get-Location) 'data\tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
$env:PIP_CACHE_DIR = Join-Path (Get-Location) 'data\cache\pip'
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

Anda dapat memakai executable `.venv` langsung tanpa mengaktifkan environment.
Jika ingin command `python` yang lebih pendek:

```powershell
.\.venv\Scripts\Activate.ps1
```

Jika kebijakan PowerShell menolak aktivasi, gunakan
`.\.venv\Scripts\python.exe` untuk seluruh command; tidak perlu mengubah
kebijakan global komputer.

Periksa environment sebelum membangun artefak:

```powershell
.\.venv\Scripts\python.exe check_environment.py
.\.venv\Scripts\python.exe -m pytest
```

`check_environment.py` menjalankan `pip check`, pemeriksaan import, validasi pin
revision, dan pembandingan adapter terhadap helper SemanticChunker yang dipin.
Pemeriksaan ini tidak memproses PDF atau mengunduh/menjalankan model E5.
Tes memakai PDF/teks/tokenizer/vektor sintetis, termasuk FAISS nyata, tanpa kueri
penelitian atau model E5 unduhan. Tes integrasi membangun korpus kecil, kelima
kondisi, indeks, retrieval, dan memuat ulang indeks. Hasil terakhir: 122 kasus uji lulus (26 Oktober 2026) dengan pengaturan lingkungan berikut.

Pada komputer ini, tes berhenti dengan galat `Fatal Python error: Aborted` pada tes pertama
yang memakai FAISS kecuali dua variabel lingkungan di bawah diatur. Penyebab pastinya belum
diselidiki; dugaannya konflik pustaka OpenMP antara FAISS dan PyTorch di Windows.

```powershell
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:OMP_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m pytest
```

`KMP_DUPLICATE_LIB_OK=TRUE` melewati pemeriksaan pustaka OpenMP ganda, sehingga kebenaran
pencarian FAISS diperiksa tersendiri oleh uji pencarian-diri pada `validate.py`. Belum
dipastikan apakah pengaturan yang sama diperlukan untuk `chunk.py`, `index.py`, dan `search.py`.

OCR membutuhkan Tesseract dengan bahasa `ind` dan Poppler. Path yang ditemukan
pada komputer ini disimpan di `configs/cleaning.yaml`; sesuaikan jika dipindah:

```powershell
& 'C:\Program Files\Tesseract-OCR\tesseract.exe' --list-langs
```

`ind` harus tercantum. Poppler membutuhkan `pdfinfo.exe` dan `pdftoppm.exe`.
Render OCR default 300 DPI. Semua output aplikasi, termasuk cache model E5,
berada dalam folder aplikasi. Pemakaian pertama tokenizer/model membutuhkan
koneksi untuk mengunduh checkpoint resmi; pemrosesan dan embedding berjalan
lokal dan tidak memerlukan API generatif.

## Menjalankan tiap tahap

Dengan environment aktif:

```powershell
python prepare.py
python chunk.py
python index.py
python validate.py
python search.py
python search.py --query "teks kueri Anda" --condition C3
python search.py --query "teks kueri Anda" --all --k 10 --full-text
python retrieve.py --queries data/queries.json
```

`python search.py` membuka sesi pencarian terminal: ketik pertanyaan pada
`Pertanyaan >` lalu tekan Enter. Ringkasan C1–C5 tampil sebelum panel hasil,
dengan skor teratas dan sumber tiap kondisi. Hasil top-10 ditampilkan untuk
C1–C5 dalam panel terpisah, termasuk skor, dokumen, halaman, struktur, dan
cuplikan teks. Teks mengikuti lebar terminal dan warna aktif bila didukung.
Skor di layar dibulatkan ke enam desimal; nilai asli pada hasil retrieval dan
file JSONL tetap dipertahankan. Sesudah hasil tampil, masukkan
pertanyaan berikutnya. Model dan indeks dimuat sekali untuk satu sesi.
Ketik `/keluar` atau `/exit`, atau tekan Ctrl+C, untuk selesai.
Pertanyaan kosong diminta ulang; input yang terlalu panjang menampilkan error
dan Anda dapat mengetik pertanyaan lain. Mode ini tidak memerlukan file kueri.

Untuk menampilkan teks chunk lengkap, jalankan `python search.py --full-text`.
Untuk satu kondisi saja, misalnya C3, jalankan `python search.py --condition C3`.
Opsi `--k 5` membatasi hasil tiap kondisi menjadi lima. Pencarian langsung
dengan `--query` dan pembacaan `--results` tetap memakai C3 bila kondisi tidak
ditentukan. Hasil pencarian interaktif ditampilkan di terminal; command batch
`retrieve.py` di bawah digunakan bila ingin menyimpan hasil JSONL.

Pilihan dalam sesi terminal:

| Perintah | Fungsi |
|---|---|
| `/bantuan` | Menampilkan daftar pilihan |
| `/mode C3` | Memilih satu kondisi yang sudah dimuat |
| `/mode semua` | Memilih semua kondisi yang sudah dimuat |
| `/k 5` | Mengubah jumlah hasil berikutnya, 1–10 per kondisi |
| `/teks lengkap` | Menampilkan teks chunk lengkap pada pencarian berikutnya |
| `/teks ringkas` | Menampilkan cuplikan pada pencarian berikutnya |
| `/lihat C3 1` | Membuka teks lengkap hasil nomor 1 dari pencarian terakhir |
| `/keluar` | Mengakhiri sesi |

`/lihat` membaca hasil yang sudah ada di memori, tanpa embedding ulang.
Jika sesi dimulai dengan `--condition C3`, hanya C3 yang dimuat dan tersedia
untuk `/mode`. Jalankan tanpa `--condition` bila ingin berpindah antara C1–C5.
Warna dapat dimatikan dengan `--no-color` atau environment variable `NO_COLOR`.
Output yang diarahkan ke file otomatis memakai teks tanpa kode warna.
Tampilan menggunakan standard library Python; tidak ada dependency baru.

Command `retrieve.py` memakai berapa pun kueri peneliti yang tersedia. File
`data/queries.json` saat ini berisi array kosong; isi sendiri dengan objek `qid`
unik dan `text` tidak kosong. Contoh format, dengan placeholder yang perlu
diganti oleh peneliti:

```json
[
  {"qid": "Q01", "text": "<kueri penelitian yang Anda tetapkan>"}
]
```

Batch membutuhkan sedikitnya 10 chunk di setiap kondisi untuk top-10 penuh.
`search.py` menampilkan jumlah aktual jika indeks lebih kecil. Input >512 token
lengkap setelah prefix dan special tokens menghasilkan error, tanpa truncation.

Untuk membaca hasil JSONL tanpa meng-embed ulang atau memuat model:

```powershell
python search.py --results data/results/results.jsonl --qid Q01 --all --full-text
python search.py --results data/results/results.jsonl --condition C3 --k 5
```

Setiap file menerima `--config PATH` dan `--help`. Entry point terpusat juga
tersedia apabila diperlukan:

```powershell
python -m chunkeval.cli prepare --config configs/config.yaml
python -m chunkeval.cli search --query "teks kueri Anda" --all
```

## Korpus dan aturan cleaning

`configs/corpus.yaml` mencatat delapan PDF D1–D8. `source_root: ../..` dihitung
relatif terhadap file YAML di folder `configs`, sehingga menunjuk ke
`W:/Projek/ASKRIPSI`. PDF asli hanya dibaca. Total halaman yang diharapkan 172.

`configs/cleaning.yaml` mempin SHA256 masing-masing PDF. D2/D3 memakai OCR.
D1 memakai deteksi kandidat raster pada mode `auto`. Konfigurasi berisi 27 replacement UTF-8 yang dibandingkan
dengan tampilan halaman PDF sumber, serta pengecualian daftar isi D1 halaman 4.
Replacement berada di `configs/replacements/`: D1 halaman 7, 8, 10, 11, 13–15,
17, 18, 21, 22, 25, 28–35; D2 halaman 26–28; D4 halaman 3; D5 halaman 6;
D6 halaman 1 dan 4. Nomor tersebut adalah nomor halaman PDF, bukan nomor cetak.
Tabel diratakan berdasarkan header/kolom/baris; `[kosong]` menyatakan sel isian
formulir yang memang kosong, bukan nol. Tahun/angka yang tercetak tidak ditebak
atau diganti agar mengikuti tahun judul dokumen. PDF sumber tetap utuh.
Aturan penghapusan lain di luar halaman tersebut masih perlu pemeriksaan bila
dibutuhkan. Halaman pertama regulasi tidak dibuang otomatis.

Preprocessing memilih satu sumber per halaman. Logo kecil tidak memaksa OCR;
lapisan teks pendek atau isi raster substantif menjadi kandidat OCR. Geometri
disimpan sebelum unwrap untuk tabel dan indent enumerasi. Halaman/tabel ambigu
menjadi `review_required`; draft dan log tersedia di `data/review/`, dan korpus
tersebut tidak diterbitkan sebagai korpus bersih valid.
`data/review/summary.json` memuat status eksekusi terakhir dan seluruh masalah
per dokumen/halaman; terminal menampilkan delapan masalah pertama saja. Setelah
berhasil menyimpan korpus, status menjadi `ready` dan daftar masalah kosong.
Draft lama tetap tersedia untuk inspeksi dan tidak digunakan oleh `chunk.py`.
Field `review_details` dalam draft
dokumen menyimpan teks terpilih sebelum cleaning, baris/word box, satuan, dan DPI
halaman yang membutuhkan pemeriksaan. Gunakan data ini bersama PDF sumber untuk
menentukan region tabel atau menulis replacement yang lengkap dan terverifikasi.
Deteksi tabel memakai keselarasan baris dan celah kolom yang konsisten, bukan
sekadar dua celah lebar dalam paragraf. Teks digital mempertahankan urutan
ekstraksi pypdf; koordinat visitor hanya dipakai sebagai metadata geometri.

Tambahkan override pada `documents.<doc_id>.pages.<nomor halaman PDF>`.
Skema berikut adalah contoh konfigurasi saja, **bukan keputusan cleaning yang
telah diverifikasi**:

```yaml
pages:
  7:
    method: ocr
    include: true
    reason: "Alasan berdasarkan pemeriksaan peneliti"
    paragraph_breaks: [3, 8]
  8:
    replacement_text_file: configs/replacements/D1-p08.txt
    reason: "Teks pengganti lengkap yang sudah diperiksa"
```

`include: false` wajib memiliki alasan. Replacement UTF-8 mengganti seluruh
halaman, dan hash isinya dicatat. Nomor `paragraph_breaks` mengacu baris setelah
rekonstruksi tabel/replacement dan normalisasi NFC/LF, sebelum unwrap.
Koreksi kecil yang sudah diverifikasi juga dapat berupa
`corrections: [{old: "teks persis", new: "koreksi persis"}]`.
Penghapusan repetitif memakai `header_patterns`, `footer_patterns`, atau
`remove_line_patterns` berupa regex yang cocok dengan seluruh baris; aturan
dapat ditempatkan pada default/dokumen/halaman. `remove_page_numbers` default
false; aktifkan hanya setelah memeriksa baris tepi halaman yang ditargetkan.

Untuk region tabel, gunakan `table_regions` di override halaman dengan
`units: pdf_points` atau `ocr_pixels`. Pixel wajib mencantumkan `dpi` yang sama
dengan render. Koordinat memakai asal kiri atas; `bounds` region adalah
`[x0,y0,x1,y1]`, dan `bounds` kolom `[x0,x1]`. Isi `columns` dengan `name` dan
rentang yang diverifikasi, `header_rows`, serta `merged_header` bila ada.
Rekonstruksi mengganti region sekali dalam teks kanonik dan mempertahankan
pasangan header-kolom-nilai. Nilai yang tidak jelas tidak ditebak.

Setelah mengganti preprocessing, aturan, replacement, teks, tokenizer, atau
parameter yang memengaruhi artefak, jalankan kembali tahap terkait:

```powershell
python prepare.py
python chunk.py
python index.py
python validate.py
```

Artefak lama ditolak melalui hash/checksum/config/provenance. Mengubah hanya
`semantic.cache_enabled` tidak mengubah identitas encoder atau chunking.
Vektor unit yang identitasnya masih sama dapat dipakai ulang.

## Kondisi dan retrieval

| Kondisi | Aturan |
|---|---|
| C1 | Fixed-size, input lengkap ≤256 token, overlap 50 token sumber |
| C2 | Fixed-size, input lengkap ≤512 token, overlap 50 token sumber |
| C3 | Paragraf/Pasal/ayat/butir, unit pendek tidak digabung, tanpa overlap |
| C4 | Semantic per dokumen percentile 95, buffer 0, tanpa overlap |
| C5 | Semantic per dokumen percentile 80, buffer 0, tanpa overlap |

Semua chunk merupakan slice karakter dari teks bersih final. C3–C5 memakai
splitter oversize pada batas kalimat, lalu batas token jika perlu. Metadata
mencatat seluruh halaman, kalimat, dan unit struktur yang beririsan. Hierarki
tidak ditambahkan ke teks embedding atau dipakai untuk memfilter retrieval.

Model/tokenizer menggunakan checkpoint yang sama,
[`fd1525a9fd15316a2d503bf26ab031a61d056e98`](https://huggingface.co/intfloat/multilingual-e5-small/tree/fd1525a9fd15316a2d503bf26ab031a61d056e98).
Prefix passage `passage: `, kueri `query: `, dan unit semantic `query: `.
Adapter memakai helper cosine dari `langchain-experimental==0.3.4`, threshold
percentile linear, dan pemisahan strict `distance > threshold`.

Satu `IndexFlatIP(384)` dibuat per kondisi, dengan vektor float32 ternormalisasi.
Search meminta seluruh row FAISS, mengurutkan `(-score,row_id)`, lalu mengambil
K pertama. Satu vektor per teks kueri dipakai bersama oleh lima indeks.
Tidak ada deduplikasi, threshold skor, reranker, atau generation LLM.

`results.jsonl` menyimpan teks chunk lengkap, skor asli FAISS, rank, offset,
halaman, token count, struktur, dan hash manifest untuk setiap kueri-kondisi.
Skor tidak diubah menjadi label relevansi. Sistem tidak membuat anotasi relevansi,
metrik penelitian, uji statistik, UI, API server, atau deployment.
