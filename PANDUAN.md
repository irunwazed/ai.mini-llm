# Panduan: ambil data → siapkan → latih → evaluasi → jalankan

Panduan praktis proyek ini, disusun **per langkah**. Penjelasan cara kerjanya (matematika, arsitektur) ada di [README.md](README.md).

Setiap langkah bisa dijalankan dengan dua cara:

- **Terminal:** dari folder proyek. Argumen tambahan ditulis setelah `--`, mis. `bun run train -- --dokumen D:/aturan`.
- **Web:** `bun run web`, lalu buka **http://localhost:3000/latih** (ada tautan "Data & training" di halaman chat). Halaman ini menjalankan perintah yang **sama** seperti di terminal, dengan progress bar, grafik loss, log lengkap, dan tombol berhenti.

## Daftar isi

0. [Alur & peta data (disimpan di mana)](#0-alur--peta-data)
1. [Langkah 1: ambil data](#1-langkah-1-ambil-data)
2. [Langkah 2: siapkan data (tokenisasi)](#2-langkah-2-siapkan-data)
3. [Langkah 3: latih](#3-langkah-3-latih)
4. [Langkah 4: evaluasi](#4-langkah-4-evaluasi)
5. [Langkah 5: jalankan chatbot](#5-langkah-5-jalankan-chatbot)
6. [Memantau & menghentikan proses](#6-memantau--menghentikan-proses)
7. [Kamus, agent, penalaran, memori (tanpa training)](#7-kamus-agent-penalaran-memori)
8. [Format file data](#8-format-file-data)
9. [Hal yang sering salah](#9-hal-yang-sering-salah)

---

## 0. Alur & peta data

```
 LANGKAH 1: AMBIL DATA            LANGKAH 2: SIAPKAN           LANGKAH 3: LATIH                 LANGKAH 4/5
 ─────────────────────            ──────────────────           ───────────────                  ───────────
 fetch-jdih / upload  → documents/ ──────────────────────────→ train        → model/       ─┐
 fetch-data           → data/mentah/wikipedia/ ─┬────────────→ index-wiki   → model-wiki/  ─┤
 upload / salin       → data/mentah/teks/      ─┤ siapkan-data                              ├→ eval*, chat, web
 make-chat-data       → data/mentah/obrolan/   ─┴→ data/siap/<dataset>/ → train-gpt → gpt-models/<model>/ ─┤
 make-reader-data     → data/mentah/pembaca/  ───────────────→ train-reader → gpt-models/<model>/ ─┘
```

| Lokasi | Isi | Dibuat oleh | Dipakai oleh |
|---|---|---|---|
| `documents/` | dokumen aturan PDF/DOCX/TXT/MD (subfolder `jdih/` = unduhan JDIH, `upload/` = upload web, `samples/` = contoh untuk soal uji) | `fetch-jdih`, upload, salin manual | `train`, `siapkan-data --sumber aturan` |
| `data/mentah/wikipedia/wiki-id.txt` (+ `sumber.txt` daftar judul) | teks Wikipedia Indonesia | `fetch-data` | `index-wiki`, `siapkan-data --sumber wikipedia`, `make-reader-data` |
| `data/mentah/teks/*.txt` | teks bahasa Indonesia milik Anda | upload, salin manual | `siapkan-data --sumber teks` |
| `data/mentah/obrolan/chat-id.jsonl` | percakapan latih (biasa + berkonteks) | `make-chat-data` | `siapkan-data --sumber obrolan` |
| `data/mentah/pembaca/reader-id.jsonl` | soal + rentang jawaban | `make-reader-data` | `train-reader`, `eval-reader` |
| `data/siap/<dataset>/` | **dataset siap latih**: `tokenizer.json`, token (`latih.bin`/`val.bin` atau `percakapan.jsonl`), `info.json` | `siapkan-data` | `train-gpt --dataset` |
| `gpt-models/<model>/` | bobot GPT, tokenizer, riwayat loss, riwayat dataset | `train-gpt`, `train-reader` | chat, web, eval |
| `model/` | indeks chatbot aturan (BM25, metadata pasal, cache per file) | `train` | chat, web, eval |
| `model-wiki/` | indeks Wikipedia lokal | `index-wiki` | chat, web |
| `progres/`, `progres/log/` | progres & log setiap proses | semua langkah | `bun run status`, panel web |
| `data/synonyms.txt`, `data/slang.txt` | kamus (bukan hasil training) | ditulis manual | chatbot |
| `evaluation/*.jsonl` | soal uji | ditulis manual | `eval*` |

Semua lokasi ditentukan di satu tempat: `src/paths.ts`. Isi folder `data/mentah/`, `data/siap/`, `documents/upload/`, `documents/jdih/`, `model*/`, `gpt-models/`, `progres/` tidak di-commit.

**Pindah dari struktur lama.** Versi lama menyimpan `data/wiki-id.txt`, `data/wiki-id.sources.txt`, `data/chat-id.jsonl`, `data/reader-id.jsonl`. Pindahkan sekali:

```
bun run migrasi-data -- --kering     # lihat apa yang akan dipindah
bun run migrasi-data                 # pindahkan
```

File yang baru diubah < 2 menit lalu (mis. sedang ditulis `fetch-data` lama) dilewati. Tunggu unduhan itu selesai, lalu jalankan lagi. Di web, peta data memberi peringatan "Masih di lokasi lama" dengan tombol **Pindahkan ke lokasi baru**. Sebelum dipindah, perintah lain tetap membaca lokasi lama, kecuali `fetch-data` yang menolak jalan agar tidak membuat dua salinan.

**Urutan minimum agar chatbot jalan:** `bun install` → `bun run train` → `bun run web`. Semua yang lain opsional; tanpanya fitur terkait dilewati, bukan error.

---

## 1. Langkah 1: ambil data

### Dokumen aturan

| Cara | Terminal | Web (/latih ①) |
|---|---|---|
| Unduh JDIH BKN | `bun run fetch-jdih` (→ `documents/jdih/`) | kartu **Unduh dokumen aturan JDIH BKN** |
| Upload | — | kartu **Upload file**, tujuan **Dokumen aturan** (→ `documents/upload/`) |
| Manual | salin file ke `documents/` (boleh bersubfolder) | — |

Format yang dibaca: `.pdf` (berteks/OCR), `.docx`, `.txt`, `.md`.

```
bun run fetch-jdih                              # crawler otomatis dari beranda, batas 50 PDF baru
bun run fetch-jdih -- --batas 0                 # semua yang ditemukan (±2500 dokumen, bisa lama)
bun run fetch-jdih -- --judul "cuti" --tahun 2022 --batas 10
bun run fetch-jdih -- --kategori P_bkn --batas 20
bun run fetch-jdih -- --url https://jdih.bkn.go.id/Detail_peraturan/breaking/2643
bun run fetch-jdih -- --kering --batas 20       # daftar saja, tanpa unduh
```

Aman dijalankan ulang: id di `documents/jdih/manifest.jsonl` dilewati, file yang ada tidak ditimpa (kecuali `--ganti`), dan antrean crawler tersimpan di `.crawl-state.json`.

→ Lanjut ke **langkah 3: latih chatbot aturan** (`bun run train`).

### Wikipedia Indonesia

```
bun run fetch-data                       # ±10 MB; artikel pilihan & bagus dulu, lalu acak
bun run fetch-data -- --target-mb 60     # tambah sampai total 60 MB (bisa dihentikan & dilanjutkan)
```

Web: kartu **Unduh Wikipedia Indonesia** (isi target MB). Hasilnya ditulis ke `data/mentah/wikipedia/wiki-id.txt`. Server Wikipedia membatasi kecepatan (HTTP 429), jadi puluhan MB bisa makan berjam-jam. Teksnya berlisensi CC BY-SA 4.0.

→ Lanjut ke **langkah 3: indeks Wikipedia** (untuk chatbot) dan/atau **langkah 2** (untuk GPT).

### Teks bahasa Indonesia milik Anda

Taruh file `.txt` (yang Anda punya hak pakainya) di `data/mentah/teks/`, atau di web pakai **Upload file** dengan tujuan **Teks bahasa Indonesia**. Nama file dibersihkan dan file yang sudah ada tidak ditimpa (diberi akhiran `-2`, `-3`, …). Batas 50 MB per file.

→ Lanjut ke **langkah 2** (`--sumber teks`).

### Data obrolan (untuk GPT obrolan)

```
bun run make-chat-data                       # obrolan biasa + 6000 percakapan berkonteks
bun run make-chat-data -- --konteks 10000    # lebih banyak percakapan berkonteks
```

Web: kartu **Buat data obrolan**. Hasil: `data/mentah/obrolan/chat-id.jsonl`. Isinya:

- **obrolan biasa** (`src/gpt/chat-data.ts`): sapaan, kabar, terima kasih, curhat, perkenalan nama, dll.;
- **percakapan berkonteks** (`src/gpt/chat-context-data.ts`): ingat fakta, pilih fakta, kata ganti, koreksi, topik lanjutan, ringkas, suasana, "belum tahu", dan **kartu konteks** (`<|konteks|> nama: …; tinggal: …`).

Nilai slot (nama, kota, pekerjaan, …) dibagi latih/uji: setiap nilai ke-5 hanya muncul di `eval-konteks`.

Menambah variasi:

- **niat obrolan baru:** `NIAT` di `src/gpt/chat-data.ts`;
- **skenario konteks baru:** `buatPercakapanKonteks` di `src/gpt/chat-context-data.ts`;
- **fakta kartu konteks baru:** `POLA` di `src/gpt/context-card.ts` (kunci sama dengan `KENAL[...].kartu`).

→ Lanjut ke **langkah 2** (`--sumber obrolan`).

### Data GPT pembaca (RAG)

```
bun run make-reader-data                         # soal otomatis dari Wikipedia
bun run make-reader-data -- --dengan-aturan      # ikut memakai dokumen aturan (butuh model/)
```

Web: kartu **Buat data GPT pembaca**. Hasil: `data/mentah/pembaca/reader-id.jsonl`. Pola pertanyaan ada di `calonDari` (`src/gpt/reader-data.ts`). Data ini langsung dipakai `train-reader` (langkah 3), tanpa langkah 2.

---

## 2. Langkah 2: siapkan data

**Untuk GPT saja.** Langkah ini membaca data mentah, membuat/memakai **tokenizer**, men-token-kan, lalu menyimpan **dataset siap latih** di `data/siap/<nama>/`. Training (langkah 3) hanya membaca hasil ini, tidak membaca teks mentah lagi. Karena itu Anda tahu persis data apa yang dilatih, dan dataset bisa dipakai berulang.

```
# tokenizer BARU (untuk model baru)
bun run siapkan-data -- --sumber wikipedia --nama wiki-v1 --huruf-asli
bun run siapkan-data -- --sumber wikipedia,teks --nama wiki-v2 --vocab 4000 --huruf-asli

# tokenizer DARI MODEL yang ada (untuk melanjutkan / fine-tune model itu)
bun run siapkan-data -- --sumber wikipedia,teks --nama wiki-v3 --tokenizer-dari wiki2
bun run siapkan-data -- --sumber obrolan --nama obrolan-v1 --tokenizer-dari wiki2
```

| Opsi | Arti |
|---|---|
| `--sumber` | `wikipedia`, `teks`, `aturan`, `obrolan` (boleh beberapa, pisah koma), atau path file/folder |
| `--nama` | nama dataset (huruf, angka, `.`, `-`, `_`) |
| `--vocab N` | ukuran kosakata tokenizer baru (bawaan 4000) |
| `--huruf-asli` | pertahankan huruf besar (tokenizer baru). Disarankan untuk pretraining |
| `--tokenizer-dari <model>` | pakai tokenizer model itu; **wajib** jika dataset akan dipakai `--resume`/`--from` model tersebut |

Web: kartu **② Siapkan data**. Centang sumber, beri nama, pilih tokenizer ("baru" atau "dari model X"). Tabel **Dataset yang sudah disiapkan** menampilkan jumlah token, asal tokenizer, dan peringatan **sumber berubah** jika data mentahnya diubah setelah disiapkan (siapkan ulang agar data baru ikut terlatih).

Isi `data/siap/<nama>/info.json`: sumber (ukuran, waktu ubah), jenis (`teks`/`obrolan`), token latih/val, asal tokenizer, dan **sidik tokenizer** (hash kosakata + merges, mis. `7819a14d-4000`). Pembagian val: teks = 10% terakhir, obrolan = setiap percakapan ke-10.

Di akhir, perintah ini mencetak saran perintah latih berikutnya.

---

## 3. Langkah 3: latih

### Chatbot aturan

```
bun run train                                   # documents/ -> model/
bun run train -- --dokumen D:/aturan            # folder lain
bun run train -- --tanpa-embedding              # lewati word2vec (lebih cepat)
```

Web: **③ → Latih chatbot aturan**. File yang tidak berubah dibaca dari cache (`model/cache/`). Word2vec otomatis dilewati jika korpus < 200 ribu token; itu normal. Ulangi setiap dokumen ditambah/diubah.

### Indeks Wikipedia (pengetahuan umum chatbot)

```
bun run index-wiki          # data/mentah/wikipedia/wiki-id.txt -> model-wiki/
```

Web: **③ → Indeks Wikipedia**. Ulangi setelah `fetch-data`.

### GPT: tiga cara

| Tujuan | Terminal | Web (③ Latih GPT) |
|---|---|---|
| **Model baru** | `bun run train-gpt -- --dataset wiki-v1 --name wiki3 --size besar --minutes 60` | **Model baru** |
| **Lanjutkan model yang sama, dengan data baru/lebih banyak** | `bun run train-gpt -- --dataset wiki-v3 --name wiki2 --resume --minutes 60` | **Lanjutkan model (resume)** |
| **Fine-tune ke model baru** (mis. pretraining → obrolan) | `bun run train-gpt -- --dataset obrolan-v1 --from wiki2 --name chat3 --steps 1500 --eval-every 200` | **Fine-tune dari model** |

**Pemeriksaan tokenizer.** Untuk `--resume` dan `--from`, sidik tokenizer dataset harus sama dengan tokenizer model. Jika berbeda, training **ditolak**, beserta perintah perbaikannya (`siapkan-data … --tokenizer-dari <model>`). Ini mencegah model dilatih dengan token yang artinya berbeda. Di web, pilihan dataset otomatis disaring ke yang cocok dengan model yang dipilih.

**Riwayat dataset.** Setiap model mencatat dataset yang pernah dipakai dan rentang langkahnya, mis. `wiki-v1 langkah 1–2000, wiki-v3 langkah 2001–3000`. Riwayat ini tampil di tabel **Model GPT yang ada** di web (dan di `gpt-models/<model>/gpt.json`, kunci `dataset`).

**Ukuran model (`--size`, hanya untuk model baru):**

| Preset | Parameter | Lapisan × dim | Konteks | Kosakata | Kecepatan |
|---|---|---|---|---|---|
| `small` (bawaan) | ±0,46 juta | 4 × 64 | 128 | 2.000 | cepat |
| `medium` | ±1 juta | 4 × 96 | 128 | 3.000 | sedang |
| `besar` | ±2,2 juta | 6 × 128, pre-LN | 256 | 4.000 | ±11 detik/langkah (CPU 16 thread) |

Kosakata sebenarnya mengikuti tokenizer dataset.

Opsi lain: `--minutes N` atau `--steps N`, `--batch 16`, `--lr 0.003`, `--dropout 0.1`, `--patience 5`, `--eval-every 100`, `--workers N`.

**Membaca log training:**

| Pola | Artinya | Tindakan |
|---|---|---|
| loss train & val turun | sedang belajar | lanjutkan (`--resume`) |
| train turun, val naik | menghafal (overfit) | tambah data / perkecil model / naikkan `--dropout` |
| keduanya macet tinggi | model kecil / lr salah | perbesar model / ubah `--lr` |

Perplexity = e^(loss val). Checkpoint yang disimpan selalu yang loss val-nya terbaik (tanda `*`). Aturan praktis: token latih ≥ 20× jumlah parameter. Preset besar ±450 token/detik, jadi 1 jam ≈ 1,6 juta token. Cicil dengan `--resume`.

**Fine-tune obrolan.** Loss hanya dihitung pada jawaban bot (masking). Kartu konteks tidak pernah terpotong. Kartu konteks hanya dipakai model yang tokenizer-nya punya `<|konteks|>` (model yang dipretrain dengan kode sekarang, mis. `wiki2` → `chat2`).

**Cara lama (masih jalan):** `bun run train-gpt -- --data data/mentah/wikipedia/wiki-id.txt --name x` membaca dan men-token-kan di memori tanpa langkah 2. Cara ini tidak memeriksa tokenizer saat `--resume`, jadi lebih baik pakai `--dataset`.

Coba hasilnya: `bun run generate -- --name wiki2 "Indonesia adalah" --length 80`, atau `bun run chat-gpt -- --name chat2 --history 4`.

### GPT pembaca (RAG)

```
bun run train-reader -- --from wiki --name reader --minutes 30
bun run train-reader -- --from reader --name reader2 --steps 1500    # lanjutkan dari pembaca yang ada
```

Web: **③ → GPT pembaca (RAG)**. Membaca `data/mentah/pembaca/reader-id.jsonl`. GPT ini **menunjuk rentang jawaban** di teks, tidak menulis bebas. Chatbot otomatis memakai pembaca terbaru.

---

## 4. Langkah 4: evaluasi

Semua bisa dijalankan dari web (**④ Evaluasi**); hasilnya tampil sebagai log di panel **Proses**.

| Perintah | Menguji | Acuan sekarang |
|---|---|---|
| `bun run typecheck` | tipe TypeScript | bersih |
| `bun run test-gpt` | turunan GPT (gradient check), overfit, KV-cache | semua LULUS |
| `bun run eval` | pencarian & jawaban aturan (`evaluation/questions.jsonl`) | jawaban 90,5%, Recall@1 85,7%, penolakan 3/3 |
| `bun run eval -- --rag` | + berapa jawaban GPT pembaca lolos verifikasi | — |
| `bun run eval-chat` (juga `-- --agent`, `-- --rag`) | percakapan "nyambung" (`evaluation/conversations.jsonl`) | 74/74 |
| `bun run eval-reasoning` | logika & matematika | 42/42 |
| `bun run eval-reader` | GPT pembaca (EM/F1 di artikel val) | EM 65%, F1 76% |
| `bun run eval-konteks -- --name chat2 --banding chat` | GPT memakai konteks (nilai uji yang tak pernah dilatih + ablasi tanpa riwayat) | `chat2` 2% (ablasi 1%), `chat` 1% |
| `bun run search -- "berapa hari cuti?"` | hasil pencarian satu pertanyaan | — |

Eval chatbot menerima `-- --soal path/file.jsonl` untuk file uji lain. Tidak ada `bun test`, lint, atau CI di proyek ini.

**Membaca `eval-konteks`.** Kolom **ablasi** menghapus riwayat dan kartu. Jika akurasi tetap, model sekadar menebak; jika turun, model benar-benar membaca konteks. Hasil `chat2` sekarang: bentuk jawabannya sudah benar (perplexity jawaban 3,98 vs 274 pada `chat`, mis. "Anda tinggal di …"), tetapi nilainya masih diambil dari hafalan latih, bukan disalin dari riwayat (2%). Model sekecil ini belum belajar "menyalin dari konteks". Lihat README bagian "GPT yang paham konteks".

**Menambah soal uji:** 1–3 soal per dokumen baru di `evaluation/questions.jsonl`, termasuk soal yang **tidak** ada aturannya (`"di_luar": true`). Jika akurasi turun, lihat daftar "Perlu diperhatikan" di keluaran `bun run eval`. Ambang keyakinan ada di `src/answer/compose.ts` (`AMBANG_TINGGI`, `AMBANG_SEDANG`, `AMBANG_KEKUATAN`).

---

## 5. Langkah 5: jalankan chatbot

**Terminal:**

```
bun run chat                          # mode otomatis
bun run chat -- --proses              # tampilkan langkah cara kerja
bun run chat -- --agent               # mode agent (+ internet)
bun run chat -- --agent --tanpa-internet
bun run chat -- --rag                 # GPT pembaca
bun run chat -- --chat-gpt chat2      # pilih model obrolan
```

Perintah di dalam chat: `:proses`, `:agent`, `:rag`, `:ingat teks`, `:memori`, `:lupa N`, `:buka N`, `:gpt teks`, `:export`, `:baru`, `:keluar`.

**Web:**

```
bun run web                           # chat: http://localhost:3000   data & training: http://localhost:3000/latih
bun run web -- --port 8080
bun run web -- --tanpa-internet       # mode agent tanpa internet
bun run web -- --chat-gpt chat        # paksa model obrolan tertentu (bawaan: yang terbaru)
bun run web -- --host 0.0.0.0         # bisa dibuka komputer lain di jaringan
```

Bawaannya server hanya mendengar `127.0.0.1` (komputer ini saja). Dengan `--host 0.0.0.0`, komputer lain bisa chat dan melihat progres, tetapi **menjalankan proses, menghentikan, dan upload tetap hanya dari komputer ini**.

Pilihan **Mode** di web: Otomatis, Hanya aturan, Hanya Wikipedia, Hanya GPT, Hanya penalaran, RAG, Agent. Klik jawaban untuk melihat panel **cara kerja**: niat, langkah agent, hasil setiap alat, curl & respons internet, peta attention GPT, dan perhatian ke konteks.

Setelah melatih model baru atau mengubah kode, matikan lalu jalankan ulang `bun run web`. Proses yang dijalankan dari web **tetap berjalan** walau server web dimatikan. Setelah web dinyalakan lagi, progres dan lognya muncul kembali.

---

## 6. Memantau & menghentikan proses

Proses panjang (`fetch-data`, `fetch-jdih`, `siapkan-data`, `train-gpt`, `train-reader`) mencatat progresnya ke `progres/<jenis>-<nama>.json`. Proses yang dijalankan dari web juga punya log lengkap di `progres/log/<id>.log` (daftarnya di `progres/jobs.json`).

**Baris progres di terminal:**

```
langkah 412/800 [██████████░░░░░░░░░░] 51%  loss 0.705 (val 0.435)  391 token/dtk  berjalan 14m 2d  sisa ±13m
```

**Menghentikan dengan aman** (menyelesaikan langkah, mengevaluasi, menyimpan checkpoint, lalu mencetak perintah lanjutan):

- **Ctrl+C sekali** di terminal prosesnya (Ctrl+C kedua = keluar paksa);
- dari terminal lain: `bun run status -- --hentikan train-gpt:chat2`;
- dari web: tombol **Hentikan dengan aman** di panel **Proses** (`/latih`) atau panel progres di halaman chat.

```
Dihentikan dengan aman di langkah 339. Checkpoint terbaik (loss val 4.2189) tersimpan di langkah 339.
Lanjutkan:  bun run train-gpt -- --dataset wiki-v3 --name wiki2 --resume --steps 100000
```

`--resume` melanjutkan dari checkpoint terbaik, termasuk momentum Adam. Unduhan cukup dijalankan ulang dengan perintah yang sama.

**Memantau:**

```
bun run status                   # sekali
bun run status -- --ikuti        # diperbarui tiap 2 detik (Ctrl+C keluar dari tampilan, proses tetap jalan)
bun run status -- --bersihkan    # hapus catatan proses yang sudah selesai/berhenti
```

Di web `/latih`, panel **Proses** menampilkan semua proses (dari web maupun terminal): progress bar, sisa waktu, grafik loss train (abu) & val (biru), dan **log berjalan** untuk proses dari web.

| Simbol | Artinya |
|---|---|
| ▶ | berjalan |
| ■ | dihentikan |
| ✓ | selesai |
| ! | terhenti tiba-tiba (checkpoint terbaik tetap ada, lanjutkan dengan `--resume`) |

**Aturan:**

- Satu model/dataset/unduhan tidak bisa dikerjakan dua proses sekaligus dari web (ditolak dengan pesan).
- Dari terminal, jangan jalankan dua `train-gpt` dengan `--name` sama.
- Cegah komputer tidur saat training `--minutes`, karena jumlah langkah dihitung dari kecepatan awal. Atau pakai `--steps`.
- Training dan web boleh jalan bersamaan, tetapi berebut CPU.

**Keamanan web.** Web hanya menjalankan perintah dari **daftar putih** (`src/web/jobs.ts`), dengan opsi bertipe (angka berentang, nama, pilihan tetap). Tidak ada shell bebas. Upload hanya ke `documents/upload/` (PDF/DOCX/TXT/MD) dan `data/mentah/teks/` (TXT), maks 50 MB per file, tanpa menimpa file yang ada.

---

## 7. Kamus, agent, penalaran, memori

Tidak perlu training; cukup jalankan ulang chat/web.

- `data/synonyms.txt`: istilah sehari-hari → istilah dokumen (`cuti hamil = cuti melahirkan`).
- `data/slang.txt`: bahasa santai → baku (`gimana|gmn = bagaimana`).
- **Agent** (`src/agent/`): pahami maksud → pakai alat → bandingkan → perbaiki → simpulkan. Internet hanya lewat API resmi: Wikipedia; DuckDuckGo (terblokir ISP di Indonesia, otomatis dilewati); Brave jika `BRAVE_API_KEY` diisi. Relasi baru ("di mana X tinggal sekarang"): `RELASI` di `src/agent/relations.ts`.
- **Penalaran** (`src/reasoning/`): soal uji di `evaluation/reasoning.jsonl`, teka-teki di `BANK` (`src/reasoning/puzzles.ts`).
- **Memori pengguna:** disimpan **manual** (`:ingat …` di terminal atau panel Memori di web) ke `data/memory.json`.

Uji dengan `bun run eval-chat` (juga `-- --agent`, `-- --rag`) dan `bun run eval-reasoning`. Kasus baru → tambahkan ke `evaluation/conversations.jsonl`.

---

## 8. Format file data

**`evaluation/questions.jsonl`:**

```json
{"tanya": "berapa hari cuti tahunan pegawai?", "file": "samples/sample-employee-leave.txt", "pasal": "Pasal 3", "harus_ada": "12 hari kerja"}
{"tanya": "siapa presiden pertama indonesia?", "di_luar": true}
```

**`evaluation/conversations.jsonl`** (`memori` = isi memori awal, `agent`/`rag` = selalu diuji di mode itu):

```json
{"nama": "nama dari memori", "memori": ["nama saya Irun"], "giliran": [{"pesan": "siapa nama saya?", "jalur": "maksud", "harus_ada": "Nama Anda Irun"}, {"pesan": "halo", "jalur": "obrolan", "tidak_ada": ["Budi"]}]}
{"nama": "agent: lanjutan", "agent": true, "giliran": [{"pesan": "apa itu virus?", "jalur": "wiki", "harus_ada": "Virus adalah"}]}
```

`harus_ada` boleh berupa daftar (salah satunya cukup). Nilai `jalur`: `aturan`, `wiki`, `internet`, `memori`, `obrolan`, `maksud`, `detail`, `klarifikasi`, `penalaran`, `tidak-ditemukan`.

**`evaluation/reasoning.jsonl`:**

```json
{"kategori": "logika dasar", "soal": "Jika hari ini hari Rabu, 100 hari lagi hari apa?", "harus_ada": ["Jumat"]}
```

**`data/mentah/obrolan/chat-id.jsonl`** (dibuat `make-chat-data`; boleh ditambah manual, satu percakapan per baris, lalu siapkan ulang di langkah 2):

```json
{"turns":[{"user":"halo"},{"bot":"Halo! Ada yang bisa saya bantu?"}]}
{"turns":[{"konteks":"nama: Zaki"},{"user":"Kamu ingat namaku?"},{"bot":"Tentu, nama Anda Zaki."}]}
```

**`data/synonyms.txt`** dan **`data/slang.txt`:**

```
bensin = bahan bakar, bbm
gimana|gmn|gmana = bagaimana
```

---

## 9. Hal yang sering salah

- **Dokumen ditambah, tetapi lupa `bun run train`.** Chatbot masih memakai indeks lama.
- **Wikipedia diunduh, tetapi lupa `bun run index-wiki`.** Pertanyaan umum belum memakai artikel baru.
- **Data mentah ditambah, tetapi dataset tidak disiapkan ulang.** `train-gpt --dataset` melatih data lama. Web menandainya **sumber berubah**; di terminal, `train-gpt` mencetak peringatan.
- **`--resume` dengan dataset bertokenizer lain.** Ditolak; siapkan dataset dengan `--tokenizer-dari <model>`.
- **Menghapus `documents/samples/`.** `bun run eval` rusak, karena soal uji merujuk dokumen contoh itu.
- **Model GPT lama + fitur baru.** Kartu konteks hanya bekerja di model yang dipretrain ulang (`wiki2` → `chat2`).
- **Web masih memakai kode/model lama.** Matikan lalu jalankan ulang `bun run web`.
- **File masih di lokasi lama.** Jalankan `bun run migrasi-data` (bagian 0).
- **(Pengembang) Flag CLI boolean baru.** Wajib didaftarkan di `FLAG` (`src/commands/args.ts`). Perintah yang boleh dijalankan web didaftarkan di `PERINTAH` (`src/web/jobs.ts`).
