# AGENTS.md

Proyek Bun + TypeScript: (1) chatbot aturan pemerintah berbasis pencarian, (2) GPT mini dilatih dari nol. Tanpa library ML.

## Commands

- `bun install` — sekali saja (bun.lock; jangan pakai npm/yarn). Berjalan di Bun, bukan Node (`Bun.serve` di web, worker thread di train-gpt).
- `bun run typecheck` — satu-satunya pemeriksaan statis (`tsc --noEmit`, cover hanya `src/` dan `lessons/`). Jalankan setiap selesai mengubah kode.
- `bun run test-gpt` — gradient check + overfit test untuk `src/gpt/`; wajib setiap mengubah forward/backward/optimizer/sampling GPT.
- Eval chatbot (semua butuh `model/` hasil `bun run train` lebih dulu):
  - `bun run eval` — akurasi dari `evaluation/questions.jsonl`.
  - `bun run eval-chat` — "nyambung" dari `evaluation/conversations.jsonl`; baseline sekarang 100% (74/74), juga dengan `-- --agent` dan `-- --rag`. Baris berisi `"agent": true` / `"rag": true` selalu diuji di mode itu (`rag` dilewati jika belum ada GPT pembaca).
  - `bun run eval-reasoning` — soal logika dari `evaluation/reasoning.jsonl`; baseline sekarang 42/42.
  - `bun run eval -- --rag` — tambahan statistik GPT pembaca (berapa rentang lolos verifikasi & benar); akurasi utama tetap 90,5%.
  - `bun run eval-reader` — EM/F1 GPT pembaca di data val (`data/mentah/pembaca/reader-id.jsonl`, dibagi per artikel); baseline F1 76%.
  - `bun run eval-konteks` — apakah GPT obrolan memakai konteks (nilai slot uji yang tidak pernah dilatih, + ablasi tanpa riwayat).
  - Ketiganya menerima `-- --soal path/file.jsonl` untuk file uji lain. Satu pertanyaan: `bun run search -- "berapa hari cuti?"`.
- `bun run fetch-jdih` — unduh PDF dari JDIH BKN (https://jdih.bkn.go.id/) ke `documents/jdih/`. Tanpa argumen = crawler otomatis dari beranda; ada `--judul/--tahun/--nomor/--tipe`, `--kategori`, `--url`, `--kering` (daftar saja), `--batas` (unduhan baru per sesi, 0 = tanpa batas). Aman dijalankan ulang: `manifest.jsonl` + `.crawl-state.json` di folder itu membuat file lama tidak diunduh lagi; file hilang di server dicatat `tidak-ada`. Setelah menambah unduhan → `bun run train`.
- Tidak ada lint, formatter, test runner (`bun test`), CI, atau pre-commit di repo ini — jangan mengarang perintah verifikasi lain.
- Argumen CLI lewat `--`: `bun run train -- --dokumen path --tanpa-embedding`. Parser di `src/commands/args.ts`; folder bawaan dokumen `documents`, model `model`.
  - **Flag boolean baru wajib didaftarkan di set `FLAG`** di `src/commands/args.ts`, kalau tidak `--flag` berikutnya akan menelan argumen sesudahnya sebagai nilai.

## Arsitektur

- Dua subsistem terpisah; perubahan di satu tidak memengaruhi yang lain:
  - **Chatbot retrieval**: `src/readers` (PDF/DOCX/TXT + cache) → `src/text` (bersih, potong per pasal, stemmer) → `src/training` (metadata, istilah, word2vec) → `src/search` (BM25 + RRF) → `src/answer` (susun jawaban + dasar hukum). State di `src/model.ts`.
  - **GPT dari nol**: `src/gpt/` (BPE, matriks, layer, Adam, sampling, checkpoint) — latih multi-worker via `SharedArrayBuffer`.
- **GPT pembaca / RAG**: `GPTConfig.spanHead` menambah kepala span (`forwardSpan`, `langkahSpan`; backward bersama di `mundur`), `src/gpt/reader.ts` (baca rentang), `src/gpt/reader-data.ts` (data otomatis). Pipa: `bun run make-reader-data` → `bun run train-reader -- --from wiki --name reader` (checkpoint `mode: "pembaca"`, ambang di `meta.pembaca`). RAG di `src/rag/` (dialog = pertanyaan lanjutan/elipsis, retrieve = pasal dirujuk + memori, pipeline = baca + verifikasi); dipakai lewat mode `"rag"` / `OpsiProses.rag`. Loop konteks percakapan (cari asli vs berkonteks) di `MesinChat.proses` aktif di semua mode.
- **Mode agent** (`src/agent/`, async): `intent.ts` (Niat) → `tools.ts` (alat = mode/sumber) → `compare.ts` (skor + kekurangan) → perbaikan di `multi-mode.ts` (maks 8 langkah) → `synthesize.ts` (kolam bukti dari SEMUA hasil, jawaban disusun agent sendiri; relasi berwaktu seperti "sekarang bekerja di mana" disimpulkan dari urutan waktu fakta oleh `reasoner.ts` + `relations.ts`). Internet lewat `src/search/internet.ts` (hanya API resmi: Wikipedia, DuckDuckGo Instant Answer tanpa kunci — hanya definisi/entitas, bukan hasil web —, Brave opsional via `BRAVE_API_KEY`; jangan scrape mesin pencari, termasuk HTML DuckDuckGo; host yang gagal TLS dicatat `hostTerblokir` dan dilewati 15 menit — DuckDuckGo diblokir ISP di Indonesia, itu normal). `bun run eval-chat -- --agent` menguji mode ini tanpa internet.
- Entry point tiap script: `src/commands/*.ts` (lihat `package.json` `scripts`; kecuali `lesson:*` → `lessons/`). `make-chat-data` (`src/commands/make-chat-data.ts`) = obrolan biasa (`src/gpt/chat-data.ts`) + percakapan berkonteks (`src/gpt/chat-context-data.ts`, nilai slot dibagi latih/uji).
- **Alur data (satu sumber kebenaran: `src/paths.ts` `LOKASI`)**: langkah 1 data MENTAH (`documents/` + `documents/jdih/` + `documents/upload/`, `data/mentah/{wikipedia,teks,obrolan,pembaca}/`) → langkah 2 `bun run siapkan-data -- --sumber wikipedia,teks --nama X [--tokenizer-dari model]` (`src/commands/prepare-data.ts`, `src/gpt/prepared.ts`) menulis `data/siap/X/` (tokenizer + token + `info.json` dengan sidik tokenizer) → langkah 3 `train-gpt -- --dataset X` (menolak `--resume`/`--from` jika sidik tokenizer beda; riwayat di `meta.dataset`). Path baru selalu lewat `LOKASI`; baca file yang mungkin belum dimigrasi dengan `lokasiBaca()`. `bun run migrasi-data` memindahkan file struktur lama (`data/wiki-id.txt` dll.; file diubah < 2 menit dilewati). `--data <file>` lama masih didukung.
- **Web** (`src/commands/web.ts`): `/` chat, `/latih` data & training (`src/web/latih.html`). Proses dari web lewat `src/web/jobs.ts`: hanya perintah di daftar putih `PERINTAH` dengan opsi bertipe — perintah/flag baru yang ingin bisa dijalankan dari web wajib ditambah di sana; spawn `detached`, keluaran langsung ke `progres/log/<id>.log`, daftar di `progres/jobs.json`. Server bawaan hanya `127.0.0.1` (`--host` untuk mengubah); rute job/upload/hentikan menolak non-lokal (`lokal(req)`). Upload hanya ke `documents/upload/` dan `data/mentah/teks/`.
- **GPT berkonteks**: preset `--size besar` (6 lapisan, dim 128, konteks 256, kosakata 4000, pre-LN); token `<|konteks|>` (kartu konteks dari `src/gpt/context-card.ts`, hanya dipakai jika tokenizer model punya token itu); KV-cache `GPT.forwardTambah` dipakai `hasilkanDariId` dan diuji `test-gpt` (harus identik dengan forward penuh). Ukur dengan `bun run eval-konteks -- --name chat2 --banding chat` (acuan: `chat` 1%, `chat2` 2% — bentuk jawaban benar tapi nilai belum disalin dari konteks; `wiki2` baru dipretrain 300 langkah).
- Panduan pemakai lengkap: `PANDUAN.md` (perintah di sana harus ikut diperbarui jika script/flag berubah).
- **Progres proses panjang**: `train-gpt`, `train-reader`, `fetch-data`, `fetch-jdih`, `siapkan-data` memakai `Pelacak` (`src/gpt/progress.ts`): baris progres, file `progres/<jenis>-<nama>.json`, berhenti aman lewat Ctrl+C / file `.stop` (`bun run status -- --hentikan jenis:nama`, tombol di panel web `/api/progres` dan `/latih`). Proses panjang baru wajib memakai `Pelacak` dan memeriksa `pelacak.berhenti` setiap langkah. `bun run status [-- --ikuti]` membaca semua progres.
- Artefak training di-gitignore, jangan di-commit: `model/` (termasuk cache per-file di `model/cache/`), `model-wiki/`, `gpt-models/`, `data/mentah/`, `data/siap/`, `data/memory.json`, `exports/`, `progres/`, `documents/jdih/` (unduhan `fetch-jdih`: PDF + manifest + state), `documents/upload/`; plus lokasi lama `data/wiki-id*.txt`, `data/reader-id.jsonl`.
  - `MesinChat` memuat GPT pembaca terbaru (`gptTerbaru("pembaca")`) otomatis; training pembaca baru dengan nama lain langsung dipakai begitu checkpoint pertama tersimpan.
  - `chat`/`search`/`eval*` gagal sampai `bun run train` pernah dijalankan; `model-wiki/` (hasil `bun run index-wiki`) dan `gpt-models/` opsional — tanpanya fitur terkait dilewati, bukan error.
  - Setiap dokumen ditambah/diubah → `bun run train` ulang (file tak berubah dibaca dari cache).
  - Jangan jalankan dua `train-gpt` dengan `--name` sama bersamaan (checkpoint saling menimpa).
- `documents/samples/` (dokumen fiktif) dirujuk `evaluation/questions.jsonl` — jangan dihapus, soal evaluasi jadi rusak.

## Konvensi

- Seluruh identifier, flag CLI, komentar, dan pesan console sengaja berbahasa Indonesia — pertahankan untuk kode baru.
- ESM `moduleResolution: NodeNext`: import file `.ts` harus ditulis berekstensi `.js` (`from "../model.js"`), atau typecheck gagal.
- Word2vec otomatis dilewati bila korpus < 200.000 token (`--min-token`); `emb === null` itu normal, bukan bug.
- README.md sudah dicek terhadap kode (nama perintah, struktur folder, lokasi file) dan akurat; angka hasil di dalamnya (loss, waktu training) bisa basi. Jika konflik, percaya kode.
