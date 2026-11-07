# AGENTS.md

Proyek Bun + TypeScript: (1) chatbot aturan pemerintah berbasis pencarian, (2) GPT mini dilatih dari nol. Tanpa library ML.

## Commands

- `bun install` — sekali saja (bun.lock; jangan pakai npm/yarn). Berjalan di Bun, bukan Node (`Bun.serve` di web, worker thread di train-gpt).
- `bun run typecheck` — satu-satunya pemeriksaan statis (`tsc --noEmit`, cover hanya `src/` dan `lessons/`). Jalankan setiap selesai mengubah kode.
- `bun run test-gpt` — gradient check + overfit test untuk `src/gpt/`; wajib setiap mengubah forward/backward/optimizer/sampling GPT.
- Eval chatbot (semua butuh `model/` hasil `bun run train` lebih dulu):
  - `bun run eval` — akurasi dari `evaluation/questions.jsonl`.
  - `bun run eval-chat` — "nyambung" dari `evaluation/conversations.jsonl`; baseline sekarang 100% (58/58).
  - `bun run eval-reasoning` — soal logika dari `evaluation/reasoning.jsonl`; baseline sekarang 42/42.
  - Ketiganya menerima `-- --soal path/file.jsonl` untuk file uji lain. Satu pertanyaan: `bun run search -- "berapa hari cuti?"`.
- Tidak ada lint, formatter, test runner (`bun test`), CI, atau pre-commit di repo ini — jangan mengarang perintah verifikasi lain.
- Argumen CLI lewat `--`: `bun run train -- --dokumen path --tanpa-embedding`. Parser di `src/commands/args.ts`; folder bawaan dokumen `documents`, model `model`.
  - **Flag boolean baru wajib didaftarkan di set `FLAG`** di `src/commands/args.ts`, kalau tidak `--flag` berikutnya akan menelan argumen sesudahnya sebagai nilai.

## Arsitektur

- Dua subsistem terpisah; perubahan di satu tidak memengaruhi yang lain:
  - **Chatbot retrieval**: `src/readers` (PDF/DOCX/TXT + cache) → `src/text` (bersih, potong per pasal, stemmer) → `src/training` (metadata, istilah, word2vec) → `src/search` (BM25 + RRF) → `src/answer` (susun jawaban + dasar hukum). State di `src/model.ts`.
  - **GPT dari nol**: `src/gpt/` (BPE, matriks, layer, Adam, sampling, checkpoint) — latih multi-worker via `SharedArrayBuffer`.
- **Mode agent** (`src/agent/multi-mode.ts`, async) menjalankan semua mode `MesinChat.proses` lalu memilih skor tertinggi; jika < 0,6 (`AMBANG_PUAS`) mencari internet lewat `src/search/internet.ts` (hanya API resmi: Wikipedia, DuckDuckGo Instant Answer tanpa kunci — hanya definisi/entitas, bukan hasil web —, Brave opsional via `BRAVE_API_KEY`; jangan scrape mesin pencari, termasuk HTML DuckDuckGo). `bun run eval-chat -- --agent` menguji mode ini tanpa internet.
- Entry point tiap script: `src/commands/*.ts` (lihat `package.json` `scripts`; kecuali `make-chat-data` → `src/gpt/chat-data.ts`, `lesson:*` → `lessons/`).
- Artefak training di-gitignore, jangan di-commit: `model/` (termasuk cache per-file di `model/cache/`), `model-wiki/`, `gpt-models/`, `data/wiki-id*.txt`, `data/memory.json`, `exports/`.
  - `chat`/`search`/`eval*` gagal sampai `bun run train` pernah dijalankan; `model-wiki/` (hasil `bun run index-wiki`) dan `gpt-models/` opsional — tanpanya fitur terkait dilewati, bukan error.
  - Setiap dokumen ditambah/diubah → `bun run train` ulang (file tak berubah dibaca dari cache).
  - Jangan jalankan dua `train-gpt` dengan `--name` sama bersamaan (checkpoint saling menimpa).
- `documents/samples/` (dokumen fiktif) dirujuk `evaluation/questions.jsonl` — jangan dihapus, soal evaluasi jadi rusak.

## Konvensi

- Seluruh identifier, flag CLI, komentar, dan pesan console sengaja berbahasa Indonesia — pertahankan untuk kode baru.
- ESM `moduleResolution: NodeNext`: import file `.ts` harus ditulis berekstensi `.js` (`from "../model.js"`), atau typecheck gagal.
- Word2vec otomatis dilewati bila korpus < 200.000 token (`--min-token`); `emb === null` itu normal, bukan bug.
- README.md sudah dicek terhadap kode (nama perintah, struktur folder, lokasi file) dan akurat; angka hasil di dalamnya (loss, waktu training) bisa basi. Jika konflik, percaya kode.
