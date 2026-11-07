# AGENTS.md

Proyek Bun + TypeScript: (1) chatbot aturan pemerintah berbasis pencarian, (2) GPT mini dilatih dari nol. Tanpa library ML.

## Commands

- `bun install` — sekali saja (bun.lock; jangan pakai npm/yarn).
- `bun run typecheck` — satu-satunya pemeriksaan umum (`tsc --noEmit`, cover hanya `src/` dan `lessons/`). Jalankan setiap selesai mengubah kode.
- `bun run test-gpt` — gradient check + overfit test untuk `src/gpt/`; wajib setiap mengubah forward/backward/optimizer GPT.
- `bun run eval` — ukur akurasi dari `evaluation/questions.jsonl`; **butuh `model/` hasil `bun run train` lebih dulu**.
- Tidak ada lint, formatter, test runner (`bun test`), CI, atau pre-commit di repo ini — jangan mengarang perintah verifikasi lain.
- Argumen CLI lewat `--`: `bun run train -- --dokumen path --tanpa-embedding`. Parser di `src/commands/args.ts`; folder bawaan dokumen `documents`, model `model`.

## README.md sebagian sudah basi — percaya kode, bukan README

- Struktur folder di README memakai nama lama; nama asli sekarang:
  `dokumen/`→`documents/`, `pelajaran/`→`lessons/`, `evaluasi/`→`evaluation/`,
  `src/baca`→`src/readers`, `src/teks`→`src/text`, `src/latih`→`src/training`,
  `src/cari`→`src/search`, `src/jawab`→`src/answer`, `src/perintah`→`src/commands`.
- Script `cek-tipe` / `pelajaran:1` tidak ada; yang benar `bun run typecheck`, `bun run lesson:1`.
- Ambang keyakinan ada di `src/answer/compose.ts` (`AMBANG_TINGGI` 0.6, `AMBANG_SEDANG` 0.35, `AMBANG_KEKUATAN` 0.9), bukan `src/jawab/susun.ts`.

## Arsitektur

- Dua subsistem terpisah; perubahan di satu tidak memengaruhi yang lain:
  - **Chatbot retrieval**: `src/readers` (PDF/DOCX/TXT + cache) → `src/text` (bersih, potong per pasal, stemmer) → `src/training` (metadata, istilah, word2vec) → `src/search` (BM25 + RRF) → `src/answer` (susun jawaban + dasar hukum). State di `src/model.ts`.
  - **GPT dari nol**: `src/gpt/` (BPE, matriks, layer, Adam, sampling, checkpoint) — latih multi-worker via `SharedArrayBuffer`.
- **Mode agent** (`src/agent/multi-mode.ts`, async) menjalankan semua mode `MesinChat.proses` lalu memilih skor tertinggi; jika < 0,6 mencari internet lewat `src/search/internet.ts` (hanya API resmi: Wikipedia, Brave opsional via `BRAVE_API_KEY`; jangan scrape mesin pencari). `bun run eval-chat -- --agent` menguji mode ini tanpa internet.
- Entry point = satu file per script di `src/commands/*.ts` (lihat `package.json` `scripts`).
- Artefak training di-gitignore dan tidak boleh di-commit: `model/` (termasuk cache baca per-file di `model/cache/`) dan `gpt-models/`. `chat`/`search`/`eval` gagal sampai `bun run train` pernah dijalankan. Setiap dokumen ditambah/diubah → `bun run train` ulang (file tak berubah dari cache).

## Konvensi

- Seluruh identifier, flag CLI, komentar, dan pesan console sengaja berbahasa Indonesia — pertahankan untuk kode baru.
- ESM `moduleResolution: NodeNext`: import file `.ts` harus ditulis berekstensi `.js` (`from "../model.js"`), atau typecheck gagal.
- Word2vec otomatis dilewati bila korpus < 200.000 token (`--min-token`); `emb === null` itu normal, bukan bug.
