/**
 * PETA DATA: satu tempat untuk semua lokasi file. Alurnya selalu sama:
 *
 *   1. AMBIL   -> MENTAH   documents/ (aturan), data/mentah/{wikipedia,teks,obrolan,pembaca}/
 *   2. SIAPKAN -> SIAP     data/siap/<dataset>/   (tokenizer + token, hasil `bun run siapkan-data`)
 *   3. LATIH   -> MODEL    gpt-models/<model>/, model/ (indeks aturan), model-wiki/ (indeks Wikipedia)
 *   4. EVALUASI           evaluation/*.jsonl
 *   Semua proses panjang mencatat progres & log di progres/.
 *
 * Lokasi lama (sebelum struktur ini) masih dibaca jika file baru belum ada; pindahkan dengan
 * `bun run migrasi-data`.
 */
import { existsSync } from "node:fs";

export const LOKASI = {
  dokumenAturan: "documents",
  dokumenUpload: "documents/upload",
  dokumenWeb: "documents/web",
  mentah: "data/mentah",
  wikiTeks: "data/mentah/wikipedia/wiki-id.txt",
  wikiJudul: "data/mentah/wikipedia/sumber.txt",
  teksSendiri: "data/mentah/teks",
  obrolan: "data/mentah/obrolan/chat-id.jsonl",
  pembaca: "data/mentah/pembaca/reader-id.jsonl",
  siap: "data/siap",
  modelAturan: "model",
  modelWiki: "model-wiki",
  modelGPT: "gpt-models",
  progres: "progres",
  log: "progres/log",
  /** kunci & token (Google Drive, alamat server) — TIDAK di-commit, jangan dicetak */
  rahasia: ".rahasia",
} as const;

/**
 * Folder yang boleh disinkron dengan Google Drive (`bun run drive`). Di Drive, isinya ada di folder
 * "LLM dari Nol" dengan path yang sama (documents/jdih/x.pdf <-> LLM dari Nol/documents/jdih/x.pdf).
 */
export const FOLDER_SINKRON = [LOKASI.dokumenAturan, LOKASI.mentah, LOKASI.siap, LOKASI.modelGPT] as const;
/** Bagian yang tidak pernah disinkron: cache mesin ini & file sementara. */
export const LEWATI_SINKRON = [/^gpt-models\/cache\//, /\.part$/, /\.stop$/, /(^|\/)\.DS_Store$/, /(^|\/)Thumbs\.db$/, /(^|\/)desktop\.ini$/];

/** Path folder sinkron yang sah: di dalam FOLDER_SINKRON, tanpa "..", hanya karakter aman. */
export function folderSinkronSah(p: string): boolean {
  const n = p.replace(/\\/g, "/").replace(/\/+$/, "");
  if (!/^[\w .\/-]{1,200}$/.test(n) || n.split("/").some((x) => x === ".." || x === "." || x === "")) return false;
  return FOLDER_SINKRON.some((f) => n === f || n.startsWith(f + "/"));
}

/** Lokasi lama -> lokasi baru (untuk `bun run migrasi-data`). */
export const LOKASI_LAMA: [string, string][] = [
  ["data/wiki-id.txt", LOKASI.wikiTeks],
  ["data/wiki-id.sources.txt", LOKASI.wikiJudul],
  ["data/chat-id.jsonl", LOKASI.obrolan],
  ["data/reader-id.jsonl", LOKASI.pembaca],
];

/** Lokasi yang dipakai untuk MEMBACA: lokasi baru, atau lokasi lama jika belum dimigrasi. */
export function lokasiBaca(baru: string): string {
  if (existsSync(baru)) return baru;
  const lama = LOKASI_LAMA.find(([, b]) => b === baru)?.[0];
  return lama && existsSync(lama) ? lama : baru;
}

/** Nama sumber mentah (untuk `siapkan-data --sumber …`) -> file/folder. */
export const SUMBER_MENTAH: Record<string, { lokasi: string; jenis: "teks" | "obrolan"; keterangan: string }> = {
  wikipedia: { lokasi: LOKASI.wikiTeks, jenis: "teks", keterangan: "teks Wikipedia Indonesia (bun run fetch-data)" },
  teks: { lokasi: LOKASI.teksSendiri, jenis: "teks", keterangan: "teks bahasa Indonesia milik Anda (upload / salin ke folder)" },
  aturan: { lokasi: LOKASI.dokumenAturan, jenis: "teks", keterangan: "isi dokumen aturan (PDF/DOCX/TXT/MD di documents/)" },
  obrolan: { lokasi: LOKASI.obrolan, jenis: "obrolan", keterangan: "percakapan latih (bun run make-chat-data)" },
};
