/**
 * Semua hasil training dalam satu objek, beserta cara menyimpan dan memuatnya.
 *
 * Isi folder model/:
 *   potongan.json      pasal + sumber + metadata peraturan + peta perubahan
 *   indeks.json        indeks BM25
 *   istilah.json       singkatan & definisi yang dipelajari dari Ketentuan Umum
 *   embedding.json     kosakata word2vec      embedding.bin     vektor kata (Float32)
 *   vektor-pasal.bin   vektor setiap pasal (Float32), untuk pencarian makna
 *   laporan.json       ringkasan training
 */
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { Indeks } from "./search/bm25.js";
import { idf } from "./search/bm25.js";
import type { Istilah } from "./training/glossary.js";
import type { InfoPeraturan } from "./training/metadata.js";
import type { Embedding } from "./training/word2vec.js";
import { keTerm } from "./text/normalize.js";
import type { Potongan } from "./types.js";

export interface Model {
  /** "aturan" = dokumen peraturan (sumber disebut "dasar hukum"); "umum" = pengetahuan umum (Wikipedia). */
  jenis?: "aturan" | "umum";
  folderDokumen: string;
  potongan: Potongan[];
  indeks: Indeks;
  info: Record<string, InfoPeraturan>;   // per file
  diubahOleh: Record<string, string[]>;  // kunci peraturan -> judul peraturan pengubah
  istilah: Istilah;
  emb: Embedding | null;
  embIdx: Map<string, number>;
  vekPasal: Float32Array | null;         // N × dim
}

/**
 * Vektor sebuah teks = rata-rata berbobot vektor kata-katanya:
 *   vek(T) = Σ_w idf(w) · v_w  lalu dinormalisasi
 * Kata langka (idf besar) lebih menentukan makna dibanding kata umum.
 */
export function vektorTeks(emb: Embedding, embIdx: Map<string, number>, ind: Indeks, teks: string | string[], keluar?: Float32Array, offset = 0): boolean {
  const out = keluar ?? new Float32Array(emb.dim);
  const terms = Array.isArray(teks) ? teks : keTerm(teks);
  let ada = false;
  for (const t of terms) {
    const i = embIdx.get(t);
    if (i === undefined) continue;
    const w = idf(ind, t);
    for (let d = 0; d < emb.dim; d++) out[offset + d] += w * emb.vec[i * emb.dim + d];
    ada = true;
  }
  let n = 0;
  for (let d = 0; d < emb.dim; d++) n += out[offset + d] ** 2;
  n = Math.sqrt(n) || 1;
  for (let d = 0; d < emb.dim; d++) out[offset + d] /= n;
  return ada;
}

const tulisBin = (path: string, a: Float32Array) => writeFileSync(path, Buffer.from(a.buffer, a.byteOffset, a.byteLength));
const bacaBin = (path: string) => {
  const b = readFileSync(path);
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};

export function simpanModel(folder: string, m: Model, laporan: unknown): void {
  mkdirSync(folder, { recursive: true });
  writeFileSync(join(folder, "potongan.json"),
    JSON.stringify({ jenis: m.jenis ?? "aturan", folderDokumen: m.folderDokumen, info: m.info, diubahOleh: m.diubahOleh, potongan: m.potongan }));
  writeFileSync(join(folder, "indeks.json"), JSON.stringify(m.indeks));
  writeFileSync(join(folder, "istilah.json"), JSON.stringify(m.istilah, null, 1));
  const fileEmb = ["embedding.json", "embedding.bin", "vektor-pasal.bin"].map((f) => join(folder, f));
  if (m.emb && m.vekPasal) {
    writeFileSync(fileEmb[0], JSON.stringify({ dim: m.emb.dim, vocab: m.emb.vocab }));
    tulisBin(fileEmb[1], m.emb.vec);
    tulisBin(fileEmb[2], m.vekPasal);
  } else {
    for (const f of fileEmb) rmSync(f, { force: true }); // jangan sampai embedding lama ikut terpakai
  }
  writeFileSync(join(folder, "laporan.json"), JSON.stringify(laporan, null, 2));
}

export function muatModel(folder: string): Model {
  if (!existsSync(join(folder, "istilah.json")))
    throw new Error(`Model belum ada (atau versi lama) di "${folder}". Jalankan dulu: bun run train`);
  const p = JSON.parse(readFileSync(join(folder, "potongan.json"), "utf8"));
  const indeks = JSON.parse(readFileSync(join(folder, "indeks.json"), "utf8"));
  const istilah = JSON.parse(readFileSync(join(folder, "istilah.json"), "utf8"));
  let emb: Embedding | null = null;
  let vekPasal: Float32Array | null = null;
  if (existsSync(join(folder, "embedding.json"))) {
    const e = JSON.parse(readFileSync(join(folder, "embedding.json"), "utf8"));
    emb = { dim: e.dim, vocab: e.vocab, vec: bacaBin(join(folder, "embedding.bin")) };
    vekPasal = bacaBin(join(folder, "vektor-pasal.bin"));
  }
  return {
    jenis: p.jenis ?? "aturan", folderDokumen: p.folderDokumen, potongan: p.potongan, info: p.info, diubahOleh: p.diubahOleh,
    indeks, istilah, emb, vekPasal,
    embIdx: new Map((emb?.vocab ?? []).map((w, i) => [w, i])),
  };
}
