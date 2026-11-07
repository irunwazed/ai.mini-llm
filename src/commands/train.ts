/**
 * TRAINING: documents/  ->  model/
 *   1. baca semua PDF / DOCX / TXT / MD (rekursif, dengan cache)
 *   2. bersihkan header/footer, potong per pasal (batang tubuh, penjelasan, lampiran)
 *   3. pelajari metadata peraturan + peta perubahan ("diubah oleh")
 *   4. pelajari singkatan & definisi dari Ketentuan Umum
 *   5. hitung statistik BM25 (inverted index)
 *   6. latih word2vec (jaringan saraf) -> vektor kata & vektor pasal
 *
 * Pakai:  bun run train
 *         bun run train -- --dokumen D:/aturan --model model
 *         bun run train -- --tanpa-embedding        (lewati langkah 6, lebih cepat)
 */
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { bacaFile, daftarFile, terlihatHasilScan } from "../readers/index.js";
import { bangunIndeks, teksIndeks } from "../search/bm25.js";
import { pelajariIstilah } from "../training/glossary.js";
import { infoDariJudul, type InfoPeraturan } from "../training/metadata.js";
import { latihWord2Vec, tetangga } from "../training/word2vec.js";
import { simpanModel, vektorTeks } from "../model.js";
import { halamanKosong } from "../text/clean.js";
import { keTerm } from "../text/normalize.js";
import { potongDokumen } from "../text/chunk.js";
import type { Potongan } from "../types.js";
import { argumen } from "./args.js";

const { opsi, folderDokumen, folderModel } = argumen();
if (!existsSync(folderDokumen)) {
  console.error(`Folder dokumen "${folderDokumen}" tidak ditemukan.`);
  process.exit(1);
}
const mulai = performance.now();
const detik = () => ((performance.now() - mulai) / 1000).toFixed(1) + " dtk";

// 1-2. Baca & potong ------------------------------------------------------------
const files = daftarFile(folderDokumen);
console.log(`[1/6] Membaca ${files.length} file dari "${folderDokumen}"`);
const potongan: Potongan[] = [];
const gagal: { file: string; alasan: string }[] = [];
const scan: string[] = [];
const sebagianScan: { file: string; halaman: number[] }[] = [];
const tanpaPasal: string[] = [];
let dariCache = 0;

for (let i = 0; i < files.length; i++) {
  try {
    const r = await bacaFile(files[i], folderDokumen, join(folderModel, "cache"));
    if (r.dariCache) dariCache++;
    if (terlihatHasilScan(r.dok)) { scan.push(r.dok.file); continue; }
    if (r.dok.adaHalaman) {
      const kosong = halamanKosong(r.dok.halaman);
      if (kosong.length >= Math.max(2, 0.2 * r.dok.halaman.length)) sebagianScan.push({ file: r.dok.file, halaman: kosong });
    }
    const pot = potongDokumen(r.dok, potongan.length);
    if (pot.every((p) => p.pasal.startsWith("Bagian "))) tanpaPasal.push(r.dok.file);
    potongan.push(...pot);
  } catch (e) {
    gagal.push({ file: files[i], alasan: (e as Error).message });
  }
  if ((i + 1) % 25 === 0 || i === files.length - 1)
    process.stdout.write(`\r      dibaca ${i + 1}/${files.length} file, ${potongan.length} potongan`);
}
console.log(`\n[2/6] Dipotong per pasal: ${potongan.length} potongan (${detik()})`);

// 3. Metadata & perubahan ------------------------------------------------------
const info: Record<string, InfoPeraturan> = {};
for (const p of potongan) info[p.file] ??= infoDariJudul(p.judul, p.file);
const diubahOleh: Record<string, string[]> = {};
for (const i of Object.values(info)) if (i.mengubah) (diubahOleh[i.mengubah] ??= []).push(i.judulPendek);
console.log(`[3/6] Metadata ${Object.keys(info).length} peraturan, ${Object.keys(diubahOleh).length} peraturan tercatat pernah diubah`);

// 4. Istilah --------------------------------------------------------------------
const istilah = pelajariIstilah(potongan);
console.log(`[4/6] Dipelajari ${istilah.definisi.length} definisi dan ${Object.keys(istilah.singkatan).length} pasangan singkatan`);

// 5. BM25 ----------------------------------------------------------------------
const indeks = bangunIndeks(potongan);
console.log(`[5/6] Indeks BM25: ${Object.keys(indeks.kamus).length} term unik (${detik()})`);

// 6. Word2vec ------------------------------------------------------------------
let emb = null;
let vekPasal: Float32Array | null = null;
if ("tanpa-embedding" in opsi) {
  console.log("[6/6] Embedding dilewati (--tanpa-embedding)");
} else {
  const kalimat = potongan.map((p) => keTerm(p.teks).filter((t) => t.startsWith("~")));
  const token = kalimat.reduce((s, k) => s + k.length, 0);
  console.log(`[6/6] Melatih word2vec pada ${token.toLocaleString("id-ID")} token...`);
  emb = latihWord2Vec(kalimat, {
    dim: Number(opsi.dim ?? 64),
    minToken: Number(opsi["min-token"] ?? 200000),
    progres: (persen, loss) => process.stdout.write(`\r      ${persen.toFixed(0).padStart(3)}%  loss = ${loss.toFixed(4)}   `),
  });
  console.log();
  if (!emb) {
    console.log("      Korpus terlalu kecil untuk embedding (butuh >= 200.000 token; ubah dengan --min-token). Dilewati, pencarian memakai BM25 saja.");
  } else {
    const embIdx = new Map(emb.vocab.map((w, i) => [w, i]));
    vekPasal = new Float32Array(potongan.length * emb.dim);
    for (const p of potongan)
      vektorTeks(emb, embIdx, indeks, keTerm(teksIndeks(p)).filter((t) => t.startsWith("~")), vekPasal, p.id * emb.dim);
    console.log(`      ${emb.vocab.length} kata dasar mendapat vektor ${emb.dim} dimensi (${detik()})`);
    const contoh = emb.vocab.slice(0, 200).filter((w) => w.length > 4).slice(0, 4);
    for (const w of contoh)
      console.log(`      mirip "${w.slice(1)}": ${tetangga(emb, embIdx, w, 4).map((n) => `${n.kata.slice(1)} (${n.cos.toFixed(2)})`).join(", ")}`);
  }
}

// Simpan -----------------------------------------------------------------------
const laporan = {
  waktu: new Date().toISOString(),
  folderDokumen: resolve(folderDokumen),
  jumlahFile: files.length,
  dariCache,
  jumlahPotongan: potongan.length,
  jumlahTerm: Object.keys(indeks.kamus).length,
  definisi: istilah.definisi.length,
  singkatan: Object.keys(istilah.singkatan).length,
  embedding: emb ? { kata: emb.vocab.length, dimensi: emb.dim } : null,
  gagal,
  pdfHasilScan: scan,
  pdfSebagianScan: sebagianScan,
  tanpaPasal,
};
simpanModel(folderModel, {
  folderDokumen: resolve(folderDokumen).split("\\").join("/"),
  potongan, indeks, info, diubahOleh, istilah, emb, vekPasal,
  embIdx: new Map(),
}, laporan);

console.log(`\nSelesai dalam ${detik()}. Model disimpan di "${folderModel}/"`);
if (tanpaPasal.length) console.log(`  tanpa pasal      : ${tanpaPasal.length} file (dipotong per ±200 kata)`);
if (scan.length) console.log(`  PDF hasil scan   : ${scan.length} file DILEWATI (butuh OCR)`);
if (sebagianScan.length) console.log(`  PDF sebagian scan: ${sebagianScan.length} file punya halaman tanpa teks (butuh OCR)`);
if (gagal.length) console.log(`  gagal dibaca     : ${gagal.length} file`);
if (scan.length || sebagianScan.length || gagal.length) console.log(`  Rincian: ${folderModel}/laporan.json`);
