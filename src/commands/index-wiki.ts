/**
 * MENGINDEKS TEKS WIKIPEDIA untuk menjawab pertanyaan fakta umum dengan RETRIEVAL.
 *
 * Model GPT kecil tidak bisa menyimpan fakta ("apa itu Obersalzberg?"). Sistem sungguhan
 * memecahkannya dengan mencari paragraf yang relevan lalu mengutipnya. Di sini kita memakai
 * mesin pencari BM25 yang sama dengan chatbot aturan:
 *   1. data/wiki-id.txt dipecah per artikel (dipisah baris kosong), judul dari wiki-id.sources.txt
 *   2. setiap artikel dipotong per ±120 kata (paragraf-paragraf berurutan)
 *   3. dibangun indeks BM25 -> folder model-wiki/
 *
 * Pakai:  bun run index-wiki         (jalankan lagi setelah bun run fetch-data)
 */
import { existsSync, readFileSync } from "node:fs";
import { bangunIndeks } from "../search/bm25.js";
import type { InfoPeraturan } from "../training/metadata.js";
import { simpanModel } from "../model.js";
import type { Potongan } from "../types.js";

const TEKS = "data/wiki-id.txt", JUDUL = "data/wiki-id.sources.txt", KELUAR = "model-wiki";
if (!existsSync(TEKS) || !existsSync(JUDUL)) {
  console.error("Data Wikipedia belum ada. Jalankan dulu: bun run fetch-data");
  process.exit(1);
}
const mulai = performance.now();
const artikel = readFileSync(TEKS, "utf8").split("\n\n").map((s) => s.trim()).filter(Boolean);
const judul = readFileSync(JUDUL, "utf8").split("\n").filter((l) => l && !l.startsWith("#"));
if (artikel.length !== judul.length)
  console.log(`Peringatan: ${artikel.length} artikel tetapi ${judul.length} judul; judul mungkin tidak cocok.`);

const potongan: Potongan[] = [];
const info: Record<string, InfoPeraturan> = {};
artikel.forEach((teks, a) => {
  const nama = judul[a] ?? `Artikel ${a + 1}`;
  const file = `wikipedia/${nama}`;
  info[file] = { jenis: "", nomor: "", tahun: null, tentang: nama, judulPendek: `Wikipedia: ${nama}`, kunci: null, mengubah: null };
  // Gabungkan paragraf berurutan sampai ±120 kata, agar satu potongan = satu topik kecil
  let isi: string[] = [], kata = 0, nomor = 1, baris = 1, barisAwal = 1;
  const tutup = () => {
    if (!isi.length) return;
    potongan.push({ id: potongan.length, file, judul: nama, bab: "", pasal: `bagian ${nomor++}`, halaman: null, baris: barisAwal, teks: isi.join("\n") });
    isi = []; kata = 0;
  };
  for (const p of teks.split("\n")) {
    if (!isi.length) barisAwal = baris;
    isi.push(p);
    kata += p.split(" ").length;
    baris++;
    if (kata >= 120) tutup();
  }
  tutup();
});

const indeks = bangunIndeks(potongan);
simpanModel(KELUAR, {
  jenis: "umum", folderDokumen: "id.wikipedia.org (CC BY-SA 4.0)", potongan, indeks, info, diubahOleh: {},
  istilah: { singkatan: {}, definisi: [] }, emb: null, vekPasal: null, embIdx: new Map(),
}, { waktu: new Date().toISOString(), artikel: artikel.length, potongan: potongan.length });
console.log(`Indeks Wikipedia: ${artikel.length} artikel, ${potongan.length} potongan, ${Object.keys(indeks.kamus).length.toLocaleString("id-ID")} term (${((performance.now() - mulai) / 1000).toFixed(1)} dtk) -> ${KELUAR}/`);
console.log(`Coba:  bun run search -- --model ${KELUAR} "apa itu Obersalzberg?"`);
