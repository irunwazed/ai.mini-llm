/**
 * LANGKAH 5: GPT mini, dilatih dari nol dalam ±1 menit
 *
 * Semua bagian dari diagram dipakai:
 *   token -> Embedding + Positional Encoding -> N× [Attention, Add&Norm, FeedForward, Add&Norm]
 *         -> Linear -> Softmax
 * Training = mengulang: ambil potongan teks acak, tebak setiap token berikutnya,
 * hitung loss (cross-entropy), backward, update Adam.
 *
 * Bandingkan dengan pelajaran 2 (n-gram): GPT tidak menyimpan tabel hitungan,
 * melainkan BOBOT yang dipelajari, dan attention bisa memakai seluruh konteks.
 *
 * Jalankan:  bun run lesson:5
 */
import { readFileSync } from "node:fs";
import { latihBPE, Tokenizer } from "../src/gpt/bpe.js";
import { GPT } from "../src/gpt/model.js";
import { Adam, jadwalLR } from "../src/gpt/optimizer.js";
import { Rng } from "../src/gpt/random.js";
import { hasilkanTeks } from "../src/gpt/text-generator.js";
import { dataTeks } from "../src/gpt/dataset.js";

const teks = readFileSync(new URL("../data/corpus.txt", import.meta.url), "utf8").toLowerCase().replace(/[ \t]+/g, " ");
const tok = new Tokenizer(latihBPE(teks, 300, undefined, 3));
// Train/val: setiap blok ke-10 menjadi data validasi (teks yang TIDAK dipakai untuk belajar)
const { train: data, val } = dataTeks(tok, teks);
const model = new GPT({ vocab: tok.ukuran, context: 32, dim: 48, heads: 4, layers: 2, ff: 192, preLN: false });
const adam = new Adam(model, { lr: 3e-3 });
const G = new Float64Array(model.ukuran);
const rng = new Rng(1);
const T = 32, B = 8, LANGKAH = 400;
const lossVal = () => {
  let s = 0, n = 0;
  for (let a = 0; a + T + 1 <= val.length; a += T) { s += model.langkah(val.subarray(a, a + T), val.subarray(a + 1, a + T + 1), null, 1); n += T; }
  return s / n;
};

console.log(`Data: ${data.length} token latih, ${val.length} token validasi, kosakata ${tok.ukuran}, model ${model.ukuran.toLocaleString("id-ID")} parameter`);
console.log(`Loss awal yang diharapkan ≈ ln(${tok.ukuran}) = ${Math.log(tok.ukuran).toFixed(2)} (menebak acak)`);
console.log("Perhatikan: loss TRAIN terus turun, tetapi loss VAL berhenti turun lalu naik = model mulai MENGHAFAL.\n");
const mulai = performance.now();
let jumlah = 0;
for (let s = 1; s <= LANGKAH; s++) {
  G.fill(0);
  let loss = 0;
  for (let b = 0; b < B; b++) {
    const a = rng.int(data.length - T - 1);
    loss += model.langkah(data.subarray(a, a + T), data.subarray(a + 1, a + T + 1), G, 1 / (B * T));
  }
  jumlah += loss / (B * T);
  adam.langkah(G, s, jadwalLR(s, LANGKAH, 3e-3, 40));
  if (s % 50 === 0) {
    console.log(`langkah ${String(s).padStart(3)}  loss train ${(jumlah / 50).toFixed(3)}  loss val ${lossVal().toFixed(3)}  (${((performance.now() - mulai) / 1000).toFixed(0)} dtk)`);
    jumlah = 0;
  }
}

console.log("\nContoh teks (suhu 0.8, top-p 0.9):");
for (const awal of [" ibu", " anak anak", " setiap hari"]) {
  console.log(`  "${awal.trim()}${hasilkanTeks(model, tok, awal, 30, { suhu: 0.8, topP: 0.9, seed: 3 })}"`);
}
console.log("\nKorpus ini kecil (±1.500 token untuk ±85 ribu parameter), jadi model banyak menghafal.");
console.log("Solusinya data jauh lebih banyak:  bun run fetch-data  lalu  bun run siapkan-data -- --sumber wikipedia --nama wiki-v1 --huruf-asli, lalu bun run train-gpt -- --dataset wiki-v1 --name wiki");
