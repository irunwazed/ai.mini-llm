/**
 * LANGKAH 3: Tokenizer BPE (Byte Pair Encoding)
 *
 * Model bahasa tidak membaca huruf atau kata, melainkan TOKEN.
 *   - per huruf  : kosakata kecil, tetapi urutan sangat panjang (model harus "mengeja")
 *   - per kata   : urutan pendek, tetapi kosakata tak terbatas ("dicutikan", "pencutiannya", ...)
 * BPE berada di tengah: mulai dari huruf, lalu berulang kali menggabungkan pasangan
 * simbol yang PALING SERING muncul berdampingan.
 *
 * Jalankan:  bun run lesson:3
 */
import { readFileSync } from "node:fs";
import { latihBPE, praTokenisasi, Tokenizer } from "../src/gpt/bpe.js";

const teks = readFileSync(new URL("../data/corpus.txt", import.meta.url), "utf8").toLowerCase().replace(/\s+/g, " ");

// 1. Pra-tokenisasi: teks dipecah menjadi "kata" + spasi di depannya
console.log("1. Pra-tokenisasi (merge tidak boleh melintasi batas kata):");
console.log("  ", praTokenisasi(" ibu membeli bayam, wortel, dan tomat.").map((s) => JSON.stringify(s)).join(" "));

// 2. Latih BPE dan tampilkan merge pertama
const bpe = latihBPE(teks, 300);
console.log(`\n2. Kosakata awal = ${bpe.vocab.length - bpe.merges.length} karakter, setelah ${bpe.merges.length} merge = ${bpe.vocab.length} token`);
console.log("   15 merge pertama (pasangan paling sering):");
bpe.merges.slice(0, 15).forEach(([a, b], i) => console.log(`   ${String(i + 1).padStart(2)}. ${JSON.stringify(a)} + ${JSON.stringify(b)} -> ${JSON.stringify(a + b)}`));
console.log("   Token terpanjang yang terbentuk:", bpe.vocab.slice().sort((a, b) => b.length - a.length).slice(0, 8).map((s) => JSON.stringify(s)).join(" "));

// 3. Encode / decode
const tok = new Tokenizer(bpe);
const kalimat = " anak anak bermain bola di lapangan sekolah.";
const ids = tok.encode(kalimat);
console.log("\n3. Encode:", JSON.stringify(kalimat));
console.log("   token:", ids.map((i) => JSON.stringify(tok.vocab[i])).join(" "));
console.log("   id   :", ids.join(" "));
console.log("   decode kembali sama?", tok.decode(ids) === kalimat);

// 4. Kata yang belum pernah dilihat tetap bisa di-encode dari potongan yang dikenal
const baru = " pemrograman komputer";
console.log("\n4. Kata baru:", JSON.stringify(baru), "->", tok.encode(baru).map((i) => JSON.stringify(tok.vocab[i])).join(" "));

// 5. Kompresi: berapa token dibanding huruf?
const semua = tok.encode(teks);
console.log(`\n5. Seluruh korpus: ${teks.length} huruf -> ${semua.length} token (${(teks.length / semua.length).toFixed(2)} huruf per token)`);
console.log("   Urutan lebih pendek = konteks GPT yang sama memuat lebih banyak makna.");
