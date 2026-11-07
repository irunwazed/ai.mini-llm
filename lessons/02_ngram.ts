/**
 * LANGKAH 2: Model N-gram + Temperature (TypeScript murni, tanpa library)
 *
 * Bigram hanya melihat 1 huruf sebelumnya. Di sini kita perpanjang konteksnya:
 *   P(x_t | x_1..x_{t-1}) ≈ P(x_t | x_{t-n}..x_{t-1})      (n = panjang konteks)
 *
 * Tiga hal yang dipelajari:
 *   1. Konteks lebih panjang  -> tebakan lebih tajam (entropi turun).
 *   2. Konteks terlalu panjang -> banyak konteks tak pernah terlihat (data sparsity),
 *      sehingga loss di data validasi justru naik. Inilah alasan kita butuh jaringan saraf.
 *   3. Temperature T mengatur seberapa "berani" model saat sampling.
 *
 * Jalankan:  bun run lesson:2
 */
import { readFileSync } from "node:fs";

// ---------------------------------------------------------------
// 0. RNG ber-seed
// ---------------------------------------------------------------
function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(7);

// ---------------------------------------------------------------
// 1. DATA: korpus Indonesia, dibagi train (90%) dan validasi (10%)
//    Validasi = teks yang TIDAK pernah dilihat model saat "belajar".
// ---------------------------------------------------------------
const teks = readFileSync(new URL("../data/corpus.txt", import.meta.url), "utf8")
  .toLowerCase()
  .replace(/\s+/g, " ");

const vocab = [...new Set(teks)].sort();
const V = vocab.length;
const stoi = new Map(vocab.map((c, i) => [c, i] as const));

const split = Math.floor(teks.length * 0.9);
const train = teks.slice(0, split);
const val = teks.slice(split);
console.log(`Vocab V = ${V} | train = ${train.length} huruf | val = ${val.length} huruf\n`);

// ---------------------------------------------------------------
// 2. MODEL N-GRAM: tabel hitungan  count(konteks, huruf berikutnya)
// ---------------------------------------------------------------
type Counts = Map<string, number[]>;

function fit(text: string, n: number): Counts {
  const counts: Counts = new Map();
  for (let t = n; t < text.length; t++) {
    const ctx = text.slice(t - n, t);
    let row = counts.get(ctx);
    if (!row) counts.set(ctx, (row = new Array<number>(V).fill(0)));
    row[stoi.get(text[t])!] += 1;
  }
  return counts;
}

// Add-k smoothing:  P(b | ctx) = (count(ctx,b) + k) / (Σ count(ctx,·) + k·V)
// Jika konteks tak pernah terlihat, hasilnya jatuh ke distribusi seragam 1/V.
function prob(counts: Counts, ctx: string, ch: string, k: number): number {
  const row = counts.get(ctx);
  const c = row ? row[stoi.get(ch)!] : 0;
  const total = row ? row.reduce((s, v) => s + v, 0) : 0;
  return (c + k) / (total + k * V);
}

// NLL rata-rata dan perplexity = exp(NLL).
// Perplexity bisa dibaca sebagai "model ragu di antara sekian huruf".
function evaluate(counts: Counts, text: string, n: number, k: number) {
  let nll = 0;
  let m = 0;
  for (let t = n; t < text.length; t++) {
    nll -= Math.log(prob(counts, text.slice(t - n, t), text[t], k));
    m++;
  }
  nll /= m;
  return { nll, ppl: Math.exp(nll) };
}

console.log("n  | NLL train | NLL val | perplexity val | konteks unik");
console.log("---+-----------+---------+----------------+-------------");
const k = 0.1;
for (let n = 1; n <= 7; n++) {
  const counts = fit(train, n);
  const tr = evaluate(counts, train, n, k);
  const va = evaluate(counts, val, n, k);
  console.log(
    `${n}  |  ${tr.nll.toFixed(4)}   | ${va.nll.toFixed(4)}  |     ${va.ppl.toFixed(2).padStart(6)}     | ${counts.size}`,
  );
}

// ---------------------------------------------------------------
// 3. SAMPLING dengan BACKOFF dan TEMPERATURE
//
//    Backoff: jika konteks panjang belum pernah terlihat, pendekkan konteksnya.
//
//    Temperature T:   p_j(T) = p_j^(1/T) / Σ_k p_k^(1/T)
//    Setara dengan softmax(z / T) pada logits z = log p.
//      T < 1  -> distribusi lebih tajam (aman, tapi bisa berulang-ulang)
//      T = 1  -> distribusi asli
//      T > 1  -> distribusi lebih rata (kreatif, tapi makin banyak salah eja)
// ---------------------------------------------------------------
const models = Array.from({ length: 8 }, (_, n) => fit(teks, n)); // pakai seluruh korpus

function nextDist(ctx: string, n: number): number[] {
  for (let m = Math.min(n, ctx.length); m >= 0; m--) {
    const row = models[m].get(ctx.slice(ctx.length - m));
    if (row) return row;
  }
  throw new Error("tidak mungkin: konteks kosong selalu ada");
}

function sampleWithTemperature(counts: number[], T: number): number {
  const w = counts.map((c) => (c > 0 ? Math.pow(c, 1 / T) : 0));
  const s = w.reduce((a, b) => a + b, 0);
  let r = rand() * s;
  for (let j = 0; j < V; j++) {
    r -= w[j];
    if (r <= 0) return j;
  }
  return V - 1;
}

function generate(prompt: string, n: number, T: number, len = 160): string {
  let out = prompt;
  for (let t = 0; t < len; t++) {
    const j = sampleWithTemperature(nextDist(out, n), T);
    out += vocab[j];
  }
  return out;
}

console.log("\n=== Pengaruh panjang konteks (T = 1.0) ===");
for (const n of [1, 2, 3, 4, 5]) console.log(`n=${n}: ${generate("ibu ", n, 1.0)}\n`);

console.log("=== Pengaruh temperature (n = 4) ===");
for (const T of [0.5, 1.0, 1.5]) console.log(`T=${T}: ${generate("anak ", 4, T)}\n`);

// ---------------------------------------------------------------
// 4. CEK: apakah model hanya MENYALIN korpus?
//    Hitung persentase potongan 20 huruf hasil generate yang ada persis di korpus.
// ---------------------------------------------------------------
function copyRate(sample: string, w = 20): number {
  let hit = 0;
  let total = 0;
  for (let i = 0; i + w <= sample.length; i++, total++) if (teks.includes(sample.slice(i, i + w))) hit++;
  return (100 * hit) / total;
}
console.log("=== Seberapa banyak model menyalin korpus? (potongan 20 huruf) ===");
for (const n of [3, 5, 7]) {
  const s = generate("kami ", n, 1.0, 400);
  console.log(`n=${n}: ${copyRate(s).toFixed(1)}% potongan identik dengan korpus`);
}
