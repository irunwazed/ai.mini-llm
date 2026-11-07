/**
 * UJI GPT: memastikan semua turunan yang ditulis tangan benar.
 *
 * 1. Gradient check: untuk sampel parameter θ dari SETIAP tensor,
 *      numerik  = ( L(θ+ε) − L(θ−ε) ) / 2ε
 *      analitik = hasil backward
 *      galat relatif = |a − n| / (|a| + |n|)   harus < 1e-5
 *    Diuji untuk post-LN (sesuai gambar) dan pre-LN, juga kepala GPT pembaca (span awal/akhir).
 * 2. Uji overfit: model harus bisa menghafal satu kalimat (loss -> hampir 0).
 *    Jika gagal, ada yang salah di forward/backward/optimizer.
 * 3. Uji overfit pembaca: model harus bisa belajar menunjuk rentang jawaban di satu contoh.
 *
 * Pakai:  bun run test-gpt
 */
import { latihBPE, Tokenizer } from "../gpt/bpe.js";
import { GPT, type GPTConfig } from "../gpt/model.js";
import { Adam } from "../gpt/optimizer.js";
import { Rng } from "../gpt/random.js";
import { sampel } from "../gpt/sampling.js";

let gagal = 0;

function gradientCheck(preLN: boolean, dropout = false, masking = false, span = false) {
  const c: GPTConfig = { vocab: 13, context: 7, dim: 8, heads: 2, layers: 2, ff: 16, preLN, spanHead: span };
  const m = new GPT(c, undefined, 7);
  const rng = new Rng(3);
  const T = 6;
  const ids = Array.from({ length: T }, () => rng.int(c.vocab));
  // masking: target −1 diabaikan (seperti token pengguna saat fine-tuning percakapan)
  const target = Array.from({ length: T }, (_, t) => (masking && t % 2 === 0 ? -1 : rng.int(c.vocab)));
  const G = new Float64Array(m.ukuran);
  if (dropout) m.aturDropout(0.3, new Rng(9).next); // buat mask acak sekali...
  // span: posisi 0–1 = pertanyaan, 2 = <|bot|> ("tidak ada jawaban"), jawaban = token 3..5
  const hitung = (g: Float64Array | null, skala: number) => (span ? m.langkahSpan(ids, 3, 5, 2, g, skala) : m.langkah(ids, target, g, skala));
  hitung(G, 1 / T);
  if (dropout) m.aturDropout(0.3, undefined, true); // ...lalu bekukan agar L(θ±ε) memakai mask yang sama
  const lossRata = () => hitung(null, 1) / T;

  let terburuk = 0, namaTerburuk = "";
  for (const [nama, t] of m.tensors()) {
    const n = t.rows * t.cols;
    for (let s = 0; s < Math.min(4, n); s++) {
      const i = t.off + rng.int(n);
      const asli = m.P[i], eps = 1e-5;
      m.P[i] = asli + eps; const lp = lossRata();
      m.P[i] = asli - eps; const lm = lossRata();
      m.P[i] = asli;
      const num = (lp - lm) / (2 * eps), ana = G[i];
      if (Math.abs(num) < 1e-9 && Math.abs(ana) < 1e-9) continue;
      const rel = Math.abs(ana - num) / (Math.abs(ana) + Math.abs(num));
      if (rel > terburuk) { terburuk = rel; namaTerburuk = nama; }
    }
  }
  const ok = terburuk < 1e-5;
  if (!ok) gagal++;
  const label = `${preLN ? "pre-LN " : "post-LN"}${dropout ? " + dropout" : ""}${masking ? " + masking" : ""}${span ? " + pembaca" : ""}`;
  console.log(`  ${label.padEnd(28)}: galat relatif terburuk = ${terburuk.toExponential(2)} (${namaTerburuk})  ${ok ? "LULUS" : "GAGAL"}`);
}

function ujiOverfit(preLN: boolean) {
  const kalimat = " pegawai berhak atas cuti tahunan selama dua belas hari kerja.";
  const tok = new Tokenizer(latihBPE(kalimat, 40));
  const ids = tok.encode(kalimat);
  const c: GPTConfig = { vocab: tok.ukuran, context: 64, dim: 32, heads: 4, layers: 2, ff: 64, preLN };
  const m = new GPT(c, undefined, 1);
  const G = new Float64Array(m.ukuran);
  const opt = new Adam(m, { lr: 3e-3, weightDecay: 0 });
  const x = ids.slice(0, -1), y = ids.slice(1);
  let loss = 0;
  for (let s = 1; s <= 300; s++) {
    G.fill(0);
    loss = m.langkah(x, y, G, 1 / x.length) / x.length;
    opt.langkah(G, s);
  }
  // Generate secara greedy dari token pertama
  const keluar = [ids[0]];
  const rng = new Rng(1);
  while (keluar.length < ids.length) {
    m.forward(keluar);
    const V = c.vocab, z = m.logits.subarray((keluar.length - 1) * V, keluar.length * V);
    keluar.push(sampel(z, { suhu: 0.01, topK: 1 }, rng));
  }
  const teks = tok.decode(keluar);
  const ok = loss < 0.05 && teks === kalimat;
  if (!ok) gagal++;
  console.log(`  ${preLN ? "pre-LN " : "post-LN"}: loss akhir = ${loss.toFixed(4)}, hafal = ${teks === kalimat}  ${ok ? "LULUS" : "GAGAL"}`);
  if (teks !== kalimat) console.log(`     hasil: "${teks}"`);
}

function ujiOverfitPembaca() {
  const c: GPTConfig = { vocab: 20, context: 16, dim: 32, heads: 4, layers: 2, ff: 64, preLN: false, spanHead: true };
  const m = new GPT(c, undefined, 2);
  const G = new Float64Array(m.ukuran);
  const opt = new Adam(m, { lr: 3e-3, weightDecay: 0 });
  const ids = [1, 2, 3, 0, 5, 6, 7, 8, 9, 10, 11, 12]; // 0 = <|bot|> di posisi 3
  let loss = 0;
  for (let s = 1; s <= 200; s++) { G.fill(0); loss = m.langkahSpan(ids, 6, 8, 3, G, 1); opt.langkah(G, s); }
  const Z = m.forwardSpan(ids);
  const argmaks = (kol: number) => { let b = 3; for (let t = 3; t < ids.length; t++) if (Z[t * 2 + kol] > Z[b * 2 + kol]) b = t; return b; };
  const ok = loss < 0.05 && argmaks(0) === 6 && argmaks(1) === 8;
  if (!ok) gagal++;
  console.log(`  pembaca: loss akhir = ${loss.toFixed(4)}, rentang = ${argmaks(0)}..${argmaks(1)} (target 6..8)  ${ok ? "LULUS" : "GAGAL"}`);
}

console.log("1. Gradient check (setiap tensor, Float64):");
gradientCheck(false);
gradientCheck(true);
gradientCheck(false, true);
gradientCheck(true, true);
gradientCheck(false, false, true);
gradientCheck(false, false, false, true);
gradientCheck(true, true, false, true);
console.log("2. Uji overfit satu kalimat (300 langkah Adam):");
ujiOverfit(false);
ujiOverfit(true);
console.log("3. Uji overfit GPT pembaca (200 langkah Adam):");
ujiOverfitPembaca();
console.log(gagal ? `\n${gagal} uji GAGAL` : "\nSemua uji LULUS");
process.exit(gagal ? 1 : 0);
