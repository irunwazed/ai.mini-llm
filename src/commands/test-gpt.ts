/**
 * UJI GPT: memastikan semua turunan yang ditulis tangan benar.
 *
 * 1. Gradient check: untuk sampel parameter θ dari SETIAP tensor,
 *      numerik  = ( L(θ+ε) − L(θ−ε) ) / 2ε
 *      analitik = hasil backward
 *      galat relatif = |a − n| / (|a| + |n|)   harus < 1e-5
 *    Diuji untuk post-LN (sesuai gambar) dan pre-LN.
 * 2. Uji overfit: model harus bisa menghafal satu kalimat (loss -> hampir 0).
 *    Jika gagal, ada yang salah di forward/backward/optimizer.
 *
 * Pakai:  bun run test-gpt
 */
import { latihBPE, Tokenizer } from "../gpt/bpe.js";
import { GPT, type GPTConfig } from "../gpt/model.js";
import { Adam } from "../gpt/optimizer.js";
import { Rng } from "../gpt/random.js";
import { sampel } from "../gpt/sampling.js";

let gagal = 0;

function gradientCheck(preLN: boolean, dropout = false, masking = false) {
  const c: GPTConfig = { vocab: 13, context: 7, dim: 8, heads: 2, layers: 2, ff: 16, preLN };
  const m = new GPT(c, undefined, 7);
  const rng = new Rng(3);
  const T = 6;
  const ids = Array.from({ length: T }, () => rng.int(c.vocab));
  // masking: target −1 diabaikan (seperti token pengguna saat fine-tuning percakapan)
  const target = Array.from({ length: T }, (_, t) => (masking && t % 2 === 0 ? -1 : rng.int(c.vocab)));
  const G = new Float64Array(m.ukuran);
  if (dropout) m.aturDropout(0.3, new Rng(9).next); // buat mask acak sekali...
  m.langkah(ids, target, G, 1 / T);
  if (dropout) m.aturDropout(0.3, undefined, true); // ...lalu bekukan agar L(θ±ε) memakai mask yang sama
  const lossRata = () => m.langkah(ids, target, null, 1) / T;

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
  const label = `${preLN ? "pre-LN " : "post-LN"}${dropout ? " + dropout" : ""}${masking ? " + masking" : ""}`;
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

console.log("1. Gradient check (setiap tensor, Float64):");
gradientCheck(false);
gradientCheck(true);
gradientCheck(false, true);
gradientCheck(true, true);
gradientCheck(false, false, true);
console.log("2. Uji overfit satu kalimat (300 langkah Adam):");
ujiOverfit(false);
ujiOverfit(true);
console.log(gagal ? `\n${gagal} uji GAGAL` : "\nSemua uji LULUS");
process.exit(gagal ? 1 : 0);
