/**
 * UJI BACKEND GPU (jalankan di server ber-GPU setelah `bun run build-gpu`):
 *
 * 1. CPU vs GPU: loss & gradien SETIAP tensor harus sama dengan versi CPU (yang sudah lulus gradient check).
 *    GPU memakai float32, jadi toleransi: loss < 1e-4, gradien (galat relatif per tensor) < 2e-3.
 *    Diuji post-LN & pre-LN, dengan masking (target −1) dan urutan berbeda panjang (padding).
 * 2. Satu langkah AdamW di GPU = di CPU.
 * 3. Dropout: loss berbeda per seed, tetap wajar.
 * 4. Overfit: GPU harus bisa menghafal satu kalimat.
 * 5. Benchmark: token/detik GPU vs 1 thread CPU untuk preset "besar".
 *
 * Pakai:  bun run test-gpu
 */
import { GPT, type GPTConfig } from "../gpt/model.js";
import { Adam } from "../gpt/optimizer.js";
import { Rng } from "../gpt/random.js";
import { GPTCuda, gpuTersedia, type UrutanGPU } from "../gpu/cuda.js";

if (!gpuTersedia()) { console.error("Backend GPU belum dikompilasi. Jalankan dulu: bun run build-gpu"); process.exit(1); }
let gagal = 0;
const lapor = (ok: boolean, teks: string) => { if (!ok) gagal++; console.log(`  ${teks}  ${ok ? "LULUS" : "GAGAL"}`); };

// Satu proses hanya boleh punya satu model GPU -> setiap uji membuat model, memakai, lalu menutupnya.
function denganGPU<T>(model: GPT, bMaks: number, f: (g: GPTCuda) => T): T {
  const g = new GPTCuda(model, bMaks);
  try { return f(g); } finally { g.tutup(); }
}

function urutanAcak(c: GPTConfig, rng: Rng, panjang: number[], masking: boolean): UrutanGPU[] {
  return panjang.map((T) => ({
    x: Int32Array.from({ length: T }, () => rng.int(c.vocab)),
    y: Int32Array.from({ length: T }, (_, t) => (masking && t % 3 === 0 ? -1 : rng.int(c.vocab))),
  }));
}

console.log("1. CPU vs GPU (loss & gradien setiap tensor)");
for (const preLN of [false, true]) for (const masking of [false, true]) {
  const c: GPTConfig = { vocab: 37, context: 24, dim: 32, heads: 4, layers: 2, ff: 64, preLN };
  const cpu = new GPT(c, undefined, 5);
  const rng = new Rng(11);
  const urutan = urutanAcak(c, rng, [24, 17, 9], masking); // panjang berbeda -> padding di GPU
  const nTarget = urutan.reduce((s, u) => s + Array.from(u.y).filter((v) => v >= 0).length, 0), skala = 1 / nTarget;
  const Gc = new Float64Array(cpu.ukuran);
  let lc = 0;
  for (const u of urutan) lc += cpu.langkah(u.x, u.y, Gc, skala);
  const Gg = new Float64Array(cpu.ukuran);
  const lg = denganGPU(cpu, 4, (g) => { g.nolGrad(); const l = g.langkah(urutan, skala, 0, 1); g.ambilGrad(Gg); return l; });
  const relLoss = Math.abs(lc - lg) / Math.abs(lc);
  let terburuk = 0, namaT = "";
  for (const [nama, t] of cpu.tensors()) {
    let selisih = 0, besar = 0;
    for (let i = t.off; i < t.off + t.rows * t.cols; i++) { selisih += (Gc[i] - Gg[i]) ** 2; besar += Gc[i] ** 2; }
    const rel = Math.sqrt(selisih) / (Math.sqrt(besar) + 1e-12);
    if (besar > 1e-20 && rel > terburuk) { terburuk = rel; namaT = nama; }
  }
  lapor(relLoss < 1e-4 && terburuk < 2e-3, `${preLN ? "pre-LN " : "post-LN"}${masking ? " + masking" : "          "}: loss CPU ${lc.toFixed(5)} GPU ${lg.toFixed(5)} (rel ${relLoss.toExponential(1)}), gradien terburuk ${terburuk.toExponential(2)} (${namaT})`);
}

console.log("2. Satu langkah AdamW");
{
  const c: GPTConfig = { vocab: 29, context: 16, dim: 32, heads: 2, layers: 2, ff: 48, preLN: true };
  const cpu = new GPT(c, undefined, 3);
  const P0 = Float64Array.from(cpu.P);
  const urutan = urutanAcak(c, new Rng(4), [16, 16], false);
  const skala = 1 / 32;
  // Kedua Adam diberi gradien YANG SAMA (gradien GPU). Jika gradien CPU dipakai, selisih float32 vs float64 pada
  // gradien yang hampir nol diperbesar oleh m/(√v+ε) — itu sifat Adam, bukan kesalahan kernel.
  const Pg = new Float64Array(cpu.ukuran), G = new Float64Array(cpu.ukuran);
  denganGPU(cpu, 2, (g) => { g.nolGrad(); g.langkah(urutan, skala, 0, 1); g.ambilGrad(G); g.adam(1, 3e-3, { beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0.1, clip: 1 }); g.ambilParam(Pg); });
  for (let i = 0; i < G.length; i++) G[i] = Math.fround(G[i]);
  new Adam(cpu, { lr: 3e-3, weightDecay: 0.1 }).langkah(G, 1, 3e-3);
  let maks = 0, gerak = 0;
  for (let i = 0; i < cpu.ukuran; i++) { maks = Math.max(maks, Math.abs(cpu.P[i] - Pg[i])); gerak = Math.max(gerak, Math.abs(cpu.P[i] - P0[i])); }
  lapor(maks < 1e-5 && gerak > 1e-4, `selisih parameter terbesar ${maks.toExponential(2)} (langkahnya sendiri ${gerak.toExponential(2)})`);
}

console.log("3. Dropout");
{
  const c: GPTConfig = { vocab: 31, context: 16, dim: 32, heads: 4, layers: 2, ff: 64, preLN: true };
  const cpu = new GPT(c, undefined, 9);
  const urutan = urutanAcak(c, new Rng(2), [16, 16, 16], false);
  const [l0, l1, l2] = denganGPU(cpu, 3, (g) => { g.nolGrad(); return [g.langkah(urutan, 1, 0, 1), g.langkah(urutan, 1, 0.2, 1), g.langkah(urutan, 1, 0.2, 2)]; });
  lapor(l1 !== l2 && Number.isFinite(l1) && Math.abs(l1 - l0) / l0 < 0.5, `loss tanpa dropout ${l0.toFixed(3)}, dropout seed 1 ${l1.toFixed(3)}, seed 2 ${l2.toFixed(3)}`);
}

console.log("4. Overfit satu kalimat di GPU");
{
  const c: GPTConfig = { vocab: 20, context: 16, dim: 32, heads: 4, layers: 2, ff: 64, preLN: true };
  const cpu = new GPT(c, undefined, 1);
  const kal = Int32Array.from({ length: 16 }, (_, i) => (i * 7 + 3) % 20);
  const u = [{ x: kal.subarray(0, 15), y: kal.subarray(1, 16) }];
  const akhir = denganGPU(cpu, 1, (g) => {
    let l = 0;
    for (let t = 1; t <= 300; t++) { g.nolGrad(); l = g.langkah(u, 1 / 15, 0, t) / 15; g.adam(t, 1e-2, { beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0, clip: 1 }); }
    return l;
  });
  lapor(akhir < 0.05, `loss akhir ${akhir.toFixed(4)}`);
}

console.log("5. Benchmark preset besar (dim 128, 6 lapisan, konteks 256, kosakata 4000)");
{
  const c: GPTConfig = { vocab: 4000, context: 256, dim: 128, heads: 4, layers: 6, ff: 512, preLN: true };
  const cpu = new GPT(c, undefined, 1);
  const rng = new Rng(7);
  const B = Number(process.env.BATCH ?? 64);
  const urutan = urutanAcak(c, rng, Array(B).fill(256), false);
  const tg = denganGPU(cpu, B, (g) => {
    console.log(`  GPU: ${g.info}`);
    g.nolGrad(); g.langkah(urutan, 1 / (B * 256), 0.1, 0); // pemanasan
    const t0 = performance.now();
    for (let i = 0; i < 5; i++) { g.nolGrad(); g.langkah(urutan, 1 / (B * 256), 0.1, i + 1); g.adam(i + 1, 1e-3, { beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0.1, clip: 1 }); }
    return (performance.now() - t0) / 5;
  });
  const G = new Float64Array(cpu.ukuran);
  const t1 = performance.now();
  cpu.langkah(urutan[0].x, urutan[0].y, G, 1 / 256);
  const tc = performance.now() - t1;
  const tokGPU = (B * 256) / (tg / 1000), tokCPU = 256 / (tc / 1000);
  console.log(`  GPU: ${tg.toFixed(0)} ms/langkah (batch ${B}) = ${Math.round(tokGPU).toLocaleString("id-ID")} token/dtk`);
  console.log(`  CPU 1 thread: ${tc.toFixed(0)} ms/urutan = ${Math.round(tokCPU).toLocaleString("id-ID")} token/dtk  -> GPU ±${Math.round(tokGPU / tokCPU)}× satu thread CPU`);
}

console.log(gagal ? `\n${gagal} uji GAGAL` : "\nSemua uji GPU LULUS");
process.exit(gagal ? 1 : 0);
