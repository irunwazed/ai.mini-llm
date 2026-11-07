/**
 * LANGKAH 1: Model Bahasa Bigram (dari nol, TypeScript murni, tanpa library)
 *
 * Tujuan LLM: memodelkan peluang token berikutnya.
 *   P(x_1, ..., x_T) = Π_t P(x_t | x_1, ..., x_{t-1})
 * Bigram = versi paling sederhana: token berikutnya hanya bergantung pada 1 token sebelumnya.
 *   P(x_t | x_1..x_{t-1}) ≈ P(x_t | x_{t-1})
 *
 * Jalankan:  bun run lesson:1
 */

// ---------------------------------------------------------------
// 0. UTILITAS: RNG ber-seed (agar hasil bisa diulang) + distribusi normal
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
const rand = mulberry32(42);

// Box-Muller: dua bilangan uniform -> satu bilangan normal N(0, 1)
function randn(): number {
  const u = 1 - rand();
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

type Matrix = number[][];
const zeros = (r: number, c: number): Matrix =>
  Array.from({ length: r }, () => new Array<number>(c).fill(0));

// ---------------------------------------------------------------
// 1. DATA & TOKENISASI (karakter -> angka)
// ---------------------------------------------------------------
const teks =
  "saya suka makan nasi goreng. saya suka minum teh manis. " +
  "kamu suka makan mie ayam. kami suka minum kopi susu. " +
  "dia suka makan sate ayam. mereka suka minum es teh. ";

const vocab = [...new Set(teks)].sort();
const V = vocab.length; // ukuran kosakata
const stoi = new Map(vocab.map((c, i) => [c, i] as const));
const itos = new Map(vocab.map((c, i) => [i, c] as const));
const data = [...teks].map((c) => stoi.get(c)!);

// pasangan (input, target): x_t -> x_{t+1}
const X = data.slice(0, -1);
const Y = data.slice(1);
const N = X.length;
console.log(`Ukuran vocab V = ${V}, jumlah pasangan N = ${N}`);

// ---------------------------------------------------------------
// 2. CARA A: HITUNG FREKUENSI (Maximum Likelihood Estimation)
//    P(b | a) = count(a, b) / Σ_b' count(a, b')
// ---------------------------------------------------------------
const C = zeros(V, V);
for (let i = 0; i < N; i++) C[X[i]][Y[i]] += 1;

const Pcount = C.map((row) => {
  const smoothed = row.map((v) => v + 1); // +1 = Laplace smoothing
  const total = smoothed.reduce((s, v) => s + v, 0);
  return smoothed.map((v) => v / total);
});

// Loss = Negative Log Likelihood rata-rata
//   L = -(1/N) Σ_i log P(y_i | x_i)
let nllCount = 0;
for (let i = 0; i < N; i++) nllCount -= Math.log(Pcount[X[i]][Y[i]]);
nllCount /= N;
console.log(`[Cara A: hitung]  NLL = ${nllCount.toFixed(4)}`);

// ---------------------------------------------------------------
// 3. CARA B: JARINGAN SARAF + GRADIENT DESCENT
//    Parameter W berukuran (V, V).
//    logits   z = onehot(x) · W  ==  W[x]        (ambil baris ke-x)
//    softmax  p_j = exp(z_j) / Σ_k exp(z_k)
//    loss     L = -log p_y                        (cross-entropy)
//    turunan  ∂L/∂z = p - onehot(y)
//    update   W <- W - lr · ∂L/∂W
// ---------------------------------------------------------------
function softmax(z: number[]): number[] {
  const m = Math.max(...z); // dikurangi max demi stabilitas numerik, hasil tidak berubah
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

function forwardBackward(W: Matrix): { loss: number; dW: Matrix } {
  const dW = zeros(V, V);
  let loss = 0;
  for (let i = 0; i < N; i++) {
    const p = softmax(W[X[i]]); // forward
    loss -= Math.log(p[Y[i]]);
    // backward: ∂L/∂z = (p - onehot(y)) / N
    for (let j = 0; j < V; j++) {
      const dz = (p[j] - (j === Y[i] ? 1 : 0)) / N;
      dW[X[i]][j] += dz; // z = W[x], jadi gradien z mengalir ke baris W[x]
    }
  }
  return { loss: loss / N, dW };
}

const W: Matrix = Array.from({ length: V }, () =>
  Array.from({ length: V }, () => 0.01 * randn()),
);
const lr = 5.0;

for (let step = 0; step <= 300; step++) {
  const { loss, dW } = forwardBackward(W);
  for (let a = 0; a < V; a++)
    for (let b = 0; b < V; b++) W[a][b] -= lr * dW[a][b];
  if (step % 50 === 0)
    console.log(`[Cara B: gradien] step ${String(step).padStart(3)}  loss = ${loss.toFixed(4)}`);
}

// ---------------------------------------------------------------
// 4. CEK GRADIEN SECARA NUMERIK (bukti bahwa rumus p - y benar)
//    ∂L/∂W_ij ≈ (L(W + ε) - L(W - ε)) / (2ε)
// ---------------------------------------------------------------
{
  const i = stoi.get("s")!;
  const j = stoi.get("a")!;
  const eps = 1e-5;
  const analitik = forwardBackward(W).dW[i][j];
  const orig = W[i][j];
  W[i][j] = orig + eps;
  const Lplus = forwardBackward(W).loss;
  W[i][j] = orig - eps;
  const Lminus = forwardBackward(W).loss;
  W[i][j] = orig;
  const numerik = (Lplus - Lminus) / (2 * eps);
  console.log(
    `Cek gradien W['s','a']: analitik=${analitik.toFixed(8)}  numerik=${numerik.toFixed(8)}`,
  );
}

// ---------------------------------------------------------------
// 5. SAMPLING: bangkitkan teks baru dari distribusi p
// ---------------------------------------------------------------
function sample(p: number[]): number {
  let r = rand();
  for (let k = 0; k < p.length; k++) {
    r -= p[k];
    if (r <= 0) return k;
  }
  return p.length - 1;
}

let idx = stoi.get("s")!;
let out = "s";
for (let t = 0; t < 80; t++) {
  idx = sample(softmax(W[idx]));
  out += itos.get(idx)!;
}
console.log("Contoh teks:", out);
