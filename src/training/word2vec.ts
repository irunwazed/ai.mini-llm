/**
 * WORD2VEC (Skip-gram + Negative Sampling) dari nol — jaringan saraf pertama di proyek ini.
 *
 * Ide: "kata dikenal dari teman-temannya". Kata yang sering muncul di konteks serupa
 * ("cuti" dan "izin", "pegawai" dan "PNS") akan mendapat vektor yang berdekatan.
 *
 * Setiap kata w punya DUA vektor berdimensi d:
 *   v_w  (vektor input / embedding)   u_w  (vektor output / konteks)
 *
 * Untuk pasangan (w, c) yang benar-benar bertetangga dalam jendela teks, kita ingin
 *   P(D=1 | w, c) = σ(v_w · u_c)  besar,        σ(x) = 1 / (1 + e^-x)
 * dan untuk k kata acak n_1..n_k (negative samples) kita ingin σ(v_w · u_n) kecil.
 *
 * Loss per pasangan:
 *   L = -log σ(v_w·u_c) - Σ_i log σ(-v_w·u_ni)
 *
 * Turunannya (sama bentuknya dengan softmax di pelajaran 1: "prediksi - label"):
 *   g = σ(v_w·u_x) - label        label = 1 untuk c, 0 untuk n_i
 *   ∂L/∂u_x = g · v_w
 *   ∂L/∂v_w = Σ_x g · u_x
 * Update SGD:  θ <- θ - lr · ∂L/∂θ,  lr turun linear dari lr0 ke ~0.
 *
 * Trik standar:
 *   - Negative sampling dari distribusi unigram^0.75 (kata langka sedikit dinaikkan peluangnya)
 *   - Subsampling kata sangat sering: dibuang dengan peluang 1 - sqrt(t / f(w))
 */
import { mulberry32 } from "../gpt/random.js";

export interface Embedding {
  dim: number;
  vocab: string[];
  vec: Float32Array; // vocab.length × dim, sudah dinormalisasi (panjang 1)
}

export interface OpsiW2V {
  dim?: number; window?: number; neg?: number; minCount?: number;
  lr0?: number; maksLangkahToken?: number; seed?: number; minToken?: number;
  progres?: (persen: number, loss: number) => void;
}

export function latihWord2Vec(kalimat: string[][], opsi: OpsiW2V = {}): Embedding | null {
  const dim = opsi.dim ?? 64, W = opsi.window ?? 5, K = opsi.neg ?? 5;
  const minCount = opsi.minCount ?? 3, lr0 = opsi.lr0 ?? 0.025;
  const rand = mulberry32(opsi.seed ?? 1);

  // 1. Kosakata
  const freq = new Map<string, number>();
  for (const s of kalimat) for (const w of s) freq.set(w, (freq.get(w) ?? 0) + 1);
  const vocab = [...freq].filter(([, c]) => c >= minCount).sort((a, b) => b[1] - a[1]).map(([w]) => w);
  const V = vocab.length;
  // Korpus terlalu kecil: vektor tidak akan bermakna dan justru mengacaukan peringkat
  if (V < 200 || [...freq.values()].reduce((a, b) => a + b, 0) < (opsi.minToken ?? 200000)) return null;
  const idx = new Map(vocab.map((w, i) => [w, i]));
  const hitung = vocab.map((w) => freq.get(w)!);
  const totalToken = hitung.reduce((a, b) => a + b, 0);
  const data = kalimat.map((s) => Int32Array.from(s.map((w) => idx.get(w) ?? -1).filter((i) => i >= 0)));

  // 2. Tabel negative sampling: unigram^0.75
  const UKURAN = 1 << 20;
  const tabel = new Int32Array(UKURAN);
  const pangkat = hitung.map((c) => Math.pow(c, 0.75));
  const z = pangkat.reduce((a, b) => a + b, 0);
  for (let i = 0, j = 0, kum = pangkat[0] / z; i < UKURAN; i++) {
    tabel[i] = j;
    if (i / UKURAN > kum && j < V - 1) kum += pangkat[++j] / z;
  }
  // Peluang mempertahankan kata (subsampling), t = 1e-4
  const simpan = hitung.map((c) => Math.min(1, Math.sqrt(1e-4 / (c / totalToken)) + 1e-4 / (c / totalToken)));

  // 3. Inisialisasi: v acak kecil, u nol
  const v = new Float32Array(V * dim).map(() => (rand() - 0.5) / dim);
  const u = new Float32Array(V * dim);
  const gradV = new Float32Array(dim);

  const epoch = Math.max(1, Math.min(15, Math.floor((opsi.maksLangkahToken ?? 1e7) / totalToken)));
  const totalLangkah = epoch * totalToken;
  let langkah = 0, lossJumlah = 0, lossN = 0, laporBerikut = 0;
  const sigmoid = (x: number) => (x > 8 ? 1 : x < -8 ? 0 : 1 / (1 + Math.exp(-x)));

  for (let e = 0; e < epoch; e++) {
    for (const s of data) {
      const kal = s.filter((w) => rand() < simpan[w]);
      for (let pos = 0; pos < kal.length; pos++) {
        langkah += s.length / Math.max(1, kal.length) || 1;
        const lr = Math.max(lr0 * 1e-4, lr0 * (1 - langkah / totalLangkah));
        const w = kal[pos], ow = w * dim;
        const b = 1 + Math.floor(rand() * W); // jendela dinamis 1..W
        for (let c = pos - b; c <= pos + b; c++) {
          if (c < 0 || c >= kal.length || c === pos) continue;
          gradV.fill(0);
          for (let k = 0; k <= K; k++) {
            const x = k === 0 ? kal[c] : tabel[(rand() * UKURAN) | 0];
            if (k > 0 && x === kal[c]) continue;
            const label = k === 0 ? 1 : 0, ox = x * dim;
            let dot = 0;
            for (let d = 0; d < dim; d++) dot += v[ow + d] * u[ox + d];
            const p = sigmoid(dot);
            lossJumlah -= Math.log(label ? p + 1e-7 : 1 - p + 1e-7); lossN++;
            const g = (p - label) * lr;               // (prediksi - label) · lr
            for (let d = 0; d < dim; d++) {
              gradV[d] += g * u[ox + d];              // kumpulkan ∂L/∂v_w
              u[ox + d] -= g * v[ow + d];             // u_x <- u_x - lr·g·v_w
            }
          }
          for (let d = 0; d < dim; d++) v[ow + d] -= gradV[d];
        }
      }
      const persen = (100 * langkah) / totalLangkah;
      if (opsi.progres && persen >= laporBerikut) {
        opsi.progres(Math.min(100, persen), lossJumlah / Math.max(1, lossN));
        lossJumlah = 0; lossN = 0; laporBerikut += 10;
      }
    }
  }

  // 4. Mean-centering: semua vektor word2vec cenderung condong ke satu arah bersama
  //    (arah "frekuensi"), sehingga kata yang tak berhubungan pun punya cos ≈ 0.9.
  //    Mengurangi rata-rata  v_w <- v_w - mean(v)  menghapus arah bersama itu.
  //    Lalu normalisasi panjang = 1 agar cosinus = dot product.
  const rata = new Float64Array(dim);
  for (let i = 0; i < V; i++) for (let d = 0; d < dim; d++) rata[d] += v[i * dim + d] / V;
  for (let i = 0; i < V; i++) for (let d = 0; d < dim; d++) v[i * dim + d] -= rata[d];
  for (let i = 0; i < V; i++) {
    let n = 0;
    for (let d = 0; d < dim; d++) n += v[i * dim + d] ** 2;
    n = Math.sqrt(n) || 1;
    for (let d = 0; d < dim; d++) v[i * dim + d] /= n;
  }
  return { dim, vocab, vec: v };
}

/** k tetangga terdekat (cosinus) dari sebuah kata. */
export function tetangga(emb: Embedding, idx: Map<string, number>, kata: string, k = 5): { kata: string; cos: number }[] {
  const i = idx.get(kata);
  if (i === undefined) return [];
  const { dim, vec, vocab } = emb;
  const hasil: { kata: string; cos: number }[] = [];
  for (let j = 0; j < vocab.length; j++) {
    if (j === i) continue;
    let dot = 0;
    for (let d = 0; d < dim; d++) dot += vec[i * dim + d] * vec[j * dim + d];
    if (hasil.length < k || dot > hasil[hasil.length - 1].cos) {
      hasil.push({ kata: vocab[j], cos: dot });
      hasil.sort((a, b) => b.cos - a.cos);
      if (hasil.length > k) hasil.pop();
    }
  }
  return hasil;
}
