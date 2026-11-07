/**
 * LANGKAH 4: Causal Self-Attention dengan angka kecil
 *
 * Attention menjawab: "untuk menebak kata berikutnya, token mana yang perlu saya lihat?"
 * Setiap token i membuat tiga vektor dari dirinya:
 *   Query  q_i  : "saya sedang mencari apa?"
 *   Key    k_j  : "saya berisi apa?"
 *   Value  v_j  : "informasi yang saya berikan jika dipilih"
 *
 *   skor  S_ij = q_i · k_j / √d          (cocok-tidaknya pencarian i dengan isi j)
 *   mask  S_ij = −∞ untuk j > i          (CAUSAL: tidak boleh mengintip token masa depan)
 *   bobot P_i  = softmax(S_i)             (jumlahnya 1)
 *   hasil O_i  = Σ_j P_ij · v_j           (rata-rata berbobot dari Value)
 *
 * Jalankan:  bun run lesson:4
 */
import { attentionBackward, attentionForward } from "../src/gpt/layers.js";
import { Rng } from "../src/gpt/random.js";

const T = 4, D = 4, H = 1;
const kata = ["pegawai", "berhak", "atas", "cuti"];
const rng = new Rng(5);
const acak = () => Float64Array.from({ length: T * D }, () => Math.round(rng.normal() * 100) / 100);
const Q = acak(), K = acak(), V = acak();
const baris = (A: Float64Array, i: number, n = D) => "[" + Array.from(A.subarray(i * n, i * n + n)).map((x) => x.toFixed(2).padStart(5)).join(" ") + "]";

console.log("Token:", kata.join(" | "));
console.log("\n1. Q, K, V (biasanya hasil X·W_Q, X·W_K, X·W_V; di sini acak)");
kata.forEach((w, i) => console.log(`   ${w.padEnd(8)} q=${baris(Q, i)}  k=${baris(K, i)}  v=${baris(V, i)}`));

// 2. Skor & mask ditampilkan manual agar terlihat
console.log("\n2. Skor S = Q·Kᵀ/√d dengan mask causal (−∞ = tidak boleh dilihat)");
for (let i = 0; i < T; i++) {
  const s = [];
  for (let j = 0; j < T; j++) {
    if (j > i) { s.push("   −∞"); continue; }
    let d = 0;
    for (let k = 0; k < D; k++) d += Q[i * D + k] * K[j * D + k];
    s.push((d / Math.sqrt(D)).toFixed(2).padStart(5));
  }
  console.log(`   ${kata[i].padEnd(8)} ${s.join(" ")}`);
}

// 3. Softmax & keluaran memakai fungsi yang sama dengan GPT
const P = new Float64Array(H * T * T), O = new Float64Array(T * D);
attentionForward(Q, K, V, T, D, H, P, O);
console.log("\n3. Bobot perhatian P = softmax(S) (setiap baris berjumlah 1)");
kata.forEach((w, i) => console.log(`   ${w.padEnd(8)} ${baris(P, i, T)}   -> "${w}" paling memperhatikan "${kata[Array.from(P.subarray(i * T, i * T + T)).indexOf(Math.max(...P.subarray(i * T, i * T + i + 1)))]}"`));
console.log("\n4. Keluaran O = P·V (token pertama hanya bisa melihat dirinya sendiri)");
kata.forEach((w, i) => console.log(`   ${w.padEnd(8)} ${baris(O, i)}`));

// 5. Gradient check: misalkan L = Σ O ⊙ R (R acak). Bandingkan turunan analitik vs numerik untuk Q.
const R = acak();
const L = () => { attentionForward(Q, K, V, T, D, H, P, O); return O.reduce((s, o, i) => s + o * R[i], 0); };
L();
const dQ = new Float64Array(T * D), dK = new Float64Array(T * D), dV = new Float64Array(T * D);
attentionBackward(Q, K, V, P, R, T, D, H, dQ, dK, dV, new Float64Array(T));
console.log("\n5. Gradient check ∂L/∂Q (analitik dari rumus Jacobian softmax vs numerik):");
for (const idx of [5, 10, 14]) {
  const asli = Q[idx], eps = 1e-6;
  Q[idx] = asli + eps; const lp = L();
  Q[idx] = asli - eps; const lm = L();
  Q[idx] = asli;
  console.log(`   Q[${idx}]: analitik ${dQ[idx].toFixed(8)}  numerik ${((lp - lm) / (2 * eps)).toFixed(8)}`);
}
console.log("\nCatatan: Q[0..3] (token pertama) gradiennya 0, karena token pertama hanya melihat dirinya sendiri (P = 1).");
