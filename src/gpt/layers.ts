/**
 * LAPISAN-LAPISAN TRANSFORMER, masing-masing dengan forward dan backward manual.
 *
 * Konvensi: X berukuran T×D (T token, D dimensi), row-major Float64Array.
 * Fungsi backward MENAMBAHKAN (+=) ke array gradien, karena gradien dari beberapa
 * jalur (mis. residual) harus dijumlahkan.
 *
 * Setiap rumus turunan di sini diverifikasi dengan gradient check numerik
 * (lihat src/commands/test-gpt.ts):
 *   ∂L/∂θ ≈ ( L(θ+ε) − L(θ−ε) ) / 2ε
 */
import { matmul, matmulAT, matmulBT, type Vec } from "./matrix.js";

// ============================================================================
// Positional encoding sinusoidal (tanpa parameter), sesuai "Attention Is All You Need":
//   PE(pos, 2i)   = sin(pos / 10000^(2i/D))
//   PE(pos, 2i+1) = cos(pos / 10000^(2i/D))
// Setiap posisi mendapat "sidik jari" gelombang dengan frekuensi berbeda-beda,
// sehingga model bisa membedakan urutan token. Untuk jarak k yang tetap,
// PE(pos+k) adalah rotasi linear dari PE(pos), jadi posisi relatif mudah dipelajari.
// ============================================================================
export function positionalEncoding(C: number, D: number): Vec {
  const pe = new Float64Array(C * D);
  for (let pos = 0; pos < C; pos++)
    for (let i = 0; i < D; i += 2) {
      const sudut = pos / Math.pow(10000, i / D);
      pe[pos * D + i] = Math.sin(sudut);
      if (i + 1 < D) pe[pos * D + i + 1] = Math.cos(sudut);
    }
  return pe;
}

// ============================================================================
// Linear:  Y = X·W + b      X: T×I, W: I×O, b: O, Y: T×O
//   ∂L/∂W = Xᵀ·∂L/∂Y      ∂L/∂b = Σ_t ∂L/∂Y_t      ∂L/∂X = ∂L/∂Y·Wᵀ
// ============================================================================
export function linearForward(X: Vec, T: number, I: number, O: number, P: Vec, wOff: number, bOff: number, Y: Vec): void {
  for (let t = 0; t < T; t++) for (let o = 0; o < O; o++) Y[t * O + o] = P[bOff + o];
  matmul(X, P, Y, T, I, O, 0, wOff, 0);
}

export function linearBackward(X: Vec, dY: Vec, T: number, I: number, O: number, P: Vec, G: Vec, wOff: number, bOff: number, dX: Vec | null): void {
  matmulAT(X, dY, G, T, I, O, 0, 0, wOff);
  for (let t = 0; t < T; t++) for (let o = 0; o < O; o++) G[bOff + o] += dY[t * O + o];
  if (dX) matmulBT(dY, P, dX, T, O, I, 0, wOff, 0);
}

// ============================================================================
// Layer Normalization (per baris):
//   μ = (1/D)Σx      σ² = (1/D)Σ(x−μ)²      x̂ = (x−μ)/√(σ²+ε)      y = γ·x̂ + β
// Backward (diturunkan dengan aturan rantai; r = 1/√(σ²+ε)):
//   ∂L/∂γ = Σ_t dy·x̂      ∂L/∂β = Σ_t dy      g = dy·γ
//   ∂L/∂x = (r/D) · ( D·g − Σg − x̂·Σ(g·x̂) )
// ============================================================================
export const LN_EPS = 1e-5;

export function layerNormForward(X: Vec, T: number, D: number, P: Vec, gOff: number, bOff: number, Y: Vec, Xhat: Vec, rstd: Vec): void {
  for (let t = 0; t < T; t++) {
    const o = t * D;
    let mu = 0;
    for (let d = 0; d < D; d++) mu += X[o + d];
    mu /= D;
    let v = 0;
    for (let d = 0; d < D; d++) v += (X[o + d] - mu) ** 2;
    const r = 1 / Math.sqrt(v / D + LN_EPS);
    rstd[t] = r;
    for (let d = 0; d < D; d++) {
      const xh = (X[o + d] - mu) * r;
      Xhat[o + d] = xh;
      Y[o + d] = P[gOff + d] * xh + P[bOff + d];
    }
  }
}

export function layerNormBackward(dY: Vec, T: number, D: number, P: Vec, G: Vec, gOff: number, bOff: number, Xhat: Vec, rstd: Vec, dX: Vec): void {
  for (let t = 0; t < T; t++) {
    const o = t * D;
    let sg = 0, sgx = 0;
    for (let d = 0; d < D; d++) {
      const dy = dY[o + d], xh = Xhat[o + d];
      G[gOff + d] += dy * xh;
      G[bOff + d] += dy;
      const g = dy * P[gOff + d];
      sg += g;
      sgx += g * xh;
    }
    const r = rstd[t] / D;
    for (let d = 0; d < D; d++) {
      const g = dY[o + d] * P[gOff + d];
      dX[o + d] += r * (D * g - sg - Xhat[o + d] * sgx);
    }
  }
}

// ============================================================================
// GELU (aproksimasi tanh, dipakai GPT-2):
//   gelu(x) = ½x(1 + tanh(u)),   u = √(2/π)(x + 0.044715x³)
//   gelu'(x) = ½(1 + tanh u) + ½x(1 − tanh²u)·√(2/π)(1 + 3·0.044715x²)
// ============================================================================
const SQRT_2_PI = Math.sqrt(2 / Math.PI);

export function geluForward(X: Vec, Y: Vec, n: number): void {
  for (let i = 0; i < n; i++) {
    const x = X[i];
    Y[i] = 0.5 * x * (1 + Math.tanh(SQRT_2_PI * (x + 0.044715 * x * x * x)));
  }
}

export function geluBackward(X: Vec, dY: Vec, dX: Vec, n: number): void {
  for (let i = 0; i < n; i++) {
    const x = X[i];
    const th = Math.tanh(SQRT_2_PI * (x + 0.044715 * x * x * x));
    const turunan = 0.5 * (1 + th) + 0.5 * x * (1 - th * th) * SQRT_2_PI * (1 + 3 * 0.044715 * x * x);
    dX[i] += dY[i] * turunan;
  }
}

// ============================================================================
// Causal Multi-Head Self-Attention (inti), setelah proyeksi Q, K, V (masing-masing T×D).
// D dibagi ke H head berdimensi d_h = D/H. Untuk setiap head:
//   S_ij = (Q_i · K_j) / √d_h                    skor: seberapa token i "memperhatikan" token j
//   S_ij = −∞ jika j > i                         CAUSAL: token tidak boleh melihat masa depan
//   P_i  = softmax(S_i)                          bobot perhatian, jumlahnya 1
//   O_i  = Σ_j P_ij · V_j                         campuran nilai dari token-token sebelumnya
// Backward:
//   dP_ij = dO_i · V_j            dV_j += Σ_i P_ij · dO_i
//   dS_ij = P_ij · (dP_ij − Σ_k P_ik dP_ik)          (Jacobian softmax)
//   dQ_i += Σ_j dS_ij · K_j / √d_h      dK_j += Σ_i dS_ij · Q_i / √d_h
// P disimpan (H×T×T) untuk backward.
// ============================================================================
export function attentionForward(Q: Vec, K: Vec, V: Vec, T: number, D: number, H: number, Pa: Vec, O: Vec): void {
  const dh = D / H, skala = 1 / Math.sqrt(dh);
  O.fill(0, 0, T * D);
  for (let h = 0; h < H; h++) {
    const c = h * dh, pOff = h * T * T;
    for (let i = 0; i < T; i++) {
      let maks = -Infinity;
      for (let j = 0; j <= i; j++) {
        let s = 0;
        for (let d = 0; d < dh; d++) s += Q[i * D + c + d] * K[j * D + c + d];
        s *= skala;
        Pa[pOff + i * T + j] = s;
        if (s > maks) maks = s;
      }
      let jumlah = 0;
      for (let j = 0; j <= i; j++) {
        const e = Math.exp(Pa[pOff + i * T + j] - maks); // dikurangi maks agar exp tidak meledak
        Pa[pOff + i * T + j] = e;
        jumlah += e;
      }
      for (let j = 0; j <= i; j++) {
        const p = Pa[pOff + i * T + j] / jumlah;
        Pa[pOff + i * T + j] = p;
        for (let d = 0; d < dh; d++) O[i * D + c + d] += p * V[j * D + c + d];
      }
      for (let j = i + 1; j < T; j++) Pa[pOff + i * T + j] = 0; // posisi masa depan
    }
  }
}

export function attentionBackward(Q: Vec, K: Vec, V: Vec, Pa: Vec, dO: Vec, T: number, D: number, H: number, dQ: Vec, dK: Vec, dV: Vec, dP: Vec): void {
  const dh = D / H, skala = 1 / Math.sqrt(dh);
  for (let h = 0; h < H; h++) {
    const c = h * dh, pOff = h * T * T;
    for (let i = 0; i < T; i++) {
      let titik = 0; // Σ_k P_ik dP_ik
      for (let j = 0; j <= i; j++) {
        const p = Pa[pOff + i * T + j];
        let dp = 0;
        for (let d = 0; d < dh; d++) {
          dp += dO[i * D + c + d] * V[j * D + c + d];
          dV[j * D + c + d] += p * dO[i * D + c + d];
        }
        dP[j] = dp;
        titik += p * dp;
      }
      for (let j = 0; j <= i; j++) {
        const ds = Pa[pOff + i * T + j] * (dP[j] - titik) * skala;
        if (ds === 0) continue;
        for (let d = 0; d < dh; d++) {
          dQ[i * D + c + d] += ds * K[j * D + c + d];
          dK[j * D + c + d] += ds * Q[i * D + c + d];
        }
      }
    }
  }
}

// ============================================================================
// Dropout (regularisasi): saat training, setiap elemen dimatikan dengan peluang p.
//   m_i = 0 (peluang p)  atau  1/(1−p) (peluang 1−p)       y = x ⊙ m       ∂L/∂x = ∂L/∂y ⊙ m
// Pembagi 1/(1−p) menjaga nilai harapan E[y] = x, jadi saat inferensi dropout cukup dimatikan.
// Model tidak bisa bergantung pada satu jalur tertentu -> lebih sulit menghafal (overfitting).
// ============================================================================
export function buatMaskDropout(mask: Vec, n: number, p: number, acak: () => number): void {
  const skala = 1 / (1 - p);
  for (let i = 0; i < n; i++) mask[i] = acak() < p ? 0 : skala;
}

export function kaliMask(X: Vec, mask: Vec, n: number): void {
  for (let i = 0; i < n; i++) X[i] *= mask[i];
}

// ============================================================================
// Softmax + Cross-Entropy (keluaran model), sama seperti pelajaran 1:
//   p = softmax(z)      L = −log p[target]      ∂L/∂z = p − onehot(target)
// `skala` = 1/(jumlah token dalam batch), karena loss dirata-rata.
// Target −1 = DIABAIKAN (masking): dipakai saat fine-tuning percakapan agar model
// hanya belajar dari jawaban bot, bukan dari kalimat pengguna.
// Mengembalikan jumlah −log p (belum dirata-rata).
// ============================================================================
export function softmaxCrossEntropy(Z: Vec, target: ArrayLike<number>, T: number, V: number, dZ: Vec | null, skala: number): number {
  let loss = 0;
  for (let t = 0; t < T; t++) {
    const o = t * V;
    if (target[t] < 0) {
      if (dZ) dZ.fill(0, o, o + V);
      continue;
    }
    let maks = -Infinity;
    for (let v = 0; v < V; v++) if (Z[o + v] > maks) maks = Z[o + v];
    let jumlah = 0;
    for (let v = 0; v < V; v++) jumlah += Math.exp(Z[o + v] - maks);
    const logJumlah = Math.log(jumlah) + maks;
    loss += logJumlah - Z[o + target[t]];
    if (dZ) {
      for (let v = 0; v < V; v++) dZ[o + v] = Math.exp(Z[o + v] - logJumlah) * skala;
      dZ[o + target[t]] -= skala;
    }
  }
  return loss;
}
