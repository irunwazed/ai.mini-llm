/**
 * OPTIMIZER AdamW + gradient clipping + jadwal learning rate.
 *
 * Adam menyimpan dua rata-rata bergerak per parameter:
 *   m ← β1·m + (1−β1)·g          (momentum: arah rata-rata gradien)
 *   v ← β2·v + (1−β2)·g²         (skala: besar rata-rata gradien)
 * Koreksi bias (karena m, v mulai dari 0):
 *   m̂ = m / (1−β1^t)      v̂ = v / (1−β2^t)
 * Update:
 *   θ ← θ − lr · ( m̂ / (√v̂ + ε)  +  λ·θ )        λ·θ = weight decay (hanya matriks)
 * Pembagian dengan √v̂ membuat setiap parameter punya "langkah" yang ternormalisasi,
 * sehingga parameter jarang-diperbarui (mis. embedding token langka) tetap belajar.
 *
 * Gradient clipping: jika ‖g‖₂ > c, maka g ← g · c/‖g‖₂  (mencegah lonjakan yang merusak).
 *
 * Jadwal learning rate:
 *   warmup  : lr naik linear dari 0 ke lrMaks selama `warmup` langkah pertama
 *   cosine  : lalu turun mengikuti ½(1 + cos(π·progres)) sampai 10% lrMaks
 */
import type { Vec } from "./matrix.js";
import type { GPT } from "./model.js";

export interface OpsiAdam { lr: number; beta1?: number; beta2?: number; eps?: number; weightDecay?: number; clip?: number }

export class Adam {
  readonly m: Float64Array;
  readonly v: Float64Array;
  private readonly decay: Uint8Array;
  private readonly P: Vec;
  readonly o: Required<OpsiAdam>;

  constructor(model: GPT, opsi: OpsiAdam) {
    this.o = { beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0.01, clip: 1.0, ...opsi };
    this.P = model.P;
    this.m = new Float64Array(model.ukuran);
    this.v = new Float64Array(model.ukuran);
    this.decay = new Uint8Array(model.ukuran);
    for (const t of model.tensors().values()) if (t.decay) this.decay.fill(1, t.off, t.off + t.rows * t.cols);
  }

  /** Satu langkah update. `t` dimulai dari 1. Mengembalikan norma gradien sebelum clipping. */
  langkah(G: Vec, t: number, lr = this.o.lr): number {
    const { beta1: b1, beta2: b2, eps, weightDecay: wd, clip } = this.o;
    let norma = 0;
    for (let i = 0; i < G.length; i++) norma += G[i] * G[i];
    norma = Math.sqrt(norma);
    const k = clip > 0 && norma > clip ? clip / norma : 1;
    const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
    const P = this.P, m = this.m, v = this.v;
    for (let i = 0; i < P.length; i++) {
      const g = G[i] * k;
      m[i] = b1 * m[i] + (1 - b1) * g;
      v[i] = b2 * v[i] + (1 - b2) * g * g;
      let u = m[i] / c1 / (Math.sqrt(v[i] / c2) + eps);
      if (this.decay[i]) u += wd * P[i];
      P[i] -= lr * u;
    }
    return norma;
  }
}

export function jadwalLR(langkah: number, total: number, lrMaks: number, warmup: number): number {
  if (langkah <= warmup) return (lrMaks * langkah) / Math.max(1, warmup);
  const progres = Math.min(1, (langkah - warmup) / Math.max(1, total - warmup));
  return lrMaks * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * progres)));
}
