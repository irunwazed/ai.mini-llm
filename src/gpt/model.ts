/**
 * GPT (decoder-only transformer) dari nol, mengikuti diagram:
 *
 *   token ──► Embedding ──(+)── Positional Encoding (sinusoidal)
 *                          │
 *        ┌─────────────────▼──────────────────┐
 *   N×   │ Causal Multi-Head Attention          │
 *        │ Add & Layer Norm    h = LN1(x + A(x)) │   <- post-LN (default, sesuai gambar)
 *        │ Feed Forward                         │
 *        │ Add & Layer Norm    y = LN2(h + F(h)) │
 *        └─────────────────┬──────────────────┘
 *                     Linear ──► Softmax ──► peluang token berikutnya
 *
 * Opsi pre-LN (seperti GPT-2 / repo w3cj):
 *   h = x + A(LN1(x)),   y = h + F(LN2(h)),   lalu Layer Norm akhir sebelum Linear.
 * Pre-LN biasanya lebih stabil untuk model yang dalam.
 *
 * Semua parameter disimpan dalam SATU Float64Array; setiap tensor hanya punya offset.
 * Optimizer, checkpoint, dan worker paralel cukup memegang satu array itu.
 */
import {
  attentionBackward, attentionForward, buatMaskDropout, geluBackward, kaliMask, geluForward, layerNormBackward, layerNormForward,
  linearBackward, linearForward, positionalEncoding, softmaxCrossEntropy,
} from "./layers.js";
import type { Vec } from "./matrix.js";
import { Rng } from "./random.js";

export interface GPTConfig {
  vocab: number;   // ukuran kosakata BPE
  context: number; // panjang konteks maksimum (token)
  dim: number;     // dimensi model D
  heads: number;   // jumlah head attention H (D harus habis dibagi H)
  layers: number;  // jumlah blok N
  ff: number;      // dimensi tersembunyi feed forward
  preLN: boolean;  // false = post-LN (sesuai gambar)
}

export interface TensorInfo { off: number; rows: number; cols: number; decay: boolean }

/** Menyusun letak setiap tensor di dalam array parameter. */
export function susunParameter(c: GPTConfig): { tensor: Map<string, TensorInfo>; ukuran: number } {
  const tensor = new Map<string, TensorInfo>();
  let off = 0;
  const tambah = (nama: string, rows: number, cols: number, decay: boolean) => {
    tensor.set(nama, { off, rows, cols, decay });
    off += rows * cols;
  };
  const D = c.dim, F = c.ff;
  tambah("tok", c.vocab, D, true);
  for (let l = 0; l < c.layers; l++) {
    for (const p of ["q", "k", "v", "o"]) { tambah(`l${l}.w${p}`, D, D, true); tambah(`l${l}.b${p}`, 1, D, false); }
    tambah(`l${l}.ln1g`, 1, D, false); tambah(`l${l}.ln1b`, 1, D, false);
    tambah(`l${l}.w1`, D, F, true); tambah(`l${l}.b1`, 1, F, false);
    tambah(`l${l}.w2`, F, D, true); tambah(`l${l}.b2`, 1, D, false);
    tambah(`l${l}.ln2g`, 1, D, false); tambah(`l${l}.ln2b`, 1, D, false);
  }
  if (c.preLN) { tambah("lnfg", 1, D, false); tambah("lnfb", 1, D, false); }
  tambah("wout", D, c.vocab, true);
  tambah("bout", 1, c.vocab, false);
  return { tensor, ukuran: off };
}

/**
 * Inisialisasi:
 *   embedding token  ~ N(0, 1/D)        (dikali √D saat dipakai -> skala ≈ 1, sebanding dengan PE)
 *   matriks bobot    ~ N(0, 2/(in+out)) Xavier/Glorot: menjaga variansi sinyal antar lapisan
 *   bias = 0,  γ (Layer Norm) = 1,  β = 0
 */
export function inisialisasi(c: GPTConfig, P: Vec, seed = 1): void {
  const rng = new Rng(seed);
  const { tensor } = susunParameter(c);
  for (const [nama, t] of tensor) {
    const n = t.rows * t.cols;
    if (nama === "tok") for (let i = 0; i < n; i++) P[t.off + i] = rng.normal() / Math.sqrt(c.dim);
    else if (/ln\dg$|lnfg$/.test(nama)) P.fill(1, t.off, t.off + n);
    else if (t.rows > 1) {
      const std = Math.sqrt(2 / (t.rows + t.cols));
      for (let i = 0; i < n; i++) P[t.off + i] = rng.normal() * std;
    } else P.fill(0, t.off, t.off + n);
  }
}

/** Nilai antara satu blok yang disimpan saat forward untuk dipakai backward. */
interface Cache {
  x: Vec;                          // masukan blok
  u: Vec;                          // masukan attention (post-LN: x, pre-LN: LN1(x))
  q: Vec; k: Vec; v: Vec; p: Vec; o: Vec; a: Vec;
  s1: Vec;                         // x + a
  h: Vec;                          // post-LN: LN1(s1)   pre-LN: s1
  w: Vec;                          // masukan feed forward (post-LN: h, pre-LN: LN2(h))
  f1: Vec; g: Vec; f2: Vec;
  s2: Vec;                         // post-LN: h + f2
  y: Vec;                          // keluaran blok
  ln1hat: Vec; ln1r: Vec; ln2hat: Vec; ln2r: Vec;
}

export class GPT {
  readonly c: GPTConfig;
  readonly P: Vec;
  readonly ukuran: number;
  readonly logits: Vec;            // T×V hasil forward terakhir
  private readonly t: Map<string, TensorInfo>;
  private readonly pe: Vec;
  private readonly x0: Vec;
  private readonly cache: Cache[];
  private readonly akhir: Vec; private readonly lnfhat: Vec; private readonly lnfr: Vec;
  // ruang kerja gradien
  private readonly dLogits: Vec; private readonly dY: Vec; private readonly dH: Vec; private readonly dS: Vec;
  private readonly dU: Vec; private readonly dO: Vec; private readonly dA: Vec;
  private readonly dQ: Vec; private readonly dK: Vec; private readonly dV: Vec;
  private readonly dG: Vec; private readonly dF1: Vec; private readonly dPbaris: Vec;
  // Dropout: aktif hanya saat training (aturDropout). m0 = embedding, ma/mf = cabang attention/FFN.
  private dropoutP = 0;
  private acak: () => number = Math.random;
  private maskBeku = false;
  private readonly m0: Vec; private readonly ma: Vec[]; private readonly mf: Vec[];

  /** Nyalakan dropout dengan peluang p (0 = mati). `beku` = pakai ulang mask terakhir (untuk gradient check). */
  aturDropout(p: number, acak?: () => number, beku = false): void {
    this.dropoutP = p;
    if (acak) this.acak = acak;
    this.maskBeku = beku;
  }

  constructor(c: GPTConfig, P?: Vec, seed = 1) {
    if (c.dim % c.heads !== 0) throw new Error(`dim (${c.dim}) harus habis dibagi heads (${c.heads})`);
    this.c = c;
    const { tensor, ukuran } = susunParameter(c);
    this.t = tensor;
    this.ukuran = ukuran;
    if (P && P.length !== ukuran) throw new Error(`ukuran parameter ${P.length} ≠ ${ukuran}`);
    this.P = P ?? new Float64Array(ukuran);
    if (!P) inisialisasi(c, this.P, seed);
    this.pe = positionalEncoding(c.context, c.dim);
    const T = c.context, D = c.dim, F = c.ff, H = c.heads, V = c.vocab;
    const f = (n: number) => new Float64Array(n);
    this.x0 = f(T * D);
    this.cache = Array.from({ length: c.layers }, () => ({
      x: f(T * D), u: f(T * D), q: f(T * D), k: f(T * D), v: f(T * D), p: f(H * T * T), o: f(T * D), a: f(T * D),
      s1: f(T * D), h: f(T * D), w: f(T * D), f1: f(T * F), g: f(T * F), f2: f(T * D), s2: f(T * D), y: f(T * D),
      ln1hat: f(T * D), ln1r: f(T), ln2hat: f(T * D), ln2r: f(T),
    }));
    this.akhir = f(T * D); this.lnfhat = f(T * D); this.lnfr = f(T);
    this.logits = f(T * V); this.dLogits = f(T * V);
    this.dY = f(T * D); this.dH = f(T * D); this.dS = f(T * D); this.dU = f(T * D);
    this.dO = f(T * D); this.dA = f(T * D); this.dQ = f(T * D); this.dK = f(T * D); this.dV = f(T * D);
    this.dG = f(T * F); this.dF1 = f(T * F); this.dPbaris = f(T);
    this.m0 = f(T * D);
    this.ma = Array.from({ length: c.layers }, () => f(T * D));
    this.mf = Array.from({ length: c.layers }, () => f(T * D));
  }

  private get dropoutAktif(): boolean {
    return this.dropoutP > 0;
  }

  private mask(m: Vec, n: number): void {
    if (!this.maskBeku) buatMaskDropout(m, n, this.dropoutP, this.acak);
  }

  off(nama: string): number {
    return this.t.get(nama)!.off;
  }

  tensors(): Map<string, TensorInfo> {
    return this.t;
  }

  /**
   * INTIP isi model (bahan belajar): jalankan forward pada `ids`, lalu ambil
   *   - embedding token (×√D), positional encoding, dan jumlahnya (masukan lapisan 1)
   *   - vektor Query & Key lapisan 1 head 1 (untuk simulasi perhitungan attention)
   *   - matriks bobot attention P (T×T) untuk setiap lapisan dan head
   * Hanya `dim` angka pertama setiap vektor yang diambil agar ringkas.
   */
  intip(ids: ArrayLike<number>, dim = 8) {
    const aktif = this.dropoutP;
    this.dropoutP = 0;
    this.forward(ids);
    this.dropoutP = aktif;
    const { dim: D, heads: H } = this.c;
    const T = ids.length, dh = D / H, sD = Math.sqrt(D), tok = this.off("tok");
    const potong = (a: Vec, off: number, n: number) => Array.from(a.subarray(off, off + n));
    return {
      D, H, dh, T,
      embedding: Array.from({ length: T }, (_, t) => potong(this.P, tok + ids[t] * D, dim).map((x) => x * sD)),
      posisi: Array.from({ length: T }, (_, t) => potong(this.pe, t * D, dim)),
      masukan: Array.from({ length: T }, (_, t) => potong(this.x0, t * D, dim)),
      q0: Array.from({ length: T }, (_, t) => potong(this.cache[0].q, t * D, dh)),
      k0: Array.from({ length: T }, (_, t) => potong(this.cache[0].k, t * D, dh)),
      attention: this.cache.map((L) => Array.from({ length: H }, (_, h) =>
        Array.from({ length: T }, (_, i) => potong(L.p, h * T * T + i * T, T)))),
    };
  }

  /** Forward: mengisi this.logits (T×V) untuk setiap posisi. */
  forward(ids: ArrayLike<number>): void {
    const { dim: D, heads: H, ff: F, vocab: V, preLN } = this.c;
    const T = ids.length;
    if (T > this.c.context) throw new Error(`panjang ${T} melebihi konteks ${this.c.context}`);
    const P = this.P, sD = Math.sqrt(D), tok = this.off("tok"), n = T * D;

    // Embedding + Positional Encoding:  x0_t = √D · E[id_t] + PE_t
    for (let t = 0; t < T; t++)
      for (let d = 0; d < D; d++) this.x0[t * D + d] = sD * P[tok + ids[t] * D + d] + this.pe[t * D + d];
    if (this.dropoutAktif) { this.mask(this.m0, n); kaliMask(this.x0, this.m0, n); }

    let x = this.x0;
    for (let l = 0; l < this.c.layers; l++) {
      const L = this.cache[l], o = (s: string) => this.off(`l${l}.${s}`);
      L.x.set(x.subarray(0, n));
      // --- Causal Multi-Head Attention
      if (preLN) layerNormForward(L.x, T, D, P, o("ln1g"), o("ln1b"), L.u, L.ln1hat, L.ln1r);
      else L.u.set(L.x.subarray(0, n));
      linearForward(L.u, T, D, D, P, o("wq"), o("bq"), L.q);
      linearForward(L.u, T, D, D, P, o("wk"), o("bk"), L.k);
      linearForward(L.u, T, D, D, P, o("wv"), o("bv"), L.v);
      attentionForward(L.q, L.k, L.v, T, D, H, L.p, L.o);
      linearForward(L.o, T, D, D, P, o("wo"), o("bo"), L.a);
      if (this.dropoutAktif) { this.mask(this.ma[l], n); kaliMask(L.a, this.ma[l], n); }
      // --- Add (& Norm)
      for (let i = 0; i < n; i++) L.s1[i] = L.x[i] + L.a[i];
      if (preLN) {
        L.h.set(L.s1.subarray(0, n));
        layerNormForward(L.h, T, D, P, o("ln2g"), o("ln2b"), L.w, L.ln2hat, L.ln2r);
      } else {
        layerNormForward(L.s1, T, D, P, o("ln1g"), o("ln1b"), L.h, L.ln1hat, L.ln1r);
        L.w.set(L.h.subarray(0, n));
      }
      // --- Feed Forward:  F(w) = GELU(w·W1 + b1)·W2 + b2
      linearForward(L.w, T, D, F, P, o("w1"), o("b1"), L.f1);
      geluForward(L.f1, L.g, T * F);
      linearForward(L.g, T, F, D, P, o("w2"), o("b2"), L.f2);
      if (this.dropoutAktif) { this.mask(this.mf[l], n); kaliMask(L.f2, this.mf[l], n); }
      // --- Add (& Norm)
      if (preLN) for (let i = 0; i < n; i++) L.y[i] = L.h[i] + L.f2[i];
      else {
        for (let i = 0; i < n; i++) L.s2[i] = L.h[i] + L.f2[i];
        layerNormForward(L.s2, T, D, P, o("ln2g"), o("ln2b"), L.y, L.ln2hat, L.ln2r);
      }
      x = L.y;
    }
    // --- (Layer Norm akhir untuk pre-LN) lalu Linear -> logits
    if (preLN) layerNormForward(x, T, D, P, this.off("lnfg"), this.off("lnfb"), this.akhir, this.lnfhat, this.lnfr);
    else this.akhir.set(x.subarray(0, n));
    linearForward(this.akhir, T, D, V, P, this.off("wout"), this.off("bout"), this.logits);
  }

  /**
   * Forward + loss + (opsional) backward. Gradien DITAMBAHKAN ke G.
   * skala = 1/(jumlah token dalam batch) agar loss & gradien merupakan rata-rata.
   * Mengembalikan jumlah −log p(target) untuk urutan ini.
   */
  langkah(ids: ArrayLike<number>, target: ArrayLike<number>, G: Vec | null, skala: number): number {
    this.forward(ids);
    const { dim: D, heads: H, ff: F, vocab: V, preLN } = this.c;
    const T = ids.length, n = T * D, P = this.P;
    const loss = softmaxCrossEntropy(this.logits, target, T, V, G ? this.dLogits : null, skala);
    if (!G) return loss;

    // Linear keluaran
    this.dY.fill(0, 0, n);
    linearBackward(this.akhir, this.dLogits, T, D, V, P, G, this.off("wout"), this.off("bout"), this.dY);
    if (preLN) {
      this.dS.fill(0, 0, n);
      layerNormBackward(this.dY, T, D, P, G, this.off("lnfg"), this.off("lnfb"), this.lnfhat, this.lnfr, this.dS);
      this.dY.set(this.dS.subarray(0, n));
    }

    // Blok dari belakang ke depan. dY = ∂L/∂(keluaran blok)
    for (let l = this.c.layers - 1; l >= 0; l--) {
      const L = this.cache[l], o = (s: string) => this.off(`l${l}.${s}`);
      // --- Add (& Norm) kedua  ->  dH (∂L/∂h) dan dS (∂L/∂f2)
      this.dS.fill(0, 0, n);
      if (preLN) this.dS.set(this.dY.subarray(0, n)); // y = h + f2
      else layerNormBackward(this.dY, T, D, P, G, o("ln2g"), o("ln2b"), L.ln2hat, L.ln2r, this.dS); // y = LN2(s2)
      this.dH.set(this.dS.subarray(0, n)); // jalur residual: ∂s2/∂h = 1  (pre-LN: ∂y/∂h = 1)
      if (this.dropoutAktif) kaliMask(this.dS, this.mf[l], n); // f2 = FFN(w) ⊙ m  ->  ∂L/∂FFN = ∂L/∂f2 ⊙ m
      // --- Feed Forward backward: dS adalah ∂L/∂f2
      this.dG.fill(0, 0, T * F);
      linearBackward(L.g, this.dS, T, F, D, P, G, o("w2"), o("b2"), this.dG);
      this.dF1.fill(0, 0, T * F);
      geluBackward(L.f1, this.dG, this.dF1, T * F);
      this.dU.fill(0, 0, n); // dipakai sementara sebagai ∂L/∂w
      linearBackward(L.w, this.dF1, T, D, F, P, G, o("w1"), o("b1"), this.dU);
      if (preLN) layerNormBackward(this.dU, T, D, P, G, o("ln2g"), o("ln2b"), L.ln2hat, L.ln2r, this.dH); // w = LN2(h)
      else for (let i = 0; i < n; i++) this.dH[i] += this.dU[i]; // w = h
      // --- Add (& Norm) pertama -> ∂L/∂s1 (disimpan di dA, karena s1 = x + a)
      this.dA.fill(0, 0, n);
      if (preLN) this.dA.set(this.dH.subarray(0, n));
      else layerNormBackward(this.dH, T, D, P, G, o("ln1g"), o("ln1b"), L.ln1hat, L.ln1r, this.dA);
      // ∂L/∂x dari jalur residual = ∂L/∂s1 ; simpan ke dY (dY lama sudah tidak dipakai)
      this.dY.set(this.dA.subarray(0, n));
      if (this.dropoutAktif) kaliMask(this.dA, this.ma[l], n); // a = MHA(u) ⊙ m
      // --- Attention backward
      this.dO.fill(0, 0, n);
      linearBackward(L.o, this.dA, T, D, D, P, G, o("wo"), o("bo"), this.dO);
      this.dQ.fill(0, 0, n); this.dK.fill(0, 0, n); this.dV.fill(0, 0, n);
      attentionBackward(L.q, L.k, L.v, L.p, this.dO, T, D, H, this.dQ, this.dK, this.dV, this.dPbaris);
      this.dU.fill(0, 0, n); // ∂L/∂u
      linearBackward(L.u, this.dQ, T, D, D, P, G, o("wq"), o("bq"), this.dU);
      linearBackward(L.u, this.dK, T, D, D, P, G, o("wk"), o("bk"), this.dU);
      linearBackward(L.u, this.dV, T, D, D, P, G, o("wv"), o("bv"), this.dU);
      if (preLN) layerNormBackward(this.dU, T, D, P, G, o("ln1g"), o("ln1b"), L.ln1hat, L.ln1r, this.dY); // u = LN1(x)
      else for (let i = 0; i < n; i++) this.dY[i] += this.dU[i]; // u = x
    }

    // Embedding:  x0 = (√D·E[id] + PE) ⊙ m0   ->   ∂L/∂E[id] += √D · (∂L/∂x0 ⊙ m0)
    if (this.dropoutAktif) kaliMask(this.dY, this.m0, n);
    const tok = this.off("tok"), sD = Math.sqrt(D);
    for (let t = 0; t < T; t++) {
      const e = tok + ids[t] * D;
      for (let d = 0; d < D; d++) G[e + d] += sD * this.dY[t * D + d];
    }
    return loss;
  }
}
