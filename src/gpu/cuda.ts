/**
 * PEMBUNGKUS BACKEND GPU: memanggil build/libgptcuda.so (hasil `bun run build-gpu` dari src/gpu/gpt.cu)
 * lewat bun:ffi. Model CPU (GPT) tetap menjadi "sumber kebenaran" untuk checkpoint & sampling:
 * bobot diunggah sekali ke GPU, training berjalan seluruhnya di GPU, lalu bobot diambil kembali saat disimpan.
 */
import { dlopen, FFIType, suffix } from "bun:ffi";
import { existsSync } from "node:fs";
import { join } from "node:path";
import type { GPT } from "../gpt/model.js";

export const FILE_LIB = join(process.cwd(), "build", `libgptcuda.${suffix}`);
export const gpuTersedia = (): boolean => existsSync(FILE_LIB);

const { i32, f32, ptr } = FFIType;
function muat() {
  return dlopen(FILE_LIB, {
    gc_buat: { args: [ptr, ptr, i32, i32, ptr], returns: i32 },
    gc_isi_param: { args: [ptr], returns: i32 },
    gc_ambil_param: { args: [ptr], returns: i32 },
    gc_ambil_grad: { args: [ptr], returns: i32 },
    gc_isi_adam: { args: [ptr, ptr], returns: i32 },
    gc_ambil_adam: { args: [ptr, ptr], returns: i32 },
    gc_nol_grad: { args: [], returns: i32 },
    gc_langkah: { args: [ptr, ptr, i32, i32, f32, f32, i32, ptr], returns: i32 },
    gc_loss: { args: [ptr, ptr, i32, i32, ptr], returns: i32 },
    gc_adam: { args: [i32, f32, f32, f32, f32, f32, f32, ptr], returns: i32 },
    gc_info: { args: [ptr, i32], returns: i32 },
    gc_galat: { args: [ptr, i32], returns: i32 },
    gc_hapus: { args: [], returns: i32 },
  });
}

export interface UrutanGPU { x: ArrayLike<number>; y: ArrayLike<number> }

/** Urutan offset tensor yang diharapkan gpt.cu (lihat komentar gc_buat). */
const PER_LAPISAN = ["wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo", "ln1g", "ln1b", "w1", "b1", "w2", "b2", "ln2g", "ln2b"];

export class GPTCuda {
  private readonly lib: ReturnType<typeof muat>;
  readonly info: string;
  readonly bMaks: number;
  private readonly ukuran: number;
  private readonly hasil = new Float64Array(1);

  constructor(model: GPT, bMaks: number) {
    if (!gpuTersedia()) throw new Error(`Backend GPU belum dikompilasi (${FILE_LIB}). Di server ber-GPU NVIDIA jalankan: bun run build-gpu`);
    if (model.c.spanHead) throw new Error("GPT pembaca (span head) belum didukung GPU; latih di CPU.");
    this.lib = muat();
    this.bMaks = bMaks;
    this.ukuran = model.ukuran;
    const c = model.c, t = model.tensors();
    const o = (n: string) => t.get(n)?.off ?? -1;
    const off = [o("tok")];
    for (let l = 0; l < c.layers; l++) for (const n of PER_LAPISAN) off.push(o(`l${l}.${n}`));
    off.push(o("lnfg"), o("lnfb"), o("wout"), o("bout"));
    const decay = new Uint8Array(model.ukuran);
    for (const x of t.values()) if (x.decay) decay.fill(1, x.off, x.off + x.rows * x.cols);
    const cfg = new Int32Array([c.vocab, c.context, c.dim, c.heads, c.layers, c.ff, c.preLN ? 1 : 0, bMaks]);
    this.cek(this.lib.symbols.gc_buat(cfg, new Int32Array(off), off.length, model.ukuran, decay), "membuat model di GPU");
    this.isiParam(model.P);
    const buf = new Uint8Array(300);
    this.lib.symbols.gc_info(buf, buf.length);
    this.info = new TextDecoder().decode(buf.subarray(0, buf.indexOf(0) < 0 ? buf.length : buf.indexOf(0)));
  }

  private cek(kode: number, apa: string): void {
    if (kode === 0) return;
    const buf = new Uint8Array(600);
    this.lib.symbols.gc_galat(buf, buf.length);
    const pesan = new TextDecoder().decode(buf.subarray(0, Math.max(0, buf.indexOf(0))));
    throw new Error(`GPU gagal ${apa}: ${pesan || "(tanpa pesan)"}`);
  }

  isiParam(P: Float64Array): void { this.cek(this.lib.symbols.gc_isi_param(P), "mengunggah bobot"); }
  ambilParam(P: Float64Array): void { this.cek(this.lib.symbols.gc_ambil_param(P), "mengambil bobot"); }
  ambilGrad(G: Float64Array): void { this.cek(this.lib.symbols.gc_ambil_grad(G), "mengambil gradien"); }
  isiAdam(m: Float64Array, v: Float64Array): void { this.cek(this.lib.symbols.gc_isi_adam(m, v), "mengunggah state Adam"); }
  ambilAdam(m: Float64Array, v: Float64Array): void { this.cek(this.lib.symbols.gc_ambil_adam(m, v), "mengambil state Adam"); }
  nolGrad(): void { this.cek(this.lib.symbols.gc_nol_grad(), "mengosongkan gradien"); }

  /** Susun urutan (panjang boleh beda) menjadi B×T: token pengisi 0, target −1 (tidak dihitung). */
  private batch(urutan: UrutanGPU[]): { ids: Int32Array; tgt: Int32Array; B: number; T: number } {
    const B = urutan.length, T = Math.max(...urutan.map((u) => u.x.length));
    const ids = new Int32Array(B * T), tgt = new Int32Array(B * T).fill(-1);
    urutan.forEach((u, b) => { for (let t = 0; t < u.x.length; t++) { ids[b * T + t] = u.x[t]; tgt[b * T + t] = u.y[t]; } });
    return { ids, tgt, B, T };
  }

  /** Forward + backward untuk semua urutan (gradien DITAMBAHKAN). Mengembalikan jumlah −log p(target). */
  langkah(urutan: UrutanGPU[], skala: number, dropout: number, seed: number): number {
    let total = 0;
    for (let i = 0; i < urutan.length; i += this.bMaks) {
      const { ids, tgt, B, T } = this.batch(urutan.slice(i, i + this.bMaks));
      this.cek(this.lib.symbols.gc_langkah(ids, tgt, B, T, skala, dropout, (seed + i) | 0, this.hasil), "menghitung langkah");
      total += this.hasil[0];
    }
    return total;
  }

  /** Hanya loss (tanpa backward & dropout) untuk evaluasi. */
  loss(urutan: UrutanGPU[]): number {
    let total = 0;
    for (let i = 0; i < urutan.length; i += this.bMaks) {
      const { ids, tgt, B, T } = this.batch(urutan.slice(i, i + this.bMaks));
      this.cek(this.lib.symbols.gc_loss(ids, tgt, B, T, this.hasil), "menghitung loss");
      total += this.hasil[0];
    }
    return total;
  }

  /** Satu langkah AdamW di GPU. Mengembalikan norma gradien sebelum clipping. */
  adam(t: number, lr: number, o: { beta1: number; beta2: number; eps: number; weightDecay: number; clip: number }): number {
    this.cek(this.lib.symbols.gc_adam(t, lr, o.beta1, o.beta2, o.eps, o.weightDecay, o.clip, this.hasil), "langkah Adam");
    return this.hasil[0];
  }

  tutup(): void { this.lib.symbols.gc_hapus(); this.lib.close(); }
}
