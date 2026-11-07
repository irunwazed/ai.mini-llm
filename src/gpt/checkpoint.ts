/**
 * Menyimpan dan memuat model GPT:
 *   gpt-models/<nama>/gpt.json      konfigurasi, tokenizer BPE, riwayat loss, info training
 *   gpt-models/<nama>/weights.bin   bobot (Float32, cukup presisi untuk inferensi & lanjut training)
 *   gpt-models/<nama>/optimizer.bin momentum Adam m dan v (untuk --resume)
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { BPE } from "./bpe.js";
import type { Vec } from "./matrix.js";
import type { GPTConfig } from "./model.js";

export const FOLDER_GPT = "gpt-models";

export interface Riwayat { langkah: number; lossTrain: number; lossVal: number; detik: number }
export interface MetaGPT {
  mode?: "teks" | "chat" | "pembaca";   // chat = fine-tune percakapan; pembaca = GPT pembaca (kepala span)
  dasar?: string;           // nama model pretrain asal (untuk fine-tune)
  config: GPTConfig;
  bpe: BPE;
  hurufKecil: boolean;
  langkah: number;
  lossValTerbaik: number;
  riwayat: Riwayat[];
  data: string;
  jumlahToken: number;
  waktu: string;
  /** Khusus pembaca: ambang keyakinan hasil kalibrasi + skor val. */
  pembaca?: { ambang: number; em: number; f1: number; akurasiKosong: number; contohVal: number };
}

const keF32 = (a: Vec) => Buffer.from(Float32Array.from(a).buffer);
const dariF32 = (b: Buffer) => Float64Array.from(new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));

export function simpanGPT(nama: string, meta: MetaGPT, P: Vec, adam?: { m: Vec; v: Vec }): string {
  const folder = join(FOLDER_GPT, nama);
  mkdirSync(folder, { recursive: true });
  writeFileSync(join(folder, "gpt.json"), JSON.stringify(meta));
  writeFileSync(join(folder, "weights.bin"), keF32(P));
  if (adam) writeFileSync(join(folder, "optimizer.bin"), Buffer.concat([keF32(adam.m), keF32(adam.v)]));
  return folder;
}

export function muatGPT(nama: string): { meta: MetaGPT; P: Float64Array; adam?: { m: Float64Array; v: Float64Array } } {
  const folder = join(FOLDER_GPT, nama);
  if (!existsSync(join(folder, "gpt.json"))) throw new Error(`Model GPT "${nama}" belum ada. Jalankan dulu: bun run train-gpt -- --data <data> --name ${nama}`);
  const meta = JSON.parse(readFileSync(join(folder, "gpt.json"), "utf8")) as MetaGPT;
  const P = dariF32(readFileSync(join(folder, "weights.bin")));
  let adam;
  if (existsSync(join(folder, "optimizer.bin"))) {
    const mv = dariF32(readFileSync(join(folder, "optimizer.bin")));
    adam = { m: mv.slice(0, P.length), v: mv.slice(P.length) };
  }
  return { meta, P, adam };
}

/** Nama model GPT yang paling baru dilatih; `mode` = hanya model ber-mode itu (mis. "chat"). */
export function gptTerbaru(mode?: "teks" | "chat" | "pembaca"): string | null {
  if (!existsSync(FOLDER_GPT)) return null;
  const modeDari = (n: string) => (JSON.parse(readFileSync(join(FOLDER_GPT, n, "gpt.json"), "utf8")) as MetaGPT).mode ?? "teks";
  const daftar = readdirSync(FOLDER_GPT)
    .filter((n) => existsSync(join(FOLDER_GPT, n, "gpt.json")))
    .filter((n) => !mode || modeDari(n) === mode)
    .map((n) => ({ n, t: statSync(join(FOLDER_GPT, n, "gpt.json")).mtimeMs }))
    .sort((a, b) => b.t - a.t);
  return daftar[0]?.n ?? null;
}
