/**
 * DATASET SIAP LATIH: hasil langkah "siapkan data" (bun run siapkan-data), disimpan di data/siap/<nama>/.
 *
 *   info.json        sumber (lokasi, ukuran, waktu ubah), jenis teks/obrolan, jumlah token,
 *                    asal tokenizer + SIDIK tokenizer, huruf kecil/asli
 *   tokenizer.json   BPE yang dipakai men-token-kan data ini
 *   teks    : latih.bin / val.bin          token Int32 (pembagian per blok 256 token, blok ke-10 = val)
 *   obrolan : percakapan.jsonl             token + penanda token yang dilatih, BELUM dipotong ke konteks
 *
 * Training (bun run train-gpt -- --dataset <nama>) membaca ini, tanpa membaca/men-token-kan teks mentah lagi.
 * SIDIK tokenizer mencegah salah pasang: model hanya boleh dilatih dengan data yang di-token-kan
 * memakai tokenizer yang sama (id token 517 harus berarti potongan kata yang sama).
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { LOKASI } from "../paths.js";
import type { BPE } from "./bpe.js";
import type { PercakapanToken } from "./dataset.js";

export interface SumberDataset { nama: string; lokasi: string; ukuran: number; diubah: string }
export interface InfoDataset {
  nama: string; jenis: "teks" | "obrolan"; dibuat: string;
  sumber: SumberDataset[];
  hurufKecil: boolean;
  tokenizer: { asal: string; sidik: string; ukuran: number };
  token: { latih: number; val: number };
  percakapan?: number;
  keterangan: string[];
}

/** Sidik tokenizer: hash FNV-1a dari kosakata + urutan merge. Sama = tokenizer identik. */
export function sidikTokenizer(bpe: BPE): string {
  let h = 2166136261;
  const s = JSON.stringify([bpe.vocab, bpe.merges]);
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0).toString(16).padStart(8, "0") + "-" + bpe.vocab.length;
}

/** Ukuran total & waktu ubah terbaru sebuah file/folder (rekursif). */
export function infoLokasi(lokasi: string): { ukuran: number; diubah: string; berkas: number } {
  if (!existsSync(lokasi)) return { ukuran: 0, diubah: "", berkas: 0 };
  const st = statSync(lokasi);
  if (st.isFile()) return { ukuran: st.size, diubah: st.mtime.toISOString(), berkas: 1 };
  let ukuran = 0, terbaru = 0, berkas = 0;
  for (const f of readdirSync(lokasi, { recursive: true }) as string[]) {
    const p = join(lokasi, f);
    try { const s = statSync(p); if (s.isFile()) { ukuran += s.size; berkas++; terbaru = Math.max(terbaru, s.mtimeMs); } } catch { /* abaikan */ }
  }
  return { ukuran, diubah: terbaru ? new Date(terbaru).toISOString() : "", berkas };
}

const folder = (nama: string) => join(LOKASI.siap, nama);

export function simpanDataset(info: InfoDataset, bpe: BPE, data: { latih?: Int32Array; val?: Int32Array; percakapan?: PercakapanToken[] }): string {
  const d = folder(info.nama);
  mkdirSync(d, { recursive: true });
  writeFileSync(join(d, "tokenizer.json"), JSON.stringify(bpe));
  if (data.latih && data.val) {
    writeFileSync(join(d, "latih.bin"), Buffer.from(data.latih.buffer, data.latih.byteOffset, data.latih.byteLength));
    writeFileSync(join(d, "val.bin"), Buffer.from(data.val.buffer, data.val.byteOffset, data.val.byteLength));
  }
  if (data.percakapan) writeFileSync(join(d, "percakapan.jsonl"), data.percakapan.map((p) => JSON.stringify({ k: p.kartu ? 1 : 0, g: p.potongan.map((q) => [q.ids, q.latih.join("")]) })).join("\n") + "\n");
  writeFileSync(join(d, "info.json"), JSON.stringify(info, null, 2));
  return d;
}

export function muatDataset(nama: string): { info: InfoDataset; bpe: BPE; latih?: Int32Array; val?: Int32Array; percakapan?: PercakapanToken[] } {
  const d = folder(nama);
  if (!existsSync(join(d, "info.json"))) throw new Error(`Dataset "${nama}" belum ada di ${LOKASI.siap}/. Buat dulu: bun run siapkan-data -- --sumber … --nama ${nama}`);
  const info = JSON.parse(readFileSync(join(d, "info.json"), "utf8")) as InfoDataset;
  const bpe = JSON.parse(readFileSync(join(d, "tokenizer.json"), "utf8")) as BPE;
  const bin = (f: string) => { const b = readFileSync(join(d, f)); return new Int32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
  if (info.jenis === "teks") return { info, bpe, latih: bin("latih.bin"), val: bin("val.bin") };
  const percakapan = readFileSync(join(d, "percakapan.jsonl"), "utf8").split(/\r?\n/).filter((l) => l.trim()).map((l) => {
    const j = JSON.parse(l) as { k: number; g: [number[], string][] };
    return { kartu: j.k === 1, potongan: j.g.map(([ids, m]) => ({ ids, latih: [...m].map(Number) })) };
  });
  return { info, bpe, percakapan };
}

/** Semua dataset siap latih + apakah sumbernya sudah berubah sejak disiapkan. */
export function daftarDataset(): (InfoDataset & { sumberBerubah: string[] })[] {
  if (!existsSync(LOKASI.siap)) return [];
  return readdirSync(LOKASI.siap).filter((n) => existsSync(join(folder(n), "info.json"))).map((n) => {
    const info = JSON.parse(readFileSync(join(folder(n), "info.json"), "utf8")) as InfoDataset;
    const sumberBerubah = info.sumber.filter((s) => { const x = infoLokasi(s.lokasi); return x.ukuran !== s.ukuran || x.diubah !== s.diubah; }).map((s) => s.nama);
    return { ...info, sumberBerubah };
  }).sort((a, b) => b.dibuat.localeCompare(a.dibuat));
}
