/**
 * Menghasilkan teks secara AUTOREGRESIF, seperti GPT:
 *   1. ubah awal kalimat menjadi token
 *   2. forward -> logits posisi terakhir -> sampling satu token
 *   3. tambahkan token itu ke urutan, ulangi
 * Jika urutan melebihi konteks, hanya C token terakhir yang dipakai (jendela geser).
 */
import { KHUSUS, UNK, type Tokenizer } from "./bpe.js";
import type { GPT } from "./model.js";
import { Rng } from "./random.js";
import { sampel, type OpsiSampel } from "./sampling.js";

export interface OpsiGenerate extends OpsiSampel {
  seed?: number;
  berhentiDi?: (teksSejauhIni: string, idTerakhir: number) => boolean;
}

/** Id token yang tidak boleh muncul di teks biasa (<unk> dan token percakapan). */
export function tokenTerlarang(tok: Tokenizer): number[] {
  return [UNK, ...KHUSUS].map((t) => tok.idToken(t)).filter((x): x is number => x !== undefined);
}

export function hasilkanDariId(
  model: GPT, tok: Tokenizer, ids: number[], panjang: number, opsi: OpsiGenerate, onToken?: (teks: string) => void,
): { teks: string; ids: number[] } {
  const rng = new Rng(opsi.seed ?? Date.now() % 2147483647);
  const V = model.c.vocab, C = model.c.context;
  const baru: number[] = [];
  let keluar = "";
  let sudah = 0; // jumlah token di cache yang masih berlaku
  for (let n = 0; n < panjang; n++) {
    const jendela = ids.slice(-C);
    // KV-cache: selama jendela belum bergeser, cukup hitung token baru. Jika jendela bergeser
    // (melebihi konteks), posisi semua token berubah sehingga harus dihitung ulang penuh.
    if (sudah > 0 && ids.length <= C) model.forwardTambah(jendela, sudah);
    else model.forward(jendela);
    sudah = jendela.length;
    const z = model.logits.subarray((jendela.length - 1) * V, jendela.length * V);
    const id = sampel(z, { ...opsi, riwayat: ids.slice(-64) }, rng);
    ids.push(id);
    baru.push(id);
    if (opsi.berhentiDi?.(keluar, id)) break;
    const s = tok.decode([id]);
    keluar += s;
    onToken?.(s);
  }
  return { teks: keluar, ids: baru };
}

export function hasilkanTeks(
  model: GPT, tok: Tokenizer, awal: string, panjang: number, opsi: OpsiGenerate, onToken?: (teks: string) => void,
): string {
  const ids = tok.encode(awal);
  if (ids.length === 0) ids.push(tok.encode(" ")[0] ?? 0);
  return hasilkanDariId(model, tok, ids, panjang, { larang: tokenTerlarang(tok), ...opsi }, onToken).teks;
}
