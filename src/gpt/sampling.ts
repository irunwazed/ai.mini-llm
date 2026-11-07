/**
 * SAMPLING: memilih token berikutnya dari logits z (satu baris, panjang V).
 *
 *   Repetition penalty r: token yang baru saja muncul dikurangi peluangnya
 *       z_i ← z_i / r  (jika z_i > 0)   atau   z_i ← z_i · r  (jika z_i < 0)
 *       -> mencegah model terjebak mengulang "anak anak anak ..."
 *   Temperature T:  p_i = exp(z_i / T) / Σ_j exp(z_j / T)
 *       T < 1 -> lebih yakin dan berulang,   T > 1 -> lebih acak/kreatif
 *   Top-k:  hanya k token dengan peluang terbesar yang boleh dipilih
 *   Top-p (nucleus):  ambil token terurut dari peluang terbesar sampai Σp ≥ P,
 *       buang sisanya ("ekor panjang" token aneh), lalu normalisasi ulang.
 */
import type { Vec } from "./matrix.js";
import type { Rng } from "./random.js";

export interface OpsiSampel {
  suhu?: number; topK?: number; topP?: number;
  penaltiUlang?: number;        // repetition penalty (1 = mati)
  riwayat?: ArrayLike<number>;  // token terakhir yang dikenai penalti
  larang?: number[];            // token yang tidak boleh dipilih
  /** Untuk belajar: dipanggil dengan 5 kandidat teratas (setelah suhu & penalti) dan token terpilih. */
  catat?: (kandidat: { id: number; p: number }[], dipilih: number) => void;
}

export function sampel(zAsli: Vec, opsi: OpsiSampel, rng: Rng): number {
  const T = Math.max(1e-4, opsi.suhu ?? 1);
  const V = zAsli.length;
  const z = Float64Array.from(zAsli);
  const r = opsi.penaltiUlang ?? 1;
  if (r !== 1 && opsi.riwayat) {
    const sudah = new Set<number>();
    for (let i = 0; i < opsi.riwayat.length; i++) sudah.add(opsi.riwayat[i]);
    for (const id of sudah) z[id] = z[id] > 0 ? z[id] / r : z[id] * r;
  }
  for (const id of opsi.larang ?? []) z[id] = -Infinity;

  let maks = -Infinity;
  for (let i = 0; i < V; i++) if (z[i] > maks) maks = z[i];
  const idx = Array.from({ length: V }, (_, i) => i);
  const p = new Float64Array(V);
  let jumlah = 0;
  for (let i = 0; i < V; i++) { p[i] = Math.exp((z[i] - maks) / T); jumlah += p[i]; }
  for (let i = 0; i < V; i++) p[i] /= jumlah;

  idx.sort((a, b) => p[b] - p[a]);
  let batas = Math.min(V, opsi.topK && opsi.topK > 0 ? opsi.topK : V);
  if (opsi.topP && opsi.topP < 1) {
    let kum = 0;
    for (let k = 0; k < batas; k++) {
      kum += p[idx[k]];
      if (kum >= opsi.topP) { batas = k + 1; break; }
    }
  }
  let massa = 0;
  for (let k = 0; k < batas; k++) massa += p[idx[k]];
  let u = rng.next() * massa;
  let dipilih = idx[batas - 1];
  for (let k = 0; k < batas; k++) {
    u -= p[idx[k]];
    if (u <= 0) { dipilih = idx[k]; break; }
  }
  opsi.catat?.(idx.slice(0, 5).map((id) => ({ id, p: p[id] })), dipilih);
  return dipilih;
}
