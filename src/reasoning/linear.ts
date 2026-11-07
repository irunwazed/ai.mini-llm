/**
 * PERSAMAAN LINEAR: "Tongkat dan bola berharga Rp11.000. Tongkat Rp10.000 lebih mahal daripada bola.
 *                    Berapa harga bola?"
 *
 * Soal diubah menjadi sistem persamaan:
 *   t + b = 11.000
 *   t − b = 10.000
 * lalu diselesaikan dengan ELIMINASI GAUSS (berlaku umum untuk n variabel):
 *   [ 1  1 | 11000 ]   baris2 − baris1   [ 1  1 | 11000 ]
 *   [ 1 −1 | 10000 ]  --------------->   [ 0 −2 | −1000 ]  ->  b = 500,  t = 10.500
 * Jawaban spontan "Rp1.000" salah: jika bola Rp1.000, tongkat Rp11.000, totalnya Rp12.000.
 * Cek: nilai yang ditemukan dimasukkan kembali ke kedua persamaan.
 */
import { angkaDigit, rupiah } from "./numbers.js";
import type { Alat } from "./types.js";

/** Eliminasi Gauss dengan pivot sebagian; mencatat setiap operasi baris. */
export function gauss(A: number[][], b: number[], langkah: string[]): number[] | null {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
    if (Math.abs(M[p][k]) < 1e-12) return null;
    if (p !== k) { [M[p], M[k]] = [M[k], M[p]]; langkah.push(`tukar baris ${k + 1} dan ${p + 1}`); }
    for (let i = k + 1; i < n; i++) {
      const f = M[i][k] / M[k][k];
      if (f === 0) continue;
      for (let j = k; j <= n; j++) M[i][j] -= f * M[k][j];
      langkah.push(`baris ${i + 1} − ${f} × baris ${k + 1}  ->  [${M[i].map((x) => +x.toFixed(4)).join(", ")}]`);
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

const RE_SELISIH = /(lebih\s+(mahal|banyak|besar|tua|berat|tinggi|panjang)|lebih\s+(murah|sedikit|kecil|muda|ringan|rendah|pendek))\s+(dari(pada)?)/i;

export const persamaan: Alat = {
  nama: "sistem persamaan linear (eliminasi Gauss)",
  pikir: "Ada jumlah total dan selisih dua benda, jadi tulis dua persamaan lalu selesaikan bersama.",
  cocok: (t) => RE_SELISIH.test(t) && /(dan|\+)/i.test(t) && angkaDigit(t).length >= 2 && /berapa/i.test(t),
  coba: (t) => {
    // pola: "<X> dan <Y> (ber)harga TOTAL ... <X> SELISIH lebih mahal daripada <Y> ... berapa (harga) <Z>?"
    const nama = t.toLowerCase().match(/(?:sebuah\s+|seekor\s+)?([a-z]+)\s+dan\s+(?:sebuah\s+|seekor\s+)?([a-z]+)/);
    const angka = angkaDigit(t);
    const sel = t.toLowerCase().match(/(?:harga\s+)?([a-z]+)\s+(?:rp\s?)?[\d.,]+\s+lebih\s+(\w+)\s+dari(?:pada)?\s+(?:harga\s+)?([a-z]+)/);
    if (!nama || !sel || angka.length < 2) return null;
    const [x, y] = [nama[1], nama[2]];
    const total = angka[0].nilai, selisih = angka[1].nilai;
    const lebihBesar = sel[1], lebih = sel[2];
    const tandaX = /(murah|sedikit|kecil|muda|ringan|rendah|pendek)/.test(lebih) ? (lebihBesar === x ? -1 : 1) : (lebihBesar === x ? 1 : -1);
    const ditanya = t.toLowerCase().match(/berapa\s+(?:harga\s+|jumlah\s+|umur\s+|berat\s+)?([a-z]+)/)?.[1] ?? y;
    const uang = /rp|harga|biaya/i.test(t);
    const tulis = (v: number) => (uang ? rupiah(v) : String(+v.toFixed(4)));
    const langkah: string[] = [];
    // x + y = total ;  x − y = ±selisih
    const hasil = gauss([[1, 1], [1, -1]], [total, tandaX * selisih], langkah);
    if (!hasil) return null;
    const [vx, vy] = hasil;
    const nilai = ditanya === x ? vx : vy;
    const okTotal = Math.abs(vx + vy - total) < 1e-6, okSelisih = Math.abs(vx - vy - tandaX * selisih) < 1e-6;
    return {
      jawaban: `${ditanya === x ? x : y} = ${tulis(nilai)}`,
      penjelasan: [
        `Misalkan ${x} = x dan ${y} = y.`,
        `Persamaan 1 (total):   x + y = ${tulis(total)}`,
        `Persamaan 2 (selisih): x − y = ${tulis(tandaX * selisih)}`,
        ...langkah.map((l) => `Eliminasi: ${l}`),
        `Hasil: ${x} = ${tulis(vx)}, ${y} = ${tulis(vy)}.`,
        `Jebakan umum: menjawab ${tulis(total - selisih)} langsung. Itu salah, karena selisihnya jadi ${tulis(Math.abs(total - 2 * (total - selisih)))}, bukan ${tulis(selisih)}.`,
      ],
      cek: { lolos: okTotal && okSelisih, teks: `${tulis(vx)} + ${tulis(vy)} = ${tulis(vx + vy)} ${okTotal ? "✓" : "✗"};  ${tulis(vx)} − ${tulis(vy)} = ${tulis(vx - vy)} ${okSelisih ? "✓" : "✗"}` },
    };
  },
};
