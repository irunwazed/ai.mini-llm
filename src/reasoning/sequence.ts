/**
 * POLA BARISAN: "Angka berikutnya dari 2, 6, 12, 20, 30, ...?"
 *
 * Metode SELISIH BERTINGKAT: hitung selisih antarsuku berulang-ulang sampai konstan.
 *   2   6   12   20   30
 *     4   6    8    10        selisih tingkat 1
 *       2    2    2           selisih tingkat 2: konstan!
 * Lalu bangun kembali dari bawah: 10 + 2 = 12, 30 + 12 = 42.
 * Jika tidak ketemu, coba RASIO tetap (barisan geometri: 3, 6, 12, 24 -> ×2).
 * Cek: aturan yang ditemukan harus menghasilkan ulang semua suku yang diketahui.
 */
import { fmt } from "./numbers.js";
import type { Alat } from "./types.js";

function ambilBarisan(t: string): number[] | null {
  const m = t.match(/-?\d+(?:[.,]\d+)?(?:\s*,\s*-?\d+(?:[.,]\d+)?){3,}/);
  if (!m) return null;
  return m[0].split(/\s*,\s*/).map((x) => Number(x.replace(",", ".")));
}

export const barisan: Alat = {
  nama: "pola barisan (selisih bertingkat / rasio)",
  pikir: "Ada deret angka dan ditanya angka berikutnya, jadi cari aturan polanya lewat selisih atau rasio.",
  cocok: (t) => !!ambilBarisan(t) && /(berikut|selanjutnya|pola|deret|barisan|\.\.\.|…)/i.test(t),
  coba: (t) => {
    const a = ambilBarisan(t);
    if (!a) return null;
    // Selisih bertingkat
    const tingkat: number[][] = [a];
    while (tingkat.at(-1)!.length > 1 && tingkat.length <= 4) {
      const s = tingkat.at(-1)!;
      const d = s.slice(1).map((x, i) => x - s[i]);
      tingkat.push(d);
      if (d.every((x) => x === d[0]) && d.length >= 2) {
        // bangun kembali suku berikutnya dari tingkat terbawah
        let tambah = d[0];
        for (let k = tingkat.length - 2; k >= 0; k--) tambah = tingkat[k].at(-1)! + tambah;
        const penjelasan = tingkat.map((row, k) => `${k === 0 ? "barisan  " : `selisih ${k}`}: ${row.map(fmt).join(", ")}`);
        penjelasan.push(`Selisih tingkat ${tingkat.length - 1} konstan (${fmt(d[0])}), jadi bangun kembali dari bawah: suku berikutnya = ${fmt(tambah)}.`);
        return {
          jawaban: fmt(tambah),
          penjelasan,
          cek: { lolos: true, teks: `aturan selisih tingkat ${tingkat.length - 1} = ${fmt(d[0])} menghasilkan ulang semua ${a.length} suku yang diketahui ✓` },
        };
      }
    }
    // Rasio tetap
    if (a.every((x) => x !== 0)) {
      const r = a.slice(1).map((x, i) => x / a[i]);
      if (r.every((x) => Math.abs(x - r[0]) < 1e-9)) {
        const berikut = a.at(-1)! * r[0];
        return {
          jawaban: fmt(berikut),
          penjelasan: [`rasio antarsuku: ${r.map(fmt).join(", ")} (tetap ×${fmt(r[0])})`, `suku berikutnya = ${fmt(a.at(-1)!)} × ${fmt(r[0])} = ${fmt(berikut)}`],
          cek: { lolos: true, teks: `setiap suku = suku sebelumnya × ${fmt(r[0])} ✓` },
        };
      }
    }
    return null;
  },
};
