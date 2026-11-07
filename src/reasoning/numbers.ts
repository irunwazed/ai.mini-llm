/**
 * MEMBACA ANGKA DARI TEKS bahasa Indonesia:
 *   "Rp11.000" -> 11000     "1,5" -> 1.5     "lima" -> 5     "dua puluh" -> 20
 * Titik = pemisah ribuan, koma = desimal (kebiasaan Indonesia).
 */
const SATUAN: Record<string, number> = {
  nol: 0, satu: 1, se: 1, dua: 2, tiga: 3, empat: 4, lima: 5, enam: 6, tujuh: 7, delapan: 8, sembilan: 9,
  sepuluh: 10, sebelas: 11, seratus: 100, seribu: 1000,
};

export interface Angka { nilai: number; teks: string; indeks: number }

/** Semua angka (digit) di teks, berurutan. */
export function angkaDigit(teks: string): Angka[] {
  const hasil: Angka[] = [];
  for (const m of teks.matchAll(/(?:Rp\s?)?\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?/g)) {
    const bersih = m[0].replace(/^Rp\s?/, "").replace(/\./g, "").replace(",", ".");
    hasil.push({ nilai: Number(bersih), teks: m[0], indeks: m.index ?? 0 });
  }
  return hasil;
}

/** Ganti kata bilangan sederhana dengan digit ("lima apel" -> "5 apel", "dua puluh" -> "20"). */
export function kataKeAngka(teks: string): string {
  return teks
    .replace(/\b(dua|tiga|empat|lima|enam|tujuh|delapan|sembilan)\s+puluh(?:\s+(satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan))?\b/gi,
      (_, a, b) => String(SATUAN[a.toLowerCase()] * 10 + (b ? SATUAN[b.toLowerCase()] : 0)))
    .replace(/\b(dua|tiga|empat|lima|enam|tujuh|delapan|sembilan)\s+belas\b/gi, (_, a) => String(10 + SATUAN[a.toLowerCase()]))
    .replace(/\b(nol|satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh|sebelas|seratus|seribu)\b/gi, (k) => String(SATUAN[k.toLowerCase()]));
}

export const fmt = (x: number) => (Number.isInteger(x) ? x.toLocaleString("id-ID") : x.toLocaleString("id-ID", { maximumFractionDigits: 4 }));
export const rupiah = (x: number) => `Rp${Math.round(x).toLocaleString("id-ID")}`;
