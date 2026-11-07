/**
 * Stemmer ringan bahasa Indonesia (buatan sendiri, tanpa kamus).
 *
 * Bahasa Indonesia membentuk kata dengan imbuhan:
 *   [awalan] + kata dasar + [akhiran] + [kata ganti milik] + [partikel]
 *   contoh:  per + timbang + an + nya      -> "pertimbangannya"
 *            me  + tetap   + kan           -> "menetapkan"  (t luluh menjadi n)
 *
 * Urutan pelepasan (versi sederhana algoritma Nazief-Adriani):
 *   1. partikel        -lah -kah -tah -pun
 *   2. kata ganti      -ku -mu -nya
 *   3. akhiran         -kan -an -i
 *   4. awalan (maks 2) di- ke- se- ter- ber- per- pe- me- (dengan peluluhan)
 * Setiap langkah hanya dilakukan jika sisa kata minimal 3 huruf.
 * Tanpa kamus, hasilnya kadang bukan kata baku. Itu tidak masalah selama konsisten
 * antara dokumen dan pertanyaan.
 */
const VOKAL = /^[aiueo]/;

function lepasAkhiran(w: string): string {
  for (const p of ["lah", "kah", "tah", "pun"])
    if (w.endsWith(p) && w.length - p.length >= 4) { w = w.slice(0, -p.length); break; }
  for (const p of ["nya", "ku", "mu"])
    if (w.endsWith(p) && w.length - p.length >= 4) { w = w.slice(0, -p.length); break; }
  if (w.endsWith("kan") && w.length - 3 >= 3) return w.slice(0, -3);
  if (w.endsWith("an") && w.length - 2 >= 4) return w.slice(0, -2);
  // -i hanya jika didahului konsonan ("memenuhi" -> "memenuh"), agar "pegawai" tetap utuh
  if (/[^aiueo]i$/.test(w) && w.length - 1 >= 4) return w.slice(0, -1);
  return w;
}

function lepasAwalan(w: string): string {
  const potong = (n: number, ganti = "") => {
    const sisa = ganti + w.slice(n);
    return sisa.length >= 3 ? sisa : null;
  };
  let r: string | null = null;
  if (w.startsWith("meng") || w.startsWith("peng")) r = potong(4);           // mengatur -> atur
  else if (w.startsWith("meny") || w.startsWith("peny")) r = potong(4, "s"); // menyusun -> susun
  else if (w.startsWith("mem") || w.startsWith("pem"))
    r = VOKAL.test(w.slice(3)) ? potong(3, "p") : potong(3);                 // memakai -> pakai, membayar -> bayar
  else if (w.startsWith("men") || w.startsWith("pen"))
    r = VOKAL.test(w.slice(3)) ? potong(3, "t") : potong(3);                 // menetapkan -> tetap, mendapat -> dapat
  else if (w.startsWith("ber") || w.startsWith("ter") || w.startsWith("per")) r = potong(3);
  else if (w.startsWith("me") || w.startsWith("pe")) r = potong(2);          // melapor -> lapor
  else if (w.startsWith("di") || w.startsWith("ke") || w.startsWith("se")) r = potong(2);
  return r ?? w;
}

const cache = new Map<string, string>();

export function stem(kata: string): string {
  const ada = cache.get(kata);
  if (ada) return ada;
  let w = kata;
  if (w.length > 4) {
    w = lepasAkhiran(w);
    for (let i = 0; i < 2; i++) {
      const baru = lepasAwalan(w);
      if (baru === w) break;
      w = baru;
    }
  }
  cache.set(kata, w);
  return w;
}
