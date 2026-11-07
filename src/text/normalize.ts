/**
 * Normalisasi & tokenisasi teks bahasa Indonesia.
 *
 * Teks mentah -> daftar "term" yang dipakai indeks.
 * Setiap kata menghasilkan dua term:
 *   1. kata asli          (mis. "ditangguhkan") -> kecocokan persis bernilai tinggi
 *   2. "~" + kata dasar   (mis. "~tangguh")     -> "tangguhkan", "ditangguhkan", "penangguhan" saling cocok
 * Dengan begitu kesalahan stemmer tidak menghapus kecocokan persis.
 */
import { STOPWORDS } from "./stopwords.js";
import { stem } from "./stemmer.js";

/** Huruf kecil, buang tanda baca, pecah per kata. Angka dipertahankan (penting untuk "pasal 5", "12 hari"). */
export function tokenisasi(teks: string): string[] {
  return teks
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "") // buang aksen
    .replace(/[^a-z0-9]+/g, " ")
    .split(" ")
    .filter((t) => t.length > 0);
}

/**
 * Term untuk SATU kata: [kata, ~kataDasar]. Kosong jika stopword.
 * Kata dasar selalu ditulis, walau sama dengan kata aslinya, agar
 * "lama" di pertanyaan cocok dengan "lamanya" di dokumen (keduanya punya "~lama").
 */
export function termKata(t: string): string[] {
  if (STOPWORDS.has(t)) return [];
  if (/^\d+$/.test(t)) return [t];
  const s = stem(t);
  if (STOPWORDS.has(s)) return /(kah|lah|tah|pun)$/.test(t) ? [] : [t]; // "bolehkah" dibuang, kata lain tetap
  return s === t ? [t, "~" + t] : [t, "~" + s];
}

/** Token -> term untuk indeks / kueri (stopword dibuang, ditambah kata dasar). */
export function keTerm(teks: string): string[] {
  return tokenisasi(teks).flatMap(termKata);
}
