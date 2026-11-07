/**
 * KONTEKS PERCAKAPAN: memahami pertanyaan lanjutan dari jalur mana pun (aturan, Wikipedia, internet).
 *
 *   kata ganti  "kapan DIA lahir?"   + subjek "Soekarno"   -> "kapan Soekarno lahir?"
 *   -nya        "siapa istriNYA?"    + subjek "Soekarno"   -> "siapa istri Soekarno?"
 *   elipsis     "syaratnya?"         + topik "cuti besar"  -> dicari dengan konteks pertanyaan sebelumnya
 *
 * Penulisan ulang hanya CALON. Mesin percakapan mencari dengan kueri asli DAN kueri berkonteks,
 * lalu memakai yang hasil pencariannya lebih kuat. Jadi jika pengguna ganti topik ("berapa denda
 * keterlambatan?"), kueri asli yang menang dan konteks lama tidak ikut tercampur.
 */
import type { Jawaban } from "../answer/compose.js";
import { keTerm } from "../text/normalize.js";
import { stem } from "../text/stemmer.js";

const KATA_GANTI = /\b(dia|ia|beliau|mereka|tersebut|itu tadi|yang tadi)\b/i;
const NYA = /\b([a-z]{3,})nya\b/i;
/** Kata pembingkai: bentuk pertanyaan, bukan topik ("SYARATnya?", "berapa LAMA?"). */
export const PEMBINGKAI = new Set("syarat persyaratan cara prosedur mekanisme ketentuan aturan lama jumlah sanksi waktu batas hak kewajiban dasar".split(" "));

/** Jumlah kata TOPIK di pertanyaan (tanpa stopword dan tanpa kata pembingkai). */
export function jumlahKataTopik(pesan: string): number {
  return keTerm(pesan).filter((t) => !t.startsWith("~") && !PEMBINGKAI.has(stem(t.replace(/nya$/, "")))).length;
}

/** Kata topik pertanyaan sebelumnya ("berapa lama cuti besar?" -> "cuti besar"), untuk mengisi elipsis. */
export function kataTopik(pertanyaan: string): string {
  return keTerm(pertanyaan).filter((t) => !t.startsWith("~") && !PEMBINGKAI.has(stem(t))).join(" ");
}

/** Pertanyaan memakai kata ganti / -nya yang merujuk subjek sebelumnya ("kapan dia lahir?", "istrinya?"). */
export function adaRujukan(pesan: string): boolean {
  return KATA_GANTI.test(pesan) || (NYA.test(pesan) && !/^(bagaimana|apa|siapa|berapa)$/i.test(pesan.match(NYA)![1]));
}

/** Penjelasan MENGAPA pesan dianggap merujuk topik sebelumnya (untuk panel cara kerja), atau null. */
export function jenisRujukan(pesan: string): string | null {
  const g = pesan.match(KATA_GANTI);
  if (g) return `kata ganti "${g[0]}" merujuk sesuatu yang dibahas sebelumnya`;
  const n = pesan.match(NYA);
  if (n && !/^(bagaimana|apa|siapa|berapa)$/i.test(n[1])) return `akhiran -nya pada "${n[0]}" (= ${n[1]} milik/dari subjek sebelumnya)`;
  const k = jumlahKataTopik(pesan);
  if (k <= 1) return `elipsis: hanya ${k} kata topik di pesan, bagian yang hilang diisi topik sebelumnya`;
  return null;
}

/** Pertanyaan ini kemungkinan bergantung pada topik sebelumnya? */
export function mungkinLanjutan(pesan: string): boolean {
  return KATA_GANTI.test(pesan) || NYA.test(pesan) || jumlahKataTopik(pesan) <= 1;
}

/** Ganti kata ganti / akhiran -nya dengan subjek. Jika tidak ada, subjek ditambahkan di akhir. */
export function tulisUlang(pesan: string, subjek: string): string {
  if (!subjek) return pesan;
  if (KATA_GANTI.test(pesan)) return pesan.replace(KATA_GANTI, subjek);
  const m = pesan.match(NYA);
  if (m && !/^(bagaimana|apa|siapa|berapa)$/i.test(m[1])) return pesan.replace(NYA, `$1 ${subjek}`);
  return `${pesan.replace(/[?.!]+$/, "")} ${subjek}?`;
}

const PERINGKAT = { rendah: 0, sedang: 1, tinggi: 2 } as const;

/**
 * Bandingkan dua hasil pencarian: berkonteks vs asli. Berkonteks dipakai hanya jika jelas lebih kuat:
 * keyakinan lebih tinggi, atau keyakinan sama tetapi cakupan pasal teratas ≥ 10 poin lebih besar.
 */
export function konteksLebihBaik(konteks: Jawaban, asli: Jawaban): boolean {
  const a = PERINGKAT[konteks.keyakinan], b = PERINGKAT[asli.keyakinan];
  if (a !== b) return a > b;
  if (a === 0) return false;
  return (konteks.hasil[0]?.cakupan ?? 0) > (asli.hasil[0]?.cakupan ?? 0) + 0.1;
}
