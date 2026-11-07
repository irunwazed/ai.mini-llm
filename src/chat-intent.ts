/**
 * PENGENAL NIAT OBROLAN: pesan "lucu juga kamu" -> niat "pujian".
 *
 * Contoh kalimat setiap niat diambil dari dataset percakapan (src/gpt/chat-data.ts, NIAT).
 * Kemiripan pesan dengan contoh dihitung memakai himpunan kata dasar:
 *   Jaccard(A, B) = |A ∩ B| / |A ∪ B|
 * dan bonus 0.9 jika SEMUA kata contoh ada di pesan ("selamat pagi" di dalam
 * "selamat pagi kak, gimana kabarnya?"). Jika dua niat sama kuat, yang cocok di
 * bagian AKHIR pesan menang (pertanyaan biasanya diletakkan di akhir).
 *
 * Dipakai untuk dua hal:
 *   1. memutuskan apakah pesan memang obrolan (bukan pertanyaan fakta / teks acak)
 *   2. memeriksa balasan GPT: niat balasan harus cocok dengan niat pesan
 */
import { NIAT } from "./gpt/chat-data.js";
import { stem } from "./text/stemmer.js";
import { tokenisasi } from "./text/normalize.js";
import { normalisasiSlang } from "./text/slang.js";

export interface HasilNiat { niat: string; skor: number; contoh: string }

const kata = (s: string) => tokenisasi(normalisasiSlang(s).teks).map((t) => stem(t));

/** Skor minimal agar pesan dianggap benar-benar obrolan dengan niat tertentu. */
export const AMBANG_NIAT = 0.6;

const contohUser = Object.entries(NIAT).flatMap(([niat, n]) => n.user.map((u) => ({ niat, kata: kata(u), teks: u })));
const contohBot = Object.entries(NIAT).flatMap(([niat, n]) => n.bot.map((b) => ({ niat, kata: new Set(kata(b)), teks: b })));

/** Niat-niat yang saling setara untuk keperluan pemeriksaan (semua bentuk salam). */
const KELUARGA: Record<string, string> = { pagi: "salam", siang: "salam", sore: "salam", malam: "salam", salamIslam: "salam", tes: "salam" };
export const keluarga = (n: string) => KELUARGA[n] ?? n;

/** Niat pesan pengguna. */
export function niatPesan(pesan: string): HasilNiat | null {
  const k = kata(pesan);
  if (!k.length) return null;
  const himpunan = new Set(k);
  let terbaik: (HasilNiat & { posisi: number }) | null = null;
  for (const c of contohUser) {
    if (!c.kata.length) continue;
    const irisan = c.kata.filter((x) => himpunan.has(x)).length;
    const gabungan = new Set([...k, ...c.kata]).size;
    let skor = irisan / gabungan;
    if (irisan === c.kata.length) skor = Math.max(skor, 0.9);
    const posisi = Math.max(...c.kata.map((x) => k.lastIndexOf(x)));
    if (!terbaik || skor > terbaik.skor || (skor === terbaik.skor && posisi > terbaik.posisi))
      terbaik = { niat: c.niat, skor, contoh: c.teks, posisi };
  }
  return terbaik && terbaik.skor > 0 ? { niat: terbaik.niat, skor: terbaik.skor, contoh: terbaik.contoh } : null;
}

/** Niat sebuah balasan bot: balasan contoh yang paling mirip. */
export function niatBalasan(balasan: string): HasilNiat | null {
  const k = new Set(kata(balasan));
  let terbaik: HasilNiat | null = null;
  for (const c of contohBot) {
    const irisan = [...c.kata].filter((x) => k.has(x)).length;
    const skor = irisan / new Set([...k, ...c.kata]).size;
    if (!terbaik || skor > terbaik.skor) terbaik = { niat: c.niat, skor, contoh: c.teks };
  }
  return terbaik;
}

/** Balasan contoh dari sebuah niat (dipakai bila semua balasan GPT ditolak). */
export function balasanTemplat(niat: string, urutan = 0): string {
  const b = NIAT[niat]?.bot ?? [];
  return b[urutan % Math.max(1, b.length)] ?? "Baik. Ada yang bisa saya bantu?";
}
