/**
 * NORMALISASI BAHASA SANTAI: "gimana kalo gak dapet cuti?" -> "bagaimana kalau tidak dapet cuti?"
 *
 * Orang mengetik dengan singkatan dan bahasa sehari-hari, sedangkan dokumen ditulis baku.
 * Kamus di data/slang.txt mengganti setiap kata santai menjadi bentuk bakunya sebelum pesan
 * diproses oleh pencarian, pemahaman maksud, maupun GPT.
 * Penggantian dilakukan per KATA UTUH, tanpa mengubah huruf besar/kecil kata lain.
 */
import { existsSync, readFileSync } from "node:fs";

let kamus: Map<string, string> | null = null;

function muat(): Map<string, string> {
  if (kamus) return kamus;
  kamus = new Map();
  const path = process.env.SLANG ?? "data/slang.txt";
  if (!existsSync(path)) return kamus;
  for (const baris of readFileSync(path, "utf8").split(/\r?\n/)) {
    const b = baris.trim();
    if (!b || b.startsWith("#") || !b.includes("=")) continue;
    const [kiri, kanan] = b.split("=").map((s) => s.trim());
    for (const k of kiri.split("|")) if (k) kamus.set(k.toLowerCase(), kanan);
  }
  return kamus;
}

export function normalisasiSlang(teks: string): { teks: string; ganti: [string, string][] } {
  const m = muat();
  const ganti: [string, string][] = [];
  const hasil = teks.replace(/[\p{L}\p{N}]+/gu, (kata) => {
    const baku = m.get(kata.toLowerCase());
    if (!baku || baku.toLowerCase() === kata.toLowerCase()) return kata;
    ganti.push([kata, baku]);
    return baku;
  });
  return { teks: hasil, ganti };
}
