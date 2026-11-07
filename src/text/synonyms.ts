/**
 * Perluasan pertanyaan dengan sinonim (query expansion).
 *
 * Masalah pencarian leksikal (berbasis kata): "mobil" tidak akan cocok dengan "kendaraan".
 * Solusi paling sederhana: kamus padanan. Kata di pertanyaan diperluas menjadi
 * kelompok { kata asli, padanan 1, padanan 2, ... }. Kelompok dianggap cocok jika
 * salah satu anggotanya ada di pasal.
 * (Nanti, embedding dari jaringan saraf bisa mempelajari kemiripan ini sendiri dari data.)
 */
import { existsSync, readFileSync } from "node:fs";
import { stem } from "./stemmer.js";

let kamus: Map<string, string[]> | null = null;

function muat(): Map<string, string[]> {
  if (kamus) return kamus;
  kamus = new Map();
  const path = process.env.SINONIM ?? "data/synonyms.txt";
  if (!existsSync(path)) return kamus;
  for (const baris of readFileSync(path, "utf8").split(/\r?\n/)) {
    const b = baris.trim();
    if (!b || b.startsWith("#") || !b.includes("=")) continue;
    const [kiri, kanan] = b.split("=");
    const padanan = kanan.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);
    const k = kiri.trim().toLowerCase();
    kamus.set(k, [...(kamus.get(k) ?? []), ...padanan]);
  }
  return kamus;
}

/** Padanan untuk satu kata (dicari dari bentuk asli lalu kata dasarnya). */
export function padanan(kata: string): string[] {
  const m = muat();
  return m.get(kata) ?? m.get(stem(kata)) ?? [];
}
