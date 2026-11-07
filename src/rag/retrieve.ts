/**
 * AMBIL KONTEKS untuk RAG: bukan hanya pasal teratas, tetapi juga
 *   - beberapa pasal berikutnya dari hasil pencarian
 *   - pasal yang DIRUJUK  ("sebagaimana dimaksud dalam Pasal 5" -> Pasal 5 dari peraturan yang sama)
 *   - bagian lanjutan pasal yang sama ("Pasal 7 (bagian 2)")
 *   - catatan memori pengguna yang relevan ("saya PNS golongan III")
 * Setiap potongan diberi alasan masuk, untuk ditampilkan di panel cara kerja.
 */
import type { Jawaban } from "../answer/compose.js";
import type { CatatanMemori } from "../memory.js";
import type { Model } from "../model.js";
import { keTerm } from "../text/normalize.js";
import type { Potongan } from "../types.js";

export type AlasanKonteks = "hasil cari" | "pasal dirujuk" | "lanjutan pasal" | "catatan Anda";
export interface PotonganKonteks { teks: string; label: string; alasan: AlasanKonteks; skor: number; potongan?: Potongan }

const dasarPasal = (p: string) => p.replace(/\s*\(bagian \d+\)$/, "");

export function ambilKonteks(model: Model, j: Jawaban, opsi: { maks?: number; memori?: CatatanMemori[] } = {}): PotonganKonteks[] {
  const maks = opsi.maks ?? 6;
  const keluar: PotonganKonteks[] = [];
  const sudah = new Set<number>();
  const label = (p: Potongan) => `${model.info[p.file]?.judulPendek ?? p.file} · ${p.pasal}`;
  const masuk = (p: Potongan, alasan: AlasanKonteks, skor: number) => {
    if (sudah.has(p.id) || keluar.length >= maks) return;
    sudah.add(p.id);
    keluar.push({ teks: p.teks, label: label(p), alasan, skor, potongan: p });
  };

  // Catatan memori pengguna lebih dulu: konteks pribadi singkat, sering menentukan jawaban
  for (const c of opsi.memori ?? []) keluar.push({ teks: c.teks, label: `Catatan Anda #${c.id}`, alasan: "catatan Anda", skor: 1 });

  const atas = j.hasil.filter((h) => h.potongan.pasal !== "Pembukaan").slice(0, 3);
  for (const h of atas) {
    const p = h.potongan;
    masuk(p, "hasil cari", h.skor);
    // bagian lanjutan pasal yang sama
    const lanjut = model.potongan[p.id + 1];
    if (lanjut && lanjut.file === p.file && dasarPasal(lanjut.pasal) === dasarPasal(p.pasal) && lanjut.pasal !== p.pasal) masuk(lanjut, "lanjutan pasal", h.skor * 0.8);
    // pasal yang dirujuk di dalam teks pasal ini
    for (const m of p.teks.matchAll(/\b(?:dimaksud\s+(?:dalam|pada)\s+)?Pasal\s+(\d+[A-Z]?)\b/g)) {
      const nomor = `Pasal ${m[1]}`;
      if (nomor === dasarPasal(p.pasal)) continue;
      const dirujuk = model.potongan.find((q) => q.file === p.file && dasarPasal(q.pasal) === nomor);
      if (dirujuk) masuk(dirujuk, "pasal dirujuk", h.skor * 0.6);
    }
  }
  return keluar;
}

/**
 * Catatan memori yang relevan untuk pertanyaan: memuat kata topik pertanyaan,
 * atau berupa fakta pribadi ("saya …") saat pertanyaannya juga tentang diri pengguna.
 */
export function memoriRelevan(catatan: CatatanMemori[], pertanyaan: string, pribadi: boolean): CatatanMemori[] {
  const q = new Set(keTerm(pertanyaan));
  return catatan.filter((c) => {
    const t = keTerm(c.teks);
    return t.some((x) => q.has(x)) || (pribadi && /\b(saya|aku)\b/i.test(c.teks) && !/\bnama\b/i.test(c.teks));
  }).slice(-3);
}
