/**
 * MEMORI yang disimpan MANUAL oleh pengguna (bukan otomatis).
 *
 *   terminal:  :ingat nama saya Irun     :memori      :lupa 2      :lupa semua
 *   web     :  panel "Memori" (tambah / hapus), atau tombol "simpan" di pesan Anda
 *
 * Catatan disimpan di data/memory.json, jadi tetap ada setelah chatbot ditutup.
 * Memori dipakai dengan dua cara:
 *   1. DICARI seperti dokumen: "siapa nama saya?" cocok dengan catatan "nama saya Irun",
 *      lalu dijawab dengan sudut pandang dibalik: "Nama Anda Irun."
 *   2. NAMA pengguna diberikan ke GPT obrolan sebagai giliran perkenalan di awal prompt,
 *      sehingga sapaan bisa menyebut nama ("Halo, Irun!").
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { keTerm, termKata, tokenisasi } from "./text/normalize.js";

export interface CatatanMemori { id: number; teks: string; waktu: string }
export interface HasilMemori { catatan: CatatanMemori; cakupan: number; jawaban: string; cocok: string[] }

/** Kata benda yang lazim diberi akhiran -ku ("namaku", "rumahku"). Dibatasi agar "buku"/"berlaku" tidak ikut. */
const BENDA_KU =
  "nama|rumah|kantor|anak|istri|suami|ibu|ayah|adik|kakak|kucing|anjing|mobil|motor|hobi|teman|pacar|keluarga|pekerjaan|kerja|tugas|janji|jadwal|alamat|nomor|umur|usia|kota|sekolah|kampus|atasan|bos|makanan|minuman|warna|pesan|catatan";
const RE_KU = new RegExp(`\\b(${BENDA_KU})ku\\b`, "gi");

/** Kata ganti orang pertama: tanda bahwa pertanyaan menyangkut diri pengguna. */
export const PRIBADI = new RegExp(`\\b(saya|aku|gue|gw)\\b|\\b(${BENDA_KU})ku\\b`, "i");

export class Memori {
  constructor(readonly path = "data/memory.json") {}

  semua(): CatatanMemori[] {
    if (!existsSync(this.path)) return [];
    try { return JSON.parse(readFileSync(this.path, "utf8")); } catch { return []; }
  }

  private simpan(daftar: CatatanMemori[]): void {
    mkdirSync(dirname(this.path), { recursive: true });
    writeFileSync(this.path, JSON.stringify(daftar, null, 2));
  }

  tambah(teks: string): CatatanMemori {
    const daftar = this.semua();
    const c = { id: Math.max(0, ...daftar.map((x) => x.id)) + 1, teks: teks.trim().slice(0, 500), waktu: new Date().toISOString() };
    daftar.push(c);
    this.simpan(daftar);
    return c;
  }

  hapus(id: number): boolean {
    const daftar = this.semua();
    const sisa = daftar.filter((c) => c.id !== id);
    this.simpan(sisa);
    return sisa.length !== daftar.length;
  }

  hapusSemua(): void {
    this.simpan([]);
  }

  /**
   * Cari catatan yang paling cocok. Cakupan = bagian kata topik pertanyaan yang ada di catatan
   * (kata asli atau kata dasarnya). Catatan dianggap menjawab jika cakupan ≥ 60%.
   */
  cari(pertanyaan: string): HasilMemori | null {
    const kata = tokenisasi(pertanyaan).filter((t) => termKata(t).length > 0);
    if (!kata.length) return null;
    // skor = cakupan + 0.05 per kata yang cocok PERSIS ("nama" lebih cocok dengan "nama saya Irun"
    // daripada dengan "kucingku bernama Mimi", walau kata dasarnya sama)
    let terbaik: (HasilMemori & { skor: number }) | null = null;
    for (const c of this.semua()) {
      const ada = new Set(keTerm(c.teks));
      const cocok = kata.filter((t) => termKata(t).some((x) => ada.has(x)));
      const cakupan = cocok.length / kata.length;
      const skor = cakupan + 0.05 * kata.filter((t) => ada.has(t)).length;
      if (cakupan >= 0.6 && (!terbaik || skor > terbaik.skor || (skor === terbaik.skor && c.id > terbaik.catatan.id)))
        terbaik = { catatan: c, cakupan, cocok, skor, jawaban: ubahSudutPandang(c.teks) };
    }
    return terbaik;
  }

  /** Nama pengguna dari catatan seperti "nama saya Irun", "panggil aku Irun", "namaku Irun". */
  nama(): string | null {
    for (const c of this.semua().slice().reverse()) {
      const m = c.teks.match(/\b(?:nama\s+(?:saya|aku|gue)(?:\s+adalah)?|namaku(?:\s+adalah)?|panggil\s+(?:saya|aku))\s+([A-Za-zÀ-ÿ'-]+)/i);
      if (m) return m[1].charAt(0).toUpperCase() + m[1].slice(1);
    }
    return null;
  }
}

/** "nama saya Irun" -> "Nama Anda Irun."  (sudut pandang pengguna dibalik untuk jawaban bot) */
export function ubahSudutPandang(s: string): string {
  let t = s.trim()
    .replace(/\b(saya|aku|gue|gw)\b/gi, "Anda")
    .replace(RE_KU, (_, a) => `${a} Anda`);
  t = t.charAt(0).toUpperCase() + t.slice(1);
  return /[.!?]$/.test(t) ? t : t + ".";
}
