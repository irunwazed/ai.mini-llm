/**
 * AGENT PENALARAN: loop "Pikir -> Aksi -> Hasil -> Cek", diulang sampai jawaban lolos pemeriksaan.
 *
 * Pada asisten AI besar, "otak" yang memilih langkah adalah model bahasa raksasa. GPT kita
 * terlalu kecil untuk itu, jadi otaknya dibuat berbasis aturan:
 *   1. PIKIR : kenali jenis soal dari pola kalimat, lalu pilih alat yang cocok
 *   2. AKSI  : jalankan alat (kalkulator, kalender, barisan, persamaan, laju kerja, silogisme, teka-teki)
 *   3. HASIL : catat jawaban + langkah hitungnya
 *   4. CEK   : masukkan jawaban kembali ke soal (atau hitung ulang dengan cara lain);
 *              jika gagal, kembali ke langkah 1 dengan alat berikutnya
 * Semua langkah dicatat agar terlihat di panel "cara kerja".
 */
import { cariEkspresi, hitung, tafsiranLain, urai } from "./calculator.js";
import { kalender } from "./calendar.js";
import { persamaan } from "./linear.js";
import { fmt, kataKeAngka } from "./numbers.js";
import { tekaTeki } from "./puzzles.js";
import { barisan } from "./sequence.js";
import { silogisme } from "./syllogism.js";
import type { Alat, HasilAlat } from "./types.js";
import { bandingBerat, lajuKerja, selangWaktu } from "./word-problems.js";

export interface LangkahAgent { pikir: string; aksi: string; hasil: string; cek: string; rincian: string[] }
export interface HasilAgent { terjawab: boolean; teks: string; langkah: LangkahAgent[]; alat?: string }

/** "8 dibagi 2" -> "8 ÷ 2", "lima kali tiga" -> "5 × 3" */
function kataOperator(t: string): string {
  return kataKeAngka(t)
    .replace(/(\d)\s*(ditambah|tambah|plus)\s*(\d)/gi, "$1 + $3")
    .replace(/(\d)\s*(dikurangi|kurang|minus)\s*(\d)/gi, "$1 − $3")
    .replace(/(\d)\s*(dikali(kan)?|kali)\s*(\d)/gi, "$1 × $4")
    .replace(/(\d)\s*(dibagi|bagi)\s*(\d)/gi, "$1 ÷ $3")
    .replace(/(\d)\s*pangkat\s*(\d)/gi, "$1 ^ $2");
}

const kalkulator: Alat = {
  nama: "kalkulator ekspresi (urutan operasi)",
  pikir: "Ada ekspresi hitung, jadi uraikan menurut urutan operasi: kurung, pangkat, kali/bagi kiri ke kanan, lalu tambah/kurang.",
  cocok: (t) => !!cariEkspresi(t),
  coba: (t) => {
    const e = cariEkspresi(t);
    if (!e) return null;
    const langkah: string[] = [];
    const v = hitung(urai(e), langkah);
    // Cek independen: hitung ulang memakai kalkulator bawaan JavaScript (perkalian tersirat dibuat eksplisit)
    const js = e.replace(/[×xX]/g, "*").replace(/[÷:]/g, "/").replace(/[−–]/g, "-").replace(/\^/g, "**").replace(/(\d)\.(\d{3})\b/g, "$1$2").replace(/,/g, ".").replace(/(\d|\))\s*\(/g, "$1*(");
    let pembanding = NaN;
    try { pembanding = Function(`"use strict"; return (${js});`)() as number; } catch { /* abaikan */ }
    const lain = tafsiranLain(e);
    return {
      jawaban: `${e} = ${fmt(v)}`,
      penjelasan: langkah,
      cek: { lolos: Math.abs(pembanding - v) < 1e-9, teks: `dihitung ulang dengan kalkulator bawaan: ${fmt(pembanding)} ${Math.abs(pembanding - v) < 1e-9 ? "✓" : "✗"}` },
      catatan: lain ? `Penulisan ini ambigu. Dengan aturan kiri-ke-kanan hasilnya ${fmt(v)}; jika perkalian tersirat dikerjakan lebih dulu, ${lain.teks} = ${fmt(lain.nilai)}. Tulis dengan kurung yang jelas agar tidak ambigu.` : undefined,
    };
  },
};

/** Urutan penting: teka-teki jebakan diperiksa lebih dulu agar tidak "dihitung" secara keliru. */
const ALAT: Alat[] = [tekaTeki, silogisme, kalender, barisan, persamaan, lajuKerja, selangWaktu, bandingBerat, kalkulator];

/** Apakah ada alat penalaran yang mengenali pesan ini sebagai soal hitung/logika? */
export function adaAlatPenalaran(pesan: string): boolean {
  const t = kataOperator(pesan);
  return ALAT.some((a) => { try { return a.cocok(t); } catch { return false; } });
}

export function jalankanAgent(pesan: string, maksLangkah = 6): HasilAgent | null {
  const t = kataOperator(pesan);
  const calon = ALAT.filter((a) => { try { return a.cocok(t); } catch { return false; } });
  if (!calon.length) return null;
  const langkah: LangkahAgent[] = [];
  for (const a of calon.slice(0, maksLangkah)) {
    let r: HasilAlat | null = null, galat = "";
    try { r = a.coba(t); } catch (e) { galat = (e as Error).message; }
    if (!r) {
      langkah.push({ pikir: a.pikir, aksi: a.nama, hasil: galat ? `gagal: ${galat}` : "alat tidak bisa membaca soal ini", cek: "gagal -> coba alat lain", rincian: [] });
      continue;
    }
    const cek = r.cek ? `${r.cek.teks}` : "tidak ada pemeriksaan otomatis untuk jenis ini";
    if (r.cek && !r.cek.lolos) {
      langkah.push({ pikir: a.pikir, aksi: a.nama, hasil: r.jawaban, cek: `${cek} -> jawaban DITOLAK, coba alat lain`, rincian: r.penjelasan });
      continue;
    }
    langkah.push({ pikir: a.pikir, aksi: a.nama, hasil: r.jawaban, cek, rincian: r.penjelasan });
    const teks = [
      r.jawaban.endsWith(".") ? r.jawaban : r.jawaban + ".",
      "",
      "Penjelasan:",
      ...r.penjelasan.map((p) => `• ${p}`),
      ...(r.catatan && !r.catatan.startsWith("teka-teki") ? ["", `Catatan: ${r.catatan}`] : []),
    ].join("\n");
    return { terjawab: true, teks, langkah, alat: a.nama + (r.catatan?.startsWith("teka-teki") ? ` — ${r.catatan}` : "") };
  }
  return { terjawab: false, teks: "", langkah };
}
