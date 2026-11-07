/**
 * Memotong dokumen aturan menjadi potongan per PASAL.
 *
 * Kenapa per pasal? BM25 bekerja paling baik jika satu "dokumen" membahas satu hal.
 * Peraturan Indonesia sudah terstruktur:  BAB -> Bagian -> Pasal -> ayat (1), (2), ...
 * Jadi pasal adalah satuan makna yang alami, dan sekaligus rujukan sumber yang jelas.
 *
 * Sebuah peraturan punya sampai tiga wilayah, dan nomor pasal bisa berulang di tiap wilayah:
 *   1. Batang tubuh   "Pasal 5"                 -> label "Pasal 5"
 *   2. PENJELASAN     "Pasal 5  Cukup jelas."   -> label "Penjelasan Pasal 5" ("Cukup jelas" dibuang)
 *   3. LAMPIRAN       "A. CUTI TAHUNAN"         -> label "Lampiran: A. CUTI TAHUNAN"
 * Peraturan perubahan memakai angka Romawi: "Pasal I", "Pasal II".
 *
 * Aturan lain:
 *   - Rujukan di tengah kalimat ("dimaksud dalam Pasal 5") tidak memotong;
 *     hanya baris yang isinya persis "Pasal N".
 *   - Teks sebelum pasal pertama = "Pembukaan" (judul, menimbang, mengingat).
 *   - Potongan panjang dipecah jendela 300 kata (overlap 50).
 *   - Dokumen tanpa pasal sama sekali dipecah jendela 200 kata (overlap 50).
 */
import type { Dokumen, Potongan } from "../types.js";
import { bersihkanHalaman, perbaikiOCR } from "./clean.js";

const RE_PASAL = /^pasal\s+(\d+[a-z]?|[ivxlc]+)\s*$/i;
const RE_BAB = /^bab\s+[ivxlcdm\d]+\b/i;
const RE_BAGIAN =
  /^(bagian|paragraf)\s+(kesatu|pertama|kedua|ketiga|keempat|kelima|keenam|ketujuh|kedelapan|kesembilan|kesepuluh|\w+belas|\d+)\s*$/i;
const RE_PENJELASAN = /^penjelasan(\s+atas.*)?$/i;
const RE_LAMPIRAN = /^LAMPIRAN\b/; // huruf besar: judul lampiran, bukan kata "lampiran" di kalimat
const RE_JUDUL_LAMPIRAN = /^(?:[A-Z]|[IVXL]+)\.\s+[A-Z][A-Z0-9 ,/()&-]{3,}$/;
const RE_JENIS = /(UNDANG-UNDANG|PERATURAN|KEPUTUSAN|INSTRUKSI|SURAT EDARAN|QANUN)/i;

interface Baris { teks: string; halaman: number; nomor: number }
type Wilayah = "batang" | "penjelasan" | "lampiran";

function jendela(kata: string[], ukuran: number, overlap: number): string[] {
  if (kata.length <= ukuran) return [kata.join(" ")];
  const hasil: string[] = [];
  for (let i = 0; i < kata.length; i += ukuran - overlap) {
    hasil.push(kata.slice(i, i + ukuran).join(" "));
    if (i + ukuran >= kata.length) break;
  }
  return hasil;
}

/** Judul = baris dari kata jenis peraturan pertama sampai sebelum "DENGAN RAHMAT"/"Menimbang"/BAB/Pasal. */
function ambilJudul(baris: Baris[]): string {
  const awal: string[] = [];
  let mulai = false;
  for (const b of baris.slice(0, 40)) {
    // Berhenti hanya SETELAH judul mulai terbaca: di PDF hasil OCR, label "Menimbang :"
    // kadang muncul di urutan paling atas, sebelum judulnya.
    if (mulai && (/^(dengan rahmat|menimbang|mengingat|menetapkan)/i.test(b.teks) || RE_BAB.test(b.teks) || RE_PASAL.test(b.teks))) break;
    if (!mulai && !RE_JENIS.test(b.teks)) continue;
    if (!mulai && /^(menimbang|mengingat|menetapkan)/i.test(b.teks)) continue;
    mulai = true;
    awal.push(b.teks);
    if (awal.length >= 8) break;
  }
  if (awal.length === 0) return ""; // tidak ada judul resmi: nama file dipakai sebagai label
  const j = awal.join(" ").replace(/\s+/g, " ").trim();
  return j.length > 220 ? j.slice(0, 217) + "..." : j;
}

export function potongDokumen(dok: Dokumen, idAwal: number): Potongan[] {
  // 1. Bersihkan header/footer, lalu ratakan semua halaman menjadi daftar baris bernomor
  const halaman = dok.adaHalaman ? bersihkanHalaman(dok.halaman) : dok.halaman;
  const baris: Baris[] = [];
  let nomor = 0;
  halaman.forEach((isi, h) => {
    for (const mentah of isi.split(/\r?\n/)) {
      nomor++;
      // buang heading markdown, rapikan spasi, perbaiki kesalahan OCR umum
      const teks = perbaikiOCR(mentah.replace(/^#+\s*/, "").replace(/\s+/g, " ").trim());
      if (teks) baris.push({ teks, halaman: h + 1, nomor });
    }
  });
  if (baris.length === 0) return [];

  const judul = ambilJudul(baris);
  const hasil: Potongan[] = [];
  let id = idAwal;
  const tambah = (bab: string, label: string, isi: Baris[], ukuran = 300) => {
    if (isi.length === 0) return;
    label = label || "Pengantar bab";
    const kata = isi.map((b) => b.teks).join("\n").split(/ +/);
    if (label.startsWith("Penjelasan Pasal") && kata.length <= 6) return; // "Cukup jelas."
    const bagian = jendela(kata, ukuran, 50);
    bagian.forEach((teks, i) =>
      hasil.push({
        id: id++,
        file: dok.file,
        judul,
        bab,
        pasal: bagian.length > 1 ? `${label} (bagian ${i + 1})` : label,
        halaman: dok.adaHalaman ? isi[0].halaman : null,
        baris: isi[0].nomor,
        teks,
      }),
    );
  };

  if (!baris.some((b) => RE_PASAL.test(b.teks))) {
    // 2a. Dokumen bebas (tanpa pasal): jendela 200 kata
    for (let i = 0; i < baris.length; ) {
      // kumpulkan ±200 kata agar halaman/baris awal tiap potongan tetap tercatat
      const isi: Baris[] = [];
      let n = 0;
      while (i < baris.length && n < 200) { isi.push(baris[i]); n += baris[i].teks.split(" ").length; i++; }
      tambah("", `Bagian ${hasil.length + 1}`, isi, 400);
    }
    return hasil;
  }

  // 2b. Dokumen berpasal
  let wilayah: Wilayah = "batang";
  let bab = "";
  let label = "Pembukaan";
  let isi: Baris[] = [];
  let tungguJudulBab = false;
  const awali = (baru: string, b: Baris | null) => { tambah(bab, label, isi); label = baru; isi = b ? [b] : []; };

  for (const b of baris) {
    if (RE_PENJELASAN.test(b.teks)) { awali("Penjelasan Umum", null); wilayah = "penjelasan"; bab = ""; continue; }
    if (RE_LAMPIRAN.test(b.teks) && b.teks.length < 150) { awali("Lampiran", b); wilayah = "lampiran"; bab = ""; continue; }

    if (wilayah === "lampiran") {
      if (RE_JUDUL_LAMPIRAN.test(b.teks)) { awali(`Lampiran: ${b.teks}`, b); continue; }
      isi.push(b);
      continue;
    }

    const m = b.teks.match(RE_PASAL);
    if (m) {
      const n = m[1].toUpperCase();
      awali(wilayah === "penjelasan" ? `Penjelasan Pasal ${n}` : `Pasal ${n}`, b);
      tungguJudulBab = false;
      continue;
    }
    if (RE_BAB.test(b.teks) && wilayah === "batang") {
      awali("", null);
      bab = b.teks;
      tungguJudulBab = true; // baris berikutnya biasanya nama bab, mis. "KETENTUAN UMUM"
      continue;
    }
    if (tungguJudulBab && !RE_BAGIAN.test(b.teks)) {
      bab += " " + b.teks;
      tungguJudulBab = false;
      continue;
    }
    tungguJudulBab = false;
    if (RE_BAGIAN.test(b.teks)) continue; // judul "Bagian Kedua" hanya penanda struktur
    isi.push(b);
  }
  tambah(bab, label || "Penutup", isi);
  return hasil;
}
