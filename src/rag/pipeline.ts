/**
 * RAG = Retrieval-Augmented Generation, versi "pembaca" dari nol:
 *   1. RETRIEVAL : potongan konteks (src/rag/retrieve.ts)
 *   2. MEMBACA   : GPT pembaca menunjuk rentang jawaban di setiap potongan (src/gpt/reader.ts)
 *   3. VERIFIKASI: jawaban pembaca hanya dipakai jika
 *        - keyakinan ≥ ambang (dikalibrasi saat training, disimpan di gpt.json)
 *        - bentuknya cocok dengan pertanyaan ("berapa" -> ada angka, "kapan" -> ada waktu)
 *        - kalimat asalnya memuat ≥ 50% bobot kata topik pertanyaan
 *      Jika gagal, penyusun aturan (src/answer/compose.ts) yang menjawab, seperti sebelumnya.
 * Pertanyaan berbentuk daftar (syarat, prosedur) atau ya/tidak tetap dijawab penyusun aturan,
 * karena pembaca hanya bisa menunjuk SATU rentang pendek.
 */
import type { Jawaban } from "../answer/compose.js";
import { besaran, satuanDiminta } from "../answer/question-type.js";
import type { Tokenizer } from "../gpt/bpe.js";
import type { GPT } from "../gpt/model.js";
import { baca, type HasilBaca } from "../gpt/reader.js";
import type { CatatanMemori } from "../memory.js";
import type { Model } from "../model.js";
import { keTerm } from "../text/normalize.js";
import { ambilKonteks, type PotonganKonteks } from "./retrieve.js";

export interface Pembaca { nama: string; model: GPT; tok: Tokenizer; hurufKecil: boolean; ambang: number; f1: number }
export interface HasilRAG {
  diterima: boolean;
  jawaban: string;
  kalimat: string;
  label: string;               // sumber potongan yang dibaca
  keyakinan: number;
  alasan: string[];            // alasan diterima/ditolak
  konteks: PotonganKonteks[];
  dibaca: { label: string; keyakinan: number; jawaban: string }[]; // hasil per potongan
  baca: HasilBaca | null;      // rincian potongan terbaik (untuk peta panas)
}

const JENIS_PEMBACA = new Set(["jumlah", "waktu", "definisi", "umum"]);

export function bacaRAG(p: Pembaca, model: Model, j: Jawaban, pertanyaan: string, opsi: { memori?: CatatanMemori[] } = {}): HasilRAG {
  const konteks = ambilKonteks(model, j, { memori: opsi.memori });
  const hasil: HasilRAG = { diterima: false, jawaban: "", kalimat: "", label: "", keyakinan: 0, alasan: [], konteks, dibaca: [], baca: null };
  if (!JENIS_PEMBACA.has(j.jenis)) {
    hasil.alasan.push(`pertanyaan jenis "${j.jenis}" butuh daftar/ya-tidak -> dijawab penyusun aturan`);
    return hasil;
  }
  let terbaik: { h: HasilBaca; k: PotonganKonteks } | null = null;
  for (const k of konteks.filter((x) => x.alasan !== "catatan Anda").slice(0, 5)) {
    const h = baca(p.model, p.tok, pertanyaan, k.teks, p.hurufKecil);
    hasil.dibaca.push({ label: k.label, keyakinan: h.keyakinan, jawaban: h.jawaban });
    if (!terbaik || h.keyakinan > terbaik.h.keyakinan) terbaik = { h, k };
  }
  if (!terbaik || !terbaik.h.jawaban) { hasil.alasan.push("pembaca tidak menemukan rentang jawaban"); return hasil; }
  const { h, k } = terbaik;
  Object.assign(hasil, { jawaban: h.jawaban, kalimat: h.kalimat, label: k.label, keyakinan: h.keyakinan, baca: h });

  // Verifikasi
  if (h.keyakinan < p.ambang) hasil.alasan.push(`keyakinan ${(100 * h.keyakinan).toFixed(0)}% < ambang ${(100 * p.ambang).toFixed(0)}%`);
  if (j.jenis === "jumlah" && !/\d/.test(h.jawaban)) hasil.alasan.push(`ditanya "berapa" tetapi rentang tanpa angka`);
  const satuan = satuanDiminta(pertanyaan);
  if (j.jenis === "jumlah" && satuan && !satuan.test(h.jawaban.toLowerCase())) hasil.alasan.push(`rentang tanpa satuan yang diminta (${satuan.source})`);
  if (/\(\s*$|^\s*\)|\(\d+\)\s/.test(h.jawaban)) hasil.alasan.push("rentang terpotong (kurung tidak lengkap / nomor ayat)");
  if (j.jenis === "jumlah") {
    // rentang "berapa" harus berupa BESARAN UTUH: "3 (tiga) bulan" -> "3 bulan" lolos;
    // "6 (enam) hari kerja ke tahun" (ada kata sisa) -> tanda batas rentang tidak tepat -> ditolak
    const bersih = h.jawaban.replace(/\s*\((?:[a-z]+\s?)+\)/gi, "").replace(/\s+/g, " ").trim();
    const b = besaran(bersih).sort((x, y) => y.length - x.length)[0];
    if (!b || b.length < 0.8 * bersih.length) hasil.alasan.push(`rentang "${bersih}" bukan besaran utuh`);
    else hasil.jawaban = bersih;
  }
  if (j.jenis === "waktu" && !/\d|\b(hari|bulan|tahun|sebelum|setelah|paling lambat)\b/i.test(h.jawaban)) hasil.alasan.push("ditanya waktu tetapi rentang tanpa keterangan waktu");
  const kelompok = j.jejak.kelompok.filter((g) => g.bobot > 0);
  if (kelompok.length) {
    const ada = new Set(keTerm(h.kalimat));
    const total = kelompok.reduce((s, g) => s + g.bobot, 0);
    const cocok = kelompok.filter((g) => g.terms.some((t) => ada.has(t))).reduce((s, g) => s + g.bobot, 0);
    if (cocok / total < 0.5) hasil.alasan.push(`kalimat asal hanya memuat ${Math.round((100 * cocok) / total)}% kata topik`);
  }
  hasil.diterima = hasil.alasan.length === 0;
  if (hasil.diterima) hasil.alasan.push(`keyakinan ${(100 * h.keyakinan).toFixed(0)}% ≥ ambang ${(100 * p.ambang).toFixed(0)}%, bentuk & kata topik cocok`);
  return hasil;
}
