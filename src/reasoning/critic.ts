/**
 * PEMERIKSA RELEVANSI: apakah jawaban benar-benar menjawab pertanyaan?
 *
 * Setiap jawaban dari jalur pencarian (aturan/Wikipedia) diperiksa sebelum ditampilkan.
 * Pemeriksaannya sederhana dan bisa dijelaskan, sesuai jenis pertanyaan:
 *   "berapa ..."           -> jawaban harus memuat angka
 *   "apa itu X"            -> kalimat utama berbentuk "... adalah/merupakan ..."
 *   "apakah/bolehkah ..."  -> jawaban diawali Ya/Tidak/Boleh atau memuat dilarang/wajib/boleh
 *   "kapan ..."            -> jawaban memuat angka atau kata waktu
 *   semua                  -> kalimat utama memuat ≥ 50% kata topik pertanyaan
 * Jika gagal, mesin percakapan mencoba jalur berikutnya (loop "hasilkan -> periksa -> coba lagi").
 */
import type { Jawaban } from "../answer/compose.js";
import { besaran, satuanDiminta } from "../answer/question-type.js";
import { keTerm } from "../text/normalize.js";

export interface HasilPeriksa { lolos: boolean; alasan: string[] }

export function periksaJawaban(j: Jawaban): HasilPeriksa {
  const alasan: string[] = [];
  const utama = j.utama.join(" ");
  const pertama = j.utama[0] ?? "";
  if (!utama.trim()) return { lolos: false, alasan: ["jawaban kosong"] };

  const satuan = satuanDiminta(j.pertanyaan);
  if (j.jenis === "jumlah" && !/\d/.test(utama)) alasan.push(`ditanya "berapa", tetapi jawaban tidak memuat angka`);
  else if (j.jenis === "jumlah" && satuan && !besaran(utama).some((b) => satuan.test(b.toLowerCase())))
    alasan.push(`ditanya "berapa", tetapi jawaban tidak memuat satuan yang diminta (mis. "tahun 1973" bukan lama waktu)`);
  if (j.jenis === "definisi" && !/\b(adalah|merupakan|ialah|yaitu|yakni)\b/i.test(pertama))
    alasan.push(`ditanya definisi, tetapi kalimat utama bukan kalimat "… adalah …"`);
  if (j.jenis === "yaTidak" && !/^(ya|tidak|boleh)\b/i.test(pertama) && !/\b(dilarang|wajib|boleh|dapat|berhak|harus)\b/i.test(utama))
    alasan.push(`ditanya ya/tidak, tetapi jawaban tidak menyatakan boleh/dilarang/wajib`);
  if (j.jenis === "waktu" && !/\d|\b(paling lambat|paling lama|sebelum|setelah|sejak|hari|bulan|tahun|tanggal|pukul|jam)\b/i.test(utama))
    alasan.push(`ditanya waktu, tetapi jawaban tidak memuat keterangan waktu`);

  // Cakupan kata topik di kalimat utama
  const kelompok = j.jejak.kelompok.filter((g) => g.bobot > 0);
  if (kelompok.length) {
    const ada = new Set(keTerm(utama));
    const total = kelompok.reduce((s, g) => s + g.bobot, 0);
    const cocok = kelompok.filter((g) => g.terms.some((t) => ada.has(t))).reduce((s, g) => s + g.bobot, 0);
    if (cocok / total < 0.5) alasan.push(`kalimat utama hanya memuat ${Math.round((100 * cocok) / total)}% kata topik pertanyaan`);
  }
  return { lolos: alasan.length === 0, alasan };
}
