/**
 * PEMAHAMAN MAKSUD (intent) yang deterministik, untuk hal yang HARUS selalu benar:
 *
 *   perkenalan     "perkenalkan namaku irun", "nama saya Budi", "panggil aku Sari", "aku Irun"
 *   tanyaDiri      "siapa aku?", "siapa namaku?", "tau aku?", "kenal saya?", "kamu ingat namaku?"
 *
 * Kenapa tidak diserahkan ke GPT? Model kecil kadang salah memahami bentuk kalimat yang
 * jarang ia lihat (mis. nama huruf kecil), dan salah menyalin nama. Untuk maksud yang
 * sederhana dan penting, aturan pola lebih andal. GPT tetap dipakai untuk obrolan bebas.
 *
 * Pesan sudah dinormalisasi lebih dulu (data/slang.txt): "tau" -> "tahu", "gue" -> "saya".
 */
export type Maksud =
  | { jenis: "perkenalan"; nama: string }
  | { jenis: "tanyaDiri" };

const KATA_BUKAN_NAMA = new Set([
  "siapa", "apa", "kamu", "saya", "aku", "yang", "ini", "itu", "di", "dari", "mau", "ingin", "tanya", "bertanya", "sedang", "lagi",
  "baru", "sudah", "orang", "pegawai", "pns", "asn", "senang", "sedih", "capek", "lelah", "bingung", "sakit", "baik", "sehat",
  "lapar", "bosan", "setuju", "paham", "mengerti", "siap", "pulang", "pergi", "kembali", "juga", "tidak", "belum", "masih",
])

/** Huruf besar di awal nama: "irun" -> "Irun". */
export const rapikanNama = (s: string) => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();

export function pahamiMaksud(pesan: string): Maksud | null {
  const t = pesan.toLowerCase().replace(/[!?.,]+/g, " ").replace(/\s+/g, " ").trim();

  // Pertanyaan tentang diri pengguna
  if (
    /^(siapa(kah)?\s+(nama\s+)?(saya|aku)|siapa\s+namaku|nama(ku|\s+saya|\s+aku)\s+siapa|(apa\s+)?(kamu\s+)?(tahu|kenal|ingat)\s+(siapa\s+)?(saya|aku|namaku|nama\s+(saya|aku))|(kamu\s+)?(masih\s+)?ingat\s+(saya|aku|namaku))\b/.test(t)
  ) return { jenis: "tanyaDiri" };

  // Perkenalan: ambil kata sesudah penanda sebagai nama
  const m =
    t.match(/^(?:halo |hai )?(?:perkenalkan\s+)?(?:nama\s+(?:saya|aku)|namaku|panggil\s+(?:saya|aku))(?:\s+adalah)?\s+([a-zà-ÿ'-]{2,20})\b/) ??
    t.match(/^(?:halo |hai )?perkenalkan\s+(?:saya|aku)\s+([a-zà-ÿ'-]{2,20})\b/);
  if (m && !KATA_BUKAN_NAMA.has(m[1])) return { jenis: "perkenalan", nama: rapikanNama(m[1]) };
  // Bentuk pendek "aku Sari" hanya jika nama diawali huruf KAPITAL ("aku capek" bukan perkenalan)
  const pendek = pesan.trim().match(/^(?:[Hh]alo,?\s+|[Hh]ai,?\s+)?(?:[Ss]aya|[Aa]ku)\s+([A-ZÀ-Þ][a-zà-ÿ'-]{1,19})\s*[!.]?$/);
  if (pendek && !KATA_BUKAN_NAMA.has(pendek[1].toLowerCase())) return { jenis: "perkenalan", nama: pendek[1] };
  return null;
}
