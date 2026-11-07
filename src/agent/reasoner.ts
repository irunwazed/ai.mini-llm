/**
 * PENALARAN ATAS FAKTA: jawaban yang tersebar di beberapa artikel disimpulkan, bukan dikutip.
 *
 * Contoh: "di mana sekarang Zudan bekerja?"
 *   artikel Zudan  : "Tahun 2008 hingga 2011, ia ditunjuk sebagai Kepala Bagian …"          -> 2008–2011 (selesai)
 *   artikel Zudan  : "… merangkap sebagai Sekretaris BNPP dari Maret 2023 sampai Januari 2025" -> selesai
 *   artikel BKN    : "BKN dipimpin oleh seorang Kepala Badan yang sejak 7 Januari 2025 dijabat oleh Zudan …"
 *                                                                                           -> sejak 2025 (berjalan)
 *   kesimpulan     : "Saat ini Zudan Arif Fakrulloh bekerja sebagai Kepala Badan Kepegawaian Negara (sejak 7 Januari 2025)."
 *
 * Langkah:
 *   1. ambil semua kalimat di kolam bukti yang membahas ORANG/HAL yang ditanya (nama disebut, atau
 *      kalimat berkata ganti "Ia …" di artikel tentang orang itu) DAN memuat kata relasi
 *   2. dari setiap kalimat ambil FAKTA: nilai (jabatan/tempat) + waktu (mulai, selesai, masih berjalan)
 *   3. urutkan menurut waktu: yang masih berjalan dan paling akhir dimulai = keadaan sekarang
 *   4. tulis kesimpulan + alasan (mengapa fakta lain tidak dipakai) + riwayat bernomor sumber
 */
import { pecahKalimat } from "../answer/sentences.js";
import { pecahUmum } from "../answer/web-extract.js";
import { RE_TANGGAL } from "../gpt/reader-data.js";
import { keTerm } from "../text/normalize.js";
import type { Niat } from "./intent.js";
import { termRelasi, type Relasi } from "./relations.js";

export interface DokumenFakta { judul: string; label: string; teks: string; jenis: string; aturan?: boolean }
export interface Fakta {
  nilai: string; awal: number | null; akhir: number | null; berjalan: boolean; waktuTeks: string;
  kalimat: string; label: string; judul: string; jenis: string;
  labelLain: string[];   // sumber lain yang menyebut fakta yang sama
}
export interface HasilPenalaran { relasi: string; nama: string; fakta: Fakta[]; terpilih: Fakta; kesimpulan: string; alasan: string[] }

const TAHUN = /\b(1[89]\d{2}|20\d{2})\b/g;

/** Waktu sebuah fakta: tahun mulai, tahun selesai, masih berjalan, dan teks waktunya untuk ditampilkan. */
function waktuDari(k: string): Pick<Fakta, "awal" | "akhir" | "berjalan" | "waktuTeks"> {
  const tahun = [...k.matchAll(TAHUN)].map((m) => Number(m[1]));
  const rentang = k.match(/((?:\d{1,2}\s+)?(?:\p{Lu}\p{Ll}+\s+)?(1[89]\d{2}|20\d{2}))\s*(?:hingga|sampai|s\.d\.?|–|-)\s*((?:\d{1,2}\s+)?(?:\p{Lu}\p{Ll}+\s+)?(1[89]\d{2}|20\d{2}))/u);
  const sejak = k.match(/\bsejak\s+((?:\d{1,2}\s+)?(?:\p{Lu}\p{Ll}+\s+)?(1[89]\d{2}|20\d{2}))/u);
  const kini = /\b(saat ini|hingga kini|sampai sekarang|sampai saat ini|kini menjabat|masih menjabat)\b/i.test(k);
  if (rentang) return { awal: Number(rentang[2]), akhir: Number(rentang[4]), berjalan: false, waktuTeks: `${rentang[1]} – ${rentang[3]}` };
  if (sejak) return { awal: Number(sejak[2]), akhir: null, berjalan: true, waktuTeks: `sejak ${sejak[1]}` };
  const tgl = k.match(RE_TANGGAL)?.[1];
  return { awal: tahun.length ? Math.min(...tahun) : null, akhir: null, berjalan: kini, waktuTeks: kini ? "saat ini" : tgl ?? (tahun.length ? String(Math.min(...tahun)) : "") };
}

/** "Kepala Badan" + "Badan Kepegawaian Negara" -> "Kepala Badan Kepegawaian Negara" (kata yang bertumpuk digabung). */
function gabung(jabatan: string, organisasi: string): string {
  const j = jabatan.split(" "), o = organisasi.split(" ");
  for (let n = Math.min(j.length, o.length); n > 0; n--)
    if (j.slice(-n).join(" ").toLowerCase() === o.slice(0, n).join(" ").toLowerCase()) return [...j.slice(0, -n), ...o].join(" ");
  return `${jabatan} ${organisasi}`;
}
const rapi = (s: string) => s.replace(/\s+/g, " ").replace(/[,;:\s]+$/, "").split(" ").slice(0, 16).join(" ");
const kapital = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

const BERHENTI = String.raw`(?=,|;|\s+(?:dari|sejak|pada tahun|sampai|hingga|selama|di bawah|untuk masa|dan kemudian|menjabat|bersama|mendampingi|menggantikan|yang|ketika|saat)\b|\.\s|\.$|$)`;

/**
 * Nilai fakta pekerjaan dari satu kalimat: jabatan / tempat kerja. Hanya jika orang yang ditanya
 * benar-benar PEMEGANG jabatan itu: namanya (atau "Ia/Beliau") muncul SEBELUM "sebagai/menjabat",
 * atau jabatannya "dijabat oleh <nama>". Kalimat yang hanya menyebut namanya sebagai pihak lain
 * ("… mendampingi Menteri Keuangan Sri Mulyani") tidak dipakai.
 * posNama = posisi huruf nama di kalimat (0 untuk kalimat berkata ganti "Ia …").
 */
function nilaiPekerjaan(k: string, judul: string, posNama: number, namaKecil: string[]): string | null {
  const rendah = k.toLowerCase();
  const pimpin = k.match(/\bdipimpin oleh\s+(?:seorang\s+)?(.+?)\s+yang\b/i);
  const olehNama = rendah.match(/\bdijabat oleh\s+(.{0,40})/)?.[1] ?? "";
  if (pimpin && namaKecil.some((n) => olehNama.includes(n))) return gabung(kapital(rapi(pimpin[1])), judul.replace(/\s*\(.*\)$/, ""));
  // "dikenal/disebut/dijuluki sebagai Jokowi" adalah nama panggilan, bukan jabatan -> lewati, cari "sebagai" berikutnya
  for (const sebagai of k.matchAll(new RegExp(String.raw`\b(?:sebagai|menjadi|menjabat)\s+(?:sebagai\s+)?(.+?)${BERHENTI}`, "gi"))) {
    const sebelum = k.slice(0, sebagai.index).trim().split(/\s+/).at(-1)?.toLowerCase() ?? "";
    if (/^(dikenal|disebut|dijuluki|akrab|biasa|terkenal|populer)$/.test(sebelum) || /\badalah\b/i.test(sebagai[1])) continue;
    if (sebagai.index! > posNama && /\p{Lu}|dosen|guru|pegawai|staf|peneliti/u.test(sebagai[1])) return rapi(sebagai[1]);
  }
  const di = k.match(new RegExp(String.raw`\b(?:bekerja|berkarier|mengajar|bertugas)\s+(?:sebagai\s+\S+\s+)?(?:di|pada)\s+(.+?)${BERHENTI}`, "i"));
  if (di && di.index! > posNama) return rapi(di[1]);
  return null;
}

/** Nilai fakta umum (tempat tinggal, dll.): frasa setelah kata relasi + "di". */
function nilaiUmum(k: string): string | null {
  const m = k.match(/\b(?:di|ke)\s+((?:\p{Lu}[\p{L}'-]*\s?){1,5})/u);
  return m ? rapi(m[1]) : null;
}

export function penalaranFakta(niat: Niat, dok: DokumenFakta[]): HasilPenalaran | null {
  const relasi: Relasi | null = niat.relasi;
  if (!relasi || !niat.entitas.length) return null;
  const termRel = termRelasi(relasi);
  const tentang = (judul: string) => { const t = new Set(keTerm(judul)); return niat.entitas.every((e) => t.has(e)); };

  // 1–2. Kalimat tentang orang/hal yang ditanya + kata relasi -> fakta
  const fakta: Fakta[] = [];
  for (const d of dok) {
    const kalimat = d.aturan ? pecahKalimat(d.teks).map((x) => x.teks) : pecahUmum(d.teks);
    for (const k of kalimat) {
      const t = new Set(keTerm(k));
      // kata ganti "ia/dia/beliau" di mana pun dalam kalimat, asal di artikel tentang orang itu
      const ganti = tentang(d.judul) ? k.search(/\b(Ia|ia|Dia|dia|Beliau|beliau)\b/) : -1;
      const sebut = niat.entitas.every((e) => t.has(e)) || ganti >= 0;
      if (!sebut || ![...termRel].some((x) => t.has(x))) continue;
      const posNama = !niat.entitas.every((e) => t.has(e)) ? ganti : Math.min(...niat.entitas.map((e) => { const i = k.toLowerCase().search(new RegExp(`\\b${e}`)); return i < 0 ? Infinity : i; }));
      const nilai = relasi.nama === "pekerjaan" ? nilaiPekerjaan(k, d.judul, posNama, niat.entitas) : nilaiUmum(k);
      if (!nilai) continue;
      const f: Fakta = { nilai, ...waktuDari(k), kalimat: k, label: d.label, judul: d.judul, jenis: d.jenis, labelLain: [] };
      // fakta yang sama dari dua sumber: simpan yang keterangan waktunya lebih lengkap
      // fakta yang sama dari sumber lain ("Kepala BKN (BKN)" = "Kepala BKN"): digabung, sumbernya dicatat
      const inti = (x: string) => x.toLowerCase().replace(/\s*\([^)]*\)/g, "").trim();
      const sama = fakta.findIndex((x) => inti(x.nilai) === inti(nilai) || inti(x.nilai).includes(inti(nilai)) || inti(nilai).includes(inti(x.nilai)));
      if (sama < 0) fakta.push(f);
      else {
        const lama = fakta[sama];
        const baru = !lama.waktuTeks && f.waktuTeks ? { ...f, nilai: lama.nilai.length >= f.nilai.length ? lama.nilai : f.nilai } : lama;
        const label = [lama.label, ...lama.labelLain, f.label].filter((x, i, a) => x !== baru.label && a.indexOf(x) === i);
        fakta[sama] = { ...baru, labelLain: label };
      }
    }
  }
  if (!fakta.length) return null;

  // 3. Urutkan menurut waktu: yang paling AKHIR dimulai dulu (jika sama, yang masih berjalan).
  //    "sejak 1998" tanpa tanggal selesai TIDAK otomatis berarti masih menjabat: jika ada jabatan lain
  //    yang dimulai sesudahnya, jabatan lama itu dianggap sudah lewat.
  const awalTerakhir = Math.max(...fakta.map((f) => f.awal ?? -1));
  for (const f of fakta) if (f.berjalan && f.awal !== null && f.awal < awalTerakhir && !/saat ini|kini|sekarang|masih/i.test(f.waktuTeks)) f.berjalan = false;
  const urut = [...fakta].sort((a, b) => (b.awal ?? -1) - (a.awal ?? -1) || Number(b.berjalan) - Number(a.berjalan));
  const terpilih = urut[0];
  const judulOrang = dok.find((d) => tentang(d.judul) && d.jenis === "internet")?.judul ?? dok.find((d) => tentang(d.judul))?.judul;
  const nama = (judulOrang ?? niat.entitas.map(kapital).join(" ")).replace(/\s*\(.*\)$/, "").replace(/\s+—.*$/, "");

  // 4. Kesimpulan + alasan
  const waktu = terpilih.waktuTeks ? ` (${terpilih.waktuTeks})` : "";
  const kata = relasi.nama === "pekerjaan" ? "bekerja sebagai" : relasi.nama === "tempat tinggal" ? "tinggal di" : `${relasi.nama}:`;
  let kesimpulan: string;
  if (terpilih.berjalan) kesimpulan = `Saat ini ${nama} ${kata} ${terpilih.nilai}${waktu}.`;
  else if (niat.mintaKini || relasi.berwaktu) kesimpulan = `Yang paling akhir tercatat: ${nama} ${kata} ${terpilih.nilai}${waktu}. Sumber yang terkumpul tidak menyebut keadaan sesudahnya.`;
  else kesimpulan = `${nama} ${kata} ${terpilih.nilai}${waktu}.`;
  const alasan: string[] = [
    `Ditemukan ${fakta.length} fakta "${relasi.nama}" tentang ${nama} di ${new Set(fakta.map((f) => f.label)).size} sumber.`,
    terpilih.berjalan
      ? `"${terpilih.nilai}" dipakai karena masih berjalan (${terpilih.waktuTeks || "tertulis saat ini"}, tanpa tanggal selesai)${urut.filter((f) => f.berjalan).length > 1 ? " dan dimulai paling akhir" : ""}.`
      : terpilih.awal ? `"${terpilih.nilai}" dipakai karena dimulai paling akhir (${terpilih.awal}).` : `Tidak ada keterangan waktu; dipakai fakta yang paling cocok.`,
    ...urut.slice(1).map((f) => f.akhir ? `"${f.nilai}" sudah berakhir (${f.waktuTeks}).` : f.berjalan ? `"${f.nilai}" juga tertulis berjalan, tetapi dimulai lebih awal (${f.waktuTeks}).` : `"${f.nilai}": ${f.waktuTeks ? `tercatat ${f.waktuTeks}, lebih awal` : "waktunya tidak disebut"}.`),
  ];
  return { relasi: relasi.nama, nama, fakta: urut, terpilih, kesimpulan, alasan };
}
