/**
 * RELASI PERTANYAAN: kata kerja yang ditanya punya banyak bentuk di teks.
 *   "di mana Zudan BEKERJA?"  -> di artikel tertulis "menjabat", "dijabat oleh", "ditunjuk sebagai", "Kepala …"
 *   "kapan ia LAHIR?"         -> "dilahirkan", "kelahiran"
 * Setiap relasi = kata pemicu di pertanyaan + kelompok kata yang boleh menggantikannya di kalimat bukti.
 * Relasi "berwaktu" (pekerjaan, tempat tinggal) bisa berubah dari waktu ke waktu, jadi jawabannya
 * harus DISIMPULKAN dari urutan waktu semua fakta (lihat reasoner.ts), bukan diambil dari satu kalimat.
 */
import { keTerm } from "../text/normalize.js";

export interface Relasi { nama: string; pemicu: RegExp; kata: string[]; berwaktu: boolean }

export const RELASI: Relasi[] = [
  {
    nama: "pekerjaan", berwaktu: true,
    pemicu: /\b(bekerja|kerja|kerjanya|jabatan|jabatannya|menjabat|pekerjaan|pekerjaannya|profesi|berkarier|karier|posisi|posisinya|berkantor|kantor|kantornya|bertugas|tugas)\b/i,
    kata: ["bekerja", "menjabat", "dijabat", "jabatan", "kepala", "direktur", "sekretaris", "ketua", "menteri", "gubernur", "bupati", "wali", "dosen",
      "pegawai", "ditunjuk", "dilantik", "diangkat", "memimpin", "dipimpin", "karier", "kariernya", "pejabat", "anggota", "duta", "komisaris",
      "rektor", "dekan", "pelaksana", "merangkap", "bertugas", "staf", "deputi", "asisten", "inspektur", "kepala"],
  },
  { nama: "tempat tinggal", berwaktu: true, pemicu: /\b(tinggal|domisili|menetap|bermukim)\b/i, kata: ["tinggal", "menetap", "berdomisili", "bermukim", "pindah"] },
  { nama: "kelahiran", berwaktu: false, pemicu: /\b(lahir|kelahiran|dilahirkan)\b/i, kata: ["lahir", "dilahirkan", "kelahiran"] },
  { nama: "kematian", berwaktu: false, pemicu: /\b(meninggal|wafat|tutup usia|gugur)\b/i, kata: ["meninggal", "wafat", "tutup", "gugur"] },
  { nama: "pasangan", berwaktu: false, pemicu: /\b(istri|istrinya|suami|suaminya|menikah|pasangan)\b/i, kata: ["istri", "suami", "menikah", "menikahi", "pernikahan", "pasangan"] },
  { nama: "pendidikan", berwaktu: false, pemicu: /\b(kuliah|sekolah|pendidikan|lulusan|lulus|belajar)\b/i, kata: ["pendidikan", "kuliah", "lulus", "lulusan", "sarjana", "magister", "doktor", "sekolah", "universitas", "menempuh"] },
];

/** Kata waktu: menunjukkan KAPAN yang dimaksud ("sekarang"), bukan kata yang wajib muncul di kalimat bukti. */
export const KATA_WAKTU = new Set(["sekarang", "saat", "kini", "terakhir", "terbaru", "dulu", "pernah", "masih", "lalu", "terkini"]);
export const MINTA_KINI = /\b(sekarang|saat ini|kini|terakhir|terbaru|terkini|masih)\b/i;

export function relasiDari(pesan: string): Relasi | null {
  return RELASI.find((r) => r.pemicu.test(pesan)) ?? null;
}

/** Bentuk kata dasar setiap anggota relasi, untuk dicocokkan dengan kalimat. */
const cacheTerm = new Map<Relasi, Set<string>>();
export function termRelasi(r: Relasi): Set<string> {
  let s = cacheTerm.get(r);
  if (!s) { s = new Set(r.kata.flatMap((k) => keTerm(k))); cacheTerm.set(r, s); }
  return s;
}

/** Apakah kata pertanyaan `w` terpenuhi oleh kalimat (kata itu sendiri, atau anggota relasinya)? */
export function kataTerpenuhi(w: string, termKalimat: Set<string>, relasi: Relasi | null, kataPemicu: string[]): boolean {
  if (keTerm(w).some((t) => termKalimat.has(t))) return true;
  if (relasi && kataPemicu.includes(w)) { for (const t of termRelasi(relasi)) if (termKalimat.has(t)) return true; }
  return false;
}
