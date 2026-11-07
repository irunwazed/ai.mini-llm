/**
 * PENJAWAB EKSTRAKTIF UNTUK HALAMAN INTERNET.
 * Halaman hasil pencarian internet tidak ada di indeks, jadi dibuat "BM25 mini" seketika:
 *   1. pecah semua halaman menjadi kalimat
 *   2. IDF setiap kata pertanyaan dihitung dari kalimat-kalimat itu: idf = ln(1 + (N − df + 0,5) / (df + 0,5))
 *   3. skor kalimat = jumlah idf kata pertanyaan yang muncul × faktor jenis pertanyaan
 *      (definisi suka "X adalah …", "berapa" butuh angka, "kapan" butuh keterangan waktu)
 *   4. cakupan = idf kata yang cocok / idf semua kata pertanyaan  (harus ≥ 50% agar lolos cek)
 *   5. nama diri di pertanyaan ("Soekarno", "Rinjani") WAJIB ada di kalimat, dan halaman yang judulnya
 *      cocok dengan nama itu diutamakan (agar "kapan Soekarno lahir?" tidak dijawab dengan Romy Soekarno)
 * Kalimat dikutip apa adanya, tidak dikarang.
 */
import { keTerm } from "../text/normalize.js";
import { jenisPertanyaan } from "./question-type.js";
import { pecahKalimat } from "./sentences.js";

export interface KalimatWeb { teks: string; halaman: number; skor: number; cakupan: number; cocok: string[] }

/** Kata pembingkai pertanyaan hitungan: "berapa JUMLAH provinsi" -> yang ditanya "provinsi". */
const PEMBINGKAI = new Set(["jumlah", "jumlahnya", "banyak", "banyaknya", "total"]);

export function kataPertanyaan(pertanyaan: string): string[] {
  const kata = [...new Set(keTerm(pertanyaan).filter((t) => !t.startsWith("~")))];
  const inti = kata.filter((k) => !PEMBINGKAI.has(k));
  return inti.length ? inti : kata;
}

/** Nama diri: kata berhuruf kapital (bukan kata pertama kalimat) yang bukan stopword. */
export function namaDiri(pertanyaan: string): string[] {
  const hasil: string[] = [];
  for (const m of pertanyaan.matchAll(/(?<=\s)[A-Z][\p{L}'-]+/gu)) {
    const t = keTerm(m[0]).find((x) => !x.startsWith("~"));
    if (t && !hasil.includes(t)) hasil.push(t);
  }
  return hasil;
}

export function pilihKalimatWeb(pertanyaan: string, halaman: { teks: string; judul?: string }[], n = 2, syarat?: (k: KalimatWeb) => boolean): { kalimat: KalimatWeb[]; kata: string[]; jumlahKalimat: number } {
  const kata = kataPertanyaan(pertanyaan);
  const wajib = namaDiri(pertanyaan);
  const kataSet = new Set(keTerm(pertanyaan));
  // Kecocokan judul: bagian kata judul yang ada di pertanyaan ("Soekarno" 100%, "Romy Soekarno" 50%)
  const cocokJudul = halaman.map((h) => {
    const k = keTerm(h.judul ?? "").filter((t) => !t.startsWith("~"));
    return k.length ? k.filter((t) => kataSet.has(t)).length / k.length : 0;
  });
  const varian = kata.map((k) => keTerm(k)); // [kata, ~kata-dasar]
  const jenis = jenisPertanyaan(pertanyaan);
  const semua: { teks: string; halaman: number; urutan: number; term: Set<string> }[] = [];
  halaman.forEach((h, i) =>
    pecahKalimat(h.teks.replace(/\n+/g, ". ").replace(/\.\s*\./g, "."))
      .filter((k) => k.teks.length >= 25 && k.teks.length <= 450)
      .forEach((k, urutan) => semua.push({ teks: k.teks, halaman: i, urutan, term: new Set(keTerm(k.teks)) })));
  if (!kata.length || !semua.length) return { kalimat: [], kata, jumlahKalimat: semua.length };

  const N = semua.length;
  const idf = varian.map((v) => {
    const df = semua.filter((s) => v.some((t) => s.term.has(t))).length;
    return Math.log(1 + (N - df + 0.5) / (df + 0.5));
  });
  const totalIdf = idf.reduce((a, b) => a + b, 0) || 1;

  const nilai = semua.map((s) => {
    const cocokIdx = varian.map((v, i) => (v.some((t) => s.term.has(t)) ? i : -1)).filter((i) => i >= 0);
    const dasar = cocokIdx.reduce((a, i) => a + idf[i], 0);
    let faktor = 1;
    if (jenis === "definisi" && /\b(adalah|merupakan|ialah)\b/i.test(s.teks)) faktor *= 1.5;
    if (jenis === "jumlah" && !/\d/.test(s.teks)) faktor *= 0.3;
    // "berapa X?" paling sering dijawab "<angka> X" ("38 provinsi", "3.726 meter")
    if (jenis === "jumlah" && kata.some((k) => new RegExp(String.raw`\d[\d.,]*\s+(?:\S+\s+)?${k}`, "i").test(s.teks))) faktor *= 2;
    if (jenis === "waktu" && !/\d|\b(abad|tahun|bulan|tanggal|hari)\b/i.test(s.teks)) faktor *= 0.3;
    if (s.urutan === 0) faktor *= 1.1;                 // kalimat pembuka biasanya ringkasan
    if (wajib.some((w) => !s.term.has(w))) faktor *= 0.2; // nama diri yang ditanya tidak disebut
    faktor *= 0.6 + 0.6 * cocokJudul[s.halaman];       // halaman tentang subjek yang ditanya
    faktor *= 1 - 0.05 * Math.min(s.halaman, 4);       // halaman peringkat atas sedikit diutamakan
    const adaNama = wajib.every((w) => s.term.has(w));
    return { teks: s.teks, halaman: s.halaman, skor: dasar * faktor, cakupan: adaNama ? dasar / totalIdf : Math.min(0.49, dasar / totalIdf), cocok: cocokIdx.map((i) => kata[i]) };
  }).filter((x) => x.skor > 0).sort((a, b) => b.skor - a.skor);

  // Jika ada syarat (mis. semua kata kunci harus ada), pilih yang terbaik di antara kalimat yang memenuhinya
  const lolos = syarat ? nilai.filter(syarat) : nilai;
  const pilih: KalimatWeb[] = [];
  for (const k of lolos.length ? lolos : nilai) {
    if (pilih.length >= n) break;
    if (pilih.length && k.skor < 0.8 * pilih[0].skor) break; // kalimat kedua hanya jika hampir sama kuat
    if (!pilih.some((p) => p.teks === k.teks)) pilih.push(k);
  }
  return { kalimat: pilih, kata, jumlahKalimat: N };
}
