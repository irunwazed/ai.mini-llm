/**
 * Membersihkan teks hasil ekstraksi PDF dari "noise" tata letak.
 *
 * Header/footer (mis. "www.peraturan.go.id", "2017, No.1861", "-3-") muncul di hampir
 * setiap halaman. Kita DETEKSI otomatis secara statistik, bukan dengan daftar tetap:
 *   1. normalisasi baris: angka diganti "#"  ->  "-3-" dan "-4-" menjadi "-#-"
 *   2. hitung di berapa halaman baris itu muncul
 *   3. baris pendek yang muncul di >= 30% halaman dianggap header/footer dan dibuang
 * Baris struktur (Pasal, BAB, Bagian) tidak pernah dibuang walau sering muncul.
 */
const STRUKTUR = /^(pasal|bab|bagian|paragraf|lampiran|penjelasan)\b/i;

const kunci = (baris: string) => baris.trim().replace(/\d+/g, "#").replace(/\s+/g, " ").toLowerCase();

export function bersihkanHalaman(halaman: string[]): string[] {
  const hitung = new Map<string, number>();
  if (halaman.length >= 3) {
    for (const h of halaman) {
      const unik = new Set(h.split(/\r?\n/).map(kunci).filter((k) => k && k.length <= 80));
      for (const k of unik) hitung.set(k, (hitung.get(k) ?? 0) + 1);
    }
  }
  const batas = Math.max(3, Math.ceil(0.3 * halaman.length));
  return halaman.map((h) =>
    h
      .split(/\r?\n/)
      .filter((b) => {
        const t = b.trim();
        if (/^-\s*\d+\s*-$/.test(t)) return false; // nomor halaman "-3-"
        if (STRUKTUR.test(t)) return true;
        return (hitung.get(kunci(t)) ?? 0) < batas;
      })
      .join("\n"),
  );
}

/**
 * Perbaikan kesalahan OCR yang umum di PDF hasil scan (banyak dokumen JDIH seperti ini):
 *   "( 1 )"          -> "(1)"            nomor ayat berspasi
 *   "Pasal 4 1"      -> "Pasal 41"       nomor pasal terpecah
 *   "d . Tender"     -> "d. Tender"      nomor huruf berspasi
 *   "Rpl00. 000.000" -> "Rp100.000.000"  huruf l/I/O terbaca sebagai angka, titik ribuan berspasi
 */
export function perbaikiOCR(baris: string): string {
  return baris
    .replace(/\(\s*(\d+)\s*\)/g, "($1)")
    .replace(/^(pasal)\s+(\d)\s+(\d)(\s+\d)?\s*$/i, (_, p, a, b, c) => `${p} ${a}${b}${(c ?? "").trim()}`)
    .replace(/^([a-z])\s+\.\s/, "$1. ")
    .replace(/\bRp\s?([lIO\d][lIO\d.\s]*\d)/g, (m) => m.replace(/[lI]/g, "1").replace(/O/g, "0").replace(/\.\s+(?=\d{3})/g, "."))
    .replace(/(\d)\.\s(\d{3})\b/g, "$1.$2");
}

/** Halaman yang hampir tanpa teks (kemungkinan gambar hasil scan). Nomor 1-based. */
export function halamanKosong(halaman: string[]): number[] {
  const hasil: number[] = [];
  halaman.forEach((h, i) => {
    if (h.replace(/\s/g, "").length < 40) hasil.push(i + 1);
  });
  return hasil;
}
