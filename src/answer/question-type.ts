/**
 * Mengenali JENIS pertanyaan, karena bentuk jawaban yang baik berbeda-beda:
 *   "berapa hari cuti tahunan?"      -> jumlah    -> jawab angkanya dulu: "12 hari kerja."
 *   "bolehkah mobil dinas dipakai?"  -> yaTidak   -> jawab "Ya." / "Tidak." dulu
 *   "apa itu ASN?"                   -> definisi  -> ambil dari Ketentuan Umum
 *   "apa syarat cuti besar?"         -> syarat    -> daftar ketentuan
 *   "bagaimana cara mengajukan cuti?"-> prosedur  -> daftar langkah sesuai urutan pasal
 *   "apa sanksi ..."                 -> sanksi
 *   "kapan / paling lambat ..."      -> waktu
 */
export type Jenis = "jumlah" | "yaTidak" | "definisi" | "syarat" | "prosedur" | "sanksi" | "waktu" | "umum";

export function jenisPertanyaan(q: string): Jenis {
  const t = q.toLowerCase().trim();
  // "apa itu X", "X itu apa?", "X apa itu?", "X adalah?", "siapa X?" (pendek), "tau tentang X?"
  if (
    /^(apa\s+(itu|yang\s+dimaksud(\s+dengan)?)|pengertian|definisi|arti(nya)?\b|(tau|tahu|kenal)\s+(tentang|soal|dengan)?|(jelaskan|ceritakan|terangkan)\s+(tentang|soal|mengenai))/.test(t) ||
    /\b(itu apa|apa itu|artinya apa|adalah apa|adalah|itu siapa|siapa itu|ialah|merupakan)\s*\??$/.test(t) ||
    /^siapa(kah)?\s+\S+(\s+\S+){0,3}\s*\??$/.test(t)
  ) return "definisi";
  if (/\b(syarat|persyaratan|kriteria|siapa\s+(saja\s+)?yang\s+berhak)\b/.test(t)) return "syarat";
  if (/\b(bagaimana\s+(cara|prosedur|mekanisme|tata\s*cara)|cara\s+\w+|tata\s*cara|prosedur|langkah|mekanisme|alur)\b/.test(t)) return "prosedur";
  if (/\bberapa\b/.test(t)) return "jumlah";
  if (/\b(kapan|paling\s+lambat|tenggat|batas\s+waktu)\b/.test(t)) return "waktu";
  if (/\b(sanksi|hukuman|denda|akibat(nya)?|konsekuensi)\b/.test(t)) return "sanksi";
  if (/\b(besaran|jumlah|nilai|tarif|batas)\b/.test(t)) return "jumlah";
  if (/^(apakah|apa\s+boleh|bolehkah|bisakah|dapatkah|haruskah|wajibkah|perlukah)\b|\b(boleh|bisa)\s+(tidak|nggak|gak)\b/.test(t)) return "yaTidak";
  return "umum";
}

/** Satuan yang diharapkan dari pertanyaan "berapa ...". */
export function satuanDiminta(q: string): RegExp | null {
  const t = q.toLowerCase();
  if (/berapa\s+(hari|lama)/.test(t)) return /hari|bulan|tahun|minggu/;
  if (/berapa\s+(tinggi|panjang|jarak|jauh|luas|dalam|lebar|berat)/.test(t)) return /\b(m|meter|mdpl|km|kilometer|cm|mm|kg|kilogram|ton|gram|ha|hektare|km2|km²|mil)\b/;
  if (/berapa\s+(bulan)/.test(t)) return /bulan/;
  if (/berapa\s+(tahun)/.test(t)) return /tahun/;
  if (/berapa\s+(jam|menit)/.test(t)) return /jam|menit/;
  // "besar" hanya jika ditanya "berapa besar"/"besarnya" (bukan "cuti besar")
  if (/(uang|biaya|harga|nilai|tarif|berapa besar|besarnya|rupiah|gaji|tunjangan)/.test(t)) return /rp|persen|%/i;
  if (/(persen|potongan|denda)/.test(t)) return /%|persen|permil|\//;
  if (/(liter|bensin|bahan bakar)/.test(t)) return /liter/;
  return null;
}

/** Semua besaran (angka + satuan) dalam sebuah kalimat. */
export function besaran(s: string): string[] {
  const re =
    /Rp\s?\d{1,3}(?:\.\d{3})*(?:,\d+)?(?:\s(?:ribu|juta|miliar))?|\d+(?:[.,]\d+)?\s?(?:%|persen\b)|\d+\/\d+\s*\(?[a-z ]*permil\)?|\d+(?:[.,]\d+)?\s(?:hari kerja|hari kalender|hari|minggu|bulan|tahun|jam(?:\s\d+\smenit)?|menit|liter|kali|orang|kilometer|km)\b/gi;
  // Buang "Nomor 58 Tahun 2013" dan "tahun 2017" (itu identitas/tanggal, bukan besaran)
  const bersih = s.replace(/\b(nomor|no\.)\s*\d+\s+tahun\s+\d{4}/gi, " ").replace(/\b\d+\s+tahun\s+\d{4}\b/gi, " ");
  return bersih.match(re) ?? [];
}

/** Polaritas kalimat untuk pertanyaan ya/tidak. */
export function polaritas(s: string): "larang" | "wajib" | "boleh" | null {
  const t = s.toLowerCase();
  if (/\b(dilarang|larangan|tidak\s+(boleh|dapat|diperbolehkan|diizinkan|berhak|diberikan))\b/.test(t)) return "larang";
  if (/\b(wajib|harus|diwajibkan)\b/.test(t)) return "wajib";
  if (/\b(boleh|dapat|berhak|diperbolehkan|diizinkan|diberikan)\b/.test(t)) return "boleh";
  return null;
}
