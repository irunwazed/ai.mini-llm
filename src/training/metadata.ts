/**
 * Metadata peraturan dari judulnya:
 *   "PERATURAN PEMERINTAH REPUBLIK INDONESIA NOMOR 11 TAHUN 2017 TENTANG MANAJEMEN PEGAWAI NEGERI SIPIL"
 *     -> jenis "PERATURAN PEMERINTAH", nomor "11", tahun 2017, tentang "Manajemen Pegawai Negeri Sipil"
 *
 * Dipakai untuk:
 *   1. kutipan dasar hukum yang rapi ("Peraturan Pemerintah Nomor 11 Tahun 2017 tentang ...")
 *   2. mendeteksi PERUBAHAN: judul "Perubahan atas Peraturan BKN Nomor 24 Tahun 2017"
 *      berarti aturan lama itu sudah diubah, dan pengguna perlu diberi tahu.
 */
export interface InfoPeraturan {
  jenis: string;
  nomor: string;
  tahun: number | null;
  tentang: string;
  judulPendek: string;
  kunci: string | null;       // "PERATURAN PEMERINTAH|11|2017"
  mengubah: string | null;    // kunci peraturan yang diubah oleh dokumen ini
}

const RE_JUDUL =
  /((?:UNDANG-UNDANG|PERATURAN|KEPUTUSAN|INSTRUKSI|SURAT EDARAN|QANUN)(?:\s+[A-Z-]+){0,6}?)\s+(?:REPUBLIK INDONESIA\s+)?NOMOR\s+([\w./-]+)\s+TAHUN\s+(\d{4})\s+TENTANG\s+(.+)/i;
const RE_UBAH =
  /PERUBAHAN\s+(?:\w+\s+)?ATAS\s+((?:UNDANG-UNDANG|PERATURAN|KEPUTUSAN)(?:\s+[A-Z-]+){0,6}?)\s+(?:REPUBLIK INDONESIA\s+)?NOMOR\s+([\w./-]+)\s+TAHUN\s+(\d{4})/i;

const KECIL = new Set(["dan", "atau", "tentang", "atas", "di", "ke", "dari", "pada", "dalam", "yang", "untuk", "bagi", "serta"]);

export function kapital(s: string): string {
  return s
    .toLowerCase()
    .split(" ")
    .map((w, i) => (i > 0 && KECIL.has(w) ? w : w.charAt(0).toUpperCase() + w.slice(1)))
    .join(" ")
    .replace(/\b(Ri|Asn|Pns|Bkn|Dpr|Dprd|Apbn|Apbd|Tni|Polri)\b/g, (m) => m.toUpperCase());
}

const bersihJenis = (j: string) => j.toUpperCase().replace(/\s+/g, " ").trim();
export const kunciPeraturan = (jenis: string, nomor: string, tahun: string | number) =>
  `${bersihJenis(jenis)}|${String(nomor).toUpperCase()}|${tahun}`;

export function infoDariJudul(judul: string, file: string): InfoPeraturan {
  const j = judul.replace(/\s+/g, " ");
  const m = j.match(RE_JUDUL);
  if (!m) {
    const nama = file.split("/").pop()!.replace(/\.[^.]+$/, "").replace(/[-_]+/g, " ");
    return { jenis: "", nomor: "", tahun: null, tentang: "", judulPendek: j ? kapital(j.slice(0, 90)) : nama, kunci: null, mengubah: null };
  }
  const [, jenis, nomor, tahun, sisa] = m;
  const tentang = sisa
    .replace(/\(.*$/, "")                 // buang "(DOKUMEN FIKTIF ...)" dsb.
    .replace(/\b(DENGAN RAHMAT|PRESIDEN|MENTERI|KEPALA)\b.*$/i, "")
    .split(" ").slice(0, 20).join(" ").trim();
  const ubah = j.match(RE_UBAH);
  return {
    jenis: kapital(jenis),
    nomor,
    tahun: Number(tahun),
    tentang: kapital(tentang),
    judulPendek: `${kapital(jenis)} Nomor ${nomor} Tahun ${tahun} tentang ${kapital(tentang)}`,
    kunci: kunciPeraturan(jenis, nomor, tahun),
    mengubah: ubah ? kunciPeraturan(ubah[1], ubah[2], ubah[3]) : null,
  };
}
