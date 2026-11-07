/**
 * Memecah pasal menjadi kalimat/ayat dan merapikannya agar enak dibaca sebagai jawaban.
 * Merapikan = membuang hal yang penting bagi naskah hukum tetapi mengganggu jawaban:
 *   "(2) Lamanya cuti tahunan adalah 12 (dua belas) hari kerja dalam satu tahun."
 *    ->  "Lamanya cuti tahunan adalah 12 hari kerja dalam satu tahun."
 * Isi dan angkanya tidak pernah diubah.
 */
export interface Kalimat { teks: string; asli: string; ayat: string | null; urutan: number }

const ANGKA_KATA =
  "(?:se|nol|satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh|sebelas|belas|puluh|ratus|ribu|juta|miliar|triliun|koma|seratus|seribu|sejuta)";
const RE_ANGKA_EJAAN = new RegExp(`\\s*\\((?:${ANGKA_KATA}\\s*)+\\)`, "gi");

// Rujukan pasal: hanya token rujukan yang dibuang, isi kalimat setelahnya tetap utuh.
const REF = String.raw`(?:Pasal\s+\d+[A-Z]?|ayat\s*\(\d+[a-z]?\)|huruf\s+[a-z]\b|angka\s+\d+)`;
const RE_RUJUKAN = new RegExp(
  String.raw`,?\s*sebagaimana dimaksud (?:dalam|pada)\s+${REF}(?:\s*(?:,|dan|atau|sampai dengan)?\s*${REF})*`,
  "gi",
);

export function pecahKalimat(teks: string): Kalimat[] {
  const bagian = teks
    .replace(/^pasal\s+\S+\s*/i, "")
    .replace(/\s+/g, " ")
    // Potong setelah . ; :  atau sebelum "(2) " / "b. " sebagai awal ayat/huruf baru.
    // Tetapi JANGAN potong di rujukan: "dimaksud pada ayat (1) dan ayat (2)", "huruf a dan b".
    .split(/(?<=[.;:])\s+(?=[(A-Za-z0-9])|(?<!\b(?:ayat|dan|atau|pada|dalam|huruf|angka|pasal|dengan|,))\s+(?=\(\d+[a-z]?\)\s)|(?<!\b(?:huruf|dan|atau|,))\s+(?=[a-z]\.\s)/i)
    .map((s) => s.trim())
    .filter(Boolean);
  const hasil: Kalimat[] = [];
  let ayat: string | null = null;
  bagian.forEach((s, i) => {
    const m = s.match(/^\((\d+[a-z]?)\)\s/);
    if (m) ayat = m[1];
    if (s.split(" ").length < 4) return;
    hasil.push({ asli: s, teks: rapikan(s), ayat, urutan: i });
  });
  return hasil;
}

export function rapikan(s: string): string {
  let t = s
    .replace(/^\(\d+[a-z]?\)\s*/, "")              // nomor ayat
    .replace(/^(?:[a-z]|\d{1,2})\.\s+/, "")         // nomor huruf/angka daftar
    .replace(/(Rp\s?[\d.,]+)\s*\([a-z ]+rupiah\)/gi, "$1")
    .replace(RE_ANGKA_EJAAN, "")                    // "12 (dua belas)" -> "12"
    .replace(RE_RUJUKAN, "")                        // "sebagaimana dimaksud pada ayat (1) dan ayat (2)"
    .replace(/\s+/g, " ")
    .trim()
    .replace(/[;,:]\s*(dan|atau)?$/i, "")
    .replace(/\s+(dan|atau)$/i, "");
  if (!t) return t;
  t = t.charAt(0).toUpperCase() + t.slice(1);
  return /[.!?]$/.test(t) ? t : t + ".";
}

/** Jaccard dua kalimat (untuk membuang kalimat yang hampir sama). */
export function mirip(a: string, b: string): number {
  const A = new Set(a.toLowerCase().split(/\W+/)), B = new Set(b.toLowerCase().split(/\W+/));
  let irisan = 0;
  for (const x of A) if (B.has(x)) irisan++;
  return irisan / (A.size + B.size - irisan || 1);
}
