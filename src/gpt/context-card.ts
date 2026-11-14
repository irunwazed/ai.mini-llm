/**
 * KARTU KONTEKS: ringkasan fakta tentang pengguna yang selalu ditaruh di awal prompt GPT obrolan.
 *
 *   <|konteks|> nama: Sari; tinggal: Medan; pekerjaan: guru<|end|><|user|> …
 *
 * Konteks GPT terbatas (256 token): giliran lama akhirnya terpotong. Fakta penting dari giliran yang
 * terpotong, dari memori yang disimpan pengguna, atau dari sesi lain tetap sampai ke model lewat kartu ini.
 * Fakta diambil dengan pola kalimat sederhana (deterministik, bisa dijelaskan), urut dari yang lama ke baru:
 * fakta yang lebih baru (atau koreksi "eh salah, maksudku …") menimpa yang lama.
 * Kunci kartu sama persis dengan data latih (src/gpt/chat-context-data.ts) agar model tahu cara memakainya.
 */
import { pahamiMaksud } from "../intent.js";

export interface FaktaPengguna { nama?: string; tinggal?: string; pekerjaan?: string; hobi?: string; "makanan favorit"?: string }
type Kunci = keyof FaktaPengguna;

const AKHIR = String.raw`(?=\s*(?:[,.!?;]|$|\s(?:dan|tapi|terus|lalu|juga|sih|ya|kok)\b))`;
const POLA: [Kunci, RegExp][] = [
  ["tinggal", new RegExp(String.raw`\b(?:aku|saya|gue)\s+(?:tinggal di|dari|orang|berasal dari)\s+([\p{L}-]+(?:\s[\p{L}-]+)?)${AKHIR}`, "iu")],
  ["tinggal", new RegExp(String.raw`\brumah(?:ku| saya)\s+di\s+([\p{L}-]+(?:\s[\p{L}-]+)?)${AKHIR}`, "iu")],
  ["pekerjaan", new RegExp(String.raw`\b(?:aku|saya)\s+(?:kerja sebagai|bekerja sebagai|bekerja jadi|kerja jadi|seorang)\s+([\p{L} ]{3,30}?)${AKHIR}`, "iu")],
  ["pekerjaan", new RegExp(String.raw`\bpekerjaan(?:ku| saya)\s+([\p{L} ]{3,30}?)${AKHIR}`, "iu")],
  ["hobi", new RegExp(String.raw`\b(?:hobi(?:ku| saya)|kegemaran(?:ku| saya))\s+([\p{L} ]{3,30}?)${AKHIR}`, "iu")],
  ["makanan favorit", new RegExp(String.raw`\bmakanan favorit(?:ku| saya)\s+([\p{L} -]{3,30}?)${AKHIR}`, "iu")],
  ["makanan favorit", new RegExp(String.raw`\b(?:aku|saya)\s+paling suka makan\s+([\p{L} -]{3,30}?)${AKHIR}`, "iu")],
];
const KOREKSI = new RegExp(String.raw`\b(?:maksud(?:ku| saya)|ralat,?|bukan [\p{L}-]+,?\s*tapi)\s+(?:aku tinggal di\s+)?([\p{L}-]+(?:\s[\p{L}-]+)?)${AKHIR}`, "iu");
const kapital = (s: string) => s.replace(/(^|\s)(\p{L})/gu, (_, a, b) => a + b.toUpperCase());

/** Ambil fakta dari pesan-pesan pengguna (urut lama -> baru) + catatan memori. */
export function faktaPengguna(pesanPengguna: string[], opsi: { namaMemori?: string | null; catatanMemori?: string[] } = {}): FaktaPengguna {
  const f: FaktaPengguna = {};
  let terakhir: Kunci | null = null;
  for (const teks of [...(opsi.catatanMemori ?? []), ...pesanPengguna]) {
    const m = pahamiMaksud(teks);
    if (m?.jenis === "perkenalan") { f.nama = m.nama; terakhir = "nama"; }
    for (const [k, re] of POLA) {
      const x = teks.match(re);
      if (x) { f[k] = k === "tinggal" ? kapital(x[1].trim()) : x[1].trim(); terakhir = k; }
    }
    const kor = teks.match(KOREKSI);
    if (kor && terakhir) f[terakhir] = terakhir === "tinggal" || terakhir === "nama" ? kapital(kor[1].trim()) : kor[1].trim();
  }
  if (opsi.namaMemori) f.nama = opsi.namaMemori;
  return f;
}

/** Teks kartu ("nama: Sari; tinggal: Medan"), atau null jika belum ada fakta. */
export function kartuKonteks(f: FaktaPengguna): string | null {
  const urutan: Kunci[] = ["nama", "tinggal", "pekerjaan", "hobi", "makanan favorit"];
  const isi = urutan.filter((k) => f[k]).map((k) => `${k}: ${f[k]}`);
  return isi.length ? isi.join("; ") : null;
}
