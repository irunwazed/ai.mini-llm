/**
 * LANGKAH 2 AGENT: ALAT. Setiap alat = satu mode atau satu sumber.
 *
 *   lokal    : otomatis, penalaran, aturan, rag, wiki-lokal, memori, obrolan-gpt   (lewat MesinChat)
 *   internet : wiki-subjek, wiki-kata-kunci, wiki-artikel, duckduckgo, brave        (hanya API resmi)
 *
 * Alat internet berbagi satu "keranjang" halaman: setiap alat menambah halaman, lalu kalimat terbaik
 * dipilih dari SEMUA halaman yang sudah terkumpul (BM25 mini, src/answer/web-extract.ts).
 */
import { pilihKalimatWeb } from "../answer/web-extract.js";
import type { LangkahProses, MesinChat, Mode, OpsiProses, Respons } from "../chat-engine.js";
import { artikelWikipedia, adaKunciWeb, cariDuckDuckGo, cariWeb, cariWikipedia, hostTerblokir, type HalamanWeb, type PanggilanHttp } from "../search/internet.js";
import { keTerm } from "../text/normalize.js";
import { kalimatCukup } from "./compare.js";
import type { Niat } from "./intent.js";

export type NamaAlat = "otomatis" | "penalaran" | "aturan" | "rag" | "wiki-lokal" | "memori" | "obrolan-gpt"
  | "wiki-subjek" | "wiki-kata-kunci" | "wiki-artikel" | "duckduckgo" | "brave";
export const ALAT_INTERNET: NamaAlat[] = ["wiki-subjek", "wiki-kata-kunci", "wiki-artikel", "duckduckgo", "brave"];

export interface Aksi { alat: NamaAlat; kueri: string; pikir: string }
export interface KonteksAlat { mesin: MesinChat; opsi: OpsiProses; halaman: HalamanWeb[] }

const MODE: Partial<Record<NamaAlat, Mode>> = { otomatis: "otomatis", penalaran: "penalaran", aturan: "aturan", rag: "rag", "wiki-lokal": "wiki", "obrolan-gpt": "gpt" };

export function alatTersedia(a: NamaAlat, ctx: KonteksAlat, internet: boolean): boolean {
  if (a === "rag") return !!ctx.mesin.pembaca;
  if (a === "wiki-lokal") return !!ctx.mesin.wiki;
  if (a === "obrolan-gpt") return !!ctx.mesin.gpt;
  if (a === "brave") return internet && adaKunciWeb();
  if (ALAT_INTERNET.includes(a)) return internet && !alasanDilewati(a);
  return true;
}

const HOST: Partial<Record<NamaAlat, string>> = {
  "wiki-subjek": "id.wikipedia.org", "wiki-kata-kunci": "id.wikipedia.org", "wiki-artikel": "id.wikipedia.org",
  duckduckgo: "api.duckduckgo.com", brave: "api.search.brave.com",
};
/** Alasan alat internet sedang dilewati (situsnya terblokir jaringan), atau null. */
export function alasanDilewati(a: NamaAlat): string | null {
  const h = HOST[a];
  return h ? hostTerblokir(h) : null;
}

const tidakAda = (teks: string, proses: LangkahProses[], topik: Respons["topik"]): Respons => ({ jalur: "tidak-ditemukan", teks, sumber: [], proses, topik });

/** Jalankan satu aksi. `kueri` boleh berbeda dari pesan asli (hasil penulisan ulang/perbaikan). */
export async function jalankanAlat(aksi: Aksi, niat: Niat, ctx: KonteksAlat): Promise<Respons> {
  const { mesin, opsi } = ctx;
  const mode = MODE[aksi.alat];
  if (mode) return mesin.proses(aksi.kueri, { ...opsi, mode });
  if (aksi.alat === "memori") {
    const proses: LangkahProses[] = [];
    const r = mesin.dariMemori(aksi.kueri, proses, "Agent: cari di memori pengguna");
    return r ? { ...r, topik: opsi.topik ?? null } : tidakAda("Tidak ada catatan memori yang cocok.", proses, opsi.topik ?? null);
  }
  return alatInternet(aksi, niat, ctx);
}

async function alatInternet(aksi: Aksi, niat: Niat, ctx: KonteksAlat): Promise<Respons> {
  const t0 = performance.now();
  let baru: HalamanWeb[] = [], galat = "";
  const panggilan: PanggilanHttp[] = []; // curl + respons asli setiap permintaan HTTP
  try {
    if (aksi.alat === "wiki-subjek") baru = await cariWikipedia(aksi.kueri, 3, "id", panggilan);
    else if (aksi.alat === "wiki-kata-kunci") baru = await cariWikipedia(aksi.kueri, 5, "id", panggilan);
    else if (aksi.alat === "duckduckgo") baru = await cariDuckDuckGo(aksi.kueri, 5, panggilan);
    else if (aksi.alat === "brave") baru = await cariWeb(aksi.kueri, 5, panggilan);
    else if (aksi.alat === "wiki-artikel") baru = await artikelWikipedia(aksi.kueri, "id", panggilan);
  } catch (e) { galat = (e as Error).message; }
  for (const h of baru) if (!ctx.halaman.some((k) => k.url === h.url && k.asal === h.asal)) ctx.halaman.push(h);
  const proses: LangkahProses[] = [{
    tahap: "Cari di internet", ms: performance.now() - t0,
    hasil: `${aksi.alat} "${aksi.kueri}": ${galat ? `gagal (${galat})` : `${baru.length} halaman (${baru.map((h) => h.judul).slice(0, 3).join(", ") || "-"})`}; terkumpul ${ctx.halaman.length}`,
    data: { halaman: baru.map((h) => ({ judul: h.judul, url: h.url, asal: h.asal, panjang: h.teks.length, cuplikan: h.teks.replace(/\s+/g, " ").slice(0, 220) })), terkumpul: ctx.halaman.length, panggilan, galat: galat || null },
  }];
  if (galat && !ctx.halaman.length) return tidakAda(`Permintaan ke internet gagal: ${galat}`, proses, ctx.opsi.topik ?? null);
  return jawabanWeb(niat, ctx.halaman, proses, ctx.opsi.topik ?? null);
}

/**
 * Susun jawaban dari halaman internet yang terkumpul. Kalimat lolos jika memuat SEMUA kata kunci
 * (≤ 3 kata) atau ≥ 70% (lebih banyak), memuat nama diri, dan memuat angka untuk "berapa"/"kapan".
 */
export function jawabanWeb(niat: Niat, kumpulan: HalamanWeb[], proses: LangkahProses[], topikLama: Respons["topik"]): Respons {
  const cukup = kalimatCukup(niat);
  const pilih = pilihKalimatWeb(niat.kueri, kumpulan, 2, cukup);
  const pakai = pilih.kalimat.filter(cukup);
  // untuk panel cara kerja: kalimat teratas dari SEMUA halaman terkumpul + apakah lolos syarat
  const teratas = pilihKalimatWeb(niat.kueri, kumpulan, 8, undefined, true).kalimat;
  proses.push({
    tahap: "Nilai kalimat internet", ms: 0,
    hasil: `${pilih.jumlahKalimat} kalimat dari ${kumpulan.length} halaman dinilai; ${pakai.length} lolos syarat`,
    data: { kalimat: teratas.map((k) => ({ teks: k.teks.slice(0, 260), sumber: kumpulan[k.halaman].judul, skor: Number(k.skor.toFixed(2)), cocok: k.cocok, lolos: cukup(k) })) },
  });
  if (!pakai.length) {
    const k0 = pilih.kalimat[0];
    return tidakAda(k0 ? `Kalimat terdekat (belum cukup cocok): ${k0.teks}` : "Tidak ada kalimat yang memuat kata pertanyaan.", proses, topikLama);
  }
  const nomor = new Map<number, number>();
  for (const k of pakai) if (!nomor.has(k.halaman)) nomor.set(k.halaman, nomor.size + 1);
  const sumber = [...nomor].map(([h, n]) => ({ nomor: n, label: `${kumpulan[h].judul} — ${kumpulan[h].url} (${kumpulan[h].asal})`, teks: kumpulan[h].teks }));
  const teks = [
    pakai.map((k) => `${k.teks} [${nomor.get(k.halaman)}]`).join(" "), "",
    "Sumber (internet):", ...sumber.map((s) => `  [${s.nomor}] ${s.label}`), "",
    "Catatan: kutipan dari internet, bukan dasar hukum. Periksa sumbernya.",
  ].join("\n");
  proses.push({ tahap: "Susun jawaban", hasil: `${pakai.length} kalimat dikutip dari ${sumber.length} halaman (BM25 mini atas kalimat halaman)`, ms: 0 });
  const utama = kumpulan[pakai[0].halaman];
  return {
    jalur: "internet", teks, sumber, proses,
    topik: { pertanyaan: niat.pesan, jalur: "internet", subjek: utama.judul.replace(/\s*\(.*\)$/, ""), sumberLabel: sumber[0].label, sumberTeks: utama.teks.slice(0, 6000) },
    keyakinan: pakai[0].cakupan >= 0.8 ? "tinggi" : "sedang",
    label: "kutipan dari internet (API resmi), bukan dasar hukum",
  };
}

/** Judul artikel yang paling cocok untuk dibaca lengkap: frasa berhuruf kapital, atau judul halaman terkumpul. */
export function judulArtikel(niat: Niat, kumpulan: HalamanWeb[]): string | null {
  // frasa berhuruf kapital di tengah kalimat, atau di awal kalimat jika ≥ 2 kata ("Joko Widodo sekarang …")
  const frasa = niat.kueri.match(/(?<=\s)[A-Z][\p{L}'-]+(?:\s+[A-Z][\p{L}'-]+)*|^[A-Z][\p{L}'-]+(?:\s+[A-Z][\p{L}'-]+)+/u)?.[0];
  if (frasa) return frasa;
  const kataQ = new Set(keTerm(niat.kueri));
  const nilai = (h: HalamanWeb) => { const k = keTerm(h.judul).filter((t) => !t.startsWith("~")); return k.length ? k.filter((t) => kataQ.has(t)).length / k.length : 0; };
  const wiki = kumpulan.filter((h) => h.asal === "Wikipedia");
  return wiki.length ? wiki.reduce((a, h) => (nilai(h) > nilai(a) ? h : a)).judul : null;
}
