/**
 * MENGAMBIL ISI ARTIKEL/KONTEN dari website apa pun (berita, blog, situs resmi) ke documents/web/.
 *
 * Tiga cara memberi sumber (boleh digabung, hasil digabung & tanpa duplikat):
 *   1. URL langsung : --url <halaman> atau URL sebagai argumen bebas (boleh banyak).
 *      Halaman artikel langsung diekstrak; halaman daftar/beranda ditelusuri sampai
 *      --batas artikel terkumpul (tautan beranchor panjang atau pola URL berita).
 *   2. RSS/Atom     : --rss <url> -> daftar artikel dari feed (news.detik.com, CNN, WordPress, ...)
 *   3. Daftar file  : --daftar path/file.txt (satu URL per baris, baris # = komentar)
 *
 * EKSTRAKSI (tanpa library HTML):
 *   - metadata: JSON-LD (NewsArticle/headline/datePublished) -> OpenGraph -> <h1>/<title>
 *   - isi: blok <article> yang valid, atau semua <p> substansial setelah menu/footer/iklan
 *     dibuang; minimal 200 huruf. Halaman yang tampak artikel tapi kosong dicatat "tanpa-isi".
 *
 * AMAN DIJALANKAN ULANG — tidak mengambil dua kali:
 *   - URL di documents/web/manifest.jsonl dilewati sebelum halaman dibuka;
 *   - file .md yang sudah ada tidak ditimpa (kecuali --ganti);
 *   - antrean + halaman daftar terlanjur dibaca tersimpan di .crawl-state.json (resume).
 *   --batas = artikel baru per sesi (bawaan 20; 0 = tanpa batas). --kering = daftar saja.
 *
 * ETIKA: robots.txt per host dihormati (blok Disallow untuk UA umum dicocokkan dengan pola),
 *   jeda 1 detik antar permintaan, UA khas proyek, retry 429/5xx. Untuk pemakaian
 *   personal/edukatif; konten berita tetap milik penerbitnya.
 *
 * Progres: Pelacak -> baris hidup + progres/fetch-web-<mode>.json (bun run status, panel web).
 * Ctrl+C = berhenti dengan aman (state disimpan); jalankan lagi untuk melanjutkan.
 *
 * Pakai:  bun run fetch-web -- https://www.bkn.go.id/category/publikasi/berita/ --batas 10
 *         bun run fetch-web -- --url https://news.detik.com/berita/d-8750810/contoh
 *         bun run fetch-web -- --rss https://news.detik.com/rss --batas 30
 *         bun run fetch-web -- --daftar urls.txt --kering
 *         bun run fetch-web                tanpa argumen = lanjutkan sisa antrean run lama
 * Setelah menambah artikel: bun run train  (supaya bisa dijawab chatbot).
 */
import { createHash } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Pelacak } from "../gpt/progress.js";
import { LOKASI } from "../paths.js";
import { argumen } from "./args.js";

const MENTAH = LOKASI.dokumenWeb;
const MANIFEST = join(MENTAH, "manifest.jsonl");
const STATE = join(MENTAH, ".crawl-state.json");
const JEDA_MS = 1000;
const WAKTU = 15000;
const HEADERS = { "User-Agent": "llm-from-scratch-learning/0.1 (educational project; low request rate)" };
const TOKEN_ROBOTS = "llm-from-scratch-learning"; // token UA (sebelum "/") untuk mencari blok User-agent di robots.txt

interface Entri { url: string; judul: string; paksa: boolean }
interface Artikel { judul: string; tanggal: string; paragraf: string[]; deskripsi: string }
interface Simpanan {
  antrean: { url: string; judul: string }[];
  dikunjungi: string[];
  gagal: { url: string; judul: string }[];
  waktu: string;
}

// ------------------------------------------------------------------------------------------
// ARGUMEN
const { opsi, teks } = argumen();
const KERING = opsi.kering === "true";
const GANTI = opsi.ganti === "true";
const BATASMENTAH = Number(opsi.batas ?? 20);
if (!Number.isFinite(BATASMENTAH) || BATASMENTAH < 0) {
  console.error(`Nilai --batas tidak valid: ${opsi.batas} (pakai angka; 0 = tanpa batas).`);
  process.exit(1);
}
const BATAS = BATASMENTAH === 0 ? Infinity : BATASMENTAH;

const token = teks.split(/\s+/).filter(Boolean);
const urlBebas = token.filter((t) => /^https?:\/\//i.test(t));
for (const t of token) {
  if (!/^https?:\/\//i.test(t)) {
    console.error(`Argumen bukan URL: ${t}\n`);
    tampilkanCaraPakai();
    process.exit(1);
  }
}
const daftarUrl = [...(opsi.url ? [opsi.url] : []), ...urlBebas];
for (const u of daftarUrl) {
  try { new URL(u); } catch { console.error(`URL tidak valid: ${u}`); process.exit(1); }
}

function tampilkanCaraPakai(): void {
  console.error(`Cara pakai:
  bun run fetch-web -- <url-halaman>                  ambil artikel, atau telusuri halaman daftar
  bun run fetch-web -- --url <url> --batas 30         batas artikel baru per sesi (bawaan 20)
  bun run fetch-web -- --rss https://news.detik.com/rss --batas 20
  bun run fetch-web -- --daftar urls.txt --kering     daftar URL dari file, pratinjau tanpa simpan
  bun run fetch-web                                   lanjutkan sisa antrean run sebelumnya
Keluaran: documents/web/<host>/<judul>.md + manifest.jsonl (aman dijalankan ulang).`);
}

// ------------------------------------------------------------------------------------------
// UTILITAS HTTP & HTML
class GalatPermanen extends Error {}
const tidur = (ms: number) => new Promise((r) => setTimeout(r, ms));
const stempel = () => new Date().toISOString();

async function ambilHtml(url: string, accept = "text/html"): Promise<string> {
  let galat = "gagal";
  for (let coba = 1; coba <= 3; coba++) {
    let r: Response;
    try {
      r = await fetch(url, { headers: { ...HEADERS, Accept: accept }, signal: AbortSignal.timeout(WAKTU) });
    } catch (e) { galat = (e as Error).message; if (coba < 3) await tidur(1500 * coba); continue; }
    if (r.status === 429 || r.status >= 500) { galat = `HTTP ${r.status}`; if (coba < 3) await tidur(1500 * coba); continue; }
    if (r.status === 400 || r.status === 404 || r.status === 410) throw new GalatPermanen(`HTTP ${r.status}`);
    if (!r.ok) throw new Error(`HTTP ${r.status} dari ${new URL(url).host}`);
    return await r.text();
  }
  throw new Error(galat);
}

const kodeAman = (n: number) => (n >= 1 && n <= 0x10ffff && !(n >= 0xd800 && n <= 0xdfff)) ? String.fromCodePoint(n) : "";
/** Teks HTML -> teks polos: buang tag, decode entity (numerik & umum), rapikan spasi. */
const bersihTag = (s: string) => s.replace(/<[^>]+>/g, " ")
  .replace(/&#x([0-9a-f]+);/gi, (_, n) => kodeAman(parseInt(n, 16)))
  .replace(/&#(\d+);/g, (_, n) => kodeAman(Number(n)))
  .replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
  .replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&rsquo;|&lsquo;/g, "'").replace(/&rdquo;|&ldquo;/g, '"')
  .replace(/&hellip;/g, "…").replace(/&mdash;/g, "—").replace(/&ndash;/g, "–").replace(/&raquo;|&laquo;/g, "»")
  .replace(/\s+/g, " ").trim();

/** Buang blok yang bukan isi artikel: skrip, gaya, menu, footer, sidebar, bentuk, komentar. */
function buangNoise(h: string): string {
  let s = h.replace(/<!--[\s\S]*?-->/g, "");
  for (const t of ["script", "style", "noscript", "template", "nav", "footer", "aside", "form", "iframe", "svg", "button"])
    s = s.replace(new RegExp(`<${t}[\\s>][\\s\\S]*?<\\/${t}>`, "gi"), "");
  return s;
}

function jsonLd(h: string): Record<string, any>[] {
  const out: Record<string, any>[] = [];
  for (const m of h.matchAll(/<script[^>]*application\/ld\+json[^>]*>([\s\S]*?)<\/script>/gi)) {
    try {
      const j = JSON.parse(m[1]);
      for (const o of Array.isArray(j) ? j : [j]) {
        if (o && typeof o === "object") { out.push(o); if (Array.isArray(o["@graph"])) out.push(...o["@graph"]); }
      }
    } catch { /* JSON rusak: lewati */ }
  }
  return out;
}

function meta(h: string, nama: string): string {
  const n = nama.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const m = h.match(new RegExp(`<meta[^>]+(?:property|name)="${n}"[^>]+content="([^"]*)"`, "i"))
    ?? h.match(new RegExp(`<meta[^>]+content="([^"]*)"[^>]+(?:property|name)="${n}"`, "i"));
  return m ? bersihTag(m[1]) : "";
}

const teksTag = (h: string, re: RegExp) => { const m = re.exec(h); return m ? bersihTag(m[1]) : ""; };

function tanggalIso(s: unknown): string {
  if (typeof s !== "string" || !s) return "";
  const m = /(\d{4})-(\d{2})-(\d{2})/.exec(s);
  if (m) return `${m[1]}-${m[2]}-${m[3]}`;
  const d = new Date(s);
  return isNaN(d.getTime()) ? "" : d.toISOString().slice(0, 10);
}

const JENIS_LD = new Set(["NewsArticle", "Article", "BlogPosting", "ReportageNewsArticle", "AnalysisNewsArticle", "OpinionNewsArticle", "LiveBlogPosting"]);
const KATA_SAMPAH = /^(baca juga|baca selengkapnya|selengkapnya|advertisement|iklan|rekomendasi|berita lainnya|share|bagikan|copyright|©)/i;

/** Semua <p> berisi dari satu potongan HTML: buang yang pendek, sampah, dan duplikat. */
function paragrafDari(h: string, minimal = 40): string[] {
  const out: string[] = [];
  const lihat = new Set<string>();
  for (const m of h.matchAll(/<p[\s>][\s\S]*?<\/p>/gi)) {
    const t = bersihTag(m[0].replace(/^<p[^>]*>/i, "").replace(/<\/p>$/i, ""));
    if (t.length < minimal || KATA_SAMPAH.test(t) || lihat.has(t)) continue;
    lihat.add(t);
    out.push(t);
  }
  return out;
}

/**
 * Keputusan halaman: artikel (dengan isinya) atau daftar (untuk ditelusuri).
 * Kandidat artikel: satu blok <article> berisi, JSON-LD tipe artikel, atau <h1> panjang
 * dengan banyak paragraf substansial. Halaman daftar biasanya punya banyak <article> kecil
 * atau <h1> pendek ("Berita", "Nasional") — tidak dianggap artikel.
 */
function ekstrak(h: string): { artikel: Artikel | null; daftar: boolean } {
  const ld = jsonLd(h);
  const jenis = (o: Record<string, any>) => Array.isArray(o["@type"]) ? o["@type"] : [o["@type"]];
  const artLd = ld.find((o) => jenis(o).some((t) => JENIS_LD.has(String(t))));

  const judul = (typeof artLd?.headline === "string" && artLd.headline.trim())
    || meta(h, "og:title")
    || teksTag(h, /<h1[^>]*>([\s\S]*?)<\/h1>/i)
    || teksTag(h, /<title>([\s\S]*?)<\/title>/i).replace(/\s+[|–—]\s+[^|–—]+$/, "");
  const tanggal = tanggalIso(artLd?.datePublished) || tanggalIso(meta(h, "article:published_time"))
    || tanggalIso(meta(h, "og:article:published_time")) || tanggalIso(meta(h, "date")) || tanggalIso(meta(h, "pubdate"));
  const deskripsi = (typeof artLd?.description === "string" && artLd.description) || meta(h, "og:description") || meta(h, "description");

  const bersih = buangNoise(h);
  const regions = [...bersih.matchAll(/<article[\s>][\s\S]*?<\/article>/gi)].map((m) => paragrafDari(m[0], 60));
  const regionTerbaik = regions.sort((a, b) => b.join("").length - a.join("").length)[0];
  const semua = paragrafDari(bersih);
  const substansial = semua.filter((p) => p.length >= 60);
  const h1 = teksTag(bersih, /<h1[^>]*>([\s\S]*?)<\/h1>/i);

  const regionYa = regions.length === 1 && !!regionTerbaik && regionTerbaik.length >= 3 && regionTerbaik.join("").length >= 800;
  const ldYa = !!artLd;
  const h1Ya = h1.length >= 15 && substansial.length >= 5 && semua.join("").length >= 1200;
  if (!regionYa && !ldYa && !h1Ya) return { artikel: null, daftar: true };

  // pilih isi: blok <article> jika berisi (1 blok tunggal, atau 1–2 blok dengan teks memadai)
  let paragraf: string[];
  if (regionYa) paragraf = regionTerbaik;
  else if (regions.length >= 1 && regions.length <= 2 && regionTerbaik && regionTerbaik.join("").length >= 600) paragraf = regionTerbaik;
  else paragraf = semua;
  if (!judul || paragraf.join("").length < 200) return { artikel: null, daftar: false }; // tampak artikel tapi isinya kosong
  return { artikel: { judul, tanggal, paragraf, deskripsi }, daftar: false };
}

// ------------------------------------------------------------------------------------------
// PENELUSURAN HALAMAN DAFTAR
const GENERIK = new Set(["go.id", "co.id", "or.id", "ac.id", "sch.id", "web.id", "my.id", "biz.id", "desa.id", "mil.id",
  "com", "net", "org", "co", "or", "ac", "sch", "go", "mil", "gov", "edu"]);
/** "www.kompas.com" -> "kompas.com" (subdomain boleh); "www.bkn.go.id" -> tetap (domain generik). */
function basisHost(host: string): string {
  const seg = host.split(".");
  if (seg.length <= 2) return host;
  const dua = seg.slice(-2).join(".");
  return GENERIK.has(dua) ? host : dua;
}
const asalDasar = new Set<string>();
const boleh = (host: string) => [...asalDasar].some((d) => host === d || host.endsWith(`.${d}`));

const RE_ASSET = /\.(css|js|mjs|png|jpe?g|gif|svg|webp|ico|pdf|zip|rar|xml|rss|json|mp[34]|webm|woff2?|ttf|eot|map)(?:$|\?)/i;
const RE_URL_BERITA = /\/d-\d+|\/\d{4}\/\d{2}\/|\/\d{6,}[-/]/;

/** Tautan yang tampak artikel di satu halaman; URL relatif dihitung terhadap halaman itu. */
function temukanTautan(h: string, dasar: string): { url: string; judul: string }[] {
  const hasil = new Map<string, string>();
  for (const m of h.matchAll(/<a[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi)) {
    let u: URL;
    try { u = new URL(m[1].replace(/&amp;/g, "&"), dasar); } catch { continue; }
    if (u.protocol !== "http:" && u.protocol !== "https:") continue;
    if (hasil.has(u.href)) continue;
    const judul = bersihTag(m[2]);
    if (!RE_URL_BERITA.test(u.pathname) && judul.length < 25) continue; // anchor pendek = menu/bukan artikel
    if (RE_ASSET.test(u.pathname)) continue;
    hasil.set(u.href, judul);
  }
  return [...hasil].map(([url, judul]) => ({ url, judul }));
}

// ------------------------------------------------------------------------------------------
// ROBOTS.TXT
function polaRobots(p: string): RegExp {
  let s = "";
  for (let i = 0; i < p.length; i++) {
    const c = p[i];
    if (c === "*") s += ".*";
    else if (c === "$" && i === p.length - 1) s += "$";
    else if (/[.+?^${}()|[\]\\]/.test(c)) s += "\\" + c;
    else s += c;
  }
  try { return new RegExp(s); } catch { return /(?!)/; }
}

/** Parser robots.txt: kelompok User-agent, pilih blok untuk UA kita atau "*", paling spesifik menang. */
function parserRobots(teks: string): (path: string) => boolean {
  const grup: { agen: string[]; aturan: { pola: string; izin: boolean }[] }[] = [];
  let sekarang: (typeof grup)[number] | null = null;
  let aturanAktif = false;
  for (const mentah of teks.split(/\r?\n/)) {
    const baris = mentah.replace(/#.*$/, "").trim();
    if (!baris) continue;
    const m = /^([A-Za-z-]+)\s*:\s*(.*)$/.exec(baris);
    if (!m) continue;
    const kunci = m[1].toLowerCase(), nilai = m[2].trim();
    if (kunci === "user-agent") {
      if (!sekarang || aturanAktif) { sekarang = { agen: [], aturan: [] }; grup.push(sekarang); aturanAktif = false; }
      sekarang.agen.push(nilai.toLowerCase());
    } else if ((kunci === "disallow" || kunci === "allow") && sekarang) {
      aturanAktif = true;
      if (nilai) sekarang.aturan.push({ pola: nilai, izin: kunci === "allow" });
    } // crawl-delay & lainnya diabaikan (jeda tetap 1 detik)
  }
  const pilih = grup.find((g) => g.agen.includes(TOKEN_ROBOTS)) ?? grup.find((g) => g.agen.includes("*"));
  if (!pilih || !pilih.aturan.length) return () => true;
  const aturan = pilih.aturan.map((a) => ({ ...a, re: polaRobots(a.pola) }));
  return (path) => {
    const cocok = aturan.filter((x) => x.re.test(path));
    if (!cocok.length) return true;
    const terpanjang = Math.max(...cocok.map((x) => x.pola.length));
    return cocok.filter((x) => x.pola.length === terpanjang).some((x) => x.izin); // seri -> Allow menang
  };
}

const robotsCache = new Map<string, (path: string) => boolean>();
async function robotsIzin(u: URL): Promise<(path: string) => boolean> {
  const ada = robotsCache.get(u.origin);
  if (ada) return ada;
  let izin: (path: string) => boolean = () => true;
  try {
    const r = await fetch(`${u.origin}/robots.txt`, { headers: HEADERS, signal: AbortSignal.timeout(8000) });
    if (r.ok) izin = parserRobots(await r.text());
  } catch { /* robots tidak terbaca -> anggap boleh (kebiasaan umum) */ }
  robotsCache.set(u.origin, izin);
  return izin;
}

// ------------------------------------------------------------------------------------------
// STATE & MANIFEST
function muatState(): Simpanan {
  try {
    const j = JSON.parse(readFileSync(STATE, "utf8")) as Partial<Simpanan>;
    if (Array.isArray(j.antrean) && Array.isArray(j.dikunjungi))
      return { antrean: j.antrean, dikunjungi: j.dikunjungi, gagal: Array.isArray(j.gagal) ? j.gagal : [], waktu: j.waktu ?? "" };
  } catch { /* belum ada state: mulai baru */ }
  return { antrean: [], dikunjungi: [], gagal: [], waktu: "" };
}
const state = muatState();
const dikunjungi = new Set(state.dikunjungi);
const gagalDulu = new Map(state.gagal.map((g) => [g.url, g.judul]));
const urlSudah = new Set<string>();
if (existsSync(MANIFEST)) {
  for (const baris of readFileSync(MANIFEST, "utf8").split("\n")) {
    try { const j = JSON.parse(baris); if (typeof j.url === "string") urlSudah.add(j.url); } catch { /* baris komentar / rusak */ }
  }
}
const catat = (baris: Record<string, unknown>) => appendFileSync(MANIFEST, JSON.stringify(baris) + "\n");

// ------------------------------------------------------------------------------------------
// ANTREAN & BENIH
const antrean: Entri[] = [];
/** Sisa antrean run lama: hanya diproses setelah benih + tautan segar habis (prioritas rendah). */
const cadangan: Entri[] = [];
const dalamAntrean = new Set<string>();
const diproses = new Set<string>();

function tambah(url: string, judul: string, paksa: boolean): boolean {
  if (dalamAntrean.has(url) || diproses.has(url)) return false;
  if (!paksa && (dikunjungi.has(url) || (!GANTI && urlSudah.has(url)))) return false;
  dalamAntrean.add(url);
  antrean.push({ url, judul, paksa });
  return true;
}

/** Sama seperti tambah, tapi masuk antrean cadangan (diproses paling akhir). */
function tambahCadangan(url: string, judul: string): void {
  if (dalamAntrean.has(url) || diproses.has(url)) return;
  if (dikunjungi.has(url) || (!GANTI && urlSudah.has(url))) return;
  dalamAntrean.add(url);
  cadangan.push({ url, judul, paksa: false });
}

/** Item RSS/Atom: <item>/<entry> dengan <link> (teks) atau <link href>; CDATA dibersihkan dulu. */
function itemFeed(feed: string): { url: string; judul: string }[] {
  const out: { url: string; judul: string }[] = [];
  for (const mentah of [...feed.matchAll(/<(?:item|entry)[\s\S]*?<\/(?:item|entry)>/gi)].map((m) => m[0])) {
    const blok = mentah.replace(/<!\[CDATA\[|\]\]>/g, "");
    const link = /<link[^>]+href="([^"]+)"/i.exec(blok)?.[1]
      ?? /<link[^>]*>\s*([^\s<][\s\S]*?)\s*<\/link>/i.exec(blok)?.[1]?.trim()
      ?? /<guid[^>]*>(https?:[^\s<]+)/i.exec(blok)?.[1];
    if (!link) continue;
    const judul = bersihTag(teksTag(blok, /<title[^>]*>([\s\S]*?)<\/title>/i));
    let u: URL;
    try { u = new URL(link); } catch { continue; }
    if (u.protocol !== "http:" && u.protocol !== "https:") continue;
    out.push({ url: u.href, judul });
  }
  return out;
}

async function siapkanBenih(): Promise<void> {
  if (opsi.rss) {
    const feed = await ambilHtml(opsi.rss, "application/rss+xml, application/atom+xml, application/xml, text/xml, */*");
    const item = itemFeed(feed);
    if (!item.length) console.log(`  Peringatan: tidak ada item di feed ${opsi.rss}`);
    for (const t of item) { asalDasar.add(basisHost(new URL(t.url).host)); tambah(t.url, t.judul, true); }
  }
  if (opsi.daftar) {
    let isi: string;
    try { isi = readFileSync(opsi.daftar, "utf8"); }
    catch { console.error(`File daftar tidak bisa dibaca: ${opsi.daftar}`); process.exit(1); }
    for (const baris of isi.split(/\r?\n/)) {
      const u = baris.trim();
      if (!u || u.startsWith("#")) continue;
      let p: URL;
      try { p = new URL(u); } catch { console.error(`Baris bukan URL di ${opsi.daftar}: ${u}`); continue; }
      asalDasar.add(basisHost(p.host));
      tambah(p.href, "", true);
    }
  }
  for (const u of daftarUrl) {
    const p = new URL(u);
    asalDasar.add(basisHost(p.host));
    tambah(p.href, "", true);
  }
}

// ------------------------------------------------------------------------------------------
// PEMROSESAN
let pelacak: Pelacak | null = null;
let halaman = 0, ditemukan = 0, diunduh = 0, dilewati = 0, tanpaIsi = 0, dilarang = 0, tidakAda = 0, gagal = 0, pratinjau = 0;

const maju = () => KERING ? pratinjau : diunduh;
const batasTercapai = () => maju() >= BATAS;

function catatGagal(url: string, judul: string): void {
  gagalDulu.delete(url);
  gagalDulu.set(url, judul);
  while (gagalDulu.size > 500) gagalDulu.delete(gagalDulu.keys().next().value!);
}

const slug = (s: string) => (s.toLowerCase().normalize("NFKD").replace(/[^\w\s-]/g, "").replace(/\s+/g, "-")
  .replace(/-+/g, "-").replace(/^-+|-+$/g, "").slice(0, 80) || "artikel");

/** Nama file dari segmen terakhir URL (unik per host); kalau kosong pakai sidik URL. */
function namaFile(u: URL, judul: string): string {
  const seg = u.pathname.split("/").filter(Boolean).pop() ?? "";
  const dasar = slug(seg.replace(/\.(html?|php)$/i, "") || judul);
  const sidik = createHash("sha1").update(u.href).digest("hex").slice(0, 6);
  const kandidat = `${dasar}.md`;
  return kandidat === ".md" ? `artikel-${sidik}.md` : kandidat;
}

async function simpan(e: Entri, artikel: Artikel): Promise<void> {
  const u = new URL(e.url);
  if (KERING) {
    pratinjau++;
    console.log(`  [${pratinjau}] ${artikel.judul} — ${e.url}`);
    pelacak?.langkah(pratinjau, { catatan: artikel.judul.slice(0, 70) });
    return;
  }
  const file = join(MENTAH, u.hostname, namaFile(u, artikel.judul));
  if (existsSync(file) && !GANTI) {
    // nama sudah terpakai (kemungkinan URL lain dengan segmen sama): beri sidik URL supaya tidak menimpa
    const sidik = createHash("sha1").update(e.url).digest("hex").slice(0, 6);
    const ganti = join(MENTAH, u.hostname, namaFile(u, artikel.judul).replace(/\.md$/, `-${sidik}.md`));
    if (existsSync(ganti)) {
      catat({ url: e.url, judul: artikel.judul, tanggal: artikel.tanggal, file: ganti, hasil: "ada", waktu: stempel() });
      urlSudah.add(e.url); dilewati++;
      return;
    }
    return tulisArtikel(e, artikel, ganti);
  }
  return tulisArtikel(e, artikel, file);
}

async function tulisArtikel(e: Entri, artikel: Artikel, file: string): Promise<void> {
  const u = new URL(e.url);
  const md = `# ${artikel.judul}\n\n- Sumber: ${e.url}\n- Tanggal: ${artikel.tanggal || "-"}\n- Diambil: ${stempel()}\n- Situs: ${u.hostname}\n\n`
    + `${artikel.paragraf.join("\n\n")}\n`;
  mkdirSync(join(MENTAH, u.hostname), { recursive: true });
  writeFileSync(file, md);
  urlSudah.add(e.url); diunduh++;
  catat({ url: e.url, judul: artikel.judul, tanggal: artikel.tanggal, file, ukuran: md.length, jumlahParagraf: artikel.paragraf.length, hasil: "ok", waktu: stempel() });
  gagalDulu.delete(e.url);
  pelacak?.langkah(diunduh, { catatan: artikel.judul.slice(0, 70) });
  console.log(`  [${diunduh}${BATAS === Infinity ? "" : `/${BATAS}`}] ${artikel.judul.slice(0, 80)} (${artikel.paragraf.length} paragraf, ${Math.round(md.length / 1024)} KB)`);
}

// ------------------------------------------------------------------------------------------
// UTAMA
async function utama(): Promise<void> {
  await siapkanBenih();
  for (const [url, judul] of gagalDulu) tambahCadangan(url, judul); // gagal run lama dicoba lagi
  for (const e of state.antrean) tambahCadangan(e.url, e.judul);    // sisa antrean run lama dilanjutkan
  if (!antrean.length && !cadangan.length) { tampilkanCaraPakai(); process.exit(1); }

  if (!KERING) {
    mkdirSync(MENTAH, { recursive: true });
    if (!existsSync(MANIFEST)) appendFileSync(MANIFEST, "# Manifest artikel web (hasil bun run fetch-web). Satu objek JSON per baris.\n");
  }
  const adaSumber = !!(daftarUrl.length || opsi.rss || opsi.daftar);
  const mode = adaSumber ? [daftarUrl.length ? "url" : "", opsi.rss ? "rss" : "", opsi.daftar ? "daftar" : ""].filter(Boolean).join("+") : "lanjut";
  console.log(`Mode ${mode}${KERING ? " (kering: daftar saja)" : ""} · batas ${BATAS === Infinity ? "tanpa batas" : `${BATAS} artikel baru`} · ${urlSudah.size} URL sudah tercatat · sumber: ${antrean.length}${cadangan.length ? ` (+${cadangan.length} antrean lama)` : ""}`);
  if (!adaSumber && cadangan.length) console.log(`Tidak ada sumber baru; melanjutkan ${cadangan.length} antrean dari run sebelumnya.`);
  console.log("(Ctrl+C = berhenti dengan aman · bisa dilanjutkan kapan saja · pantau: bun run status -- --ikuti)");

  pelacak = new Pelacak("fetch-web", mode, {
    total: BATAS === Infinity ? null : BATAS, satuan: KERING ? "artikel" : "tersimpan", satuanKecepatan: "artikel",
  });

  while ((antrean.length || cadangan.length) && !batasTercapai() && !pelacak.berhenti) {
    const e = antrean.shift() ?? cadangan.shift()!;
    if (diproses.has(e.url)) continue;
    diproses.add(e.url);
    if (!GANTI && urlSudah.has(e.url)) { dilewati++; continue; } // sudah diambil: jangan buka halamannya
    if (dikunjungi.has(e.url) && !e.paksa) continue;             // halaman daftar terlanjur dibaca run lama

    const u = new URL(e.url);
    const izin = await robotsIzin(u);
    if (!izin(u.pathname + u.search)) {
      dilarang++;
      console.log(`  ! dilewati robots.txt: ${e.url}`);
      continue;
    }

    await tidur(JEDA_MS);
    let html: string;
    try { html = await ambilHtml(e.url); }
    catch (err) {
      if (err instanceof GalatPermanen) {
        catat({ url: e.url, judul: e.judul, hasil: "tidak-ada", keterangan: (err as Error).message, waktu: stempel() });
        urlSudah.add(e.url); tidakAda++;
        console.log(`  - halaman tidak ada: ${e.url} (${(err as Error).message})`);
      } else {
        gagal++;
        catatGagal(e.url, e.judul);
        simpanState();
        console.log(`\n  ! gagal membaca ${e.url}: ${(err as Error).message}`);
      }
      continue;
    }

    const { artikel, daftar } = ekstrak(html);
    if (artikel) await simpan(e, artikel);
    else if (!daftar) {
      catat({ url: e.url, judul: e.judul, hasil: "tanpa-isi", waktu: stempel() });
      urlSudah.add(e.url); tanpaIsi++;
      console.log(`  - tampak artikel tapi isinya kosong/tipis: ${e.url}`);
    } else {
      halaman++;
      dikunjungi.add(e.url);
      gagalDulu.delete(e.url);
      for (const t of temukanTautan(html, e.url)) {
        let p: URL;
        try { p = new URL(t.url); } catch { continue; }
        if (!boleh(p.host)) continue;
        if (tambah(p.href, t.judul, false)) ditemukan++;
      }
      pelacak.langkah(maju(), { catatan: `daftar ${halaman} · antrian ${antrean.length + cadangan.length} · simpan ${diunduh}` });
      simpanState();
    }
  }

  const berhenti = pelacak.berhenti;
  simpanState();
  pelacak.tutup(berhenti ? "berhenti" : "selesai",
    KERING ? `${pratinjau} artikel dipratinjau` : `${diunduh} artikel tersimpan, ${urlSudah.size} total tercatat`);

  console.log(`\nRingkasan (mode ${mode}):`);
  console.log(`  halaman daftar  : ${halaman}`);
  console.log(`  tautan ditemukan: ${ditemukan}`);
  if (KERING) console.log(`  dipratinjau     : ${pratinjau} (tanpa simpan)`);
  else console.log(`  artikel tersimpan: ${diunduh} -> ${MENTAH}/`);
  console.log(`  dilewati (ada)  : ${dilewati}`);
  if (dilarang) console.log(`  robots.txt      : ${dilarang} dilewati (dilarang aturan situs)`);
  if (tanpaIsi) console.log(`  tanpa isi       : ${tanpaIsi}`);
  if (tidakAda) console.log(`  tidak ada (404) : ${tidakAda}`);
  if (gagal) console.log(`  gagal           : ${gagal} (akan dicoba lagi run berikutnya)`);
  if (berhenti) console.log(`\nDihentikan dengan aman. Jalankan perintah yang sama untuk melanjutkan; yang sudah ada tidak diambil ulang.`);
  else if (!KERING && diunduh) console.log(`\nJangan lupa: bun run train  (agar artikel baru bisa dijawab chatbot)`);
}

function simpanState(): void {
  if (KERING) return; // pratinjau tidak mengubah apa pun
  const baru: Simpanan = {
    antrean: [...cadangan, ...antrean].map(({ url, judul }) => ({ url, judul })),
    dikunjungi: [...dikunjungi],
    gagal: [...gagalDulu].map(([url, judul]) => ({ url, judul })),
    waktu: stempel(),
  };
  try { writeFileSync(STATE, JSON.stringify(baru)); } catch { /* tidak fatal */ }
}

utama().catch((e) => {
  simpanState();
  pelacak?.tutup("gagal", (e as Error).message);
  console.error(`\nGagal: ${(e as Error).message}`);
  process.exit(1);
});
