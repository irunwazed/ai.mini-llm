/**
 * PENCARIAN INTERNET (hanya API resmi; tidak ada layanan AI).
 *
 *   1. Wikipedia Indonesia (API MediaWiki, gratis, tanpa kunci):
 *        - cariWikipedia     : cari judul yang cocok + paragraf pembuka (intro) beberapa artikel
 *        - artikelWikipedia  : teks lengkap satu artikel (dipakai bila intro belum memuat jawabannya)
 *   2. DuckDuckGo Instant Answer (API resmi, gratis, tanpa kunci):
 *        cariDuckDuckGo — definisi/entitas (Abstract, Answer, RelatedTopics).
 *        INI BUKAN hasil web penuh: untuk pertanyaan biasa hasilnya sering kosong,
 *        dan Abstract biasanya bersumber Wikipedia (jadi pelengkap, bukan pengganti).
 *   3. Brave Search API (opsional, butuh kunci gratis/berbayar milik Anda sendiri):
 *        set variabel lingkungan BRAVE_API_KEY, lalu cuplikan hasil web ikut dipakai.
 *
 * Halaman hasil mesin pencari biasa (Google/DuckDuckGo/Bing) TIDAK di-scrape: itu melanggar
 * ketentuan layanan mereka dan diblokir oleh pemeriksaan bot.
 * Kata kunci pertanyaan dikirim ke layanan di atas, jadi jangan menanyakan data pribadi lewat mode agent.
 *
 * Setiap permintaan HTTP dicatat (PanggilanHttp): perintah curl yang setara + respons asli server,
 * supaya terlihat di panel cara kerja persis apa yang dikirim dan diterima. Kunci API disamarkan.
 */

export interface HalamanWeb { judul: string; url: string; teks: string; asal: string }

const UA = "llm-dari-nol/1.0 (proyek belajar; Bun + TypeScript)";
const WAKTU_TUNGGU = 8000;
/** Satu permintaan HTTP: perintah curl setara + respons asli (dipotong jika sangat panjang). */
export interface PanggilanHttp {
  url: string; curl: string; status: number; ms: number; ukuran: number;
  tipe: string; respons: string; dipotong: boolean; dariCache?: boolean; galat?: string;
}
const MAKS_RESPONS = 12000;
const RAHASIA = /token|key|authorization/i;
const cache = new Map<string, { hal: HalamanWeb[]; panggilan: PanggilanHttp[] }>();

/** Perintah curl yang setara dengan permintaan ini (header rahasia disamarkan). */
function keCurl(url: string, header: Record<string, string>): string {
  const h = Object.entries(header).map(([k, v]) => ` \\\n  -H "${k}: ${RAHASIA.test(k) ? "***" : v.replace(/"/g, '\\"')}"`).join("");
  return `curl -s "${url}"${h}`;
}

/**
 * Situs yang koneksi amannya (TLS) gagal dianggap TERBLOKIR selama 15 menit, supaya agent tidak
 * membuang waktu mencoba ulang. Di Indonesia ini biasanya karena situs diblokir penyedia internet
 * (Internet Positif): DNS dialihkan ke halaman blokir atau koneksi diputus saat nama situs terlihat,
 * sehingga muncul galat seperti "certificate has expired" atau "handshake failure". Ini masalah
 * jaringan, bukan kode; aplikasi hanya melewati situs tersebut.
 */
const LAMA_BLOKIR = 15 * 60 * 1000;
const terblokir = new Map<string, { sampai: number; alasan: string }>();
const GALAT_TLS = /certificate|handshake|ssl|tls|self.signed|unable to verify|ECONNRESET|socket connection was closed/i;

/** Alasan sebuah host sedang dilewati (null jika boleh dipakai). */
export function hostTerblokir(host: string): string | null {
  const b = terblokir.get(host);
  if (!b) return null;
  if (Date.now() > b.sampai) { terblokir.delete(host); return null; }
  return b.alasan;
}

function jelaskanGalat(host: string, pesan: string): string {
  if (!GALAT_TLS.test(pesan)) return pesan;
  return `koneksi aman ke ${host} gagal ("${pesan}"). Biasanya karena situs ini diblokir atau diganggu jaringan/penyedia internet (mis. Internet Positif), bukan karena kode. Situs ini dilewati selama 15 menit.`;
}

async function json(url: string, header: Record<string, string> = {}, log?: PanggilanHttp[]): Promise<any> {
  const semua = { "User-Agent": UA, Accept: "application/json", ...header };
  const t0 = performance.now();
  const host = new URL(url).host;
  const catat: PanggilanHttp = { url, curl: keCurl(url, semua), status: 0, ms: 0, ukuran: 0, tipe: "", respons: "", dipotong: false };
  log?.push(catat);
  const blokir = hostTerblokir(host);
  if (blokir) { catat.galat = `tidak dikirim: ${blokir}`; throw new Error(`${host} dilewati — ${blokir}`); }
  let r: Response;
  try { r = await fetch(url, { headers: semua, signal: AbortSignal.timeout(WAKTU_TUNGGU) }); }
  catch (e) {
    const pesan = (e as Error).message;
    catat.ms = Math.round(performance.now() - t0);
    catat.galat = jelaskanGalat(host, pesan);
    if (GALAT_TLS.test(pesan)) terblokir.set(host, { sampai: Date.now() + LAMA_BLOKIR, alasan: catat.galat });
    throw new Error(catat.galat);
  }
  const teks = await r.text();
  Object.assign(catat, { status: r.status, ms: Math.round(performance.now() - t0), ukuran: teks.length, tipe: r.headers.get("content-type") ?? "" });
  let rapi = teks;
  try { rapi = JSON.stringify(JSON.parse(teks), null, 2); } catch { /* bukan JSON: tampilkan apa adanya */ }
  catat.dipotong = rapi.length > MAKS_RESPONS;
  catat.respons = catat.dipotong ? rapi.slice(0, MAKS_RESPONS) : rapi;
  if (!r.ok) { catat.galat = `HTTP ${r.status}`; throw new Error(`HTTP ${r.status} dari ${new URL(url).host}`); }
  return JSON.parse(teks);
}

/** Hasil yang sama tidak diminta ulang; panggilan aslinya tetap dicatat dengan tanda "dari cache". */
async function dariCache(kunci: string, f: (log: PanggilanHttp[]) => Promise<HalamanWeb[]>, log?: PanggilanHttp[]): Promise<HalamanWeb[]> {
  const ada = cache.get(kunci);
  if (ada) { log?.push(...ada.panggilan.map((p) => ({ ...p, dariCache: true }))); return ada.hal; }
  const lokal: PanggilanHttp[] = [];
  try {
    const hal = await f(lokal);
    if (cache.size > 200) cache.delete(cache.keys().next().value!);
    cache.set(kunci, { hal, panggilan: lokal });
    return hal;
  } finally { log?.push(...lokal); }
}

/** Bersihkan teks Wikipedia: buang judul bagian ("== Sejarah ==") dan baris kosong berlebih. */
function rapikanWiki(t: string): string {
  return t.replace(/^=+[^=\n]+=+\s*$/gm, "").replace(/\n{2,}/g, "\n").trim();
}

export function cariWikipedia(kueri: string, n = 3, bahasa = "id", log?: PanggilanHttp[]): Promise<HalamanWeb[]> {
  return dariCache(`wiki:${bahasa}:${kueri}`, async (lg) => {
    const p = new URLSearchParams({
      action: "query", format: "json", formatversion: "2", redirects: "1",
      generator: "search", gsrsearch: kueri, gsrlimit: String(n),
      prop: "extracts|info", inprop: "url", explaintext: "1", exintro: "1", exlimit: String(n),
    });
    const j = await json(`https://${bahasa}.wikipedia.org/w/api.php?${p}`, {}, lg);
    const hal: { index: number; title: string; fullurl: string; extract?: string }[] = j?.query?.pages ?? [];
    return hal.sort((a, b) => a.index - b.index).filter((x) => x.extract?.trim())
      .map((x) => ({ judul: x.title, url: x.fullurl, teks: rapikanWiki(x.extract!), asal: "Wikipedia" }));
  }, log);
}

export function artikelWikipedia(judul: string, bahasa = "id", log?: PanggilanHttp[]): Promise<HalamanWeb[]> {
  return dariCache(`artikel:${bahasa}:${judul}`, async (lg) => {
    const p = new URLSearchParams({ action: "query", format: "json", formatversion: "2", redirects: "1", titles: judul, prop: "extracts|info", inprop: "url", explaintext: "1" });
    const j = await json(`https://${bahasa}.wikipedia.org/w/api.php?${p}`, {}, lg);
    const x = j?.query?.pages?.[0];
    return x?.extract ? [{ judul: x.title, url: x.fullurl, teks: rapikanWiki(x.extract).slice(0, 250000), asal: "Wikipedia (artikel lengkap)" }] : [];
  }, log);
}

export const adaKunciWeb = () => !!process.env.BRAVE_API_KEY;

/**
 * DuckDuckGo Instant Answer (api.duckduckgo.com): API resmi, gratis, tanpa kunci.
 * Mengembalikan Abstract/Answer/RelatedTopics — definisi & entitas saja, bukan hasil web.
 * Kueri pertanyaan biasa sering kosong; jangan mengandalkan DDG sebagai satu-satunya sumber.
 */
export function cariDuckDuckGo(kueri: string, n = 5, log?: PanggilanHttp[]): Promise<HalamanWeb[]> {
  return dariCache(`ddg:${kueri}`, async (lg) => {
    const p = new URLSearchParams({ q: kueri, format: "json", no_html: "1", skip_disambig: "1" });
    const j = await json(`https://api.duckduckgo.com/?${p}`, {}, lg);
    const html = (s: string) => s.replace(/<[^>]+>/g, "").replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/\s+/g, " ").trim();
    const hal: HalamanWeb[] = [];

    const abstrak = html(j.AbstractText || j.Abstract || "");
    if (abstrak) {
      const asal = j.AbstractSource ? `DuckDuckGo (via ${j.AbstractSource})` : "DuckDuckGo";
      hal.push({ judul: html(j.AbstractHeading || j.Heading || kueri), url: j.AbstractURL || `https://duckduckgo.com/?q=${encodeURIComponent(kueri)}`, teks: abstrak, asal });
    }
    const jawab = html(j.Answer || "");
    if (jawab && !abstrak.includes(jawab)) hal.push({ judul: `Jawaban: ${kueri}`, url: `https://duckduckgo.com/?q=${encodeURIComponent(kueri)}`, teks: jawab, asal: "DuckDuckGo" });

    const topik = (j.RelatedTopics ?? []).flatMap((t: any) => t.Topics ?? [t]);
    for (const t of topik.slice(0, n)) {
      const teks = html(t.Text || "");
      if (teks && t.FirstURL) hal.push({ judul: teks.split(" - ")[0].slice(0, 80), url: t.FirstURL, teks, asal: "DuckDuckGo" });
    }
    return hal;
  }, log);
}

export function cariWeb(kueri: string, n = 5, log?: PanggilanHttp[]): Promise<HalamanWeb[]> {
  const kunci = process.env.BRAVE_API_KEY;
  if (!kunci) return Promise.resolve([]);
  return dariCache(`brave:${kueri}`, async (lg) => {
    const p = new URLSearchParams({ q: kueri, count: String(n), country: "id", search_lang: "id" });
    const j = await json(`https://api.search.brave.com/res/v1/web/search?${p}`, { "X-Subscription-Token": kunci }, lg);
    const hasil: { title: string; url: string; description?: string; extra_snippets?: string[] }[] = j?.web?.results ?? [];
    const html = (s: string) => s.replace(/<[^>]+>/g, "").replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/\s+/g, " ").trim();
    return hasil.map((x) => ({ judul: html(x.title), url: x.url, teks: html([x.description ?? "", ...(x.extra_snippets ?? [])].join(" ")), asal: "Web (Brave Search)" })).filter((x) => x.teks);
  }, log);
}
