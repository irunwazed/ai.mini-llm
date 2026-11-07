/**
 * PENCARIAN INTERNET (hanya API resmi; tidak ada layanan AI).
 *
 *   1. Wikipedia Indonesia (API MediaWiki, gratis, tanpa kunci):
 *        - cariWikipedia     : cari judul yang cocok + paragraf pembuka (intro) beberapa artikel
 *        - artikelWikipedia  : teks lengkap satu artikel (dipakai bila intro belum memuat jawabannya)
 *   2. Brave Search API (opsional, butuh kunci gratis/berbayar milik Anda sendiri):
 *        set variabel lingkungan BRAVE_API_KEY, lalu cuplikan hasil web ikut dipakai.
 *
 * Halaman hasil mesin pencari biasa (Google/DuckDuckGo/Bing) TIDAK di-scrape: itu melanggar
 * ketentuan layanan mereka dan diblokir oleh pemeriksaan bot.
 * Kata kunci pertanyaan dikirim ke layanan di atas, jadi jangan menanyakan data pribadi lewat mode agent.
 */

export interface HalamanWeb { judul: string; url: string; teks: string; asal: string }

const UA = "llm-dari-nol/1.0 (proyek belajar; Bun + TypeScript)";
const WAKTU_TUNGGU = 8000;
const cache = new Map<string, HalamanWeb[]>();

async function json(url: string, header: Record<string, string> = {}): Promise<any> {
  const r = await fetch(url, { headers: { "User-Agent": UA, Accept: "application/json", ...header }, signal: AbortSignal.timeout(WAKTU_TUNGGU) });
  if (!r.ok) throw new Error(`HTTP ${r.status} dari ${new URL(url).host}`);
  return r.json();
}

async function dariCache(kunci: string, f: () => Promise<HalamanWeb[]>): Promise<HalamanWeb[]> {
  const ada = cache.get(kunci);
  if (ada) return ada;
  const h = await f();
  if (cache.size > 200) cache.delete(cache.keys().next().value!);
  cache.set(kunci, h);
  return h;
}

/** Bersihkan teks Wikipedia: buang judul bagian ("== Sejarah ==") dan baris kosong berlebih. */
function rapikanWiki(t: string): string {
  return t.replace(/^=+[^=\n]+=+\s*$/gm, "").replace(/\n{2,}/g, "\n").trim();
}

export function cariWikipedia(kueri: string, n = 3, bahasa = "id"): Promise<HalamanWeb[]> {
  return dariCache(`wiki:${bahasa}:${kueri}`, async () => {
    const p = new URLSearchParams({
      action: "query", format: "json", formatversion: "2", redirects: "1",
      generator: "search", gsrsearch: kueri, gsrlimit: String(n),
      prop: "extracts|info", inprop: "url", explaintext: "1", exintro: "1", exlimit: String(n),
    });
    const j = await json(`https://${bahasa}.wikipedia.org/w/api.php?${p}`);
    const hal: { index: number; title: string; fullurl: string; extract?: string }[] = j?.query?.pages ?? [];
    return hal.sort((a, b) => a.index - b.index).filter((x) => x.extract?.trim())
      .map((x) => ({ judul: x.title, url: x.fullurl, teks: rapikanWiki(x.extract!), asal: "Wikipedia" }));
  });
}

export function artikelWikipedia(judul: string, bahasa = "id"): Promise<HalamanWeb[]> {
  return dariCache(`artikel:${bahasa}:${judul}`, async () => {
    const p = new URLSearchParams({ action: "query", format: "json", formatversion: "2", redirects: "1", titles: judul, prop: "extracts|info", inprop: "url", explaintext: "1" });
    const j = await json(`https://${bahasa}.wikipedia.org/w/api.php?${p}`);
    const x = j?.query?.pages?.[0];
    return x?.extract ? [{ judul: x.title, url: x.fullurl, teks: rapikanWiki(x.extract).slice(0, 250000), asal: "Wikipedia (artikel lengkap)" }] : [];
  });
}

export const adaKunciWeb = () => !!process.env.BRAVE_API_KEY;

export function cariWeb(kueri: string, n = 5): Promise<HalamanWeb[]> {
  const kunci = process.env.BRAVE_API_KEY;
  if (!kunci) return Promise.resolve([]);
  return dariCache(`brave:${kueri}`, async () => {
    const p = new URLSearchParams({ q: kueri, count: String(n), country: "id", search_lang: "id" });
    const j = await json(`https://api.search.brave.com/res/v1/web/search?${p}`, { "X-Subscription-Token": kunci });
    const hasil: { title: string; url: string; description?: string; extra_snippets?: string[] }[] = j?.web?.results ?? [];
    const html = (s: string) => s.replace(/<[^>]+>/g, "").replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/\s+/g, " ").trim();
    return hasil.map((x) => ({ judul: html(x.title), url: x.url, teks: html([x.description ?? "", ...(x.extra_snippets ?? [])].join(" ")), asal: "Web (Brave Search)" })).filter((x) => x.teks);
  });
}
