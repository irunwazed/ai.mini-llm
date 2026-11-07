/**
 * MENGUNDUH TEKS WIKIPEDIA BAHASA INDONESIA untuk pretraining GPT.
 *
 * Urutan sumber:
 *   1. Kategori:Artikel pilihan dan Kategori:Artikel bagus (prosa paling rapi)
 *   2. artikel acak, sampai target ukuran tercapai
 * Agar sopan ke server, wikitext diambil 50 artikel per permintaan (prop=revisions),
 * berurutan dengan jeda 1 detik, dan menghormati header Retry-After saat dibatasi.
 * Markup wiki dibersihkan sendiri: templat {{...}}, tabel, referensi, berkas, tautan, dll.
 *
 * Pakai:  bun run fetch-data                      (target 10 MB)
 *         bun run fetch-data -- --target-mb 20
 * Keluaran: data/wiki-id.txt dan data/wiki-id.sources.txt (daftar judul + lisensi).
 * Bisa dihentikan dan dijalankan lagi: judul yang sudah diunduh dilewati.
 *
 * Lisensi teks Wikipedia: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
 */
import { appendFileSync, existsSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { argumen } from "./args.js";

const { opsi } = argumen();
const API = "https://id.wikipedia.org/w/api.php";
const KELUAR = "data/wiki-id.txt";
const SUMBER = "data/wiki-id.sources.txt";
const TARGET = Number(opsi["target-mb"] ?? 10) * 1024 * 1024;
const JEDA_MS = 1000;
const HEADERS = { "User-Agent": "llm-from-scratch-learning/0.1 (educational project; low request rate)" };
const BAGIAN_STOP = /^(referensi|daftar pustaka|pranala luar|catatan kaki|catatan|lihat pula|lihat juga|bacaan lanjutan|sumber|rujukan|bibliografi|galeri)$/i;

const tidur = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function api(params: Record<string, string>): Promise<any> {
  const url = `${API}?${new URLSearchParams({ format: "json", formatversion: "2", maxlag: "5", ...params })}`;
  for (let coba = 1; coba <= 8; coba++) {
    await tidur(JEDA_MS);
    try {
      const r = await fetch(url, { headers: HEADERS });
      if (r.status === 429 || r.status >= 500) {
        const tunggu = Number(r.headers.get("retry-after") ?? 0) * 1000 || 5000 * coba;
        process.stdout.write(`\n  server membatasi (HTTP ${r.status}), menunggu ${Math.round(tunggu / 1000)} detik...`);
        await tidur(tunggu);
        continue;
      }
      const j: any = await r.json();
      if (j.error?.code === "maxlag") { await tidur(5000); continue; }
      return j;
    } catch (e) {
      if (coba === 8) throw e;
      await tidur(3000 * coba);
    }
  }
  throw new Error("gagal setelah 8 percobaan");
}

async function anggotaKategori(kategori: string): Promise<string[]> {
  const judul = new Set<string>();
  let lanjut: string | undefined;
  do {
    const j = await api({
      action: "query", list: "categorymembers", cmtitle: kategori, cmlimit: "500", cmnamespace: "0|1",
      ...(lanjut ? { cmcontinue: lanjut } : {}),
    });
    for (const m of j.query?.categorymembers ?? []) judul.add(String(m.title).replace(/^Pembicaraan:/, ""));
    lanjut = j.continue?.cmcontinue;
  } while (lanjut);
  return [...judul];
}

/** Hapus pola bersarang (mis. {{ {{ }} }}) dari dalam ke luar sampai tidak berubah lagi. */
function hapusBersarang(s: string, re: RegExp): string {
  let lama;
  do { lama = s; s = s.replace(re, ""); } while (s !== lama);
  return s;
}

/** Wikitext -> teks polos. */
export function wikiKeTeks(w: string): string {
  let s = w.replace(/<!--[\s\S]*?-->/g, "");
  s = s.replace(/<ref[^>]*\/>/gi, "").replace(/<ref[^>]*>[\s\S]*?<\/ref>/gi, "");
  s = hapusBersarang(s, /\{\{[^{}]*\}\}/g);                 // templat
  s = hapusBersarang(s, /\{\|[^{}]*?\|\}/g);                // tabel
  s = hapusBersarang(s, /\[\[(?:berkas|file|gambar|image|kategori|category):[^[\]]*\]\]/gi); // berkas & kategori
  s = s.replace(/\[\[(?:berkas|file|gambar|image)[^\n]*\n/gi, "\n");
  s = s.replace(/\[\[([^|\]]*)\|([^\]]*)\]\]/g, "$2").replace(/\[\[([^\]]*)\]\]/g, "$1"); // tautan
  s = s.replace(/\[https?:[^\s\]]+\s([^\]]*)\]/g, "$1").replace(/\[?https?:\/\/\S+\]?/g, "");
  s = s.replace(/'{2,}/g, "").replace(/<[^>]+>/g, "").replace(/&nbsp;/g, " ").replace(/&[a-z]+;/g, " ");
  s = s.replace(/\(\s*[;,]?\s*\)/g, "").replace(/__[A-Z]+__/g, "");
  return s;
}

/** Bersihkan teks: buang judul bagian, bagian referensi dst., daftar, dan baris pendek. */
function bersihkan(teks: string): string {
  const keluar: string[] = [];
  for (const mentah of teks.split("\n")) {
    const b = mentah.trim();
    const judul = b.match(/^=+\s*(.*?)\s*=+$/);
    if (judul) {
      if (BAGIAN_STOP.test(judul[1])) break; // sisa artikel hanya referensi
      continue;
    }
    if (/^[*#:;|!{}]/.test(b) || b.length < 40) continue;
    keluar.push(b.replace(/\s+/g, " ").replace(/\s+([,.;:])/g, "$1"));
  }
  return keluar.join("\n");
}

/** Ambil & bersihkan wikitext untuk sampai 50 judul sekaligus. */
async function ambilBanyak(judul: string[]): Promise<{ judul: string; teks: string }[]> {
  const j = await api({ action: "query", prop: "revisions", rvprop: "content", rvslots: "main", redirects: "1", titles: judul.join("|") });
  return (j.query?.pages ?? []).map((p: any) => ({
    judul: String(p.title),
    teks: bersihkan(wikiKeTeks(p.revisions?.[0]?.slots?.main?.content ?? "")),
  }));
}

// ------------------------------------------------------------------------------------------
const sudah = new Set(existsSync(SUMBER) ? readFileSync(SUMBER, "utf8").split("\n").filter((l) => l && !l.startsWith("#")).map((l) => l.trim()) : []);
if (!existsSync(SUMBER)) writeFileSync(SUMBER, "# Sumber: id.wikipedia.org, lisensi CC BY-SA 4.0. Satu judul artikel per baris.\n");
const ukuran = () => (existsSync(KELUAR) ? statSync(KELUAR).size : 0);
const mb = (n: number) => (n / 1024 / 1024).toFixed(1);

async function unduh(daftar: string[]): Promise<number> {
  let n = 0;
  const baru = daftar.filter((t) => !sudah.has(t));
  for (let i = 0; i < baru.length && ukuran() < TARGET; i += 50) {
    for (const { judul, teks } of await ambilBanyak(baru.slice(i, i + 50))) {
      sudah.add(judul);
      if (teks.length < 1000) continue; // artikel rintisan terlalu pendek
      appendFileSync(KELUAR, teks + "\n\n");
      appendFileSync(SUMBER, judul + "\n");
      n++;
    }
    process.stdout.write(`\r  ${n} artikel, total ${mb(ukuran())} MB   `);
  }
  return n;
}

console.log(`Target ${mb(TARGET)} MB, sekarang ${mb(ukuran())} MB (${sudah.size} judul sudah diunduh).`);
for (const kat of ["Kategori:Artikel pilihan", "Kategori:Artikel bagus"]) {
  if (ukuran() >= TARGET) break;
  const daftar = await anggotaKategori(kat);
  console.log(`${kat}: ${daftar.length} judul`);
  const n = await unduh(daftar);
  console.log(`\r  ${n} artikel baru, total ${mb(ukuran())} MB   `);
}
while (ukuran() < TARGET) {
  const j = await api({ action: "query", list: "random", rnnamespace: "0", rnlimit: "50" });
  await unduh((j.query?.random ?? []).map((r: any) => String(r.title)));
}
console.log(`\nSelesai: ${KELUAR} (${mb(ukuran())} MB, ${sudah.size} judul). Lisensi: CC BY-SA 4.0.`);
