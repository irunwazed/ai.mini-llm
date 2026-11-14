/**
 * MENGUNDUH PDF dari JDIH BKN (https://jdih.bkn.go.id/) ke documents/jdih/.
 *
 * Cara memilih dokumen (hasil digabung, tanpa duplikat):
 *   1. Crawler otomatis — default saat tanpa argumen: beranda -> menu kategori + katalog
 *      (/Home/peraturan, monografi, artikel, putusan, cari_data) -> ikuti paginasi
 *      (/{kategori}/index/<n>, /Home/<katalog>/<n>) -> halaman detail (/.../breaking/<id>)
 *      -> link PDF (/common/dokumen/*.pdf). Berjalan sendiri sampai --batas tercapai
 *      atau semua halaman habis.
 *   2. Pencarian:  --judul "cuti tahunan" --tahun 2024 --nomor 12 --tipe "SURAT EDARAN" --status Berlaku
 *   3. Kategori:   --kategori P_bkn   (Undang, P_pemerintah, P_presiden, P_kabkn, S_edaran, ...)
 *   4. URL:        --url <halaman detail>  (boleh juga ditulis langsung sebagai argumen bebas)
 *
 * AMAN DIJALANKAN ULANG — tidak mengunduh dua kali:
 *   - id yang tercatat di documents/jdih/manifest.jsonl dilewati sebelum halaman detail dibuka;
 *   - file PDF yang sudah ada di disk tidak ditimpa, kecuali --ganti;
 *   - halaman daftar yang sudah dibaca & halaman yang gagal disimpan di .crawl-state.json,
 *     jadi run berikutnya melanjutkan (bukan mulai nol) dan mencoba yang gagal lagi.
 *   --batas N = jumlah unduhan BARU per sesi (bawaan 50; 0 = tanpa batas).
 *   --kering  = daftar dokumen saja, tanpa mengunduh apa pun (tidak menulis state).
 *
 * Etika server: jeda 500 ms antar permintaan, hanya host jdih.bkn.go.id, retry untuk 429/5xx.
 * Progres: baris hidup + file progres/fetch-jdih-<mode>.json -> `bun run status`, panel web.
 * Ctrl+C = berhenti dengan aman (state disimpan); jalankan perintah sama untuk melanjutkan.
 *
 * Pakai:  bun run fetch-jdih                                (crawler, batas 50 PDF baru)
 *         bun run fetch-jdih -- --batas 0                   (semua yang ditemukan)
 *         bun run fetch-jdih -- --judul "cuti" --tahun 2024 --batas 10
 *         bun run fetch-jdih -- --kategori S_edaran --batas 20
 *         bun run fetch-jdih -- --url https://jdih.bkn.go.id/Detail_peraturan/breaking/2643
 *         bun run fetch-jdih -- --kering --batas 20         (daftar saja, tanpa unduh)
 * Setelah menambah dokumen: bun run train  (supaya isi peraturan bisa dijawab chatbot).
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Pelacak } from "../gpt/progress.js";
import { argumen } from "./args.js";

const ASAL = "https://jdih.bkn.go.id";
const HOST = "jdih.bkn.go.id";
const FOLDER = "documents/jdih";
const MANIFEST = join(FOLDER, "manifest.jsonl");
const STATE = join(FOLDER, ".crawl-state.json");
const JEDA_MS = 500;
const WAKTU_HALAMAN = 15000;
const WAKTU_PDF = 60000;
const MAKS_GAGAL = 500; // halaman gagal yang diingat antar run (urutan lama dibuang)
const HEADERS = { "User-Agent": "llm-from-scratch-learning/0.1 (educational project; low request rate)" };
/** Halaman daftar lanjutan: /{kategori}/index/<n> atau /Home/<katalog>/<n>. */
const RE_HALAMAN = /\/index\/\d+\/?$|\/Home\/[A-Za-z0-9_]+\/\d+\/?$/;

type Mode = "crawler" | "cari" | "kategori" | "url";
interface Entri { url: string; judul: string; paksa: boolean }
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
const CRAWLER = opsi.crawler === "true";
const BATASMENTAH = Number(opsi.batas ?? 50);
if (!Number.isFinite(BATASMENTAH) || BATASMENTAH < 0) {
  console.error(`Nilai --batas tidak valid: ${opsi.batas} (pakai angka; 0 = tanpa batas).`);
  process.exit(1);
}
const BATAS = BATASMENTAH === 0 ? Infinity : BATASMENTAH;

const token = teks.split(/\s+/).filter(Boolean);
const urlBebas = token.filter((t) => /^https?:\/\//i.test(t));
const daftarUrl = [...(opsi.url ? [opsi.url] : []), ...urlBebas];
const judulCari = opsi.judul ?? (daftarUrl.length === 0 && token.length ? token.join(" ") : undefined);
const mode: Mode = daftarUrl.length ? "url" : opsi.kategori ? "kategori" : judulCari ? "cari" : "crawler";
/** Crawler mengikuti paginasi apa pun yang ditemukan; mode lain hanya keluarga sumbernya. */
const crawlSemua = mode === "crawler" || CRAWLER;

if (mode === "url") {
  for (const s of daftarUrl) {
    let u: URL;
    try { u = new URL(s); } catch { console.error(`URL tidak valid: ${s}`); process.exit(1); }
    if (u.hostname !== HOST && !u.hostname.endsWith(`.${HOST}`)) {
      console.error(`Hanya halaman ${HOST} yang bisa diunduh: ${s}`);
      process.exit(1);
    }
  }
}
if (mode === "kategori" && !/^[A-Za-z0-9_-]+$/.test(opsi.kategori!)) {
  console.error(`Nama kategori tidak valid: ${opsi.kategori} (contoh: P_bkn, Undang, S_edaran)`);
  process.exit(1);
}

function urlPencarian(): string {
  const p = new URLSearchParams();
  if (judulCari) p.set("judul", judulCari);
  for (const [k, v] of [["tipe", opsi.tipe], ["bentuk", opsi.bentuk], ["tahun", opsi.tahun], ["nomor", opsi.nomor], ["status", opsi.status]])
    if (v) p.set(k, v);
  const q = p.toString();
  return `${ASAL}/index.php/Home/cari_data${q ? `?${q}` : ""}`;
}
const keluargaSeed = mode === "cari" ? keluarga(new URL(urlPencarian())) : mode === "kategori" ? (opsi.kategori ?? "") : "";

// ------------------------------------------------------------------------------------------
// UTILITAS HTTP & HTML
/** Gagal permanen (file/halaman tidak ada di server) — dicatat ke manifest, tidak dicoba lagi. */
class GalatPermanen extends Error {}
const tidur = (ms: number) => new Promise((r) => setTimeout(r, ms));
const mb = (n: number) => (n / 1024 / 1024).toFixed(1);
const stempel = () => new Date().toISOString();

async function ambilHtml(url: string): Promise<string> {
  let galat = "gagal";
  for (let coba = 1; coba <= 3; coba++) {
    let r: Response;
    try {
      r = await fetch(url, { headers: { ...HEADERS, Accept: "text/html" }, signal: AbortSignal.timeout(WAKTU_HALAMAN) });
    } catch (e) { galat = (e as Error).message; if (coba < 3) await tidur(1500 * coba); continue; }
    if (r.status === 429 || r.status >= 500) { galat = `HTTP ${r.status}`; if (coba < 3) await tidur(1500 * coba); continue; }
    if (!r.ok) throw new Error(`HTTP ${r.status} dari ${HOST}`);
    return await r.text();
  }
  throw new Error(galat);
}

/** Unduh satu PDF: wajib berawalan %PDF (server kadang mengirim halaman HTML sebagai "PDF"). */
async function unduhPdf(url: string, tujuan: string): Promise<number> {
  let galat = "gagal";
  for (let coba = 1; coba <= 3; coba++) {
    let r: Response;
    try {
      r = await fetch(url, { headers: HEADERS, signal: AbortSignal.timeout(WAKTU_PDF) });
    } catch (e) { galat = (e as Error).message; if (coba < 3) await tidur(1500 * coba); continue; }
    if (r.status === 429 || r.status >= 500) { galat = `HTTP ${r.status}`; if (coba < 3) await tidur(1500 * coba); continue; }
    if (r.status === 400 || r.status === 404) throw new GalatPermanen(`HTTP ${r.status}`); // file tidak ada di server
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const buf = new Uint8Array(await r.arrayBuffer());
    if (!(buf[0] === 0x25 && buf[1] === 0x50 && buf[2] === 0x44 && buf[3] === 0x46))
      throw new Error("bukan PDF (server tidak mengirim berkas PDF)");
    writeFileSync(tujuan, buf);
    return buf.length;
  }
  throw new Error(galat);
}

const bersihTag = (s: string) => s.replace(/<[^>]+>/g, " ").replace(/&nbsp;/g, " ").replace(/&amp;/g, "&")
  .replace(/&#0?39;/g, "'").replace(/&quot;/g, '"').replace(/&rsquo;/g, "'").replace(/\s+/g, " ").trim();

/** Isi satu baris tabel metadata halaman detail, mis. Judul, Nomor Peraturan, Status. */
function meta(h: string, label: string): string {
  const re = new RegExp(`<th[^>]*>\\s*${label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*</th>\\s*<td[^>]*>([\\s\\S]*?)</td>`, "i");
  const m = h.match(re);
  return m ? bersihTag(m[1]) : "";
}
function metaPertama(h: string, label: string[]): string {
  for (const l of label) { const v = meta(h, l); if (v) return v; }
  return "";
}
const judulBreadcrumb = (h: string) => {
  const m = /breadcrumb-area[\s\S]*?<h2>([\s\S]*?)<\/h2>/i.exec(h);
  return m ? bersihTag(m[1]) : "";
};

/** URL PDF di halaman detail: /common/dokumen/*.pdf (langsung atau lewat DocumentProxy). */
function ekstrakPdf(h: string): string {
  const kandidat: string[] = [];
  for (const m of h.matchAll(/(?:href|data)="([^"]*common\/dokumen\/[^"]+?\.pdf)"/gi)) kandidat.push(m[1]);
  for (const m of h.matchAll(/DocumentProxy\/pdf\?url=([^"&]+)/gi)) {
    try { kandidat.push(decodeURIComponent(m[1])); } catch { /* bukan URL ter-encode: lewati */ }
  }
  for (const k of kandidat) {
    try {
      const u = new URL(k, ASAL);
      if ((u.hostname === HOST || u.hostname.endsWith(`.${HOST}`)) && u.pathname.toLowerCase().endsWith(".pdf")) return u.href;
    } catch { /* kandidat rusak: lewati */ }
  }
  return "";
}

const idDari = (u: URL) => { const m = u.pathname.match(/\/breaking\/(\d+)/); return m ? Number(m[1]) : null; };
/** Keluarga halaman daftar: "Home/cari_data", "Home/peraturan", "P_bkn", ... */
function keluarga(u: URL): string {
  const seg = u.pathname.replace(/^\/index\.php/, "").split("/").filter(Boolean);
  return seg[0] === "Home" ? seg.slice(0, 2).join("/") : seg[0] ?? "";
}
function slug(s: string): string {
  const j = s.toLowerCase().normalize("NFKD").replace(/[^\w\s-]/g, "").replace(/\s+/g, "-")
    .replace(/-+/g, "-").replace(/^-+|-+$/g, "").slice(0, 80);
  return j || "dokumen";
}

// ------------------------------------------------------------------------------------------
// STATE & MANIFEST (anti unduh-ulang)
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
const idSudah = new Set<number>();
if (existsSync(MANIFEST)) {
  for (const baris of readFileSync(MANIFEST, "utf8").split("\n")) {
    try { const j = JSON.parse(baris); if (typeof j.id === "number") idSudah.add(j.id); } catch { /* baris komentar / rusak */ }
  }
}
const catat = (baris: Record<string, unknown>) => appendFileSync(MANIFEST, JSON.stringify(baris) + "\n");

// ------------------------------------------------------------------------------------------
// ANTREAN & BENIH
const antrean: Entri[] = [];
const dalamAntrean = new Set<string>();
const diproses = new Set<string>();
const idDilihat = new Set<number>();

function tambah(url: string, judul: string, paksa: boolean): void {
  if (dalamAntrean.has(url) || diproses.has(url)) return;
  if (!paksa && dikunjungi.has(url)) return;
  dalamAntrean.add(url);
  antrean.push({ url, judul, paksa });
}

/** Benih crawler: beranda + semua link menu/katalog satu segmen di beranda (otomatis, bukan daftar mati). */
async function siapkanBenih(): Promise<void> {
  if (mode === "url") { for (const s of daftarUrl) tambah(s, "", true); return; }
  if (mode === "kategori") { tambah(`${ASAL}/${opsi.kategori}`, "", true); return; }
  if (mode === "cari") { tambah(urlPencarian(), "", true); return; }
  await tidur(JEDA_MS);
  const beranda = await ambilHtml(`${ASAL}/`);
  tambah(`${ASAL}/`, "Beranda", true);
  for (const m of beranda.matchAll(/href="([^"]+)"/g)) {
    let u: URL;
    try { u = new URL(m[1], ASAL); } catch { continue; }
    if (u.hostname !== HOST && !u.hostname.endsWith(`.${HOST}`)) continue;
    const p = u.pathname.replace(/^\/index\.php/, "");
    if (/^\/[A-Za-z0-9_]+\/?$/.test(p) || /^\/Home\/[A-Za-z0-9_]+\/?$/.test(p)) tambah(u.href, "", true);
  }
}

// ------------------------------------------------------------------------------------------
// PEMROSESAN
let pelacak: Pelacak | null = null;
let halaman = 0, ditemukan = 0, sudahAda = 0, diunduh = 0, dilewati = 0, tanpaPdf = 0, hilang = 0, gagal = 0, pratinjau = 0;

const maju = () => KERING ? pratinjau : diunduh;
const batasTercapai = () => maju() >= BATAS;

function catatGagal(url: string, judul: string): void {
  gagalDulu.delete(url);
  gagalDulu.set(url, judul);
  while (gagalDulu.size > MAKS_GAGAL) gagalDulu.delete(gagalDulu.keys().next().value!);
}

/** Halaman daftar: kumpulkan tautan detail (dengan judulnya) + paginasi lanjutan. */
function prosesDaftar(html: string): void {
  const baris = [...html.matchAll(/<a[^>]+href="([^"]*\/breaking\/\d+)"[^>]*>([\s\S]*?)<\/a>/g)];
  for (const m of baris) {
    let u: URL;
    try { u = new URL(m[1], ASAL); } catch { continue; }
    if (u.hostname !== HOST && !u.hostname.endsWith(`.${HOST}`)) continue;
    const id = idDari(u);
    if (id === null || idDilihat.has(id)) continue;
    idDilihat.add(id);
    if (!GANTI && idSudah.has(id)) { sudahAda++; continue; }
    ditemukan++;
    tambah(u.href, bersihTag(m[2]), false);
  }
  if (!baris.length) return; // halaman kosong = ujung daftar: jangan ikuti paginasinya
  for (const m of html.matchAll(/href="([^"]+)"/g)) {
    let u: URL;
    try { u = new URL(m[1], ASAL); } catch { continue; }
    if (u.hostname !== HOST && !u.hostname.endsWith(`.${HOST}`)) continue;
    if (!RE_HALAMAN.test(u.pathname)) continue;
    if (!crawlSemua && keluargaSeed && keluarga(u) !== keluargaSeed) continue;
    tambah(u.href, "", false);
  }
}

/** Halaman detail: ekstrak metadata + PDF, lalu unduh (atau cukup dicatat saat --kering). */
async function prosesDetail(html: string, id: number, url: string, judulEntri: string): Promise<void> {
  const pdf = ekstrakPdf(html);
  const judul = metaPertama(html, ["Judul", "Judul Peraturan"]) || judulBreadcrumb(html) || judulEntri || `Dokumen ${id}`;
  const nomor = metaPertama(html, ["Nomor", "Nomor Peraturan"]);
  const tahun = metaPertama(html, ["Tahun", "Tahun Terbit"]);
  const status = meta(html, "Status");
  const bentuk = meta(html, "Bentuk");
  const namaFile = `${slug(judul)}-${id}.pdf`;

  if (KERING) {
    pratinjau++;
    console.log(`  [${pratinjau}] ${pdf ? namaFile : "TANPA PDF"} — ${judul}`);
    pelacak?.langkah(pratinjau, { catatan: judul.slice(0, 70) });
    return;
  }
  const dasar = { id, judul, nomor, tahun, status, bentuk, url };
  if (!pdf) {
    catat({ ...dasar, hasil: "tanpa-pdf", waktu: stempel() });
    idSudah.add(id); tanpaPdf++;
    console.log(`  - id ${id}: halaman tanpa link PDF`);
    return;
  }
  const tujuan = join(FOLDER, namaFile);
  if (existsSync(tujuan) && !GANTI) {
    catat({ ...dasar, sumber: pdf, file: namaFile, hasil: "ada", waktu: stempel() });
    idSudah.add(id); dilewati++;
    return;
  }
  await tidur(JEDA_MS);
  try {
    const ukuran = await unduhPdf(pdf, tujuan);
    idSudah.add(id); diunduh++;
    catat({ ...dasar, sumber: pdf, file: namaFile, ukuran, hasil: "ok", waktu: stempel() });
    gagalDulu.delete(url);
    pelacak?.langkah(diunduh, { catatan: namaFile });
    console.log(`  [${diunduh}${BATAS === Infinity ? "" : `/${BATAS}`}] ${namaFile} (${mb(ukuran)} MB)`);
  } catch (e) {
    if (e instanceof GalatPermanen) {
      catat({ ...dasar, sumber: pdf, hasil: "tidak-ada", waktu: stempel() });
      idSudah.add(id); hilang++;
      gagalDulu.delete(url);
      console.log(`  - id ${id}: file PDF tidak ada di server (${(e as Error).message})`);
    } else {
      gagal++;
      catatGagal(url, judulEntri);
      simpanState();
      console.log(`\n  ! unduh gagal id ${id}: ${(e as Error).message}`);
    }
  }
}

// ------------------------------------------------------------------------------------------
// UTAMA
async function utama(): Promise<void> {
  if (!KERING) {
    mkdirSync(FOLDER, { recursive: true });
    if (!existsSync(MANIFEST)) appendFileSync(MANIFEST, "# Manifest unduhan JDIH BKN (https://jdih.bkn.go.id). Satu objek JSON per baris.\n");
  }
  console.log(`Mode ${mode}${KERING ? " (kering: daftar saja)" : ""} · batas ${BATAS === Infinity ? "tanpa batas" : `${BATAS} PDF baru`} · ${idSudah.size} dokumen sudah tercatat.`);
  console.log("(Ctrl+C = berhenti dengan aman · bisa dilanjutkan kapan saja · pantau: bun run status -- --ikuti)");

  pelacak = new Pelacak("fetch-jdih", mode, {
    total: BATAS === Infinity ? null : BATAS, satuan: KERING ? "dokumen" : "PDF", satuanKecepatan: "unduh",
  });

  await siapkanBenih();
  for (const [url, judul] of gagalDulu) tambah(url, judul, false); // gagal run lama dicoba lagi
  for (const e of state.antrean) tambah(e.url, e.judul, false);    // sisa antrean run lama dilanjutkan

  while (antrean.length && !batasTercapai() && !pelacak.berhenti) {
    const e = antrean.shift()!;
    if (diproses.has(e.url)) continue;
    diproses.add(e.url);
    const id = idDari(new URL(e.url, ASAL));
    if (id !== null && !GANTI && idSudah.has(id)) { dilewati++; continue; } // sudah diunduh: jangan buka halamannya

    await tidur(JEDA_MS);
    let html: string;
    try { html = await ambilHtml(e.url); }
    catch (err) {
      gagal++;
      catatGagal(e.url, e.judul); // gagal diingat di state: halaman daftar yang gagal = anaknya perlu dicari lagi run berikutnya
      simpanState();
      console.log(`\n  ! gagal membaca ${e.url}: ${(err as Error).message}`);
      continue;
    }

    if (id !== null) await prosesDetail(html, id, e.url, e.judul);
    else {
      if (dikunjungi.has(e.url) && !e.paksa) continue;
      halaman++;
      dikunjungi.add(e.url);
      gagalDulu.delete(e.url);
      prosesDaftar(html);
      // state disimpan setiap halaman daftar: crash/Ctrl+C paksa tidak menghilangkan subtree
      pelacak.langkah(maju(), { catatan: `daftar ${halaman} · antrian ${antrean.length} · unduh ${diunduh}` });
      simpanState();
    }
  }

  const berhenti = pelacak.berhenti;
  simpanState();
  pelacak.tutup(berhenti ? "berhenti" : "selesai",
    KERING ? `${pratinjau} dokumen dipratinjau` : `${diunduh} PDF baru, ${idSudah.size} total tercatat`);

  console.log(`\nRingkasan (mode ${mode}):`);
  console.log(`  halaman daftar  : ${halaman}`);
  console.log(`  dokumen baru    : ${ditemukan}${sudahAda ? `  (+${sudahAda} sudah ada)` : ""}`);
  if (KERING) console.log(`  dipratinjau     : ${pratinjau} (tanpa unduh)`);
  else console.log(`  PDF diunduh     : ${diunduh} -> ${FOLDER}/`);
  console.log(`  dilewati (sudah): ${dilewati}`);
  if (tanpaPdf) console.log(`  tanpa link PDF  : ${tanpaPdf}`);
  if (hilang) console.log(`  file hilang     : ${hilang} (tidak ada di server, dicatat supaya tidak dicoba lagi)`);
  if (gagal) console.log(`  gagal           : ${gagal} (akan dicoba lagi run berikutnya)`);
  if (berhenti) console.log(`\nDihentikan dengan aman. Jalankan perintah yang sama untuk melanjutkan; yang sudah ada tidak diunduh ulang.`);
  else if (!KERING && diunduh) console.log(`\nJangan lupa: bun run train  (agar dokumen baru bisa dijawab chatbot)`);
  else if (!KERING) console.log(`\nTidak ada unduhan baru; semuanya sudah ada di ${FOLDER}/.`);
}

function simpanState(): void {
  if (KERING) return; // pratinjau tidak mengubah apa pun
  const baru: Simpanan = {
    antrean: antrean.map(({ url, judul }) => ({ url, judul })),
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
