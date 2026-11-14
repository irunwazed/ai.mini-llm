/**
 * GOOGLE DRIVE API v3 — hanya fetch (tanpa library).
 *
 * Semua file aplikasi ada di satu folder Drive "LLM dari Nol", dengan susunan subfolder sama seperti lokal.
 * Scope drive.file: aplikasi hanya melihat file buatannya sendiri.
 */
import { renameSync, rmSync, statSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { aksesToken } from "./google-auth.js";

const API = "https://www.googleapis.com/drive/v3";
const UNGGAH = "https://www.googleapis.com/upload/drive/v3";
export const MIME_FOLDER = "application/vnd.google-apps.folder";
export const NAMA_AKAR = "LLM";
const BIDANG = "id,name,mimeType,md5Checksum,size,modifiedTime,webViewLink";
const POTONGAN = 8 * 1024 * 1024; // kelipatan 256 KB (syarat unggahan resumable)

export interface BerkasDrive { id: string; name: string; mimeType: string; md5Checksum?: string; size?: string; modifiedTime: string; webViewLink?: string }

const tidur = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Permintaan ke Drive dengan token + coba ulang (429, 5xx, batas kecepatan). */
async function minta(url: string, init: RequestInit = {}, coba = 0): Promise<Response> {
  const token = await aksesToken();
  const r = await fetch(url, { ...init, headers: { ...(init.headers as Record<string, string>), authorization: `Bearer ${token}` } });
  if (r.ok) return r;
  const teks = await r.text();
  const batas = r.status === 429 || r.status >= 500 || (r.status === 403 && /rateLimit|userRateLimit/i.test(teks));
  if (batas && coba < 6) { await tidur(Math.min(60_000, 1000 * 2 ** coba + Math.random() * 1000)); return minta(url, init, coba + 1); }
  let pesan = teks.slice(0, 300);
  try { pesan = JSON.parse(teks).error?.message ?? pesan; } catch { /* bukan JSON */ }
  throw new Error(`Google Drive ${r.status}: ${pesan}`);
}

const kutip = (s: string) => s.replace(/\\/g, "\\\\").replace(/'/g, "\\'");

/** Isi satu folder (semua halaman). */
export async function isiFolder(idFolder: string): Promise<BerkasDrive[]> {
  const hasil: BerkasDrive[] = [];
  let halaman: string | undefined;
  do {
    const q = new URLSearchParams({ q: `'${kutip(idFolder)}' in parents and trashed=false`, fields: `nextPageToken,files(${BIDANG})`, pageSize: "1000", spaces: "drive" });
    if (halaman) q.set("pageToken", halaman);
    const j = (await (await minta(`${API}/files?${q}`)).json()) as { files: BerkasDrive[]; nextPageToken?: string };
    hasil.push(...j.files);
    halaman = j.nextPageToken;
  } while (halaman);
  return hasil;
}

/** Cari folder bernama `nama` di dalam `induk` ("root" = My Drive); buat jika belum ada (dan `buat`). */
export async function folder(nama: string, induk: string, buat = true): Promise<string | null> {
  const q = new URLSearchParams({ q: `name='${kutip(nama)}' and mimeType='${MIME_FOLDER}' and '${kutip(induk)}' in parents and trashed=false`, fields: "files(id,modifiedTime)", spaces: "drive" });
  const j = (await (await minta(`${API}/files?${q}`)).json()) as { files: { id: string; modifiedTime: string }[] };
  if (j.files.length) return j.files.sort((a, b) => a.modifiedTime.localeCompare(b.modifiedTime))[0].id; // folder kembar: pakai yang tertua
  if (!buat) return null;
  const r = await minta(`${API}/files?fields=id`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name: nama, mimeType: MIME_FOLDER, parents: [induk] }) });
  return ((await r.json()) as { id: string }).id;
}

const cacheFolder = new Map<string, string>();
/** id folder Drive untuk path "a/b/c" di bawah folder akar "LLM dari Nol". */
export async function folderPath(path: string, buat = true): Promise<string | null> {
  const bagian = path.split("/").filter(Boolean);
  let id = cacheFolder.get("") ?? (await folder(NAMA_AKAR, "root", buat));
  if (!id) return null;
  cacheFolder.set("", id);
  let kini = "";
  for (const b of bagian) {
    kini = kini ? `${kini}/${b}` : b;
    const c = cacheFolder.get(kini);
    if (c) { id = c; continue; }
    const baru = await folder(b, id, buat);
    if (!baru) return null;
    cacheFolder.set(kini, baru); id = baru;
  }
  return id;
}

/** Semua file (bukan folder) di bawah path Drive, rekursif. Kunci = path relatif dari akar ("documents/x.pdf"). */
export async function daftarRekursif(path: string): Promise<{ berkas: Map<string, BerkasDrive>; kembar: string[]; lainnya: string[] }> {
  const berkas = new Map<string, BerkasDrive>(), kembar: string[] = [], lainnya: string[] = [];
  const awal = await folderPath(path, false);
  if (!awal) return { berkas, kembar, lainnya };
  const antre: [string, string][] = [[awal, path]];
  while (antre.length) {
    const [id, p] = antre.shift()!;
    for (const f of await isiFolder(id)) {
      const anak = `${p}/${f.name}`;
      if (f.mimeType === MIME_FOLDER) { cacheFolder.set(anak, f.id); antre.push([f.id, anak]); continue; }
      if (!f.md5Checksum) { lainnya.push(anak); continue; } // Google Docs/Sheets asli: tidak bisa diunduh apa adanya
      const ada = berkas.get(anak);
      if (ada) { kembar.push(anak); if (ada.modifiedTime >= f.modifiedTime) continue; }
      berkas.set(anak, f);
    }
  }
  return { berkas, kembar, lainnya };
}

function multipart(meta: object, isi: Uint8Array, mime: string): { body: Uint8Array<ArrayBuffer>; tipe: string } {
  const batas = "batas" + Math.random().toString(36).slice(2);
  const enc = new TextEncoder();
  const kepala = enc.encode(`--${batas}\r\ncontent-type: application/json; charset=UTF-8\r\n\r\n${JSON.stringify(meta)}\r\n--${batas}\r\ncontent-type: ${mime}\r\n\r\n`);
  const ekor = enc.encode(`\r\n--${batas}--`);
  const body = new Uint8Array(kepala.length + isi.length + ekor.length);
  body.set(kepala, 0); body.set(isi, kepala.length); body.set(ekor, kepala.length + isi.length);
  return { body, tipe: `multipart/related; boundary=${batas}` };
}

/**
 * Unggah file lokal ke folder Drive `idInduk`. Jika `idLama` diberikan, isi file itu diganti (id tetap).
 * File < 5 MB satu permintaan; lebih besar memakai unggahan resumable per 8 MB.
 */
export async function unggahBerkas(lokal: string, nama: string, idInduk: string, idLama?: string, kemajuan?: (byte: number) => void): Promise<BerkasDrive> {
  const ukuran = statSync(lokal).size, mime = "application/octet-stream";
  const meta = idLama ? { name: nama } : { name: nama, parents: [idInduk] };
  const dasar = idLama ? `${UNGGAH}/files/${idLama}` : `${UNGGAH}/files`, metode = idLama ? "PATCH" : "POST";
  if (ukuran < 5 * 1024 * 1024) {
    const { body, tipe } = multipart(meta, new Uint8Array(await Bun.file(lokal).arrayBuffer()), mime);
    const r = await minta(`${dasar}?uploadType=multipart&fields=${BIDANG}`, { method: metode, headers: { "content-type": tipe }, body });
    kemajuan?.(ukuran);
    return (await r.json()) as BerkasDrive;
  }
  const awal = await minta(`${dasar}?uploadType=resumable&fields=${BIDANG}`, {
    method: metode, headers: { "content-type": "application/json; charset=UTF-8", "x-upload-content-type": mime, "x-upload-content-length": String(ukuran) }, body: JSON.stringify(meta),
  });
  const sesi = awal.headers.get("location");
  if (!sesi) throw new Error("Drive tidak memberi alamat unggahan resumable");
  let posisi = 0;
  for (let coba = 0; ;) {
    const akhir = Math.min(ukuran, posisi + POTONGAN);
    const potong = new Uint8Array(await Bun.file(lokal).slice(posisi, akhir).arrayBuffer());
    const r = await fetch(sesi, { method: "PUT", headers: { "content-range": `bytes ${posisi}-${akhir - 1}/${ukuran}` }, body: potong }).catch(() => null);
    if (r && (r.status === 200 || r.status === 201)) { kemajuan?.(ukuran); return (await r.json()) as BerkasDrive; }
    if (r && r.status === 308) {
      const rentang = r.headers.get("range"); // "bytes=0-1234"
      posisi = rentang ? Number(rentang.split("-")[1]) + 1 : 0;
      kemajuan?.(posisi); coba = 0; continue;
    }
    if (++coba > 6 || (r && r.status < 500 && r.status !== 429)) throw new Error(`Unggahan ${nama} gagal: ${r ? `${r.status} ${(await r.text()).slice(0, 200)}` : "koneksi terputus"}`);
    await tidur(1000 * 2 ** coba);
    // tanya server sudah sampai mana
    const cek = await fetch(sesi, { method: "PUT", headers: { "content-range": `bytes */${ukuran}` } }).catch(() => null);
    const rentang = cek?.headers.get("range");
    posisi = rentang ? Number(rentang.split("-")[1]) + 1 : posisi;
  }
}

/** Unduh isi file Drive ke `tujuan` (lewat file .part, lalu diganti nama — tidak meninggalkan file setengah jadi). */
export async function unduhBerkas(id: string, tujuan: string): Promise<void> {
  mkdirSync(dirname(tujuan), { recursive: true });
  const r = await minta(`${API}/files/${id}?alt=media`);
  const sementara = tujuan + ".part";
  try {
    await Bun.write(sementara, r);
    renameSync(sementara, tujuan);
  } catch (e) { rmSync(sementara, { force: true }); throw e; }
}

/** Pindahkan file Drive ke tempat sampah (masih bisa dipulihkan 30 hari). */
export async function buangBerkas(id: string): Promise<void> {
  await minta(`${API}/files/${id}?fields=id`, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ trashed: true }) });
}

/** Tautan folder akar untuk dibuka di browser. */
export async function tautanAkar(): Promise<string | null> {
  const id = await folderPath("", false);
  return id ? `https://drive.google.com/drive/folders/${id}` : null;
}
