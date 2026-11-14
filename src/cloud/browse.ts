/**
 * PENJELAJAH DRIVE untuk halaman /latih: isi satu folder di Drive ("LLM dari Nol/<path>") dibandingkan
 * dengan folder lokal yang sama. Status tiap file: sama / beda / hanya di Drive / hanya di lokal.
 */
import { existsSync, readdirSync, statSync } from "node:fs";
import { FOLDER_SINKRON, folderSinkronSah, LEWATI_SINKRON } from "../paths.js";
import { folderPath, isiFolder, MIME_FOLDER, NAMA_AKAR } from "./drive.js";
import { bacaCatatan, md5Berkas } from "./sync.js";

export type StatusItem = "folder" | "sama" | "beda" | "hanya di Drive" | "hanya di lokal" | "belum diunggah" | "dokumen Google";
export interface ItemDrive { nama: string; path: string; folder: boolean; ukuran: number | null; diubah: string | null; status: StatusItem; tautan: string | null }

/** Path yang boleh dijelajah: akar (""), folder perantara ("data"), atau folder sinkron & isinya. */
export function pathJelajahSah(p: string): boolean {
  return p === "" || FOLDER_SINKRON.some((f) => f.startsWith(p + "/")) || folderSinkronSah(p);
}

/** Folder untuk aksi unggah/unduh/sinkron dari posisi ini ("" = semua). */
export function folderAksi(p: string): string {
  if (p === "") return "";
  if (folderSinkronSah(p)) return p;
  return FOLDER_SINKRON.filter((f) => f.startsWith(p + "/")).join(",");
}

const lewati = (p: string, folder: boolean) => LEWATI_SINKRON.some((r) => r.test(p + (folder ? "/" : "")));

export async function jelajahDrive(path: string): Promise<{ path: string; akar: string; tautan: string | null; folderAksi: string; item: ItemDrive[] }> {
  if (!pathJelajahSah(path)) throw new Error("Folder tidak boleh dijelajah.");
  const id = await folderPath(path, false);
  const isi = id ? await isiFolder(id) : [];
  const catatan = bacaCatatan();
  const gabung = (n: string) => (path ? `${path}/${n}` : n);
  const dalamSinkron = (p: string) => folderSinkronSah(p) || FOLDER_SINKRON.some((f) => f.startsWith(p + "/"));
  const item: ItemDrive[] = [];
  const dariDrive = new Set<string>();

  for (const f of isi) {
    const p = gabung(f.name);
    if (!dalamSinkron(p) || lewati(p, f.mimeType === MIME_FOLDER)) continue; // di luar folder sinkron: tidak ditampilkan
    dariDrive.add(f.name);
    if (f.mimeType === MIME_FOLDER) {
      item.push({ nama: f.name, path: p, folder: true, ukuran: null, diubah: f.modifiedTime, status: "folder", tautan: f.webViewLink ?? `https://drive.google.com/drive/folders/${f.id}` });
      continue;
    }
    let status: StatusItem;
    const ukuranDrive = Number(f.size ?? 0);
    if (!f.md5Checksum) status = "dokumen Google";
    else if (!existsSync(p) || !statSync(p).isFile()) status = "hanya di Drive";
    else {
      const st = statSync(p), c = catatan[p];
      if (st.size !== ukuranDrive) status = "beda";
      else if (c && c.md5 === f.md5Checksum && Math.abs(c.mtime - st.mtimeMs) < 1) status = "sama";
      else status = (await md5Berkas(p)) === f.md5Checksum ? "sama" : "beda";
    }
    item.push({ nama: f.name, path: p, folder: false, ukuran: ukuranDrive, diubah: f.modifiedTime, status, tautan: f.webViewLink ?? null });
  }

  // yang hanya ada di lokal (belum diunggah)
  const lokal = path === "" ? "." : path;
  if (existsSync(lokal) && statSync(lokal).isDirectory()) {
    for (const e of readdirSync(lokal, { withFileTypes: true })) {
      const p = gabung(e.name);
      if (dariDrive.has(e.name) || !dalamSinkron(p) || lewati(p, e.isDirectory())) continue;
      if (e.isDirectory()) item.push({ nama: e.name, path: p, folder: true, ukuran: null, diubah: null, status: "belum diunggah", tautan: null });
      else if (e.isFile()) { const st = statSync(p); item.push({ nama: e.name, path: p, folder: false, ukuran: st.size, diubah: st.mtime.toISOString(), status: "hanya di lokal", tautan: null }); }
    }
  }
  item.sort((a, b) => Number(b.folder) - Number(a.folder) || a.nama.localeCompare(b.nama));
  return { path, akar: NAMA_AKAR, tautan: id ? `https://drive.google.com/drive/folders/${id}` : null, folderAksi: folderAksi(path), item };
}
