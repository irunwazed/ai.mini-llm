/**
 * Memindai folder dokumen secara rekursif dan membaca setiap file sesuai formatnya.
 * Hasil ekstraksi disimpan di model/cache/, sehingga training ulang hanya membaca
 * file yang baru atau berubah (dicek dari ukuran + waktu modifikasi).
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { extname, join, relative } from "node:path";
import type { Dokumen } from "../types.js";
import { bacaDocx } from "./docx.js";
import { bacaPdf } from "./pdf.js";
import { bacaTeks } from "./text.js";

const JENIS: Record<string, Dokumen["jenis"]> = { ".pdf": "pdf", ".docx": "docx", ".txt": "txt", ".md": "md" };

export function daftarFile(folder: string): string[] {
  const hasil: string[] = [];
  for (const nama of readdirSync(folder, { withFileTypes: true })) {
    const p = join(folder, nama.name);
    if (nama.isDirectory()) hasil.push(...daftarFile(p));
    else if (JENIS[extname(nama.name).toLowerCase()] && !nama.name.startsWith("~$")) hasil.push(p);
  }
  return hasil.sort();
}

export async function bacaFile(path: string, folderDokumen: string, folderCache: string): Promise<{ dok: Dokumen; dariCache: boolean }> {
  const jenis = JENIS[extname(path).toLowerCase()];
  const file = relative(folderDokumen, path).split("\\").join("/");
  const st = statSync(path);
  const kunci = `${st.size}-${st.mtimeMs}`;

  mkdirSync(folderCache, { recursive: true });
  const cachePath = join(folderCache, createHash("sha1").update(file).digest("hex") + ".json");
  if (existsSync(cachePath)) {
    const c = JSON.parse(readFileSync(cachePath, "utf8"));
    if (c.kunci === kunci) return { dok: c.dok, dariCache: true };
  }

  const buf = readFileSync(path);
  const halaman =
    jenis === "pdf" ? await bacaPdf(buf) : jenis === "docx" ? await bacaDocx(buf) : bacaTeks(buf);
  const dok: Dokumen = { file, jenis, adaHalaman: jenis === "pdf", halaman };
  writeFileSync(cachePath, JSON.stringify({ kunci, dok }));
  return { dok, dariCache: false };
}

/** PDF hasil scan tidak punya lapisan teks: rata-rata < 30 huruf per halaman. */
export function terlihatHasilScan(dok: Dokumen): boolean {
  const huruf = dok.halaman.reduce((s, h) => s + h.replace(/\s/g, "").length, 0);
  return huruf / Math.max(1, dok.halaman.length) < 30;
}
