/**
 * SINKRON LOKAL <-> GOOGLE DRIVE: perbandingan 3 arah (lokal, Drive, catatan sinkron terakhir).
 *
 * Catatan sinkron (.rahasia/drive-sinkron.json) menyimpan md5 setiap file saat terakhir sama di kedua sisi.
 * Dengan catatan itu bisa dibedakan "berubah di lokal", "berubah di Drive", "baru", dan "dihapus".
 *
 *   lokal berubah, Drive sama      -> unggah
 *   lokal sama, Drive berubah      -> unduh
 *   keduanya berubah               -> KONFLIK (tidak ada yang ditimpa; versi Drive disimpan "nama (drive).ext")
 *   dihapus di salah satu sisi     -> hanya dilaporkan; ikut dihapus di sisi lain HANYA dengan --hapus
 *                                     (dan hanya jika sisi lain tidak berubah sejak sinkron terakhir)
 */
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { LEWATI_SINKRON, LOKASI } from "../paths.js";

export type Mode = "unggah" | "unduh" | "sinkron";
export type JenisAksi = "unggah" | "unduh" | "konflik" | "hapus-drive" | "hapus-lokal" | "lapor" | "sama" | "lupakan";

export interface InfoLokal { md5: string; ukuran: number; mtime: number }
export interface InfoDrive { id: string; md5: string; ukuran: number; waktu: string }
export interface Catatan { md5: string; id: string; ukuran: number; mtime: number; waktu: string }
export interface Aksi { jenis: JenisAksi; path: string; alasan: string }

export const FILE_CATATAN = join(LOKASI.rahasia, "drive-sinkron.json");

export function bacaCatatan(): Record<string, Catatan> {
  if (!existsSync(FILE_CATATAN)) return {};
  try { return (JSON.parse(readFileSync(FILE_CATATAN, "utf8")) as { berkas: Record<string, Catatan> }).berkas ?? {}; } catch { return {}; }
}
export function simpanCatatan(c: Record<string, Catatan>): void {
  mkdirSync(LOKASI.rahasia, { recursive: true });
  writeFileSync(FILE_CATATAN, JSON.stringify({ diperbarui: new Date().toISOString(), berkas: c }));
}

/**
 * Rencana aksi untuk setiap path (gabungan lokal, Drive, catatan) di bawah satu folder.
 * Fungsi murni (tanpa I/O) supaya bisa diuji.
 */
export function rencanakan(
  lokal: Map<string, InfoLokal>, drive: Map<string, InfoDrive>, catatan: Record<string, Catatan>,
  mode: Mode, opsi: { hapus?: boolean; menang?: "lokal" | "drive"; dalam?: (p: string) => boolean } = {},
): Aksi[] {
  const dalam = opsi.dalam ?? (() => true);
  const semua = new Set([...lokal.keys(), ...drive.keys(), ...Object.keys(catatan).filter(dalam)]);
  const naik = mode !== "unduh", turun = mode !== "unggah";
  const aksi: Aksi[] = [];
  for (const path of [...semua].sort()) {
    const L = lokal.get(path), D = drive.get(path), C = catatan[path];
    const tambah = (jenis: JenisAksi, alasan: string) => aksi.push({ jenis, path, alasan });
    if (L && D) {
      if (L.md5 === D.md5) { tambah("sama", "isi sama"); continue; }
      const lokalBerubah = !C || C.md5 !== L.md5, driveBerubah = !C || C.md5 !== D.md5;
      if (lokalBerubah && !driveBerubah) { naik ? tambah("unggah", "berubah di lokal") : tambah("lapor", "berubah di lokal (pakai --unggah / --sinkron)"); continue; }
      if (driveBerubah && !lokalBerubah) { turun ? tambah("unduh", "berubah di Drive") : tambah("lapor", "berubah di Drive (pakai --unduh / --sinkron)"); continue; }
      // keduanya berubah, atau belum pernah disinkron dan isinya beda
      if (opsi.menang === "lokal" && naik) { tambah("unggah", "keduanya berubah, --menang lokal"); continue; }
      if (opsi.menang === "drive" && turun) { tambah("unduh", "keduanya berubah, --menang drive"); continue; }
      if (!C && mode === "unggah") { tambah("unggah", "beda dengan Drive, mode unggah: lokal dipakai"); continue; }
      if (!C && mode === "unduh") { tambah("unduh", "beda dengan lokal, mode unduh: Drive dipakai"); continue; }
      tambah("konflik", C ? "berubah di lokal DAN di Drive" : "isi beda dan belum pernah disinkron");
      continue;
    }
    if (L && !D) {
      if (!C) { naik ? tambah("unggah", "baru di lokal") : tambah("lapor", "hanya ada di lokal"); continue; }
      // pernah sinkron, sekarang hilang dari Drive
      if (L.md5 !== C.md5) { naik ? tambah("unggah", "dihapus di Drive tetapi berubah di lokal: diunggah lagi") : tambah("lapor", "dihapus di Drive, berubah di lokal"); continue; }
      if (opsi.hapus && turun) tambah("hapus-lokal", "dihapus di Drive (--hapus)");
      else if (mode === "unggah") tambah("unggah", "dihapus di Drive: diunggah lagi (mode unggah)");
      else tambah("lapor", "dihapus di Drive; ikut hapus di lokal dengan --hapus");
      continue;
    }
    if (!L && D) {
      if (!C) { turun ? tambah("unduh", "baru di Drive") : tambah("lapor", "hanya ada di Drive"); continue; }
      if (D.md5 !== C.md5) { turun ? tambah("unduh", "dihapus di lokal tetapi berubah di Drive: diunduh lagi") : tambah("lapor", "dihapus di lokal, berubah di Drive"); continue; }
      if (opsi.hapus && naik) tambah("hapus-drive", "dihapus di lokal (--hapus; ke tempat sampah Drive)");
      else if (mode === "unduh") tambah("unduh", "dihapus di lokal: diunduh lagi (mode unduh)");
      else tambah("lapor", "dihapus di lokal; ikut hapus di Drive dengan --hapus");
      continue;
    }
    if (C) tambah("lupakan", "tidak ada di kedua sisi");
  }
  return aksi;
}

/** md5 file lokal (stream; aman untuk file besar). */
export async function md5Berkas(path: string): Promise<string> {
  const h = new Bun.CryptoHasher("md5");
  const stream = Bun.file(path).stream();
  for await (const potong of stream) h.update(potong);
  return h.digest("hex");
}

/**
 * Pindai file lokal di bawah `folder`. md5 dipakai ulang dari catatan jika ukuran & waktu ubah sama.
 * File yang baru diubah < `segarDetik` lalu (mungkin sedang ditulis) dimasukkan ke `sedangDitulis`, tidak dipindai.
 */
export async function pindaiLokal(folder: string, catatan: Record<string, Catatan>, segarDetik = 120): Promise<{ berkas: Map<string, InfoLokal>; sedangDitulis: string[] }> {
  const berkas = new Map<string, InfoLokal>(), sedangDitulis: string[] = [];
  if (!existsSync(folder)) return { berkas, sedangDitulis };
  const jalan = async (dir: string) => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const p = `${dir}/${e.name}`;
      if (LEWATI_SINKRON.some((r) => r.test(p + (e.isDirectory() ? "/" : "")))) continue;
      if (e.isDirectory()) { await jalan(p); continue; }
      if (!e.isFile()) continue;
      const st = statSync(p);
      if (Date.now() - st.mtimeMs < segarDetik * 1000) { sedangDitulis.push(p); continue; }
      const c = catatan[p];
      const md5 = c && c.ukuran === st.size && Math.abs(c.mtime - st.mtimeMs) < 1 ? c.md5 : await md5Berkas(p);
      berkas.set(p, { md5, ukuran: st.size, mtime: st.mtimeMs });
    }
  };
  if (statSync(folder).isDirectory()) await jalan(folder.replace(/\\/g, "/").replace(/\/+$/, ""));
  return { berkas, sedangDitulis };
}
