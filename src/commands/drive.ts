/**
 * GOOGLE DRIVE: tempat simpan dokumen, data mentah, dataset siap latih, dan model GPT.
 *
 *   bun run drive -- --login                 login (browser terbuka; di server tanpa browser: --tempel)
 *   bun run drive -- --status                status login + ringkasan per folder (lokal vs Drive)
 *   bun run drive -- --unggah  [--folder documents]          lokal -> Drive (file baru/berubah)
 *   bun run drive -- --unduh   [--folder data/siap/wiki-v1]  Drive -> lokal
 *   bun run drive -- --sinkron [--folder ...] [--hapus] [--menang lokal|drive]
 *   bun run drive -- --logout                cabut akses & hapus token
 *   Tambahkan --kering untuk melihat rencana tanpa mengubah apa pun.
 *
 * Folder yang boleh: documents, data/mentah, data/siap, gpt-models (atau subfoldernya). Tanpa --folder = semua.
 * Di Drive semuanya ada di folder "LLM dari Nol" (scope drive.file: aplikasi hanya melihat file buatannya).
 * Penghapusan TIDAK pernah otomatis: file yang hilang di satu sisi hanya dilaporkan, kecuali dengan --hapus.
 * File lokal yang dihapus oleh --hapus dipindah ke .rahasia/sampah/, file Drive ke tempat sampah Drive.
 */
import { existsSync, mkdirSync, renameSync, statSync } from "node:fs";
import { basename, dirname, extname, join } from "node:path";
import { buangBerkas, daftarRekursif, folderPath, tautanAkar, unduhBerkas, unggahBerkas } from "../cloud/drive.js";
import { bacaKlien, buatUrlLogin, logout, PETUNJUK_KLIEN, selesaikanLogin, statusLogin } from "../cloud/google-auth.js";
import { bacaCatatan, md5Berkas, pindaiLokal, rencanakan, simpanCatatan, type Aksi, type InfoDrive, type Mode } from "../cloud/sync.js";
import { Pelacak } from "../gpt/progress.js";
import { FOLDER_SINKRON, folderSinkronSah, LOKASI } from "../paths.js";
import { argumen } from "./args.js";

const { opsi } = argumen();
const ukuran = (b: number) => b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : b >= 1e3 ? `${(b / 1e3).toFixed(0)} KB` : `${b} B`;

function daftarFolder(): string[] {
  if (!opsi.folder) return [...FOLDER_SINKRON];
  const f = opsi.folder.split(",").map((x) => x.trim().replace(/\\/g, "/").replace(/\/+$/, "")).filter(Boolean);
  const salah = f.filter((x) => !folderSinkronSah(x));
  if (salah.length) { console.error(`Folder tidak boleh: ${salah.join(", ")}. Hanya ${FOLDER_SINKRON.join(", ")} (atau subfoldernya).`); process.exit(1); }
  return f;
}

// ---------------------------------------------------------------- login
async function masuk(): Promise<void> {
  if (!bacaKlien()) { console.error(PETUNJUK_KLIEN); process.exit(1); }
  let selesai!: (x: { state: string; code: string } | { galat: string }) => void;
  const hasil = new Promise<{ state: string; code: string } | { galat: string }>((r) => (selesai = r));
  const ambilDari = (u: URL) => {
    const code = u.searchParams.get("code"), state = u.searchParams.get("state"), err = u.searchParams.get("error");
    if (err) selesai({ galat: err });
    else if (code && state) selesai({ state, code });
    return !!(err || code);
  };
  // pendengar loopback sementara; di server tanpa browser (--tempel) pengguna menempel URL akhir dari browser
  const server = Bun.serve({
    hostname: "127.0.0.1", port: opsi.tempel === "true" ? 53682 : 0,
    fetch(req) {
      const ok = ambilDari(new URL(req.url));
      return new Response(ok ? "<meta charset=utf-8><h3>Login Google Drive selesai.</h3>Silakan kembali ke terminal / tutup tab ini." : "menunggu login…", { headers: { "content-type": "text/html; charset=utf-8" } });
    },
  });
  const { url } = buatUrlLogin(`http://127.0.0.1:${server.port}/`);
  console.log("Buka alamat ini di browser untuk login Google (akses HANYA ke file buatan aplikasi ini, folder \"LLM\"):\n");
  console.log(url + "\n");
  if (opsi.tempel === "true") console.log("Server tanpa browser: buka alamat di atas di komputer Anda. Setelah login, browser menuju alamat 127.0.0.1 yang gagal dibuka —\nSALIN seluruh alamat dari address bar itu, lalu TEMPEL di sini dan tekan Enter.");
  else {
    console.log("Menunggu login… (jika browser di komputer lain: salin alamat akhir dari address bar dan tempel di sini)");
    if (process.platform === "win32") Bun.spawn(["rundll32", "url.dll,FileProtocolHandler", url], { stdout: "ignore", stderr: "ignore" });
    else if (process.platform === "darwin") Bun.spawn(["open", url], { stdout: "ignore", stderr: "ignore" });
  }
  // baca tempelan dari terminal (tidak memblokir jika stdin kosong)
  void (async () => {
    try { for await (const baris of console) { try { if (ambilDari(new URL(baris.trim()))) break; } catch { console.log("Itu bukan alamat lengkap. Tempel seluruh alamat dari address bar (diawali http://127.0.0.1)."); } } } catch { /* tanpa stdin */ }
  })();
  const batas = setTimeout(() => selesai({ galat: "waktu habis (10 menit)" }), 10 * 60_000);
  const r = await hasil;
  clearTimeout(batas); server.stop(true);
  if ("galat" in r) { console.error(`Login gagal: ${r.galat}`); process.exit(1); }
  await selesaikanLogin(r.state, r.code);
  await folderPath("", true); // buat folder "LLM dari Nol"
  console.log(`\nLogin berhasil. Token disimpan di ${LOKASI.rahasia}/ (jangan dibagikan).`);
  console.log(`Folder Drive: ${await tautanAkar()}`);
  process.exit(0);
}

// ---------------------------------------------------------------- rencana per folder
async function rencana(folder: string, mode: Mode, catatan = bacaCatatan()) {
  const { berkas: lokal, sedangDitulis } = await pindaiLokal(folder, catatan);
  const d = await daftarRekursif(folder);
  const drive = new Map<string, InfoDrive>();
  for (const [p, f] of d.berkas) drive.set(p, { id: f.id, md5: f.md5Checksum!, ukuran: Number(f.size ?? 0), waktu: f.modifiedTime });
  const ditulis = new Set(sedangDitulis);
  for (const p of ditulis) drive.delete(p);
  const aksi = rencanakan(lokal, drive, catatan, mode, {
    hapus: opsi.hapus === "true", menang: opsi.menang === "lokal" || opsi.menang === "drive" ? opsi.menang : undefined,
    dalam: (p) => (p === folder || p.startsWith(folder + "/")) && !ditulis.has(p),
  });
  return { aksi, lokal, drive, sedangDitulis, kembar: d.kembar, lainnya: d.lainnya };
}

function ringkas(aksi: Aksi[]) {
  const n: Record<string, number> = {};
  for (const a of aksi) n[a.jenis] = (n[a.jenis] ?? 0) + 1;
  return n;
}

// ---------------------------------------------------------------- status
async function status(): Promise<void> {
  const s = statusLogin();
  console.log("Google Drive");
  console.log(`  OAuth client : ${s.klien ? "ada" : "BELUM — lihat petunjuk di bawah"}`);
  console.log(`  Login        : ${s.login ? `ya (sejak ${s.sejak?.slice(0, 16).replace("T", " ")})` : "belum  ->  bun run drive -- --login"}`);
  if (!s.klien) { console.log("\n" + PETUNJUK_KLIEN); return; }
  if (!s.login) return;
  console.log(`  Folder Drive : ${(await tautanAkar()) ?? "(belum dibuat)"}\n`);
  console.log("folder               lokal                 Drive                 belum sinkron");
  for (const f of daftarFolder()) {
    const r = await rencana(f, "sinkron");
    const n = ringkas(r.aksi);
    const ukL = [...r.lokal.values()].reduce((a, x) => a + x.ukuran, 0), ukD = [...r.drive.values()].reduce((a, x) => a + x.ukuran, 0);
    const beda = [n.unggah && `${n.unggah} unggah`, n.unduh && `${n.unduh} unduh`, n.konflik && `${n.konflik} KONFLIK`, n.lapor && `${n.lapor} perlu dicek`, r.sedangDitulis.length && `${r.sedangDitulis.length} sedang ditulis`].filter(Boolean).join(", ") || "sudah sama";
    console.log(`${f.padEnd(20)} ${`${r.lokal.size} file, ${ukuran(ukL)}`.padEnd(21)} ${`${r.drive.size} file, ${ukuran(ukD)}`.padEnd(21)} ${beda}`);
  }
  console.log("\nLihat rinciannya:  bun run drive -- --sinkron --kering");
}

// ---------------------------------------------------------------- jalankan
async function jalankan(mode: Mode): Promise<void> {
  const kering = opsi.kering === "true";
  const catatan = bacaCatatan();
  const semua: { folder: string; aksi: Aksi[]; drive: Map<string, InfoDrive>; lokal: Map<string, { md5: string; ukuran: number; mtime: number }> }[] = [];
  console.log(`${mode === "unggah" ? "Unggah lokal -> Drive" : mode === "unduh" ? "Unduh Drive -> lokal" : "Sinkron dua arah"}${kering ? " (KERING: hanya rencana)" : ""}`);
  for (const f of daftarFolder()) {
    process.stdout.write(`  memeriksa ${f} … `);
    const r = await rencana(f, mode, catatan);
    semua.push({ folder: f, aksi: r.aksi, drive: r.drive, lokal: r.lokal });
    const n = ringkas(r.aksi);
    console.log(`${r.lokal.size} lokal, ${r.drive.size} di Drive · ${Object.entries(n).filter(([k]) => k !== "sama").map(([k, v]) => `${v} ${k}`).join(", ") || "sudah sama"}`);
    for (const p of r.sedangDitulis.slice(0, 5)) console.log(`    dilewati (baru diubah, mungkin sedang ditulis): ${p}`);
    for (const p of r.kembar.slice(0, 5)) console.log(`    nama kembar di Drive (dipakai yang terbaru): ${p}`);
    for (const p of r.lainnya.slice(0, 5)) console.log(`    dilewati (dokumen Google asli, bukan file biasa): ${p}`);
  }
  const kerja = semua.flatMap((s) => s.aksi.map((a) => ({ ...a, s }))).filter((a) => a.jenis !== "sama" && a.jenis !== "lupakan");
  const LABEL: Record<string, string> = { unggah: "↑ unggah", unduh: "↓ unduh", konflik: "! KONFLIK", "hapus-drive": "✗ hapus di Drive", "hapus-lokal": "✗ hapus di lokal", lapor: "? cek" };
  if (kerja.length) console.log("");
  for (const a of kerja.slice(0, 200)) {
    const uk = a.s.lokal.get(a.path)?.ukuran ?? a.s.drive.get(a.path)?.ukuran ?? 0;
    console.log(`  ${(LABEL[a.jenis] ?? a.jenis).padEnd(17)} ${a.path}  (${ukuran(uk)}; ${a.alasan})`);
  }
  if (kerja.length > 200) console.log(`  … dan ${kerja.length - 200} lagi`);

  // catatan untuk file yang sudah sama & yang hilang di kedua sisi (tanpa jaringan)
  for (const s of semua) for (const a of s.aksi) {
    if (a.jenis === "lupakan") delete catatan[a.path];
    if (a.jenis === "sama") { const L = s.lokal.get(a.path)!, D = s.drive.get(a.path)!; catatan[a.path] = { md5: L.md5, id: D.id, ukuran: L.ukuran, mtime: L.mtime, waktu: D.waktu }; }
  }
  const dikerjakan = kerja.filter((a) => a.jenis !== "lapor");
  if (kering || !dikerjakan.length) {
    if (!kering) simpanCatatan(catatan);
    console.log(dikerjakan.length ? `\n${dikerjakan.length} aksi akan dijalankan. Jalankan tanpa --kering untuk melakukannya.` : "\nTidak ada yang perlu dipindahkan.");
    if (kerja.some((a) => a.jenis === "lapor")) console.log("Tanda \"? cek\" tidak diubah otomatis (lihat alasannya; hapus ikut dengan --hapus, atau pakai --unggah / --unduh).");
    return;
  }

  const total = dikerjakan.length;
  const pelacak = new Pelacak("drive", mode, { total, satuan: "berkas", satuanKecepatan: "MB/dtk" });
  const t0 = Date.now();
  let byte = 0, gagal = 0, konflik = 0, i = 0;
  const sampah = join(LOKASI.rahasia, "sampah", new Date().toISOString().replace(/[:.]/g, "-"));
  console.log("");
  for (const a of dikerjakan) {
    if (pelacak.berhenti) break;
    const L = a.s.lokal.get(a.path), D = a.s.drive.get(a.path);
    try {
      if (a.jenis === "unggah") {
        const induk = (await folderPath(dirname(a.path), true))!;
        const f = await unggahBerkas(a.path, basename(a.path), induk, D?.id);
        if (f.md5Checksum && f.md5Checksum !== L!.md5) throw new Error("md5 di Drive tidak sama dengan file lokal (file berubah saat diunggah?)");
        catatan[a.path] = { md5: L!.md5, id: f.id, ukuran: L!.ukuran, mtime: L!.mtime, waktu: f.modifiedTime };
        byte += L!.ukuran;
      } else if (a.jenis === "unduh") {
        await unduhBerkas(D!.id, a.path);
        const md5 = await md5Berkas(a.path);
        if (md5 !== D!.md5) throw new Error("md5 hasil unduhan tidak sama dengan Drive");
        catatan[a.path] = { md5, id: D!.id, ukuran: D!.ukuran, mtime: statSync(a.path).mtimeMs, waktu: D!.waktu };
        byte += D!.ukuran;
      } else if (a.jenis === "konflik") {
        const ext = extname(a.path), salinan = `${a.path.slice(0, a.path.length - ext.length)} (drive)${ext}`;
        if (!existsSync(salinan)) await unduhBerkas(D!.id, salinan);
        konflik++;
        console.log(`  KONFLIK ${a.path}: versi lokal tetap; versi Drive disimpan sebagai ${salinan}. Pilih salah satu, hapus yang lain, lalu sinkron lagi (atau --menang lokal|drive).`);
      } else if (a.jenis === "hapus-drive") {
        await buangBerkas(D!.id);
        delete catatan[a.path];
      } else if (a.jenis === "hapus-lokal") {
        const ke = join(sampah, a.path);
        mkdirSync(dirname(ke), { recursive: true });
        renameSync(a.path, ke);
        delete catatan[a.path];
      }
    } catch (e) {
      gagal++;
      console.log(`  GAGAL ${a.jenis} ${a.path}: ${(e as Error).message}`);
    }
    i++;
    const detik = (Date.now() - t0) / 1000;
    pelacak.langkah(i, { catatan: `${a.jenis} ${basename(a.path)}`, kecepatan: detik > 0 ? byte / 1e6 / detik : 0 });
    if (i % 10 === 0) simpanCatatan(catatan);
  }
  simpanCatatan(catatan);
  const berhenti = i < total;
  const teks = `${i}/${total} berkas, ${ukuran(byte)}${gagal ? `, ${gagal} gagal` : ""}${konflik ? `, ${konflik} konflik` : ""}`;
  pelacak.tutup(berhenti ? "berhenti" : gagal ? "gagal" : "selesai", teks);
  console.log(`\n${berhenti ? "Dihentikan" : "Selesai"}: ${teks} dalam ${Math.round((Date.now() - t0) / 1000)} detik.`);
  if (existsSync(sampah)) console.log(`File lokal yang dihapus dipindah ke ${sampah}/ (hapus sendiri jika yakin).`);
  if (mode !== "unggah" && dikerjakan.some((a) => a.jenis === "unduh" && a.path.startsWith(LOKASI.dokumenAturan + "/"))) console.log("Ada dokumen aturan baru: jalankan bun run train agar chatbot membacanya.");
  if (gagal) process.exit(1);
}

// ---------------------------------------------------------------- utama
try {
  if (opsi.login === "true") await masuk();
  else if (opsi.logout === "true") { console.log((await logout()) ? "Logout: akses dicabut dan token dihapus." : "Belum login."); }
  else if (opsi.unggah === "true") await jalankan("unggah");
  else if (opsi.unduh === "true") await jalankan("unduh");
  else if (opsi.sinkron === "true") await jalankan("sinkron");
  else await status();
} catch (e) {
  console.error((e as Error).message);
  process.exit(1);
}
