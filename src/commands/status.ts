/**
 * STATUS PROSES PANJANG: training GPT, training GPT pembaca, unduhan Wikipedia.
 * Membaca file progres/*.json yang ditulis proses itu sendiri (src/gpt/progress.ts).
 *
 * Pakai:  bun run status                                  (sekali)
 *         bun run status -- --ikuti                       (diperbarui tiap 2 detik; Ctrl+C untuk keluar)
 *         bun run status -- --hentikan train-gpt:chat2    (minta berhenti DENGAN AMAN: simpan dulu, lalu keluar)
 *         bun run status -- --bersihkan                   (hapus catatan proses yang sudah selesai/berhenti)
 */
import { rmSync } from "node:fs";
import { join } from "node:path";
import { bacaSemuaProgres, FOLDER_PROGRES, formatDurasi, mintaBerhenti, type JenisProses, type Progres } from "../gpt/progress.js";
import { argumen } from "./args.js";

const { opsi } = argumen();

if (opsi.hentikan) {
  const [jenis, nama] = opsi.hentikan.split(":");
  const ok = mintaBerhenti(jenis as JenisProses, nama ?? "");
  console.log(ok
    ? `Permintaan berhenti dikirim ke ${jenis}:${nama}. Proses akan menyelesaikan langkahnya, menyimpan, lalu keluar.`
    : `Tidak ada proses ${opsi.hentikan}. Lihat daftar dengan: bun run status`);
  process.exit(ok ? 0 : 1);
}

if (opsi.bersihkan === "true") {
  let n = 0;
  for (const p of bacaSemuaProgres()) if (p.status !== "berjalan") {
    rmSync(join(FOLDER_PROGRES, `${p.jenis}-${p.nama.replace(/[^\w.-]/g, "_")}.json`), { force: true }); n++;
  }
  console.log(`${n} catatan proses lama dihapus.`);
  process.exit(0);
}

const lalu = (iso: string) => formatDurasi((Date.now() - Date.parse(iso)) / 1000) + " lalu";

function tampil(daftar: Progres[]): string {
  if (!daftar.length) return "Belum ada proses tercatat. Proses yang mencatat progres: train-gpt, train-reader, fetch-data, fetch-jdih, fetch-web, siapkan-data.";
  const baris: string[] = [];
  for (const p of daftar) {
    const lebar = 24, isi = p.persen !== null ? Math.round((lebar * p.persen) / 100) : 0;
    const bar = p.persen !== null ? `[${"█".repeat(isi)}${"░".repeat(lebar - isi)}] ${p.persen.toFixed(0)}%` : "";
    const ikon = { berjalan: "▶", berhenti: "■", selesai: "✓", gagal: "✗", "terhenti tiba-tiba": "!" }[p.status];
    baris.push(`${ikon} ${p.jenis}:${p.nama}  (${p.status}, pid ${p.pid}, diperbarui ${lalu(p.diperbarui)})`);
    baris.push(`   ${p.satuan} ${p.satuan === "MB" ? p.langkah.toFixed(1) : p.langkah}${p.total ? ` / ${p.total}` : ""}  ${bar}`
      + `${p.status === "berjalan" && p.etaDetik !== null ? `  sisa ±${formatDurasi(p.etaDetik)}` : ""}`);
    const loss = [p.lossTrain !== null ? `train ${p.lossTrain.toFixed(3)}` : "", p.lossVal !== null ? `val ${p.lossVal.toFixed(3)}` : "", p.lossTerbaik !== null ? `terbaik ${p.lossTerbaik.toFixed(3)}` : ""].filter(Boolean).join(" · ");
    if (loss) baris.push(`   loss ${loss}${p.kecepatan ? ` · ${Math.round(p.kecepatan)} ${p.satuanKecepatan}` : ""}`);
    if (p.riwayat.length > 1) baris.push(`   loss val: ${p.riwayat.slice(-8).map((r) => r.lossVal.toFixed(2)).join(" → ")}`);
    if (p.catatan) baris.push(`   ${p.catatan}`);
    if (p.status === "berjalan") baris.push(`   hentikan dengan aman: bun run status -- --hentikan ${p.jenis}:${p.nama}`);
    if (p.status === "terhenti tiba-tiba") baris.push(`   proses mati tanpa menyimpan status akhir (mis. jendela terminal ditutup). Checkpoint terbaik tetap ada; lanjutkan dengan --resume.`);
    baris.push("");
  }
  return baris.join("\n");
}

if (opsi.ikuti === "true") {
  const segarkan = () => { process.stdout.write("\x1b[2J\x1b[H"); console.log(`Status proses — ${new Date().toLocaleTimeString("id-ID")} (Ctrl+C untuk keluar)\n`); console.log(tampil(bacaSemuaProgres())); };
  segarkan();
  setInterval(segarkan, 2000);
} else console.log(tampil(bacaSemuaProgres()));
