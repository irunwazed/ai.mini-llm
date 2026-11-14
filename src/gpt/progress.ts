/**
 * PROGRES & KONTROL untuk proses panjang (train-gpt, train-reader, fetch-data, fetch-jdih).
 *
 *   - baris progres hidup di terminal:  langkah 412/800 [██████████░░░░░░░░░░] 51%  loss 0,71 …  sisa ±13m
 *   - file progres/<jenis>-<nama>.json  -> dibaca `bun run status` (terminal lain) dan panel web
 *   - BERHENTI DENGAN AMAN: Ctrl+C sekali, file progres/<jenis>-<nama>.stop (dari `bun run status --hentikan`
 *     atau tombol di web). Proses menyelesaikan langkah yang sedang berjalan, mengevaluasi, menyimpan
 *     checkpoint, lalu keluar. Ctrl+C kedua = keluar paksa.
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export const FOLDER_PROGRES = "progres";
export type JenisProses = "train-gpt" | "train-reader" | "fetch-data" | "fetch-jdih" | "siapkan-data" | "train" | "index-wiki" | "make-chat-data" | "make-reader-data" | "eval";
export type StatusProses = "berjalan" | "berhenti" | "selesai" | "gagal" | "terhenti tiba-tiba";

export interface Progres {
  jenis: JenisProses; nama: string; pid: number; perintah: string;
  mulai: string; diperbarui: string; status: StatusProses;
  langkah: number; total: number | null; persen: number | null; satuan: string;   // "langkah" atau "MB"
  lossTrain: number | null; lossVal: number | null; lossTerbaik: number | null;
  kecepatan: number | null; satuanKecepatan: string; etaDetik: number | null;
  catatan: string; riwayat: { langkah: number; lossTrain: number; lossVal: number }[];
}

const berkas = (jenis: JenisProses, nama: string, ext: string) => join(FOLDER_PROGRES, `${jenis}-${nama.replace(/[^\w.-]/g, "_")}.${ext}`);

function hidup(pid: number): boolean {
  try { process.kill(pid, 0); return true; } catch (e) { return (e as NodeJS.ErrnoException).code === "EPERM"; }
}

/** Semua proses yang pernah/sedang berjalan, terbaru dulu. Proses "berjalan" yang pid-nya mati ditandai. */
export function bacaSemuaProgres(): Progres[] {
  if (!existsSync(FOLDER_PROGRES)) return [];
  return readdirSync(FOLDER_PROGRES).filter((f) => f.endsWith(".json")).flatMap((f) => {
    try {
      const p = JSON.parse(readFileSync(join(FOLDER_PROGRES, f), "utf8")) as Progres;
      // folder progres juga berisi file lain (mis. jobs.json milik panel web) — lewati yang bukan progres
      if (!p || typeof p !== "object" || typeof p.jenis !== "string" || typeof p.diperbarui !== "string") return [];
      if (p.status === "berjalan" && !hidup(p.pid)) p.status = "terhenti tiba-tiba";
      return [p];
    } catch { return []; }
  }).sort((a, b) => b.diperbarui.localeCompare(a.diperbarui));
}

/** Minta proses berhenti dengan aman (dibaca proses itu di langkah berikutnya). */
export function mintaBerhenti(jenis: JenisProses, nama: string): boolean {
  if (!existsSync(berkas(jenis, nama, "json"))) return false;
  writeFileSync(berkas(jenis, nama, "stop"), new Date().toISOString());
  return true;
}

export const formatDurasi = (d: number) => {
  if (!isFinite(d) || d < 0) return "-";
  const j = Math.floor(d / 3600), m = Math.floor((d % 3600) / 60), s = Math.floor(d % 60);
  return j ? `${j}j ${m}m` : m ? `${m}m ${s}d` : `${s}d`;
};

/**
 * Pelacak satu proses: menulis file progres, menggambar baris progres, dan menangani permintaan berhenti.
 * Pemakaian di loop:  if (pelacak.berhenti) break;  pelacak.langkah(n, { lossTrain, kecepatan })
 */
export class Pelacak {
  readonly p: Progres;
  private terakhirTulis = 0;
  private terakhirCetak = -1;
  private mintaStop = false;
  private readonly tty = !!process.stdout.isTTY;
  private readonly t0 = Date.now();
  private nAwal: number | null = null; // langkah pertama yang diamati (training lanjutan mulai dari langkah > 0)

  constructor(jenis: JenisProses, nama: string, opsi: { total?: number | null; satuan?: string; satuanKecepatan?: string } = {}) {
    mkdirSync(FOLDER_PROGRES, { recursive: true });
    rmSync(berkas(jenis, nama, "stop"), { force: true });
    const sekarang = new Date().toISOString();
    this.p = {
      jenis, nama, pid: process.pid, perintah: `bun run ${jenis} -- ${process.argv.slice(2).join(" ")}`,
      mulai: sekarang, diperbarui: sekarang, status: "berjalan",
      langkah: 0, total: opsi.total ?? null, persen: null, satuan: opsi.satuan ?? "langkah",
      lossTrain: null, lossVal: null, lossTerbaik: null, kecepatan: null, satuanKecepatan: opsi.satuanKecepatan ?? "token/dtk",
      etaDetik: null, catatan: "", riwayat: [],
    };
    // Ctrl+C pertama: berhenti dengan aman; kedua: keluar paksa
    process.on("SIGINT", () => {
      if (this.mintaStop) { console.log("\nKeluar paksa."); this.tutup("berhenti"); process.exit(130); }
      this.mintaStop = true;
      console.log("\n[Ctrl+C] Berhenti dengan aman setelah langkah ini (menyimpan dulu). Tekan Ctrl+C sekali lagi untuk keluar paksa.");
    });
    this.tulis(true);
  }

  /** Perlu berhenti? (Ctrl+C, atau file .stop dari `bun run status --hentikan` / tombol web) */
  get berhenti(): boolean {
    if (!this.mintaStop && existsSync(berkas(this.p.jenis, this.p.nama, "stop"))) {
      this.mintaStop = true;
      console.log("\n[status/web] Diminta berhenti: menyelesaikan langkah ini lalu menyimpan.");
    }
    return this.mintaStop;
  }

  setTotal(total: number): void { this.p.total = total; }

  /** Catat kemajuan; menggambar baris progres dan menulis file (paling sering tiap 2 detik). */
  langkah(n: number, data: { lossTrain?: number; kecepatan?: number; catatan?: string } = {}): void {
    const p = this.p;
    p.langkah = n;
    if (data.lossTrain !== undefined) p.lossTrain = data.lossTrain;
    if (data.kecepatan !== undefined) p.kecepatan = data.kecepatan;
    if (data.catatan !== undefined) p.catatan = data.catatan;
    const detik = (Date.now() - this.t0) / 1000;
    p.persen = p.total ? Math.min(100, (100 * n) / p.total) : null;
    if (this.nAwal === null) this.nAwal = Math.max(0, n - 1);
    const dikerjakan = n - this.nAwal;
    p.etaDetik = p.total && dikerjakan > 0 ? (detik / dikerjakan) * (p.total - n) : null;
    this.cetak(detik);
    this.tulis();
  }

  /** Catat hasil evaluasi (loss val) ke riwayat. */
  evaluasi(langkah: number, lossTrain: number, lossVal: number, terbaik: number): void {
    this.p.lossVal = lossVal; this.p.lossTerbaik = terbaik;
    this.p.riwayat.push({ langkah, lossTrain, lossVal });
    if (this.p.riwayat.length > 200) this.p.riwayat.shift();
    if (this.tty) process.stdout.write("\r" + " ".repeat(110) + "\r"); // bersihkan baris progres sebelum tabel dicetak
    this.tulis(true);
  }

  tutup(status: StatusProses, catatan?: string): void {
    if (catatan) this.p.catatan = catatan;
    this.p.status = status;
    if (this.tty) process.stdout.write("\n");
    this.tulis(true);
    rmSync(berkas(this.p.jenis, this.p.nama, "stop"), { force: true });
  }

  private cetak(detik: number): void {
    const p = this.p;
    const lebar = 20, isi = p.persen !== null ? Math.round((lebar * p.persen) / 100) : 0;
    const bar = p.total ? `[${"█".repeat(isi)}${"░".repeat(lebar - isi)}] ${p.persen!.toFixed(0)}%` : "";
    const baris = `${p.satuan} ${p.satuan === "MB" ? p.langkah.toFixed(1) : p.langkah}${p.total ? `/${p.total}` : ""} ${bar}`
      + `${p.lossTrain !== null ? `  loss ${p.lossTrain.toFixed(3)}` : ""}${p.lossVal !== null ? ` (val ${p.lossVal.toFixed(3)})` : ""}`
      + `${p.kecepatan ? `  ${Math.round(p.kecepatan)} ${p.satuanKecepatan}` : ""}  berjalan ${formatDurasi(detik)}`
      + `${p.etaDetik !== null ? `  sisa ±${formatDurasi(p.etaDetik)}` : ""}${p.catatan ? `  ${p.catatan}` : ""}`;
    if (this.tty) process.stdout.write("\r" + baris.slice(0, 160).padEnd(110));
    else if (p.persen !== null && Math.floor(p.persen / 5) !== this.terakhirCetak) { this.terakhirCetak = Math.floor(p.persen / 5); console.log(baris); }
  }

  private tulis(paksa = false): void {
    if (!paksa && Date.now() - this.terakhirTulis < 2000) return;
    this.terakhirTulis = Date.now();
    this.p.diperbarui = new Date().toISOString();
    try { writeFileSync(berkas(this.p.jenis, this.p.nama, "json"), JSON.stringify(this.p)); } catch { /* tidak fatal */ }
  }
}
