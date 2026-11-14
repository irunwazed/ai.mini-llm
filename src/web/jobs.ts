/**
 * PROSES DARI WEB: halaman /latih menjalankan langkah-langkah (ambil data, siapkan, latih, evaluasi)
 * dengan perintah yang SAMA seperti di terminal.
 *
 * Keamanan:
 *   - hanya perintah di DAFTAR PUTIH di bawah, dengan opsi bertipe (angka berentang, nama, pilihan, bendera);
 *     tidak ada shell, argumen dikirim sebagai array ke Bun.spawn
 *   - server web hanya menerima permintaan dari komputer ini (dicek di web.ts)
 * Setiap proses: log lengkap di progres/log/<id>.log, daftar proses di progres/jobs.json.
 * Proses yang memakai Pelacak (train-gpt, train-reader, fetch-*, siapkan-data) juga punya progress bar;
 * "hentikan" untuk proses itu = berhenti DENGAN AMAN (simpan dulu), selain itu proses dihentikan biasa.
 */
import { appendFileSync, closeSync, existsSync, mkdirSync, openSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { bacaSemuaProgres, mintaBerhenti } from "../gpt/progress.js";
import { LOKASI, SUMBER_MENTAH } from "../paths.js";

type Validasi =
  | { tipe: "angka"; min: number; max: number }
  | { tipe: "nama" }
  | { tipe: "teks"; maks: number }
  | { tipe: "pilihan"; nilai: string[] }
  | { tipe: "bendera" }
  | { tipe: "sumber" };
export interface DefPerintah { judul: string; langkah: 0 | 1 | 2 | 3 | 4; opsi: Record<string, Validasi>; kunci?: (o: Record<string, string>) => string }

const angka = (min: number, max: number): Validasi => ({ tipe: "angka", min, max });
const nama: Validasi = { tipe: "nama" }, bendera: Validasi = { tipe: "bendera" };

/** DAFTAR PUTIH: nama script di package.json -> opsi yang boleh dikirim dari web. */
export const PERINTAH: Record<string, DefPerintah> = {
  "migrasi-data": { judul: "Pindahkan data ke lokasi baru", langkah: 0, opsi: { kering: bendera }, kunci: () => "migrasi" },
  "fetch-data": { judul: "Unduh Wikipedia", langkah: 1, opsi: { "target-mb": angka(1, 2000) }, kunci: () => "fetch-data" },
  "fetch-jdih": {
    judul: "Unduh dokumen JDIH BKN", langkah: 1,
    opsi: { batas: angka(0, 100000), judul: { tipe: "teks", maks: 100 }, tahun: angka(1945, 2100), kategori: { tipe: "teks", maks: 40 }, kering: bendera },
    kunci: () => "fetch-jdih",
  },
  "make-chat-data": { judul: "Buat data obrolan", langkah: 1, opsi: { konteks: angka(0, 100000) }, kunci: () => "obrolan" },
  "make-reader-data": { judul: "Buat data GPT pembaca", langkah: 1, opsi: { "dengan-aturan": bendera, maks: angka(100, 100000) }, kunci: () => "pembaca" },
  "siapkan-data": {
    judul: "Siapkan data (tokenisasi)", langkah: 2,
    opsi: { sumber: { tipe: "sumber" }, nama, vocab: angka(100, 20000), "tokenizer-dari": nama, "huruf-asli": bendera },
    kunci: (o) => `dataset:${o.nama}`,
  },
  "train": { judul: "Latih chatbot aturan", langkah: 3, opsi: { "tanpa-embedding": bendera }, kunci: () => "model-aturan" },
  "index-wiki": { judul: "Indeks Wikipedia lokal", langkah: 3, opsi: {}, kunci: () => "model-wiki" },
  "train-gpt": {
    judul: "Latih GPT", langkah: 3,
    opsi: {
      dataset: nama, name: nama, resume: bendera, from: nama, size: { tipe: "pilihan", nilai: ["small", "medium", "besar"] },
      minutes: angka(1, 1440), steps: angka(1, 1_000_000), "eval-every": angka(5, 100000), batch: angka(1, 256),
      lr: angka(0.00001, 0.1), dropout: angka(0, 0.9), patience: angka(1, 100), "keep-case": bendera,
    },
    kunci: (o) => `model:${o.name ?? o.dataset}`,
  },
  "train-reader": { judul: "Latih GPT pembaca", langkah: 3, opsi: { from: nama, name: nama, minutes: angka(1, 1440), steps: angka(1, 1_000_000) }, kunci: (o) => `model:${o.name ?? "reader"}` },
  "eval": { judul: "Evaluasi chatbot aturan", langkah: 4, opsi: { rag: bendera } },
  "eval-chat": { judul: "Evaluasi percakapan", langkah: 4, opsi: { agent: bendera, rag: bendera } },
  "eval-reasoning": { judul: "Evaluasi penalaran", langkah: 4, opsi: {} },
  "eval-reader": { judul: "Evaluasi GPT pembaca", langkah: 4, opsi: { name: nama, maks: angka(10, 10000) } },
  "eval-konteks": { judul: "Evaluasi GPT berkonteks", langkah: 4, opsi: { name: nama, banding: nama, jumlah: angka(10, 5000) } },
};

export interface Job {
  id: string; perintah: string; judul: string; argv: string[]; opsi: Record<string, string>;
  pid: number; mulai: string; selesai: string | null; kode: number | null;
  status: "berjalan" | "selesai" | "gagal" | "dihentikan"; log: string; kunci: string | null;
}

const FILE_JOB = join(LOKASI.progres, "jobs.json");
const proses = new Map<string, ReturnType<typeof Bun.spawn>>();

function hidup(pid: number): boolean { try { process.kill(pid, 0); return true; } catch { return false; } }

export function daftarJob(): Job[] {
  if (!existsSync(FILE_JOB)) return [];
  try {
    const d = JSON.parse(readFileSync(FILE_JOB, "utf8")) as Job[];
    // proses dari sesi server sebelumnya (proses tetap jalan walau server ditutup): status akhir diambil dari
    // file progresnya jika ada, kalau tidak dianggap gagal
    const yatim = d.filter((j) => (j.status === "berjalan" || (j.status === "dihentikan" && !j.selesai)) && !proses.has(j.id) && !hidup(j.pid));
    if (yatim.length) {
      const progres = bacaSemuaProgres();
      for (const j of yatim) {
        const pr = progres.find((p) => p.pid === j.pid);
        if (j.status === "berjalan") j.status = pr?.status === "selesai" ? "selesai" : pr?.status === "berhenti" ? "dihentikan" : "gagal";
        j.selesai ??= pr?.diperbarui ?? new Date().toISOString();
      }
    }
    return d;
  } catch { return []; }
}
function simpanJob(daftar: Job[]): void {
  mkdirSync(LOKASI.progres, { recursive: true });
  writeFileSync(FILE_JOB, JSON.stringify(daftar.slice(-100), null, 1));
}
function ubahJob(id: string, ubah: Partial<Job>): void {
  const d = daftarJob(); const j = d.find((x) => x.id === id);
  if (j) { Object.assign(j, ubah); simpanJob(d); }
}

/** Periksa opsi terhadap daftar putih; kembalikan argumen CLI atau pesan galat. */
export function validasi(perintah: string, opsi: Record<string, unknown>): { argv: string[]; bersih: Record<string, string> } | { galat: string } {
  const def = PERINTAH[perintah];
  if (!def) return { galat: `perintah "${perintah}" tidak diizinkan` };
  const argv: string[] = [], bersih: Record<string, string> = {};
  for (const [k, v] of Object.entries(opsi ?? {})) {
    if (v === undefined || v === null || v === "" || v === false) continue;
    const aturan = def.opsi[k];
    if (!aturan) return { galat: `opsi --${k} tidak diizinkan untuk ${perintah}` };
    const s = String(v).trim();
    if (aturan.tipe === "bendera") { if (v !== true && s !== "true") return { galat: `--${k} harus bendera` }; argv.push(`--${k}`); bersih[k] = "true"; continue; }
    if (aturan.tipe === "angka") { const n = Number(s); if (!isFinite(n) || n < aturan.min || n > aturan.max) return { galat: `--${k} harus angka ${aturan.min}–${aturan.max}` }; }
    if (aturan.tipe === "nama" && !/^[\w.-]{1,40}$/.test(s)) return { galat: `--${k} hanya boleh huruf, angka, titik, minus, garis bawah (maks 40)` };
    if (aturan.tipe === "teks" && (s.length > aturan.maks || /[\r\n\0]/.test(s))) return { galat: `--${k} terlalu panjang / tidak valid` };
    if (aturan.tipe === "pilihan" && !aturan.nilai.includes(s)) return { galat: `--${k} harus salah satu: ${aturan.nilai.join(", ")}` };
    if (aturan.tipe === "sumber" && !s.split(",").every((x) => x in SUMBER_MENTAH)) return { galat: `--${k} harus dari: ${Object.keys(SUMBER_MENTAH).join(", ")}` };
    argv.push(`--${k}`, s); bersih[k] = s;
  }
  return { argv, bersih };
}

/** Script package.json -> perintah bun langsung (tanpa "bun run", supaya pid = proses yang dipantau). */
function scriptDari(perintah: string): string[] {
  const pkg = JSON.parse(readFileSync("package.json", "utf8")) as { scripts: Record<string, string> };
  const s = pkg.scripts[perintah];
  if (!s || !s.startsWith("bun ")) throw new Error(`script ${perintah} tidak ditemukan`);
  return s.split(" ");
}

export function mulaiJob(perintah: string, opsi: Record<string, unknown>): Job | { galat: string } {
  const v = validasi(perintah, opsi);
  if ("galat" in v) return v;
  const def = PERINTAH[perintah];
  const kunci = def.kunci ? def.kunci(v.bersih) : null;
  // satu model/dataset/unduhan tidak boleh dikerjakan dua proses sekaligus
  if (kunci) {
    const bentrok = daftarJob().find((j) => j.status === "berjalan" && j.kunci === kunci);
    if (bentrok) return { galat: `"${kunci}" sedang dikerjakan proses lain (${bentrok.judul}, ${bentrok.id}). Tunggu atau hentikan dulu.` };
  }
  const id = `${new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14)}-${perintah}`;
  mkdirSync(LOKASI.log, { recursive: true });
  const log = join(LOKASI.log, `${id}.log`);
  const cmd = [...scriptDari(perintah), ...v.argv];
  writeFileSync(log, `$ ${["bun", "run", perintah, ...(v.argv.length ? ["--", ...v.argv] : [])].join(" ")}\n`);
  // keluaran langsung ke file log (bukan pipa ke server) + detached: proses TETAP berjalan walau server web
  // ditutup/di-restart; server berikutnya membaca log & progresnya seperti biasa
  const fd = openSync(log, "a");
  const p = Bun.spawn({ cmd, cwd: process.cwd(), stdout: fd, stderr: fd, stdin: "ignore", detached: true, env: { ...process.env, FORCE_COLOR: "0" } });
  closeSync(fd);
  proses.set(id, p);
  const job: Job = { id, perintah, judul: def.judul, argv: v.argv, opsi: v.bersih, pid: p.pid, mulai: new Date().toISOString(), selesai: null, kode: null, status: "berjalan", log, kunci };
  simpanJob([...daftarJob(), job]);
  void p.exited.then((kode) => {
    proses.delete(id);
    const status = daftarJob().find((x) => x.id === id)?.status === "dihentikan" ? "dihentikan" : kode === 0 ? "selesai" : "gagal";
    ubahJob(id, { selesai: new Date().toISOString(), kode, status });
    appendFileSync(log, `\n[proses selesai, kode keluar ${kode}]\n`);
  });
  return job;
}

/** Hentikan: proses dengan progres = berhenti dengan aman (simpan dulu); lainnya dihentikan langsung. */
export function hentikanJob(id: string): { ok: boolean; cara: string } {
  const j = daftarJob().find((x) => x.id === id);
  if (!j || j.status !== "berjalan") return { ok: false, cara: "proses tidak sedang berjalan" };
  const pr = bacaSemuaProgres().find((p) => p.pid === j.pid && p.status === "berjalan");
  if (pr && mintaBerhenti(pr.jenis, pr.nama)) { ubahJob(id, { status: "dihentikan" }); return { ok: true, cara: "berhenti dengan aman (menyimpan dulu)" }; }
  const p = proses.get(id);
  try { if (p) p.kill(); else process.kill(j.pid); } catch { /* sudah mati */ }
  ubahJob(id, { status: "dihentikan" });
  return { ok: true, cara: "dihentikan" };
}

/** Potongan log mulai dari byte `dari` (untuk tampilan log berjalan). */
export function bacaLog(id: string, dari: number): { teks: string; sampai: number } | null {
  const j = daftarJob().find((x) => x.id === id);
  if (!j || !existsSync(j.log)) return null;
  const ukuran = statSync(j.log).size;
  if (dari >= ukuran) return { teks: "", sampai: ukuran };
  const b = readFileSync(j.log);
  const mulai = Math.max(dari, ukuran - 200_000); // paling banyak 200 KB sekali kirim
  return { teks: b.subarray(mulai, ukuran).toString("utf8"), sampai: ukuran };
}
