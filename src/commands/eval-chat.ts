/**
 * UJI PERCAKAPAN OTOMATIS: apakah balasan "nyambung"?
 *
 * Setiap baris evaluation/conversations.jsonl adalah satu percakapan:
 *   {"nama": "...", "memori": ["nama saya Irun"],          (opsional: isi memori awal)
 *    "giliran": [{"pesan": "siapa aku?", "jalur": "maksud", "harus_ada": "Irun", "tidak_ada": ["Budi"]}]}
 * Untuk setiap giliran diperiksa:
 *   - jalur jawaban sesuai (aturan / wiki / memori / obrolan / maksud / detail / tidak-ditemukan)
 *   - teks jawaban memuat "harus_ada" (teks, atau daftar teks yang salah satunya harus ada) dan tidak memuat kata di "tidak_ada" (huruf besar/kecil diabaikan)
 * Memori uji memakai file sementara, jadi memori Anda (data/memory.json) tidak tersentuh.
 *
 * Pakai:  bun run eval-chat
 *         bun run eval-chat -- --soal evaluation/conversations.jsonl
 *         bun run eval-chat -- --agent      (lewat mode agent, tanpa internet agar hasilnya tetap)
 */
import { readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { MesinChat, type GiliranRiwayat, type Respons, type Topik } from "../chat-engine.js";
import { jalankanModeAgent } from "../agent/multi-mode.js";
import { argumen } from "./args.js";

interface Giliran { pesan: string; jalur?: string; harus_ada?: string | string[]; tidak_ada?: string[] }
interface Percakapan { nama: string; memori?: string[]; giliran: Giliran[] }

const { opsi, folderModel } = argumen();
const daftar: Percakapan[] = readFileSync(opsi.soal ?? "evaluation/conversations.jsonl", "utf8")
  .split(/\r?\n/).filter((l) => l.trim()).map((l) => JSON.parse(l));
const fileMemori = join(tmpdir(), `llm-eval-memori-${process.pid}.json`);
const mesin = new MesinChat({ folderModel, fileMemori });

let lulus = 0, total = 0;
const gagal: string[] = [];
for (const p of daftar) {
  mesin.memori.hapusSemua();
  for (const m of p.memori ?? []) mesin.memori.tambah(m);
  let topik: Topik | null = null;
  const riwayat: GiliranRiwayat[] = [];
  for (const g of p.giliran) {
    const r: Respons = opsi.agent === "true"
      ? await jalankanModeAgent(mesin, g.pesan, { riwayat, topik, internet: false })
      : mesin.proses(g.pesan, { riwayat, topik });
    topik = r.topik;
    riwayat.push({ user: g.pesan, bot: r.teks, jalur: r.jalur });
    const teks = r.teks.toLowerCase();
    const masalah: string[] = [];
    if (g.jalur && r.jalur !== g.jalur) masalah.push(`jalur ${r.jalur}, seharusnya ${g.jalur}`);
    const harus = g.harus_ada === undefined ? [] : Array.isArray(g.harus_ada) ? g.harus_ada : [g.harus_ada];
    if (harus.length && !harus.some((h) => teks.includes(h.toLowerCase()))) masalah.push(`tidak memuat ${harus.map((h) => `"${h}"`).join(" atau ")}`);
    for (const x of g.tidak_ada ?? []) if (teks.includes(x.toLowerCase())) masalah.push(`memuat "${x}" yang seharusnya tidak ada`);
    total++;
    if (masalah.length) gagal.push(`  - [${p.nama}] "${g.pesan}": ${masalah.join("; ")}\n      jawaban: ${r.teks.split("\n")[0].slice(0, 140)}`);
    else lulus++;
  }
}
rmSync(fileMemori, { force: true });

console.log(`Percakapan: ${daftar.length} | giliran: ${total}${opsi.agent === "true" ? " | mode agent (tanpa internet)" : ""}`);
console.log(`Nyambung: ${((100 * lulus) / total).toFixed(1)}%  (${lulus}/${total})`);
if (gagal.length) console.log(`\nTidak nyambung:\n${gagal.join("\n")}`);
