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
 *         bun run eval-chat -- --rag        (lewat mode RAG: GPT pembaca + konteks memori)
 */
import { readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { MesinChat, type GiliranRiwayat, type Respons, type Topik } from "../chat-engine.js";
import { jalankanModeAgent } from "../agent/multi-mode.js";
import { argumen } from "./args.js";

interface Giliran { pesan: string; jalur?: string; harus_ada?: string | string[]; tidak_ada?: string[] }
/** rag: true = percakapan ini selalu diuji lewat mode RAG (dilewati jika GPT pembaca belum dilatih). */
/** agent: true = percakapan ini selalu diuji lewat mode agent (menguji loop perbaikan; tanpa internet kecuali --internet). */
interface Percakapan { nama: string; memori?: string[]; rag?: boolean; agent?: boolean; giliran: Giliran[] }

const { opsi, folderModel } = argumen();
const daftar: Percakapan[] = readFileSync(opsi.soal ?? "evaluation/conversations.jsonl", "utf8")
  .split(/\r?\n/).filter((l) => l.trim()).map((l) => JSON.parse(l));
const fileMemori = join(tmpdir(), `llm-eval-memori-${process.pid}.json`);
const mesin = new MesinChat({ folderModel, fileMemori });

let lulus = 0, total = 0;
const gagal: string[] = [];
let dilewati = 0;
for (const p of daftar) {
  if (p.rag && !mesin.pembaca) { dilewati++; continue; }
  mesin.memori.hapusSemua();
  for (const m of p.memori ?? []) mesin.memori.tambah(m);
  let topik: Topik | null = null;
  const riwayat: GiliranRiwayat[] = [];
  for (const g of p.giliran) {
    const r: Respons = opsi.agent === "true" || p.agent
      ? await jalankanModeAgent(mesin, g.pesan, { riwayat, topik, internet: opsi.internet === "true", rag: p.rag })
      : mesin.proses(g.pesan, { riwayat, topik, mode: opsi.rag === "true" || p.rag ? "rag" : undefined });
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

console.log(`Percakapan: ${daftar.length} | giliran: ${total}${opsi.agent === "true" ? " | mode agent (tanpa internet)" : ""}${opsi.rag === "true" ? " | mode RAG" : ""}`);
if (dilewati) console.log(`(${dilewati} percakapan RAG dilewati: belum ada GPT pembaca)`);
console.log(`Nyambung: ${((100 * lulus) / total).toFixed(1)}%  (${lulus}/${total})`);
if (gagal.length) console.log(`\nTidak nyambung:\n${gagal.join("\n")}`);
