/**
 * UJI AGENT PENALARAN: soal logika & matematika di evaluation/reasoning.jsonl.
 *   {"kategori": "...", "soal": "...", "harus_ada": ["jawaban yang benar", "atau bentuk lain"]}
 * Soal dijalankan lewat mesin percakapan lengkap, jadi pemilihan jalurnya ikut diuji.
 * Kategori "variasi angka" berisi soal yang angka/namanya diubah, untuk mengukur apakah
 * alat benar-benar menghitung (bukan menghafal jawaban soal aslinya).
 *
 * Pakai:  bun run eval-reasoning
 */
import { readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { MesinChat } from "../chat-engine.js";
import { argumen } from "./args.js";

interface Soal { kategori: string; soal: string; harus_ada: string[] }

const { opsi, folderModel } = argumen();
const daftar: Soal[] = readFileSync(opsi.soal ?? "evaluation/reasoning.jsonl", "utf8").split(/\r?\n/).filter((l) => l.trim()).map((l) => JSON.parse(l));
const fileMemori = join(tmpdir(), `llm-eval-penalaran-${process.pid}.json`);
const mesin = new MesinChat({ folderModel, fileMemori });

const perKategori = new Map<string, { benar: number; total: number }>();
const gagal: string[] = [];
for (const s of daftar) {
  const r = mesin.proses(s.soal);
  const teks = r.teks.toLowerCase();
  const benar = r.jalur === "penalaran" && s.harus_ada.some((h) => teks.includes(h.toLowerCase()));
  const k = perKategori.get(s.kategori) ?? { benar: 0, total: 0 };
  k.total++; if (benar) k.benar++;
  perKategori.set(s.kategori, k);
  if (!benar) gagal.push(`  - [${s.kategori}] ${s.soal.slice(0, 90)}\n      jalur ${r.jalur}; jawaban: ${r.teks.split("\n")[0].slice(0, 120)}`);
}
rmSync(fileMemori, { force: true });

let b = 0, t = 0;
for (const [k, v] of perKategori) { console.log(`${k.padEnd(16)} ${v.benar}/${v.total}`); b += v.benar; t += v.total; }
console.log(`${"TOTAL".padEnd(16)} ${b}/${t}  (${((100 * b) / t).toFixed(1)}%)`);
if (gagal.length) console.log(`\nGagal:\n${gagal.join("\n")}`);
