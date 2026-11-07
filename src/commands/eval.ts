/**
 * EVALUASI: seberapa sering pasal yang BENAR ditemukan, dan apakah JAWABANNYA benar?
 *
 * Isi evaluation/questions.jsonl, satu baris per soal:
 *   {"tanya": "berapa hari cuti tahunan?", "file": "samples/sample-employee-leave.txt",
 *    "pasal": "Pasal 3", "harus_ada": "12 hari"}
 *   - "pasal" boleh dikosongkan jika cukup file-nya benar
 *   - "harus_ada" (opsional): teks yang wajib muncul di bagian utama jawaban
 *   - "di_luar": true untuk soal yang memang TIDAK ada aturannya (jawaban benar = menolak)
 *
 * Metrik:
 *   Recall@k = (1/|Q|) Σ [pasal benar ada di k hasil teratas]
 *   MRR      = (1/|Q|) Σ 1/peringkat_benar     (0 jika tidak ditemukan di 10 teratas)
 *   Akurasi jawaban = proporsi soal yang jawaban utamanya memuat "harus_ada"
 *   Penolakan benar = proporsi soal "di_luar" yang dijawab "tidak ditemukan"
 *
 * Pakai:  bun run eval
 *         bun run eval -- --soal evaluation/questions.jsonl
 */
import { readFileSync } from "node:fs";
import { cari } from "../search/search.js";
import { jawab } from "../answer/compose.js";
import { muatModel } from "../model.js";
import { argumen } from "./args.js";

interface Soal { tanya: string; file?: string; pasal?: string; harus_ada?: string; di_luar?: boolean }

const { opsi, folderModel } = argumen();
const soal: Soal[] = readFileSync(opsi.soal ?? "evaluation/questions.jsonl", "utf8")
  .split(/\r?\n/).filter((s) => s.trim()).map((s) => JSON.parse(s));
const model = muatModel(folderModel);

const K = [1, 3, 5];
const benar = new Map(K.map((k) => [k, 0]));
let mrr = 0, nCari = 0, nJawab = 0, okJawab = 0, nLuar = 0, okLuar = 0;
const salah: string[] = [];
const norm = (s: string) => s.toLowerCase().replace(/\s+/g, " ");

for (const s of soal) {
  if (s.di_luar) {
    nLuar++;
    const j = jawab(model, s.tanya);
    if (j.keyakinan === "rendah") okLuar++;
    else salah.push(`  - "${s.tanya}"\n      seharusnya ditolak, tetapi dijawab (keyakinan ${j.keyakinan})`);
    continue;
  }
  nCari++;
  const { hasil } = cari(model, s.tanya, { k: 10 });
  const r = hasil.findIndex(
    (h) => h.potongan.file === s.file && (!s.pasal || h.potongan.pasal.replace(/ \(bagian \d+\)$/, "") === s.pasal),
  ) + 1;
  for (const k of K) if (r > 0 && r <= k) benar.set(k, benar.get(k)! + 1);
  if (r > 0) mrr += 1 / r;
  if (r !== 1) salah.push(`  - "${s.tanya}"\n      pasal benar (${s.file} · ${s.pasal ?? "-"}) di peringkat ${r || "> 10"}; teratas: ${hasil[0]?.potongan.file} · ${hasil[0]?.potongan.pasal}`);

  if (s.harus_ada) {
    nJawab++;
    const j = jawab(model, s.tanya);
    const utama = j.utama.join(" ");
    if (norm(utama).includes(norm(s.harus_ada))) okJawab++;
    else salah.push(`  - "${s.tanya}"\n      jawaban tidak memuat "${s.harus_ada}": ${utama.slice(0, 160) || "(tidak menjawab)"}`);
  }
}

const pct = (a: number, b: number) => (b ? `${((100 * a) / b).toFixed(1)}%` : "-");
console.log(`Soal pencarian : ${nCari}`);
for (const k of K) console.log(`  Recall@${k}    : ${pct(benar.get(k)!, nCari)}`);
console.log(`  MRR         : ${(mrr / Math.max(1, nCari)).toFixed(3)}`);
console.log(`Akurasi jawaban: ${pct(okJawab, nJawab)}  (${okJawab}/${nJawab})`);
console.log(`Penolakan benar: ${pct(okLuar, nLuar)}  (${okLuar}/${nLuar})`);
if (salah.length) console.log(`\nPerlu diperhatikan:\n${salah.join("\n")}`);
