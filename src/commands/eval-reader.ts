/**
 * UJI GPT PEMBACA pada data val (artikel yang tidak dipakai saat latihan).
 *
 * Metrik (seperti SQuAD 2.0):
 *   EM  = rentang persis sama dengan jawaban benar
 *   F1  = kecocokan kata antara rentang dan jawaban benar
 *   tolak benar = contoh tanpa jawaban yang keyakinannya < ambang
 *   tabel ambang: untuk setiap ambang keyakinan, berapa jawaban dipakai dan berapa yang benar
 *
 * Pakai:  bun run eval-reader
 *         bun run eval-reader -- --name reader --data data/reader-id.jsonl --maks 1000
 */
import { readFileSync } from "node:fs";
import { Tokenizer } from "../gpt/bpe.js";
import { gptTerbaru, muatGPT } from "../gpt/checkpoint.js";
import { GPT } from "../gpt/model.js";
import { bagiData, type ContohPembaca } from "../gpt/reader-data.js";
import { baca, f1Kata } from "../gpt/reader.js";
import { LOKASI, lokasiBaca } from "../paths.js";
import { argumen } from "./args.js";

const { opsi } = argumen();
const nama = opsi.name ?? gptTerbaru("pembaca");
if (!nama) { console.error("Belum ada GPT pembaca. Jalankan: bun run make-reader-data && bun run train-reader"); process.exit(1); }
const { meta, P } = muatGPT(nama);
const model = new GPT(meta.config, P), tok = new Tokenizer(meta.bpe);
const ambang = meta.pembaca?.ambang ?? 0.5;
const data: ContohPembaca[] = readFileSync(opsi.data ?? lokasiBaca(LOKASI.pembaca), "utf8").split(/\r?\n/).filter((l) => l.trim()).map((l) => JSON.parse(l));
const { val } = bagiData(data);
const maks = Number(opsi.maks ?? 1000);
const uji = val.filter((_, i) => i % Math.max(1, Math.floor(val.length / maks)) === 0).slice(0, maks);

const per = new Map<string, { n: number; em: number; f1: number; nKosong: number; tolak: number }>();
const kal: { yakin: number; ada: boolean; benar: boolean }[] = [];
const contoh: string[] = [];
const t0 = performance.now();
for (const d of uji) {
  const h = baca(model, tok, d.pertanyaan, d.konteks, meta.hurufKecil);
  const x = per.get(d.pola) ?? { n: 0, em: 0, f1: 0, nKosong: 0, tolak: 0 };
  const ada = d.awal >= 0;
  const f = ada ? f1Kata(h.jawaban, d.jawaban) : 0;
  if (ada) { x.n++; x.f1 += f; if (h.jawaban.toLowerCase() === d.jawaban.toLowerCase()) x.em++; }
  else { x.nKosong++; if (h.keyakinan < ambang) x.tolak++; }
  per.set(d.pola, x);
  kal.push({ yakin: h.keyakinan, ada, benar: f >= 0.5 });
  if (contoh.length < 8 && ada) contoh.push(`  T: ${d.pertanyaan}\n     benar: "${d.jawaban}" | pembaca: "${h.jawaban}" (${(100 * h.keyakinan).toFixed(0)}%) ${f >= 0.5 ? "✓" : "✗"}`);
}

console.log(`GPT pembaca "${nama}" (langkah ${meta.langkah}, ambang ${ambang}) — ${uji.length} contoh val, ${((performance.now() - t0) / uji.length).toFixed(0)} ms/contoh\n`);
console.log("pola      |   n |   EM  |   F1  | tolak benar");
let N = 0, EM = 0, F1 = 0, NK = 0, TK = 0;
for (const [p, x] of per) {
  console.log(`${p.padEnd(9)} | ${String(x.n).padStart(3)} | ${(100 * x.em / Math.max(1, x.n)).toFixed(1).padStart(5)} | ${(100 * x.f1 / Math.max(1, x.n)).toFixed(1).padStart(5)} | ${(100 * x.tolak / Math.max(1, x.nKosong)).toFixed(1).padStart(5)}% dari ${x.nKosong}`);
  N += x.n; EM += x.em; F1 += x.f1; NK += x.nKosong; TK += x.tolak;
}
console.log(`${"TOTAL".padEnd(9)} | ${String(N).padStart(3)} | ${(100 * EM / N).toFixed(1).padStart(5)} | ${(100 * F1 / N).toFixed(1).padStart(5)} | ${(100 * TK / Math.max(1, NK)).toFixed(1).padStart(5)}% dari ${NK}`);

console.log("\nAmbang keyakinan -> jawaban yang dipakai (dari contoh berjawaban) dan ketepatannya:");
for (const a of [0.3, 0.5, 0.7, 0.8, 0.9]) {
  const pakai = kal.filter((k) => k.yakin >= a);
  const benar = pakai.filter((k) => k.ada && k.benar).length;
  console.log(`  ≥ ${a.toFixed(1)}: dipakai ${pakai.length}/${kal.length}, tepat ${(100 * benar / Math.max(1, pakai.length)).toFixed(1)}%, salah tolak ${kal.filter((k) => k.ada && k.benar && k.yakin < a).length}`);
}
console.log(`\nContoh:\n${contoh.join("\n")}`);
