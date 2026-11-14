/**
 * MENGHASILKAN TEKS dengan GPT hasil training, token demi token (seperti GPT).
 *
 * Pakai:
 *   bun run generate -- --name wiki "Indonesia adalah"
 *   bun run generate -- --name wiki "Candi Borobudur" --length 150 --temperature 0.7 --stop-newline
 *
 * Opsi: --length (200 token)  --temperature (0.8)  --top-k (40)  --top-p (0.9)
 *       --repeat-penalty (1.15)  --stop-newline (berhenti di akhir paragraf)  --seed (acak)
 * Tanpa --name, model GPT teks yang paling baru dilatih yang dipakai.
 */
import { Tokenizer } from "../gpt/bpe.js";
import { gptTerbaru, muatGPT } from "../gpt/checkpoint.js";
import { GPT } from "../gpt/model.js";
import { hasilkanTeks } from "../gpt/text-generator.js";
import { argumen } from "./args.js";

const { opsi, teks } = argumen();
const nama = opsi.name ?? gptTerbaru("teks") ?? gptTerbaru();
if (!nama) {
  console.error("Belum ada model GPT. Latih dulu:  bun run siapkan-data -- --sumber wikipedia --nama wiki-v1 --huruf-asli, lalu bun run train-gpt -- --dataset wiki-v1 --name wiki");
  process.exit(1);
}
const { meta, P } = muatGPT(nama);
const model = new GPT(meta.config, P);
const tok = new Tokenizer(meta.bpe);
const awal = meta.hurufKecil ? teks.toLowerCase() : teks;
const angka = (k: string, b: number) => (opsi[k] !== undefined ? Number(opsi[k]) : b);
const berhentiBaris = opsi["stop-newline"] === "true";

console.log(`Model "${nama}" (${model.ukuran.toLocaleString("id-ID")} parameter, checkpoint langkah ${meta.langkah}, loss val ${meta.lossValTerbaik.toFixed(3)}, perplexity ${Math.exp(meta.lossValTerbaik).toFixed(1)})`);
console.log("Catatan: teks ini karangan model, bukan kutipan aturan.\n");
process.stdout.write(awal);
hasilkanTeks(model, tok, awal, angka("length", 200), {
  suhu: angka("temperature", 0.8), topK: angka("top-k", 40), topP: angka("top-p", 0.9),
  penaltiUlang: angka("repeat-penalty", 1.15),
  seed: opsi.seed !== undefined ? Number(opsi.seed) : undefined,
  berhentiDi: berhentiBaris ? (_, id) => tok.vocab[id] === "\n" : undefined,
}, (s) => process.stdout.write(s));
process.stdout.write("\n");
