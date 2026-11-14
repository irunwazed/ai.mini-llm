/**
 * BUAT DATASET PERCAKAPAN untuk fine-tuning GPT obrolan -> data/mentah/obrolan/chat-id.jsonl
 * (langkah berikutnya: bun run siapkan-data -- --sumber obrolan --nama obrolan-v1 --tokenizer-dari wiki2)
 *
 *   1. obrolan biasa (src/gpt/chat-data.ts): sapaan, terima kasih, curhat, perkenalan nama, …
 *   2. percakapan BERKONTEKS (src/gpt/chat-context-data.ts): jawaban bergantung pada giliran sebelumnya
 *      (ingat fakta, kata ganti, koreksi, topik lanjutan, ringkas, suasana, kartu konteks)
 * Percakapan berkonteks hanya memakai nilai slot "latih"; nilai "uji" disimpan untuk `bun run eval-konteks`.
 *
 * Pakai:  bun run make-chat-data
 *         bun run make-chat-data -- --konteks 6000      (jumlah percakapan berkonteks; 0 = tanpa)
 */
import { buatPercakapan } from "../gpt/chat-data.js";
import { buatPercakapanKonteks } from "../gpt/chat-context-data.js";
import { Rng } from "../gpt/random.js";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { LOKASI } from "../paths.js";
import { argumen } from "./args.js";

const { opsi } = argumen();
const biasa = buatPercakapan();
const konteks = buatPercakapanKonteks(Number(opsi.konteks ?? 6000), false, 7);
const semua: object[][] = [...biasa, ...konteks.map((p) => p.turns)];
const rng = new Rng(3);
for (let i = semua.length - 1; i > 0; i--) { const j = rng.int(i + 1); [semua[i], semua[j]] = [semua[j], semua[i]]; }
mkdirSync(dirname(LOKASI.obrolan), { recursive: true });
await Bun.write(LOKASI.obrolan, semua.map((turns) => JSON.stringify({ turns })).join("\n") + "\n");

const per = new Map<string, number>();
for (const p of konteks) per.set(p.skenario, (per.get(p.skenario) ?? 0) + 1);
console.log(`${LOKASI.obrolan}: ${semua.length} percakapan (${biasa.length} obrolan biasa + ${konteks.length} berkonteks)`);
for (const [s, n] of per) console.log(`  ${s.padEnd(15)} ${n}`);
console.log("\nContoh percakapan berkonteks:");
for (const p of konteks.slice(0, 6)) {
  console.log(`  [${p.skenario}]`);
  for (const g of p.turns) console.log(`    ${g.konteks !== undefined ? `<|konteks|> ${g.konteks}` : g.user !== undefined ? `Anda: ${g.user}` : `Bot : ${g.bot}`}`);
}
