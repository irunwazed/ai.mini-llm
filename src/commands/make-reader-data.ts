/**
 * BUAT DATA LATIH GPT PEMBACA (otomatis, tanpa label manual) -> data/reader-id.jsonl
 *
 * Pakai:  bun run make-reader-data
 *         bun run make-reader-data -- --dengan-aturan          (ikut pakai dokumen aturan dari model/)
 *         bun run make-reader-data -- --maks 6000 --negatif 0.3 --keluar data/reader-id.jsonl
 *
 * Bawaan hanya Wikipedia (data/wiki-id.txt), supaya skor penjawab aturan di `bun run eval`
 * tetap jujur (dokumen aturan tidak "dilihat" pembaca saat latihan).
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { muatModel } from "../model.js";
import { bagiData, buatDataPembaca, type Artikel } from "../gpt/reader-data.js";
import { argumen } from "./args.js";

const { opsi, folderModel } = argumen();
const artikel: Artikel[] = [];
const fileWiki = opsi.wiki ?? "data/wiki-id.txt";
if (existsSync(fileWiki)) {
  const judul = existsSync("data/wiki-id.sources.txt")
    ? readFileSync("data/wiki-id.sources.txt", "utf8").split(/\r?\n/).filter((l) => l.trim() && !l.startsWith("#")) : [];
  readFileSync(fileWiki, "utf8").split(/\n\s*\n/).forEach((teks, i) => {
    if (teks.trim()) artikel.push({ judul: judul[i] ?? `artikel ${i + 1}`, teks, kelompok: `wiki:${judul[i] ?? i}` });
  });
}
if (opsi["dengan-aturan"] === "true") {
  const m = muatModel(folderModel);
  const perFile = new Map<string, string[]>();
  for (const p of m.potongan) if (p.pasal !== "Pembukaan") perFile.set(p.file, [...(perFile.get(p.file) ?? []), p.teks]);
  for (const [file, teks] of perFile) artikel.push({ judul: m.info[file]?.tentang ?? file, teks: teks.join("\n"), kelompok: `aturan:${file}` });
}
if (!artikel.length) {
  console.error("Tidak ada teks sumber. Unduh dulu: bun run fetch-data  (atau pakai --dengan-aturan setelah bun run train)");
  process.exit(1);
}

const t0 = performance.now();
const data = buatDataPembaca(artikel, { maksPerPola: Number(opsi.maks ?? 6000), negatif: Number(opsi.negatif ?? 0.3), seed: Number(opsi.seed ?? 1) });
const keluar = opsi.keluar ?? "data/reader-id.jsonl";
writeFileSync(keluar, data.map((d) => JSON.stringify(d)).join("\n") + "\n");

const { train, val } = bagiData(data);
const perPola = new Map<string, { pos: number; neg: number }>();
for (const d of data) { const x = perPola.get(d.pola) ?? { pos: 0, neg: 0 }; d.awal >= 0 ? x.pos++ : x.neg++; perPola.set(d.pola, x); }
console.log(`${artikel.length} artikel/dokumen -> ${data.length} contoh (${train.length} latih, ${val.length} val) dalam ${((performance.now() - t0) / 1000).toFixed(1)} detik`);
for (const [p, x] of perPola) console.log(`  ${p.padEnd(9)} ${String(x.pos).padStart(6)} berjawaban  ${String(x.neg).padStart(6)} tanpa jawaban`);
console.log(`Disimpan ke ${keluar}. Contoh:`);
for (let i = 0; i < 6; i++) {
  const d = data[Math.floor((i * data.length) / 6)];
  console.log(`  [${d.pola}] T: ${d.pertanyaan}\n     J: ${d.awal >= 0 ? d.jawaban : "(tidak ada di konteks)"}\n     K: ${d.konteks.slice(0, 140)}…`);
}
console.log("\nLatih:  bun run train-reader -- --from wiki --name reader");
