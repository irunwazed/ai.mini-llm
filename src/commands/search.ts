/**
 * Menjalankan model hasil training untuk SATU pertanyaan.
 *
 * Pakai:  bun run search -- "berapa hari cuti tahunan?"
 */
import { formatJawaban, jawab } from "../answer/compose.js";
import { muatModel } from "../model.js";
import { argumen } from "./args.js";

const { teks, folderModel, k } = argumen();
if (!teks) {
  console.error('Tulis pertanyaannya, contoh:  bun run search -- "berapa hari cuti tahunan?"');
  process.exit(1);
}
const model = muatModel(folderModel);
console.log(`Pertanyaan: ${teks}\n`);
console.log(formatJawaban(model, jawab(model, teks, { k })));
