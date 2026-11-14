/**
 * NGOBROL HANYA DENGAN GPT (tanpa penjawab aturan), untuk belajar melihat
 * kemampuan model hasil fine-tuning percakapan.
 *
 * Pakai:  bun run chat-gpt                     (model chat terbaru)
 *         bun run chat-gpt -- --name chat --temperature 0.4 --history 0   (tanpa riwayat; bawaan 2 giliran)
 * Perintah:  :baru (lupakan riwayat)   :keluar
 */
import { createInterface } from "node:readline";
import { Tokenizer } from "../gpt/bpe.js";
import { balas, type Giliran } from "../gpt/chat.js";
import { gptTerbaru, muatGPT } from "../gpt/checkpoint.js";
import { GPT } from "../gpt/model.js";
import { argumen } from "./args.js";

const { opsi } = argumen();
const nama = opsi.name ?? gptTerbaru("chat");
if (!nama) {
  console.error("Belum ada model GPT percakapan. Jalankan dulu:\n  bun run siapkan-data -- --sumber obrolan --nama obrolan-v1 --tokenizer-dari wiki, lalu bun run train-gpt -- --dataset obrolan-v1 --from wiki --name chat");
  process.exit(1);
}
const { meta, P } = muatGPT(nama);
if (meta.mode !== "chat") console.log(`Peringatan: model "${nama}" belum di-fine-tune untuk percakapan; balasannya bisa aneh.`);
const model = new GPT(meta.config, P);
const tok = new Tokenizer(meta.bpe);
const suhu = opsi.temperature !== undefined ? Number(opsi.temperature) : 0.4;
const panjangRiwayat = opsi.history !== undefined ? Number(opsi.history) : 2; // sama dengan chatbot utama

console.log(`GPT "${nama}" (${model.ukuran.toLocaleString("id-ID")} parameter, dasar: ${meta.dasar ?? "-"}). Ketik :baru atau :keluar.`);
console.log("Catatan: balasan dikarang model, bukan dasar hukum.\n");
const riwayat: Giliran[] = [];
const rl = createInterface({ input: process.stdin, output: process.stdout, prompt: "Kamu> " });
rl.prompt();
rl.on("line", (baris) => {
  const pesan = baris.trim();
  if (!pesan) return rl.prompt();
  if (pesan === ":keluar" || pesan === ":q") return rl.close();
  if (pesan === ":baru") { riwayat.length = 0; console.log("Riwayat dihapus.\n"); return rl.prompt(); }
  const jawab = balas(model, tok, riwayat, pesan, { suhu }, meta.hurufKecil);
  console.log(`GPT> ${jawab}\n`);
  riwayat.push({ user: pesan, bot: jawab });
  while (riwayat.length > panjangRiwayat) riwayat.shift();
  rl.prompt();
});
rl.on("close", () => console.log("Sampai jumpa."));
