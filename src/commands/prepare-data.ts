/**
 * LANGKAH 2: SIAPKAN DATA  (ambil data -> SIAPKAN -> latih -> evaluasi)
 *
 * Membaca data MENTAH, men-token-kan, lalu menyimpan DATASET SIAP LATIH di data/siap/<nama>/.
 * Training (`bun run train-gpt -- --dataset <nama>`) membaca hasil ini, tanpa membaca teks mentah lagi.
 *
 * Sumber (--sumber, boleh beberapa dipisah koma):
 *   wikipedia  data/mentah/wikipedia/wiki-id.txt         (bun run fetch-data)
 *   teks       data/mentah/teks/                         (teks Indonesia milik Anda: upload / salin)
 *   aturan     documents/                                (PDF/DOCX/TXT/MD)
 *   obrolan    data/mentah/obrolan/chat-id.jsonl         (bun run make-chat-data)
 *   atau path file/folder lain
 *
 * Tokenizer:
 *   (bawaan)                 tokenizer BARU dilatih dari data ini (--vocab 4000)         -> untuk model baru
 *   --tokenizer-dari <model> tokenizer model yang sudah ada (gpt-models/<model>)          -> untuk --resume / --from
 *   --huruf-asli             pertahankan huruf besar (hanya untuk tokenizer baru)
 *
 * Pakai:
 *   bun run siapkan-data -- --sumber wikipedia --nama wiki-v1 --huruf-asli
 *   bun run siapkan-data -- --sumber wikipedia,teks --nama wiki-v2 --tokenizer-dari wiki2
 *   bun run siapkan-data -- --sumber obrolan --nama obrolan-v1 --tokenizer-dari wiki2
 */
import { latihBPE, Tokenizer, type BPE } from "../gpt/bpe.js";
import { FOLDER_GPT, muatGPT } from "../gpt/checkpoint.js";
import { bacaSumber, dataTeks, encodePercakapan } from "../gpt/dataset.js";
import { infoLokasi, sidikTokenizer, simpanDataset, type InfoDataset } from "../gpt/prepared.js";
import { Pelacak } from "../gpt/progress.js";
import { lokasiBaca, SUMBER_MENTAH } from "../paths.js";
import { argumen } from "./args.js";

const { opsi } = argumen();
const nama = opsi.nama;
if (!opsi.sumber || !nama || !/^[\w.-]{1,40}$/.test(nama)) {
  console.error("Pakai: bun run siapkan-data -- --sumber wikipedia[,teks,aturan,obrolan] --nama <nama-dataset> [--tokenizer-dari <model> | --vocab 4000 --huruf-asli]");
  console.error("Sumber yang dikenal:"); for (const [k, v] of Object.entries(SUMBER_MENTAH)) console.error(`  ${k.padEnd(10)} ${v.lokasi}  — ${v.keterangan}`);
  process.exit(1);
}
const pelacak = new Pelacak("siapkan-data", nama, { total: 4, satuan: "tahap" });

// 1. Sumber mentah
const daftar = opsi.sumber.split(",").map((x) => x.trim()).filter(Boolean).map((s) => {
  const dikenal = SUMBER_MENTAH[s];
  return { nama: s, lokasi: dikenal ? lokasiBaca(dikenal.lokasi) : s };
});
let bpe: BPE | null = null, hurufKecil = opsi["huruf-asli"] !== "true" && opsi["keep-case"] !== "true", asalTokenizer = "baru";
if (opsi["tokenizer-dari"]) {
  const m = muatGPT(opsi["tokenizer-dari"]);
  bpe = m.meta.bpe; hurufKecil = m.meta.hurufKecil; asalTokenizer = `model ${opsi["tokenizer-dari"]}`;
}
console.log(`[1/4] Membaca sumber mentah (${hurufKecil ? "huruf kecil" : "huruf asli"})...`);
pelacak.langkah(0, { catatan: "membaca sumber" });
const sumber = await bacaSumber(daftar.map((d) => d.lokasi).join(","), FOLDER_GPT, hurufKecil);
for (const k of sumber.keterangan) console.log(`      ${k}`);
const jenis = sumber.percakapan.length ? "obrolan" : "teks";

// 2. Tokenizer
pelacak.langkah(1, { catatan: "tokenizer" });
if (!bpe) {
  const target = Number(opsi.vocab ?? 4000);
  console.log(`[2/4] Melatih tokenizer BPE baru (maks ${target} token)...`);
  const teksBPE = jenis === "teks" ? sumber.teks : sumber.percakapan.flat().map((g) => `${g.user ?? ""} ${g.bot ?? ""}`).join("\n");
  bpe = latihBPE(teksBPE, target, (s) => console.log("      " + s));
} else console.log(`[2/4] Tokenizer dari ${asalTokenizer} (${bpe.vocab.length} token)`);
const tok = new Tokenizer(bpe);

// 3. Tokenisasi
pelacak.langkah(2, { catatan: "tokenisasi" });
console.log(`[3/4] Men-token-kan...`);
const info: InfoDataset = {
  nama, jenis, dibuat: new Date().toISOString(),
  sumber: daftar.map((d) => { const x = infoLokasi(d.lokasi); return { nama: d.nama, lokasi: d.lokasi, ukuran: x.ukuran, diubah: x.diubah }; }),
  hurufKecil, tokenizer: { asal: asalTokenizer, sidik: sidikTokenizer(bpe), ukuran: bpe.vocab.length },
  token: { latih: 0, val: 0 }, keterangan: sumber.keterangan,
};
let folder: string;
if (jenis === "teks") {
  const { train, val } = dataTeks(tok, sumber.teks);
  info.token = { latih: train.length, val: val.length };
  pelacak.langkah(3, { catatan: "menyimpan" });
  folder = simpanDataset(info, bpe, { latih: train, val });
} else {
  const percakapan = sumber.percakapan.map((g) => encodePercakapan(tok, g));
  const total = percakapan.reduce((s, p) => s + p.potongan.reduce((a, q) => a + q.ids.length, 0), 0);
  info.percakapan = percakapan.length;
  info.token = { latih: Math.round(total * 0.9), val: Math.round(total * 0.1) };
  pelacak.langkah(3, { catatan: "menyimpan" });
  folder = simpanDataset(info, bpe, { percakapan });
}

// 4. Selesai
pelacak.langkah(4, { catatan: "selesai" });
pelacak.tutup("selesai", `${jenis}, ${(info.token.latih + info.token.val).toLocaleString("id-ID")} token`);
console.log(`[4/4] Disimpan di ${folder}/`);
console.log(`      ${jenis}${info.percakapan ? `, ${info.percakapan} percakapan` : ""}, token latih ${info.token.latih.toLocaleString("id-ID")}, val ${info.token.val.toLocaleString("id-ID")}`);
console.log(`      tokenizer ${asalTokenizer}, sidik ${info.tokenizer.sidik}`);
const saran = jenis === "obrolan"
  ? `bun run train-gpt -- --dataset ${nama} --from ${opsi["tokenizer-dari"] ?? "<model-pretrain>"} --name <model-chat> --steps 1500 --eval-every 200`
  : opsi["tokenizer-dari"]
    ? `bun run train-gpt -- --dataset ${nama} --name ${opsi["tokenizer-dari"]} --resume --minutes 60      (lanjutkan model dengan data baru)`
    : `bun run train-gpt -- --dataset ${nama} --name <model-baru> --size besar --minutes 60`;
console.log(`\nLangkah berikutnya (LATIH):\n  ${saran}`);

