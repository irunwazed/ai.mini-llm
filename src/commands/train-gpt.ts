/**
 * TRAINING GPT dari nol (pretraining) atau lanjutan dari model lain (fine-tuning percakapan).
 *
 * LANGKAH 3 dari alur  ambil data -> siapkan data -> LATIH -> evaluasi  (lihat PANDUAN.md).
 * Training membaca DATASET SIAP LATIH dari data/siap/<dataset>/ (hasil `bun run siapkan-data`):
 *   bun run train-gpt -- --dataset wiki-v1 --name wiki --size besar --minutes 60      model BARU
 *   bun run train-gpt -- --dataset wiki-v2 --name wiki --resume --minutes 60          LANJUTKAN model yang sama dgn data baru
 *   bun run train-gpt -- --dataset obrolan-v1 --from wiki --name chat --steps 1500    FINE-TUNE ke model baru
 * Untuk --resume/--from, dataset harus di-token-kan dengan tokenizer model itu
 * (siapkan-data … --tokenizer-dari <model>); jika tidak, training ditolak.
 * Cara lama satu perintah tetap jalan: --data <file/folder> (baca + tokenisasi di memori).
 *
 * Opsi (bawaan dalam kurung):
 *   --dataset <nama>           dataset siap latih di data/siap/<nama>/
 *   --data <sumber[,sumber]>   (cara lama) file .txt, folder dokumen, atau .jsonl percakapan
 *   --name <nama>              nama model di gpt-models/ (nama data)
 *   --from <nama>              mulai dari bobot + tokenizer model lain (fine-tuning)
 *   --resume                   lanjutkan model --name (termasuk momentum Adam)
 *   --size small|medium|besar  preset ukuran (small; besar = dim 128, 6 lapisan, konteks 256, kosakata 4000, pre-LN); bisa ditimpa:
 *   --layers --dim --heads --context --ff --vocab
 *   --minutes (45; chat 10)    anggaran waktu; jumlah langkah dihitung dari kecepatan nyata
 *   --steps                    jumlah langkah tetap (menggantikan --minutes)
 *   --batch (16) --lr (0.003; fine-tune 0.001) --warmup --eval-every (100; chat 25)
 *   --patience (5)             early stopping: berhenti jika loss val tak membaik N evaluasi
 *   --dropout (0.1) --weight-decay (0.1) --workers (jumlah CPU − 2) --seed (1)
 *   --pre-ln --keep-case
 */
import { cpus } from "node:os";
import { basename } from "node:path";
import { latihBPE, Tokenizer } from "../gpt/bpe.js";
import { balas } from "../gpt/chat.js";
import { FOLDER_GPT, muatGPT, simpanGPT, type MetaGPT, type Riwayat } from "../gpt/checkpoint.js";
import { bacaSumber, dataChat, dataTeks, urutanChat, type Urutan } from "../gpt/dataset.js";
import { daftarDataset, muatDataset, sidikTokenizer } from "../gpt/prepared.js";
import { GPT, inisialisasi, susunParameter, type GPTConfig } from "../gpt/model.js";
import { Adam, jadwalLR } from "../gpt/optimizer.js";
import { Rng } from "../gpt/random.js";
import { hasilkanTeks } from "../gpt/text-generator.js";
import { Pelacak } from "../gpt/progress.js";
import { argumen } from "./args.js";

const { opsi } = argumen();
if (!opsi.data && !opsi.dataset) {
  console.error("Sebutkan data latih. Disarankan dua langkah terpisah:\n  1. bun run siapkan-data -- --sumber wikipedia --nama wiki-v1 --huruf-asli\n  2. bun run train-gpt -- --dataset wiki-v1 --name wiki\n(Cara lama tetap bisa: --data <file/folder>)");
  process.exit(1);
}
const nama = opsi.name ?? opsi.dataset ?? basename(opsi.data!.split(",")[0]).replace(/\.[^.]+$/, "");
const angka = (k: string, b: number) => (opsi[k] !== undefined ? Number(opsi[k]) : b);
const mulai = performance.now();
const detik = () => (performance.now() - mulai) / 1000;

const PRESET: Record<string, { dim: number; layers: number; heads: number; context: number; vocab: number; preLN?: boolean }> = {
  small: { dim: 64, layers: 4, heads: 4, context: 128, vocab: 2000 },
  medium: { dim: 96, layers: 4, heads: 4, context: 128, vocab: 3000 },
  // "besar": untuk obrolan yang paham konteks. Konteks 256 token memuat lebih banyak riwayat; kosakata 4000
  // membuat kata Indonesia lebih jarang terpecah; 6 lapisan pre-LN tetap stabil. ±2,2 juta parameter.
  besar: { dim: 128, layers: 6, heads: 4, context: 256, vocab: 4000, preLN: true },
};

// 1. Model asal (resume / fine-tune) + data --------------------------------------------
const asal = opsi.resume === "true" ? muatGPT(nama) : opsi.from ? muatGPT(opsi.from) : null;
let hurufKecil = asal ? asal.meta.hurufKecil : opsi["keep-case"] !== "true";
let mode: "teks" | "chat";
let bpe = asal?.meta.bpe;
let keteranganData: string;
let datasetSiap: ReturnType<typeof muatDataset> | null = null;
let sumber: Awaited<ReturnType<typeof bacaSumber>> | null = null;
const preset = PRESET[opsi.size ?? "small"] ?? PRESET.small;

if (opsi.dataset) {
  // LANGKAH 3 membaca hasil LANGKAH 2 (data/siap/<dataset>/): token sudah jadi, tidak membaca teks mentah lagi
  datasetSiap = muatDataset(opsi.dataset);
  const ds = datasetSiap.info;
  console.log(`[1/4] Dataset siap latih "${ds.nama}" (${ds.jenis}, disiapkan ${new Date(ds.dibuat).toLocaleString("id-ID")})`);
  for (const k of ds.keterangan) console.log(`      ${k}`);
  const berubah = daftarDataset().find((d) => d.nama === ds.nama)?.sumberBerubah ?? [];
  if (berubah.length) console.log(`      PERINGATAN: sumber ${berubah.join(", ")} berubah sejak dataset disiapkan. Siapkan ulang agar data baru ikut: bun run siapkan-data -- … --nama ${ds.nama}`);
  if (asal) {
    const sidikModel = sidikTokenizer(asal.meta.bpe);
    if (sidikModel !== ds.tokenizer.sidik || asal.meta.hurufKecil !== ds.hurufKecil) {
      const m = opsi.resume === "true" ? nama : opsi.from;
      console.error(`\nDitolak: dataset "${ds.nama}" di-token-kan dengan tokenizer ${ds.tokenizer.asal} (sidik ${ds.tokenizer.sidik}, ${ds.hurufKecil ? "huruf kecil" : "huruf asli"}),`
        + `\nsedangkan model "${m}" memakai tokenizer lain (sidik ${sidikModel}, ${asal.meta.hurufKecil ? "huruf kecil" : "huruf asli"}).`
        + `\nId token di dataset akan berarti potongan kata yang berbeda bagi model ini. Siapkan ulang dengan tokenizer model itu:`
        + `\n  bun run siapkan-data -- --sumber ${ds.sumber.map((x) => x.nama).join(",")} --nama ${ds.nama}-${m} --tokenizer-dari ${m}`);
      process.exit(1);
    }
  }
  mode = ds.jenis === "teks" ? "teks" : "chat";
  bpe = datasetSiap.bpe;
  hurufKecil = ds.hurufKecil;
  keteranganData = `dataset ${ds.nama}: ${ds.keterangan.join("; ")}`;
  console.log(`[2/4] Tokenizer dari dataset (${ds.tokenizer.asal}, ${ds.tokenizer.ukuran} token, sidik ${ds.tokenizer.sidik})`);
} else {
  // Cara lama (satu perintah): baca mentah + tokenizer + token di memori
  console.log(`[1/4] Membaca data...`);
  sumber = await bacaSumber(opsi.data!, FOLDER_GPT, hurufKecil);
  for (const k of sumber.keterangan) console.log(`      ${k}`);
  mode = sumber.percakapan.length ? "chat" : "teks";
  keteranganData = sumber.keterangan.join("; ");
  if (!bpe) {
    const target = angka("vocab", preset.vocab);
    console.log(`[2/4] Melatih tokenizer BPE (maks ${target} token, pasangan minimal muncul 10 kali)...`);
    const teksBPE = mode === "teks" ? sumber.teks : sumber.percakapan.flat().map((g) => `${g.user ?? ""} ${g.bot ?? ""}`).join("\n");
    bpe = latihBPE(teksBPE, target, (x) => console.log("      " + x));
  } else console.log(`[2/4] Tokenizer BPE dari model "${asal!.meta.dasar ?? (opsi.from ?? nama)}"`);
}
if (mode === "chat" && !asal) console.log("      PERINGATAN: fine-tune chat sebaiknya dimulai dari model pretrain (--from <nama>).");
const tok = new Tokenizer(bpe!);

// 3. Model -----------------------------------------------------------------------------
const dim = angka("dim", preset.dim);
const config: GPTConfig = asal?.meta.config ?? {
  vocab: tok.ukuran, context: angka("context", preset.context), dim, heads: angka("heads", preset.heads),
  layers: angka("layers", preset.layers), ff: angka("ff", 4 * dim), preLN: opsi["pre-ln"] === "true" || !!preset.preLN,
};
const T = config.context;
let trainTeks: Int32Array = new Int32Array(0), valTeks: Int32Array = new Int32Array(0);
let trainChat: Urutan[] = [], valChat: Urutan[] = [];
if (datasetSiap) {
  if (mode === "teks") { trainTeks = datasetSiap.latih!; valTeks = datasetSiap.val!; }
  else ({ train: trainChat, val: valChat } = urutanChat(datasetSiap.percakapan!, T));
} else if (mode === "teks") ({ train: trainTeks, val: valTeks } = dataTeks(tok, sumber!.teks));
else ({ train: trainChat, val: valChat } = dataChat(tok, sumber!.percakapan, T));
const tokenTarget = mode === "teks" ? trainTeks.length : trainChat.reduce((s, u) => s + u.y.filter((v) => v >= 0).length, 0);
console.log(`      kosakata ${tok.ukuran} token | ${mode === "teks"
  ? `train ${trainTeks.length.toLocaleString("id-ID")} token, val ${valTeks.length.toLocaleString("id-ID")} token`
  : `${trainChat.length} percakapan latih, ${valChat.length} val, ${tokenTarget.toLocaleString("id-ID")} token jawaban`}`);
if (mode === "teks" && trainTeks.length < T + 2) throw new Error(`Data terlalu sedikit untuk konteks ${T}. Tambah data atau kecilkan --context.`);
if (mode === "chat" && (trainChat.length === 0 || valChat.length === 0)) throw new Error("Data percakapan terlalu sedikit (butuh minimal 10 percakapan).");

const { ukuran } = susunParameter(config);
const bobotSAB = new SharedArrayBuffer(ukuran * 8);
const P = new Float64Array(bobotSAB);
if (asal) P.set(asal.P);
else inisialisasi(config, P, angka("seed", 1));
const model = new GPT(config, P);
const lrMaks = angka("lr", opsi.from ? 1e-3 : 3e-3);
const adam = new Adam(model, { lr: lrMaks, weightDecay: angka("weight-decay", 0.1) });
if (opsi.resume === "true" && asal?.adam) { adam.m.set(asal.adam.m); adam.v.set(asal.adam.v); }
const dropout = angka("dropout", 0.1);
const nPekerja = Math.max(1, Math.min(angka("workers", Math.max(1, cpus().length - 2)), angka("batch", 16)));
console.log(`[3/4] GPT ${config.preLN ? "pre-LN" : "post-LN (sesuai diagram)"}: ${config.layers} lapisan, dim ${config.dim}, ${config.heads} head, konteks ${T}, FF ${config.ff}`);
console.log(`      ${ukuran.toLocaleString("id-ID")} parameter | dropout ${dropout} | ${nPekerja} worker${opsi.from ? ` | fine-tune dari "${opsi.from}"` : ""}`);
if (!asal && ukuran > tokenTarget / 2)
  console.log(`      PERINGATAN: parameter (${ukuran.toLocaleString("id-ID")}) jauh lebih banyak dari token latih (${tokenTarget.toLocaleString("id-ID")}).\n      Model akan MENGHAFAL. Tambah data (bun run fetch-data) atau kecilkan model (--dim 32 --layers 2).`);

// 4. Worker paralel --------------------------------------------------------------------
const pekerja: { w: Worker; G: Float64Array }[] = [];
if (nPekerja > 1) {
  let tokenSAB: SharedArrayBuffer | null = null;
  if (mode === "teks") { tokenSAB = new SharedArrayBuffer(trainTeks.length * 4); new Int32Array(tokenSAB).set(trainTeks); }
  await Promise.all(Array.from({ length: nPekerja }, (_, i) => {
    const gSAB = new SharedArrayBuffer(ukuran * 8);
    const w = new Worker(new URL("../gpt/worker.ts", import.meta.url).href);
    pekerja.push({ w, G: new Float64Array(gSAB) });
    return new Promise<void>((ok) => {
      w.onmessage = () => ok();
      w.postMessage({ jenis: "init", config, bobot: bobotSAB, gradien: gSAB, token: tokenSAB, urutan: mode === "chat" ? trainChat : null, dropout, seed: angka("seed", 1) * 1000 + i });
    });
  }));
} else if (dropout > 0) model.aturDropout(dropout, new Rng(angka("seed", 1) + 7).next);

const G = new Float64Array(ukuran);
async function hitungBatch(kerja: number[], skala: number): Promise<number> {
  G.fill(0);
  if (pekerja.length === 0) {
    let loss = 0;
    for (const s of kerja)
      loss += mode === "chat"
        ? model.langkah(trainChat[s].x, trainChat[s].y, G, skala)
        : model.langkah(trainTeks.subarray(s, s + T), trainTeks.subarray(s + 1, s + T + 1), G, skala);
    return loss;
  }
  const bagian = pekerja.map((_, i) => kerja.filter((_, j) => j % pekerja.length === i));
  const hasil = await Promise.all(pekerja.map(({ w }, i) => new Promise<number>((ok) => {
    w.onmessage = (e) => ok(e.data.loss);
    w.postMessage({ jenis: "hitung", kerja: bagian[i], T, skala });
  })));
  for (const { G: g } of pekerja) for (let i = 0; i < ukuran; i++) G[i] += g[i];
  return hasil.reduce((a, b) => a + b, 0);
}

/** Loss validasi rata-rata per token target, dengan dropout dimatikan. */
const sumberVal = valTeks.length >= T + 2 ? valTeks : trainTeks;
const jendelaVal = mode === "teks" ? Array.from({ length: Math.min(48, Math.floor((sumberVal.length - 1) / T)) }, (_, i) =>
  Math.floor((i * (sumberVal.length - T - 1)) / Math.max(1, Math.min(48, Math.floor((sumberVal.length - 1) / T))))) : [];
function lossVal(): number {
  model.aturDropout(0);
  let s = 0, n = 0;
  if (mode === "chat") for (const u of valChat) { s += model.langkah(u.x, u.y, null, 1); n += u.y.filter((v) => v >= 0).length; }
  else for (const a of jendelaVal) { s += model.langkah(sumberVal.subarray(a, a + T), sumberVal.subarray(a + 1, a + T + 1), null, 1); n += T; }
  if (pekerja.length === 0 && dropout > 0) model.aturDropout(dropout);
  return s / Math.max(1, n);
}

function contoh(): string {
  model.aturDropout(0);
  let s: string;
  if (mode === "chat") s = ["halo", "apa kabar?"].map((q) => `"${q}" -> "${balas(model, tok, [], q, { seed: 1 }, hurufKecil)}"`).join("  |  ");
  else {
    const awal = tok.decode(Array.from(sumberVal.subarray(0, 6)));
    s = JSON.stringify(awal + hasilkanTeks(model, tok, awal, 40, { suhu: 0.8, topP: 0.9, penaltiUlang: 1.15, seed: 1 }));
  }
  if (pekerja.length === 0 && dropout > 0) model.aturDropout(dropout);
  return s;
}

// 5. Training loop ---------------------------------------------------------------------
const batch = angka("batch", 16);
const setiap = angka("eval-every", mode === "chat" ? 25 : 100);
const patience = angka("patience", 5);
const anggaran = angka("minutes", mode === "chat" ? 10 : 45) * 60;
let langkahTotal: number | null = opsi.steps ? Number(opsi.steps) : null;
const warmup = angka("warmup", langkahTotal ? Math.max(1, Math.floor(langkahTotal / 10)) : 100);
const rng = new Rng(angka("seed", 1) + (opsi.resume === "true" ? asal!.meta.langkah : 0));
const riwayat: Riwayat[] = opsi.resume === "true" ? asal!.meta.riwayat : [];
let terbaik = opsi.resume === "true" ? asal!.meta.lossValTerbaik : Infinity;
const langkahAwal = opsi.resume === "true" ? asal!.meta.langkah : 0;
let tanpaKemajuan = 0;
let jumlahLoss = 0, nLoss = 0, tokenDiproses = 0, waktuBlok = performance.now(), mulaiLatih = performance.now();

const meta = (langkah: number): MetaGPT => ({
  mode, dasar: opsi.from ?? asal?.meta.dasar, config, bpe: bpe!, hurufKecil, langkah, lossValTerbaik: terbaik, riwayat,
  data: keteranganData, jumlahToken: tokenTarget, waktu: new Date().toISOString(),
  // riwayat data yang pernah dipakai model ini (resume dengan data baru menambah baris baru)
  dataset: [...(opsi.resume === "true" ? asal!.meta.dataset ?? [] : []), { nama: opsi.dataset ?? opsi.data!, mulai: langkahAwal + 1, selesai: langkah, waktu: new Date().toISOString() }],
});

const lv0 = lossVal();
console.log(`[4/4] Training ${langkahTotal ? `${langkahTotal} langkah` : `±${Math.round(anggaran / 60)} menit`}, batch ${batch}, lr ${lrMaks} (warmup ${warmup}), early stopping ${patience} evaluasi`);
console.log(`      loss val awal ${lv0.toFixed(3)} (menebak acak ≈ ln(${tok.ukuran}) = ${Math.log(tok.ukuran).toFixed(2)})\n`);
console.log("langkah |  loss train | loss val | perplexity |      lr | token/dtk");
console.log("        (Ctrl+C = berhenti dengan aman & simpan · pantau dari terminal lain: bun run status -- --ikuti)");
// Progres hidup di terminal + file progres/train-gpt-<nama>.json (untuk bun run status dan panel web)
const pelacak = new Pelacak("train-gpt", nama, { total: langkahTotal });
let langkahSimpan = opsi.resume === "true" ? langkahAwal : 0, dihentikan = 0, alasanSelesai = "";
for (let langkah = langkahAwal + 1; langkahTotal === null || langkah <= langkahTotal; langkah++) {
  let kerja: number[], nToken: number;
  if (mode === "chat") {
    kerja = Array.from({ length: batch }, () => rng.int(trainChat.length));
    nToken = kerja.reduce((s, i) => s + trainChat[i].y.filter((v) => v >= 0).length, 0);
  } else {
    kerja = Array.from({ length: batch }, () => rng.int(trainTeks.length - T - 1));
    nToken = batch * T;
  }
  jumlahLoss += (await hitungBatch(kerja, 1 / nToken)) / nToken;
  nLoss++;
  tokenDiproses += mode === "chat" ? kerja.reduce((s, i) => s + trainChat[i].x.length, 0) : nToken;
  const lr = jadwalLR(langkah, langkahTotal ?? 1e9, lrMaks, warmup);
  adam.langkah(G, langkah, lr);
  pelacak.langkah(langkah, { lossTrain: jumlahLoss / nLoss, kecepatan: tokenDiproses / Math.max(1e-3, (performance.now() - waktuBlok) / 1000) });

  // Setelah 20 langkah, ukur kecepatan nyata dan tetapkan jumlah langkah sesuai anggaran waktu
  if (langkahTotal === null && langkah - langkahAwal === 20) {
    const perLangkah = (performance.now() - mulaiLatih) / 1000 / 20;
    langkahTotal = langkah + Math.max(setiap, Math.floor((anggaran - (performance.now() - mulaiLatih) / 1000) / (perLangkah * 1.05)));
    console.log(`        (±${perLangkah.toFixed(2)} detik/langkah -> total ${langkahTotal} langkah)`);
    pelacak.setTotal(langkahTotal);
  }

  const evalSekarang = langkah % setiap === 0 || langkah === langkahTotal || pelacak.berhenti;
  if (evalSekarang) {
    const lv = lossVal(), lt = jumlahLoss / nLoss;
    pelacak.evaluasi(langkah, lt, lv, Math.min(terbaik, lv));
    const kecepatan = tokenDiproses / ((performance.now() - waktuBlok) / 1000);
    riwayat.push({ langkah, lossTrain: lt, lossVal: lv, detik: detik() });
    const tanda = lv < terbaik - 1e-3 ? " *" : "";
    console.log(`${String(langkah).padStart(7)} | ${lt.toFixed(4).padStart(11)} | ${lv.toFixed(4).padStart(8)} | ${Math.exp(lv).toFixed(1).padStart(10)} | ${lr.toExponential(1)} | ${Math.round(kecepatan).toLocaleString("id-ID").padStart(9)}${tanda}`);
    if (lv < terbaik - 1e-3) {
      terbaik = lv;
      tanpaKemajuan = 0;
      simpanGPT(nama, meta(langkah), P, adam);
      langkahSimpan = langkah;
    } else if (++tanpaKemajuan >= patience) {
      console.log(`\nEarly stopping: loss val tidak membaik selama ${patience} evaluasi (overfitting terdeteksi). Model terbaik sudah tersimpan.`);
      alasanSelesai = "early stopping";
      break;
    }
    if (pelacak.berhenti) { dihentikan = langkah; break; }
    if (langkah % (setiap * 5) === 0 || langkah === langkahTotal) console.log(`        contoh: ${contoh()}`);
    jumlahLoss = 0; nLoss = 0; tokenDiproses = 0; waktuBlok = performance.now();
  }
}
for (const { w } of pekerja) w.terminate();
if (dihentikan) {
  const lanjut = `bun run train-gpt -- ${opsi.dataset ? `--dataset ${opsi.dataset}` : `--data ${opsi.data}`} --name ${nama} --resume${opsi.minutes ? ` --minutes ${opsi.minutes}` : ""}${opsi.steps ? ` --steps ${opsi.steps}` : ""}`;
  pelacak.tutup("berhenti", `dihentikan di langkah ${dihentikan}; checkpoint terbaik di langkah ${langkahSimpan}`);
  console.log(`\nDihentikan dengan aman di langkah ${dihentikan}. Checkpoint terbaik (loss val ${terbaik.toFixed(4)}) tersimpan di langkah ${langkahSimpan}.`);
  console.log(`Lanjutkan:  ${lanjut}`);
  process.exit(0);
}
pelacak.tutup("selesai", alasanSelesai || `selesai, loss val terbaik ${terbaik.toFixed(4)}`);

console.log(`\nSelesai dalam ${Math.round(detik() / 60)} menit. Loss val terbaik ${terbaik.toFixed(4)} (perplexity ${Math.exp(terbaik).toFixed(1)}). Langkah bertanda * = disimpan.`);
console.log(mode === "chat"
  ? `Coba ngobrol:  bun run chat-gpt -- --name ${nama}`
  : `Coba:  bun run generate -- --name ${nama} "awal kalimat"`);
