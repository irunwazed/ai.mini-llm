/**
 * TRAINING GPT PEMBACA: GPT pretrain + kepala span (awal/akhir jawaban), dilatih dari data otomatis.
 *
 * Pakai:
 *   bun run make-reader-data                                  (sekali, membuat data/reader-id.jsonl)
 *   bun run train-reader -- --from wiki --name reader         (±30 menit)
 *
 * Ctrl+C = berhenti dengan aman (checkpoint terbaik disimpan). Pantau: bun run status -- --ikuti
 *
 * Opsi (bawaan): --from (wiki) --name (reader) --data (data/reader-id.jsonl) --minutes (30) --steps
 *   --batch (16) --lr (0.001) --warmup (100) --eval-every (100) --patience (6) --dropout (0.1)
 *   --context (256) --workers (CPU − 2) --seed (1)
 *
 * Langkah:
 *   1. salin semua bobot model pretrain (embedding, attention, FFN) -> pengetahuan bahasa dipakai ulang
 *   2. konteks diperpanjang ke 256 token: positional encoding sinusoidal DIHITUNG (bukan dipelajari),
 *      jadi bobot lama tetap berlaku untuk posisi baru
 *   3. kepala span W_span (D×2) diinisialisasi acak, lalu seluruh model dilatih (fine-tune)
 *   4. setiap evaluasi: loss val, EM (persis sama), F1 (kata), akurasi "tidak ada jawaban"
 *   5. terakhir: ambang keyakinan dikalibrasi di data val dan disimpan di gpt.json
 */
import { cpus } from "node:os";
import { readFileSync } from "node:fs";
import { Tokenizer } from "../gpt/bpe.js";
import { muatGPT, simpanGPT, type MetaGPT, type Riwayat } from "../gpt/checkpoint.js";
import { GPT, susunParameter, type GPTConfig } from "../gpt/model.js";
import { Adam, jadwalLR } from "../gpt/optimizer.js";
import { Rng } from "../gpt/random.js";
import { bagiData, type ContohPembaca } from "../gpt/reader-data.js";
import { baca, encodeContoh, f1Kata, type UrutanPembaca } from "../gpt/reader.js";
import { Pelacak } from "../gpt/progress.js";
import { LOKASI, lokasiBaca } from "../paths.js";
import { argumen } from "./args.js";

const { opsi } = argumen();
const angka = (k: string, b: number) => (opsi[k] !== undefined ? Number(opsi[k]) : b);
const dari = opsi.from ?? "wiki", nama = opsi.name ?? "reader";
const mulai = performance.now();
const detik = () => (performance.now() - mulai) / 1000;

// 1. Model asal + config baru -----------------------------------------------------------
const asal = muatGPT(dari);
const tok = new Tokenizer(asal.meta.bpe);
const hurufKecil = asal.meta.hurufKecil;
const config: GPTConfig = { ...asal.meta.config, context: angka("context", 256), spanHead: true };
const { ukuran } = susunParameter(config);
const bobotSAB = new SharedArrayBuffer(ukuran * 8);
const P = new Float64Array(bobotSAB);
const model = new GPT(config, P);
{
  const acak = new GPT(config, undefined, angka("seed", 1)); // inisialisasi lengkap (termasuk kepala span)
  P.set(acak.P);
  const lama = new GPT(asal.meta.config, asal.P);
  let disalin = 0;
  for (const [n, t] of lama.tensors()) {
    const baru = model.tensors().get(n);
    if (!baru || baru.rows !== t.rows || baru.cols !== t.cols) continue;
    P.set(asal.P.subarray(t.off, t.off + t.rows * t.cols), baru.off);
    disalin++;
  }
  console.log(`[1/4] Bobot disalin dari "${dari}" (${disalin} tensor). Kepala span baru: W_span ${config.dim}×2.`);
}

// 2. Data -------------------------------------------------------------------------------
const data: ContohPembaca[] = readFileSync(opsi.data ?? lokasiBaca(LOKASI.pembaca), "utf8").split(/\r?\n/).filter((l) => l.trim()).map((l) => JSON.parse(l));
const { train: dTrain, val: dVal } = bagiData(data);
const T = config.context;
const enc = (d: ContohPembaca) => encodeContoh(tok, d.pertanyaan, d.konteks, d.awal, d.jawaban, T, hurufKecil);
const train = dTrain.map(enc).filter((u): u is UrutanPembaca => !!u);
const valPasang = dVal.map((d) => ({ d, u: enc(d) })).filter((x): x is { d: ContohPembaca; u: UrutanPembaca } => !!x.u);
const valUji = valPasang.filter((_, i) => i % Math.max(1, Math.floor(valPasang.length / 400)) === 0).slice(0, 400);
console.log(`[2/4] ${train.length} contoh latih, ${valPasang.length} val (dibuang ${data.length - train.length - valPasang.length} yang tidak muat/sejajar)`);

// 3. Worker paralel ---------------------------------------------------------------------
const lrMaks = angka("lr", 1e-3), dropout = angka("dropout", 0.1);
const adam = new Adam(model, { lr: lrMaks, weightDecay: angka("weight-decay", 0.1) });
const batch = angka("batch", 16);
const nPekerja = Math.max(1, Math.min(angka("workers", Math.max(1, cpus().length - 2)), batch));
const pekerja: { w: Worker; G: Float64Array }[] = [];
if (nPekerja > 1) {
  await Promise.all(Array.from({ length: nPekerja }, (_, i) => {
    const gSAB = new SharedArrayBuffer(ukuran * 8);
    const w = new Worker(new URL("../gpt/worker.ts", import.meta.url).href);
    pekerja.push({ w, G: new Float64Array(gSAB) });
    return new Promise<void>((ok) => {
      w.onmessage = () => ok();
      w.postMessage({ jenis: "init", config, bobot: bobotSAB, gradien: gSAB, span: train, dropout, seed: angka("seed", 1) * 1000 + i });
    });
  }));
} else if (dropout > 0) model.aturDropout(dropout, new Rng(angka("seed", 1) + 7).next);
console.log(`[3/4] ${ukuran.toLocaleString("id-ID")} parameter, konteks ${T}, ${nPekerja} worker, batch ${batch}, lr ${lrMaks}, dropout ${dropout}`);

const G = new Float64Array(ukuran);
async function hitungBatch(kerja: number[], skala: number): Promise<number> {
  G.fill(0);
  if (!pekerja.length) return kerja.reduce((s, i) => s + model.langkahSpan(train[i].x, train[i].mulai, train[i].akhir, train[i].batas, G, skala), 0);
  const bagian = pekerja.map((_, i) => kerja.filter((_, j) => j % pekerja.length === i));
  const hasil = await Promise.all(pekerja.map(({ w }, i) => new Promise<number>((ok) => {
    w.onmessage = (e) => ok(e.data.loss);
    w.postMessage({ jenis: "hitung", kerja: bagian[i], T, skala });
  })));
  for (const { G: g } of pekerja) for (let i = 0; i < ukuran; i++) G[i] += g[i];
  return hasil.reduce((a, b) => a + b, 0);
}

/** Ukur val: loss, EM, F1, akurasi "tidak ada jawaban" pada ambang 0,5; kumpulkan keyakinan untuk kalibrasi. */
function ukurVal() {
  model.aturDropout(0);
  let loss = 0, em = 0, f1 = 0, nJawab = 0, benarKosong = 0, nKosong = 0;
  const kal: { yakin: number; benar: boolean; adaJawaban: boolean }[] = [];
  for (const { d, u } of valUji) {
    loss += model.langkahSpan(u.x, u.mulai, u.akhir, u.batas, null, 1) / 2;
    const h = baca(model, tok, d.pertanyaan, d.konteks, hurufKecil);
    const adaJawaban = d.awal >= 0;
    const f = adaJawaban ? f1Kata(h.jawaban, d.jawaban) : 0;
    kal.push({ yakin: h.keyakinan, benar: f >= 0.5, adaJawaban });
    if (adaJawaban) { nJawab++; f1 += f; if (h.jawaban.toLowerCase() === d.jawaban.toLowerCase()) em++; }
    else { nKosong++; if (h.keyakinan < 0.5) benarKosong++; }
  }
  if (!pekerja.length && dropout > 0) model.aturDropout(dropout);
  return { loss: loss / valUji.length, em: em / Math.max(1, nJawab), f1: f1 / Math.max(1, nJawab), kosong: benarKosong / Math.max(1, nKosong), kal };
}

/**
 * Kalibrasi ambang: pilih ambang yang memaksimalkan "jawaban benar + penolakan benar".
 * Di atas ambang -> pakai jawaban pembaca; di bawahnya -> anggap "tidak ada jawaban" (fallback).
 */
function kalibrasi(kal: { yakin: number; benar: boolean; adaJawaban: boolean }[]): number {
  let terbaik = 0.5, skorTerbaik = -1;
  for (let a = 0.3; a <= 0.95; a += 0.05) {
    const s = kal.filter((k) => (k.yakin >= a ? k.adaJawaban && k.benar : !k.adaJawaban || !k.benar)).length;
    if (s > skorTerbaik) { skorTerbaik = s; terbaik = Number(a.toFixed(2)); }
  }
  return terbaik;
}

// 4. Training loop ----------------------------------------------------------------------
const setiap = angka("eval-every", 100), patience = angka("patience", 6);
const anggaran = angka("minutes", 30) * 60;
let langkahTotal: number | null = opsi.steps ? Number(opsi.steps) : null;
const warmup = angka("warmup", 100);
const rng = new Rng(angka("seed", 1));
const riwayat: Riwayat[] = [];
let terbaik = Infinity, tanpaKemajuan = 0, jumlahLoss = 0, nLoss = 0;
const mulaiLatih = performance.now();
let ringkas = { em: 0, f1: 0, kosong: 0, ambang: 0.5 };
const meta = (langkah: number): MetaGPT => ({
  mode: "pembaca", dasar: dari, config, bpe: asal.meta.bpe, hurufKecil, langkah, lossValTerbaik: terbaik, riwayat,
  data: opsi.data ?? lokasiBaca(LOKASI.pembaca), jumlahToken: train.reduce((s, u) => s + u.x.length, 0), waktu: new Date().toISOString(),
  pembaca: { ambang: ringkas.ambang, em: ringkas.em, f1: ringkas.f1, akurasiKosong: ringkas.kosong, contohVal: valUji.length },
});

const v0 = ukurVal();
console.log(`[4/4] Training ${langkahTotal ? `${langkahTotal} langkah` : `±${Math.round(anggaran / 60)} menit`}; sebelum latih: loss val ${v0.loss.toFixed(3)}, EM ${(100 * v0.em).toFixed(1)}%, F1 ${(100 * v0.f1).toFixed(1)}%\n`);
console.log("langkah | loss train | loss val |   EM  |   F1  | tolak benar");
console.log("        (Ctrl+C = berhenti dengan aman & simpan · pantau dari terminal lain: bun run status -- --ikuti)");
const pelacak = new Pelacak("train-reader", nama, { total: langkahTotal });
let dihentikan = 0;
for (let langkah = 1; langkahTotal === null || langkah <= langkahTotal; langkah++) {
  const kerja = Array.from({ length: batch }, () => rng.int(train.length));
  jumlahLoss += (await hitungBatch(kerja, 1 / batch)) / batch / 2;
  nLoss++;
  adam.langkah(G, langkah, jadwalLR(langkah, langkahTotal ?? 1e9, lrMaks, warmup));
  pelacak.langkah(langkah, { lossTrain: jumlahLoss / nLoss, kecepatan: batch / Math.max(1e-3, (performance.now() - mulaiLatih) / 1000 / langkah) });
  if (langkahTotal === null && langkah === 20) {
    const perLangkah = (performance.now() - mulaiLatih) / 1000 / 20;
    langkahTotal = Math.max(setiap, Math.floor(anggaran / (perLangkah * 1.12))); // ±10% waktu untuk evaluasi
    console.log(`        (±${perLangkah.toFixed(2)} detik/langkah -> total ${langkahTotal} langkah)`);
    pelacak.setTotal(langkahTotal);
  }
  if (langkah % setiap === 0 || langkah === langkahTotal || pelacak.berhenti) {
    const v = ukurVal();
    pelacak.evaluasi(langkah, jumlahLoss / nLoss, v.loss, Math.min(terbaik, v.loss));
    riwayat.push({ langkah, lossTrain: jumlahLoss / nLoss, lossVal: v.loss, detik: detik() });
    const lebihBaik = v.loss < terbaik - 1e-3;
    console.log(`${String(langkah).padStart(7)} | ${(jumlahLoss / nLoss).toFixed(4).padStart(10)} | ${v.loss.toFixed(4).padStart(8)} | ${(100 * v.em).toFixed(1).padStart(5)} | ${(100 * v.f1).toFixed(1).padStart(5)} | ${(100 * v.kosong).toFixed(1).padStart(6)}%${lebihBaik ? " *" : ""}`);
    if (lebihBaik) {
      terbaik = v.loss; tanpaKemajuan = 0;
      ringkas = { em: v.em, f1: v.f1, kosong: v.kosong, ambang: kalibrasi(v.kal) };
      simpanGPT(nama, meta(langkah), P);
    } else if (++tanpaKemajuan >= patience) { console.log(`\nEarly stopping (loss val tidak membaik ${patience} evaluasi).`); break; }
    jumlahLoss = 0; nLoss = 0;
    if (pelacak.berhenti) { dihentikan = langkah; break; }
  }
}
for (const { w } of pekerja) w.terminate();
pelacak.tutup(dihentikan ? "berhenti" : "selesai", dihentikan ? `dihentikan di langkah ${dihentikan}` : `F1 ${(100 * ringkas.f1).toFixed(1)}%`);
if (dihentikan) console.log(`\nDihentikan dengan aman di langkah ${dihentikan}; checkpoint terbaik sudah tersimpan. Lanjutkan dengan: bun run train-reader -- --from ${nama} --name ${nama}-lanjut`);
console.log(`\nSelesai dalam ${Math.round(detik() / 60)} menit. Terbaik: loss val ${terbaik.toFixed(3)}, EM ${(100 * ringkas.em).toFixed(1)}%, F1 ${(100 * ringkas.f1).toFixed(1)}%, ambang keyakinan ${ringkas.ambang}.`);
console.log(`Uji:  bun run eval-reader -- --name ${nama}`);
