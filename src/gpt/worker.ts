/**
 * WORKER untuk training data-parallel.
 *
 * Satu batch dibagi ke beberapa worker (thread CPU). Setiap worker:
 *   - membaca bobot dari SharedArrayBuffer yang SAMA dengan thread utama
 *   - menghitung forward + backward untuk bagian batch-nya (dengan dropout sendiri)
 *   - menulis gradien ke SharedArrayBuffer miliknya sendiri
 * Thread utama lalu menjumlahkan gradien semua worker dan menjalankan Adam.
 * Karena Σ gradien bagian = gradien seluruh batch, hasilnya setara dengan satu thread.
 *
 * Mode teks : pekerjaan = posisi awal jendela di aliran token bersama
 * Mode chat : pekerjaan = nomor urutan percakapan (dikirim sekali saat init)
 * Mode span : pekerjaan = nomor contoh GPT pembaca (loss awal/akhir jawaban)
 */
import { GPT, type GPTConfig } from "./model.js";
import { Rng } from "./random.js";

declare const self: Worker;

let model: GPT;
let G: Float64Array;
let token: Int32Array;
let urutan: { x: Int32Array; y: Int32Array }[] = [];
let span: { x: Int32Array; batas: number; mulai: number; akhir: number }[] = [];

self.onmessage = (e: MessageEvent) => {
  const m = e.data;
  if (m.jenis === "init") {
    model = new GPT(m.config as GPTConfig, new Float64Array(m.bobot as SharedArrayBuffer));
    G = new Float64Array(m.gradien as SharedArrayBuffer);
    if (m.token) token = new Int32Array(m.token as SharedArrayBuffer);
    if (m.urutan) urutan = m.urutan;
    if (m.span) span = m.span;
    if (m.dropout > 0) model.aturDropout(m.dropout, new Rng(m.seed).next);
    self.postMessage({ jenis: "siap" });
  } else if (m.jenis === "hitung") {
    const T: number = m.T, skala: number = m.skala;
    G.fill(0);
    let loss = 0;
    for (const s of m.kerja as number[]) {
      if (span.length) { const u = span[s]; loss += model.langkahSpan(u.x, u.mulai, u.akhir, u.batas, G, skala); }
      else if (urutan.length) loss += model.langkah(urutan[s].x, urutan[s].y, G, skala);
      else loss += model.langkah(token.subarray(s, s + T), token.subarray(s + 1, s + T + 1), G, skala);
    }
    self.postMessage({ jenis: "selesai", loss });
  }
};
