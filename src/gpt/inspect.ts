/**
 * BAHAN BELAJAR: menyusun simulasi perhitungan GPT dengan ANGKA SUNGGUHAN dari model,
 * supaya orang awam bisa melihat apa yang terjadi di dalamnya.
 *
 *   1. Token -> angka:      embedding token + positional encoding = vektor masukan
 *   2. Attention (1 token): skor q·k, dibagi √d, softmax -> seberapa besar perhatian ke tiap token
 *   3. Peta attention:      matriks bobot perhatian semua token, per lapisan & head
 *   4. Logits -> peluang:   z, z/suhu, e^(z/suhu), dibagi jumlah -> peluang token berikutnya
 */
import { END, type Tokenizer } from "./bpe.js";
import type { GPT } from "./model.js";

const r = (x: number, d = 3) => Number(x.toFixed(d));

export interface SimulasiGPT {
  D: number; H: number; dh: number; jendela: string[];
  vektor: { token: string; embedding: number[]; posisi: number[]; masukan: number[] }[];
  attentionSatu: { lapisan: number; head: number; dari: string; q: number[]; baris: { token: string; k: number[]; qk: number; skala: number; p: number }[] };
  peta: number[][][][];   // [lapisan][head][i][j]
  peluang: { suhu: number; baris: { token: string; z: number; zSuhu: number; exp: number; p: number }[]; jumlahExp: number; sisaP: number };
}

export function simulasiGPT(model: GPT, tok: Tokenizer, prompt: number[], balasan: number[], suhu: number): SimulasiGPT {
  const V = model.c.vocab;
  // --- 4. Logits -> peluang untuk token PERTAMA balasan (posisi terakhir prompt)
  const promptJendela = prompt.slice(-model.c.context);
  model.forward(promptJendela);
  const z = Array.from(model.logits.subarray((promptJendela.length - 1) * V, promptJendela.length * V));
  const T = Math.max(1e-4, suhu);
  const maks = Math.max(...z) / T;
  let jumlah = 0;
  for (const x of z) jumlah += Math.exp(x / T - maks);
  const urut = z.map((x, id) => ({ id, x })).sort((a, b) => b.x - a.x).slice(0, 6);
  const baris = urut.map(({ id, x }) => {
    const e = Math.exp(x / T - maks);
    return { token: tok.vocab[id], z: r(x), zSuhu: r(x / T), exp: r(e, 4), p: r(e / jumlah, 4) };
  });
  const peluang = { suhu, baris, jumlahExp: r(jumlah, 4), sisaP: r(1 - baris.reduce((s, b) => s + b.p, 0), 4) };

  // --- 1–3. Forward pada prompt + balasan, ambil jendela 16 token terakhir
  const idEnd = tok.idToken(END);
  const urutan = [...prompt, ...balasan.filter((id) => id !== idEnd)].slice(-model.c.context);
  const W = Math.min(16, urutan.length), awal = urutan.length - W;
  const d = model.intip(urutan, 8);
  const label = urutan.map((id) => tok.vocab[id]);
  const jendela = label.slice(awal);
  const vektor = jendela.map((token, i) => ({
    token,
    embedding: d.embedding[awal + i].map((x) => r(x)),
    posisi: d.posisi[awal + i].map((x) => r(x)),
    masukan: d.masukan[awal + i].map((x) => r(x)),
  }));

  // Simulasi attention untuk posisi AKHIR PROMPT (saat model memutuskan kata pertama balasan),
  // lapisan 1 head 1, terhadap maks 6 token sebelumnya
  const i = Math.min(promptJendela.length, urutan.length) - 1;
  const q = d.q0[i];
  const skalaAkar = Math.sqrt(d.dh);
  const indeksJ = Array.from({ length: Math.min(6, i + 1) }, (_, n) => i - n).reverse();
  const barisAtt = indeksJ.map((j) => {
    const k = d.k0[j];
    const qk = q.reduce((s, x, n) => s + x * k[n], 0);
    return { token: label[j], k: k.slice(0, 8).map((x) => r(x)), qk: r(qk), skala: r(qk / skalaAkar), p: r(d.attention[0][0][i][j], 4) };
  });

  const peta = d.attention.map((lap) => lap.map((m) => m.slice(awal).map((row) => row.slice(awal).map((x) => r(x, 3)))));
  return {
    D: d.D, H: d.H, dh: d.dh, jendela, vektor, peta, peluang,
    attentionSatu: { lapisan: 1, head: 1, dari: label[i], q: q.slice(0, 8).map((x) => r(x)), baris: barisAtt },
  };
}
