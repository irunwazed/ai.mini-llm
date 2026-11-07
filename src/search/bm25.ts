/**
 * BM25 dari nol: "training" = menghitung statistik kata dari seluruh potongan.
 *
 * Notasi:
 *   N        jumlah potongan (pasal)
 *   f(q,D)   berapa kali term q muncul di potongan D            (term frequency)
 *   n_q      berapa potongan yang mengandung q                   (document frequency)
 *   |D|      panjang potongan D (jumlah term),  avgdl = rata-rata |D|
 *
 * IDF  (Inverse Document Frequency) — kata langka lebih informatif:
 *   idf(q) = ln( 1 + (N - n_q + 0.5) / (n_q + 0.5) )
 *
 * Skor BM25 untuk pertanyaan Q dengan bobot term w_q (1 untuk kata asli, < 1 untuk perluasan):
 *   skor(D, Q) = Σ_q w_q · idf(q) · f(q,D)·(k1 + 1) / ( f(q,D) + k1·(1 - b + b·|D|/avgdl) )
 *
 *   k1 (≈1.5) = saturasi: muncul 10x tidak 10 kali lebih penting dari 1x
 *   b  (≈0.75)= normalisasi panjang: pasal panjang wajar memuat banyak kata
 *
 * Struktur data: inverted index  term -> [id, tf, id, tf, ...]
 */
import { keTerm } from "../text/normalize.js";
import type { Potongan } from "../types.js";

export interface Indeks {
  versi: 1;
  k1: number;
  b: number;
  N: number;
  avgdl: number;
  panjang: number[];
  kamus: Record<string, number[]>;
}

/**
 * Teks yang diindeks BM25: bab + label pasal + isi.
 * Judul peraturan SENGAJA tidak ikut: jika ikut, setiap pasal pendek dari peraturan
 * berjudul "Cuti PNS" akan menang untuk semua pertanyaan tentang cuti, walau isinya
 * hanya "Peraturan ini mulai berlaku pada tanggal diundangkan".
 * Judul dipakai terpisah sebagai penguat tingkat dokumen (lihat cari.ts).
 */
export function teksIndeks(p: Potongan): string {
  return `${p.bab} ${p.pasal} ${p.teks}`;
}

export function bangunIndeks(potongan: Potongan[], k1 = 1.5, b = 0.75): Indeks {
  const kamusMap = new Map<string, number[]>();
  const panjang: number[] = new Array(potongan.length);
  let total = 0;
  for (const p of potongan) {
    const terms = keTerm(teksIndeks(p));
    panjang[p.id] = terms.length;
    total += terms.length;
    const tf = new Map<string, number>();
    for (const t of terms) tf.set(t, (tf.get(t) ?? 0) + 1);
    for (const [t, f] of tf) {
      let daftar = kamusMap.get(t);
      if (!daftar) kamusMap.set(t, (daftar = []));
      daftar.push(p.id, f);
    }
  }
  return {
    versi: 1, k1, b,
    N: potongan.length,
    avgdl: total / Math.max(1, potongan.length),
    panjang,
    kamus: Object.fromEntries(kamusMap),
  };
}

export function idf(ind: Indeks, term: string): number {
  const nq = (ind.kamus[term]?.length ?? 0) / 2;
  return Math.log(1 + (ind.N - nq + 0.5) / (nq + 0.5));
}

/**
 * Rincian perhitungan BM25 untuk SATU potongan (bahan belajar):
 * untuk setiap term, tampilkan semua angka yang masuk ke rumus.
 */
export interface BarisBM25 { term: string; bobot: number; nq: number; idf: number; f: number; kontribusi: number }
export function rincianBM25(ind: Indeks, terms: Map<string, number>, id: number) {
  const panjangAsli = ind.panjang[id];
  const panjang = Math.max(panjangAsli, 0.5 * ind.avgdl);
  const norm = ind.k1 * (1 - ind.b + (ind.b * panjang) / ind.avgdl);
  const baris: BarisBM25[] = [];
  for (const [term, bobot] of terms) {
    const daftar = ind.kamus[term] ?? [];
    let f = 0;
    for (let i = 0; i < daftar.length; i += 2) if (daftar[i] === id) { f = daftar[i + 1]; break; }
    const nilaiIdf = idf(ind, term);
    baris.push({ term, bobot, nq: daftar.length / 2, idf: nilaiIdf, f, kontribusi: f ? (bobot * nilaiIdf * f * (ind.k1 + 1)) / (f + norm) : 0 });
  }
  return { N: ind.N, avgdl: ind.avgdl, panjang: panjangAsli, panjangEfektif: panjang, k1: ind.k1, b: ind.b, norm, baris, total: baris.reduce((s, x) => s + x.kontribusi, 0) };
}

/** Skor BM25 semua potongan. Hanya menyentuh potongan yang mengandung term pertanyaan. */
export function skorBM25(ind: Indeks, terms: Map<string, number>): Float64Array {
  const skor = new Float64Array(ind.N);
  for (const [q, bobot] of terms) {
    const daftar = ind.kamus[q];
    if (!daftar) continue;
    const w = bobot * idf(ind, q);
    for (let i = 0; i < daftar.length; i += 2) {
      const id = daftar[i], f = daftar[i + 1];
      // Panjang efektif minimal setengah rata-rata: pasal penunjuk yang sangat pendek
      // ("Tata cara ... tercantum dalam Lampiran") tidak boleh menang hanya karena pendek.
      const panjang = Math.max(ind.panjang[id], 0.5 * ind.avgdl);
      const norm = ind.k1 * (1 - ind.b + (ind.b * panjang) / ind.avgdl);
      skor[id] += (w * f * (ind.k1 + 1)) / (f + norm);
    }
  }
  return skor;
}
