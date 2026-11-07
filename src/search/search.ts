/**
 * PENCARIAN HIBRIDA: kata (BM25) + makna (embedding), digabung dengan Reciprocal Rank Fusion.
 *
 * 1. Perluasan pertanyaan. Setiap kata pertanyaan menjadi satu KELOMPOK term:
 *      kata asli + kata dasar + sinonim manual (data/synonyms.txt)
 *      + singkatan yang DIPELAJARI dari dokumen ("pns" <-> "pegawai negeri sipil")
 *    Ditambah term berbobot kecil (tidak dihitung dalam cakupan):
 *      + tetangga embedding word2vec (cos >= 0.6), bobot 0.3·cos
 *      + kata dari pertanyaan sebelumnya (percakapan lanjutan), bobot 0.4
 *
 * 2. Dua peringkat:
 *      r_kata  = peringkat BM25
 *      r_makna = peringkat cos(vek(pertanyaan), vek(pasal))
 *
 * 3. Reciprocal Rank Fusion (RRF):
 *      skor(D) = 1/(60 + r_kata(D)) + β/(60 + r_makna(D)),   β = 0.5
 *    RRF hanya memakai PERINGKAT, jadi tidak perlu menyamakan skala BM25 dan cosinus.
 *    Konstanta 60 meredam selisih antar peringkat atas, sehingga pasal yang bagus
 *    di kedua daftar menang atas pasal yang hanya juara di satu daftar.
 */
import type { Model } from "../model.js";
import { vektorTeks } from "../model.js";
import { tetangga } from "../training/word2vec.js";
import { keTerm, termKata, tokenisasi } from "../text/normalize.js";
import { padanan } from "../text/synonyms.js";
import type { Potongan } from "../types.js";
import { idf, skorBM25, teksIndeks } from "./bm25.js";

export interface KelompokKueri { kata: string; terms: string[]; bobot: number }
export interface Kueri { terms: Map<string, number>; kelompok: KelompokKueri[] }
export interface Hasil {
  potongan: Potongan;
  skor: number;     // BM25
  cos: number;      // kemiripan makna (0 jika tanpa embedding)
  cakupan: number;  // 0..1 bagian bobot kata pertanyaan yang ada di pasal
  kekuatan: number; // Σ idf kata yang cocok / ln(1+N): seberapa KHUSUS kata yang cocok
}

const BETA = 0.3;
const RRF_K = 60;

/**
 * Kata "pembingkai" menjelaskan BENTUK pertanyaan, bukan topiknya:
 * "apa SYARAT cuti besar" -> topiknya "cuti besar". Kata ini tetap dicari (bobot 0.5),
 * tetapi tidak dihitung dalam cakupan, agar pasal yang tidak menulis kata "syarat"
 * tetap dianggap menjawab.
 */
const PEMBINGKAI = new Set(
  "syarat persyaratan cara prosedur mekanisme tata langkah ketentuan aturan peraturan hukum dasar mendapat mendapatkan memperoleh diperoleh lama".split(" "),
);

/**
 * Pasal "rujukan": ketentuan peralihan/penutup, atau pasal yang isinya terutama
 * menyebut judul peraturan lain ("SE Nomor 01/SE/1977 tentang ... Cuti PNS dicabut").
 * Kata-katanya cocok dengan banyak pertanyaan tetapi hampir tidak pernah menjadi jawaban.
 */
const rujukanCache = new WeakMap<Potongan, boolean>();
function adalahRujukan(p: Potongan): boolean {
  let r = rujukanCache.get(p);
  if (r === undefined) {
    const judulLain = p.teks.match(/\b(peraturan|undang-undang|surat edaran|keputusan)\b[^.;]{0,120}?\bnomor\s+\S+\s+tahun\s+\d{4}/gi)?.length ?? 0;
    r = /ketentuan (penutup|peralihan)/i.test(p.bab) || judulLain >= 2 || (judulLain === 1 && p.teks.split(" ").length < 60);
    rujukanCache.set(p, r);
  }
  return r;
}

const tambah = (m: Map<string, number>, t: string, w: number) => m.set(t, Math.max(m.get(t) ?? 0, w));

export function analisisKueri(model: Model, pertanyaan: string, konteks = ""): Kueri {
  const ind = model.indeks;
  const kelompok: KelompokKueri[] = [];
  const terms = new Map<string, number>();
  const token = tokenisasi(pertanyaan);
  const sudah = new Set<string>();

  // Bobot padanan: 0.6 dibagi jumlah katanya. "pns" -> "pegawai negeri sipil" memberi
  // 3 kata × 0.2, bukan 3 kata × 1, agar satu kata pertanyaan tidak berbobot tiga kali lipat.
  const padananBerbobot = new Map<string, number>();
  token.forEach((kata) => {
    const inti = termKata(kata);
    if (inti.length === 0 || sudah.has(kata)) return;
    sudah.add(kata);
    const bingkai = PEMBINGKAI.has(kata);
    const g = new Set(inti);
    for (const t of inti) tambah(terms, t, bingkai ? 0.5 : 1);
    for (const p of [...padanan(kata), ...(model.istilah.singkatan[kata] ?? [])]) {
      const ts = keTerm(p);
      const n = Math.max(1, tokenisasi(p).length);
      for (const t of ts) { g.add(t); tambah(padananBerbobot, t, (bingkai ? 0.3 : 0.6) / n); }
    }
    kelompok.push({ kata, terms: [...g], bobot: bingkai ? 0 : Math.max(...[...g].map((t) => idf(ind, t))) });
  });
  for (const [t, w] of padananBerbobot) tambah(terms, t, w);

  // Frasa 2-4 kata yang punya singkatan ("aparatur sipil negara" -> "asn") atau sinonim
  // di data/synonyms.txt ("naik pangkat" -> "kenaikan pangkat", "cuti hamil" -> "cuti melahirkan")
  for (let n = 2; n <= 4; n++)
    for (let i = 0; i + n <= token.length; i++) {
      const frasa = token.slice(i, i + n).join(" ");
      for (const p of [...(model.istilah.singkatan[frasa] ?? []), ...padanan(frasa)]) {
        const ts = keTerm(p), m = Math.max(1, tokenisasi(p).length);
        for (const t of ts) tambah(terms, t, 0.6 / m);
      }
    }

  // Tetangga embedding (makna mirip yang dipelajari word2vec)
  if (model.emb)
    for (const g of kelompok)
      for (const t of g.terms.filter((x) => x.startsWith("~")))
        for (const n of tetangga(model.emb, model.embIdx, t, 3)) if (n.cos >= 0.75) tambah(terms, n.kata, 0.3 * n.cos);

  // Konteks percakapan sebelumnya
  for (const t of keTerm(konteks)) tambah(terms, t, 0.4);

  return { terms, kelompok };
}

function peringkat(skor: ArrayLike<number>, n: number, minimal = 0): number[] {
  const id: number[] = [];
  for (let i = 0; i < skor.length; i++) if (skor[i] > minimal) id.push(i);
  return id.sort((a, b) => skor[b] - skor[a]).slice(0, n);
}

/**
 * Dua ukuran kecocokan pasal p dengan pertanyaan:
 *   cakupan  = Σ bobot kelompok yang cocok / Σ bobot semua kelompok        (relatif, 0..1)
 *   kekuatan = Σ bobot kelompok yang cocok / ln(1 + N)                      (absolut)
 * Cakupan saja tidak cukup: "siapa presiden pertama indonesia" bisa cocok 100%
 * dengan kata-kata umum di header peraturan. Kekuatan yang rendah menandakan
 * yang cocok hanya kata-kata umum, sehingga jawaban ditolak.
 */
/**
 * Topik judul = bagian "tentang ..." saja. Bagian "Peraturan Pemerintah Republik Indonesia"
 * ada di semua judul, sehingga kata "presiden" dan "indonesia" akan selalu cocok.
 */
function topikJudul(model: Model, p: Potongan): string {
  return model.info[p.file]?.tentang || "";
}

export function nilaiCocok(model: Model, q: Kueri, p: Potongan): { cakupan: number; kekuatan: number } {
  const total = q.kelompok.reduce((s, g) => s + g.bobot, 0);
  if (total === 0) return { cakupan: 0, kekuatan: 0 };
  // Topik judul ikut ("cuti PNS" cocok dengan peraturan tentang cuti PNS)
  const ada = new Set(keTerm(topikJudul(model, p) + " " + teksIndeks(p)));
  const cocok = q.kelompok.filter((g) => g.terms.some((t) => ada.has(t))).reduce((s, g) => s + g.bobot, 0);
  return { cakupan: cocok / total, kekuatan: cocok / Math.log(1 + model.indeks.N) };
}

export function cari(model: Model, pertanyaan: string, opsi: { k?: number; konteks?: string } = {}): { hasil: Hasil[]; kueri: Kueri } {
  const k = opsi.k ?? 5;
  const kueri = analisisKueri(model, pertanyaan, opsi.konteks);
  if (kueri.kelompok.length === 0) return { hasil: [], kueri };

  // Peringkat kata
  const bm = skorBM25(model.indeks, kueri.terms);
  // Bagian yang jarang menjadi jawaban diturunkan bobotnya:
  //   Pembukaan/Penutup (menimbang, mengingat, pengesahan) hanya mendaftar judul aturan lain
  //   Penjelasan menafsirkan pasal, bukan aturannya sendiri
  model.potongan.forEach((p, i) => {
    if (p.pasal.startsWith("Pembukaan") || p.pasal.startsWith("Penutup")) bm[i] *= 0.3;
    else if (adalahRujukan(p)) bm[i] *= 0.5;
    else if (p.pasal.startsWith("Penjelasan")) bm[i] *= 0.85;
  });
  // Penguat judul (tingkat dokumen): pasal dari peraturan yang judulnya memuat kata
  // pertanyaan dinaikkan:  skor × (1 + 0.5 · cakupanJudul)
  const totalBobot = kueri.kelompok.reduce((s, g) => s + g.bobot, 0) || 1;
  const cakJudul = new Map<string, number>();
  model.potongan.forEach((p, i) => {
    if (bm[i] === 0) return;
    let c = cakJudul.get(p.file);
    if (c === undefined) {
      const ada = new Set(keTerm(topikJudul(model, p)));
      c = kueri.kelompok.filter((g) => g.terms.some((t) => ada.has(t))).reduce((s, g) => s + g.bobot, 0) / totalBobot;
      cakJudul.set(p.file, c);
    }
    bm[i] *= 1 + 0.5 * c;
  });
  const rKata = peringkat(bm, 100);

  // Peringkat makna
  const cos = new Float32Array(model.indeks.N);
  let rMakna: number[] = [];
  if (model.emb && model.vekPasal) {
    const qv = new Float32Array(model.emb.dim);
    if (vektorTeks(model.emb, model.embIdx, model.indeks, [...kueri.terms.keys()].filter((t) => t.startsWith("~")), qv)) {
      const d = model.emb.dim, V = model.vekPasal;
      for (let i = 0; i < model.indeks.N; i++) {
        let s = 0;
        for (let j = 0; j < d; j++) s += qv[j] * V[i * d + j];
        cos[i] = s;
      }
      rMakna = peringkat(cos, 30, 0.5);
    }
  }

  // Reciprocal Rank Fusion
  const rrf = new Map<number, number>();
  rKata.forEach((id, r) => rrf.set(id, (rrf.get(id) ?? 0) + 1 / (RRF_K + r + 1)));
  rMakna.forEach((id, r) => rrf.set(id, (rrf.get(id) ?? 0) + BETA / (RRF_K + r + 1)));
  const urut = [...rrf].sort((a, b) => b[1] - a[1]).slice(0, k);

  return {
    kueri,
    hasil: urut.map(([id]) => ({
      potongan: model.potongan[id],
      skor: bm[id],
      cos: cos[id],
      ...nilaiCocok(model, kueri, model.potongan[id]),
    })),
  };
}
