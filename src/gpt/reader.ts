/**
 * GPT PEMBACA: membaca konteks + pertanyaan, lalu MENUNJUK rentang jawaban di konteks.
 *
 *   urutan token:  <|user|> pertanyaan <|bot|> konteks
 *                                        ^ posisi ini = "tidak ada jawaban"
 *   untuk setiap token t: skor_awal(t), skor_akhir(t)        (kepala span, lihat GPT.forwardSpan)
 *   p_awal = softmax(skor_awal), p_akhir = softmax(skor_akhir)   (hanya posisi <|bot|> dan sesudahnya)
 *   rentang terbaik = argmax p_awal(i) · p_akhir(j),  i ≤ j ≤ i + 30
 *   keyakinan = p(rentang) / (p(rentang) + p(tidak ada jawaban))
 *
 * Karena jawaban selalu DISALIN dari konteks, pembaca tidak bisa mengarang fakta baru.
 * Konteks panjang dipecah menjadi beberapa jendela kalimat yang muat di panjang konteks model.
 */
import { pecahKalimat } from "../answer/sentences.js";
import type { Tokenizer } from "./bpe.js";
import type { GPT } from "./model.js";

export interface UrutanPembaca { x: Int32Array; batas: number; mulai: number; akhir: number }
export const PANJANG_MAKS_JAWABAN = 30;

/** Token awal: <|user|> pertanyaan <|bot|>. Mengembalikan id dan posisi <|bot|>. */
function awalan(tok: Tokenizer, pertanyaan: string): { ids: number[]; batas: number } {
  const user = tok.idToken("<|user|>"), bot = tok.idToken("<|bot|>");
  if (bot === undefined) throw new Error("tokenizer tidak punya token <|bot|>");
  const ids = [...(user !== undefined ? [user] : []), ...tok.encode(pertanyaan.slice(0, 300))];
  return { ids: [...ids, bot], batas: ids.length };
}

/**
 * Ubah satu contoh (teks + posisi karakter jawaban) menjadi token + posisi token jawaban.
 * Konteks di-encode per bagian (sebelum | jawaban | sesudah) dan dicek sama dengan encode utuh,
 * supaya posisi token jawaban pasti benar. Mengembalikan null jika tidak muat / tidak sejajar.
 */
export function encodeContoh(tok: Tokenizer, pertanyaan: string, konteks: string, awal: number, jawaban: string, maksT: number, hurufKecil = false): UrutanPembaca | null {
  const kecil = (s: string) => (hurufKecil ? s.toLowerCase() : s);
  const { ids: a, batas } = awalan(tok, kecil(pertanyaan));
  const utuh = tok.encode(kecil(konteks));
  let mulai = batas, akhir = batas;
  if (awal >= 0) {
    let s = awal;
    if (s > 0 && konteks[s - 1] === " ") s--; // spasi depan menempel ke kata ("_12")
    const sebelum = tok.encode(kecil(konteks.slice(0, s)));
    const isi = tok.encode(kecil(konteks.slice(s, awal + jawaban.length)));
    const sesudah = tok.encode(kecil(konteks.slice(awal + jawaban.length)));
    const gabung = [...sebelum, ...isi, ...sesudah];
    if (gabung.length !== utuh.length || gabung.some((v, i) => v !== utuh[i]) || !isi.length) return null;
    mulai = batas + 1 + sebelum.length;
    akhir = mulai + isi.length - 1;
    if (akhir - mulai >= PANJANG_MAKS_JAWABAN) return null;
  }
  const x = [...a, ...utuh].slice(0, maksT);
  if (akhir >= x.length) return null;
  return { x: Int32Array.from(x), batas, mulai, akhir };
}

export interface HasilBaca {
  jawaban: string;               // rentang jawaban (teks asli konteks); "" jika tidak ada
  keyakinan: number;             // 0–1
  kalimat: string;               // kalimat konteks yang memuat jawaban
  jendela: string;               // teks jendela yang dibaca
  token: string[];               // token jendela (untuk peta panas)
  pAwal: number[]; pAkhir: number[]; // peluang per token jendela (indeks 0 = "tidak ada jawaban")
  pilih: [number, number] | null;    // indeks token jawaban di `token`
  jumlahJendela: number;
}

function softmaxDari(Z: ArrayLike<number>, kol: number, dari: number, T: number): number[] {
  let maks = -Infinity;
  for (let t = dari; t < T; t++) maks = Math.max(maks, Z[t * 2 + kol]);
  const e = Array.from({ length: T - dari }, (_, i) => Math.exp(Z[(dari + i) * 2 + kol] - maks));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** Pecah konteks menjadi jendela kalimat yang muat bersama pertanyaan. */
export function jendelaKonteks(tok: Tokenizer, konteks: string, ruang: number): string[] {
  const kal = konteks.split(/\n+/).flatMap((p) => { const k = pecahKalimat(p).map((x) => x.asli); return k.length ? k : [p]; }).filter((s) => s.trim());
  const hasil: string[] = [];
  let kini: string[] = [], n = 0;
  for (const k of kal) {
    const m = tok.encode(" " + k).length;
    if (n + m > ruang && kini.length) { hasil.push(kini.join(" ")); kini = []; n = 0; }
    if (m > ruang) { hasil.push(k); continue; } // kalimat sangat panjang: dipotong saat encode
    kini.push(k); n += m;
  }
  if (kini.length) hasil.push(kini.join(" "));
  return hasil;
}

/** Baca konteks dan cari rentang jawaban terbaik di semua jendela. */
export function baca(model: GPT, tok: Tokenizer, pertanyaan: string, konteks: string, hurufKecil = false): HasilBaca {
  const kecil = (s: string) => (hurufKecil ? s.toLowerCase() : s);
  const T = model.c.context;
  const { ids: a, batas } = awalan(tok, kecil(pertanyaan));
  const daftar = jendelaKonteks(tok, konteks, T - a.length - 1);
  let terbaik: HasilBaca | null = null;
  for (const j of daftar) {
    const ctx = tok.encode(kecil(j));
    const ids = [...a, ...ctx].slice(0, T);
    const Z = model.forwardSpan(ids);
    const pA = softmaxDari(Z, 0, batas, ids.length), pB = softmaxDari(Z, 1, batas, ids.length);
    let pRentang = 0, pilih: [number, number] | null = null;
    for (let i = 1; i < pA.length; i++)
      for (let k = i; k < Math.min(pB.length, i + PANJANG_MAKS_JAWABAN); k++) {
        const p = pA[i] * pB[k];
        if (p > pRentang) { pRentang = p; pilih = [i, k]; }
      }
    const pKosong = pA[0] * pB[0];
    const keyakinan = pRentang / (pRentang + pKosong + 1e-12);
    if (terbaik && keyakinan <= terbaik.keyakinan) continue;
    const token = ids.slice(batas + 1).map((id) => tok.decode([id]));
    let jawaban = "", kalimat = "";
    if (pilih) {
      // Teks jawaban = gabungan token rentang itu sendiri (bukan potongan teks asli berdasarkan hitungan
      // karakter: pra-tokenisasi bisa membuang karakter tertentu sehingga posisinya bergeser)
      jawaban = token.slice(pilih[0] - 1, pilih[1]).join("").trim().replace(/[,;:(]$/, "").trim();
      const rendah = jawaban.toLowerCase();
      kalimat = pecahKalimat(j).map((x) => x.asli).find((s) => s.toLowerCase().includes(rendah)) ?? j;
    }
    terbaik = { jawaban, keyakinan, kalimat, jendela: j, token: ["∅", ...token], pAwal: pA, pAkhir: pB, pilih, jumlahJendela: daftar.length };
  }
  return terbaik ?? { jawaban: "", keyakinan: 0, kalimat: "", jendela: "", token: [], pAwal: [], pAkhir: [], pilih: null, jumlahJendela: 0 };
}

/** F1 tingkat kata (seperti SQuAD) antara jawaban prediksi dan jawaban benar. */
export function f1Kata(pred: string, benar: string): number {
  const a = pred.toLowerCase().match(/[\p{L}\d]+/gu) ?? [], b = benar.toLowerCase().match(/[\p{L}\d]+/gu) ?? [];
  if (!a.length || !b.length) return a.length === b.length ? 1 : 0;
  const sisa = [...b];
  let sama = 0;
  for (const w of a) { const i = sisa.indexOf(w); if (i >= 0) { sama++; sisa.splice(i, 1); } }
  if (!sama) return 0;
  const p = sama / a.length, r = sama / b.length;
  return (2 * p * r) / (p + r);
}
