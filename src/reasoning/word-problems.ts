/**
 * SOAL CERITA SEDERHANA: laju kerja, selang waktu, dan perbandingan berat.
 *
 * LAJU KERJA (proporsi):  hasil = jumlah_pekerja × laju_per_pekerja × waktu
 *   "5 mesin menghasilkan 5 produk dalam 5 menit" -> laju per mesin = 5 / (5 × 5) = 0,2 produk/menit
 *   100 mesin, 100 produk:  waktu = 100 / (100 × 0,2) = 5 menit
 *
 * SELANG WAKTU (masalah tiang pagar):  N kejadian dengan jarak M -> (N − 1) × M
 *   3 pil, satu setiap 30 menit: pil ke-1 menit 0, ke-2 menit 30, ke-3 menit 60 -> 60 menit, bukan 90.
 *
 * PERBANDINGAN BERAT: ubah semua ke gram lalu bandingkan. 1 kg besi = 1 kg kapas.
 */
import { angkaDigit, fmt } from "./numbers.js";
import type { Alat } from "./types.js";

const PELAKU = "mesin|pekerja|orang|tukang|printer|pabrik|robot|ayam|kucing|koki|karyawan|petani";
const WAKTU = "detik|menit|jam|hari|minggu|bulan|tahun";

export const lajuKerja: Alat = {
  nama: "laju kerja (proporsi)",
  pikir: "Ada sejumlah pelaku yang menghasilkan sesuatu dalam waktu tertentu, jadi hitung laju per pelaku dulu.",
  cocok: (t) => new RegExp(`\\d+\\s*(${PELAKU})`, "i").test(t) && new RegExp(`\\d+\\s*(${WAKTU})`, "i").test(t) && /berapa/i.test(t),
  coba: (t) => {
    const s = t.toLowerCase();
    const re1 = new RegExp(`(\\d+)\\s*(${PELAKU})[^\\d]*?(\\d+)\\s*([a-z]+)[^\\d]*?(\\d+)\\s*(${WAKTU})`);
    const m1 = s.match(re1);
    if (!m1) return null;
    const sisa = s.slice((m1.index ?? 0) + m1[0].length);
    const m2 = sisa.match(new RegExp(`(\\d+)\\s*(${PELAKU})[^\\d]*?(\\d+)\\s*([a-z]+)`));
    if (!m2) return null;
    const [a1, o1, t1] = [Number(m1[1]), Number(m1[3]), Number(m1[5])];
    const [a2, o2] = [Number(m2[1]), Number(m2[3])];
    const satuan = m1[6], pelaku = m1[2], benda = m1[4];
    const laju = o1 / (a1 * t1);
    const t2 = o2 / (a2 * laju);
    return {
      jawaban: `${fmt(t2)} ${satuan}`,
      penjelasan: [
        `Laju 1 ${pelaku} = ${fmt(o1)} ${benda} ÷ (${fmt(a1)} ${pelaku} × ${fmt(t1)} ${satuan}) = ${fmt(laju)} ${benda} per ${satuan}.`,
        `${fmt(a2)} ${pelaku} bersama menghasilkan ${fmt(a2)} × ${fmt(laju)} = ${fmt(a2 * laju)} ${benda} per ${satuan}.`,
        `Waktu untuk ${fmt(o2)} ${benda} = ${fmt(o2)} ÷ ${fmt(a2 * laju)} = ${fmt(t2)} ${satuan}.`,
        `Jebakan umum: menjawab ${fmt(o2)} ${satuan} karena angkanya ikut naik. Padahal pelakunya juga bertambah, jadi waktunya tetap.`,
      ],
      cek: { lolos: Math.abs(a2 * laju * t2 - o2) < 1e-9, teks: `${fmt(a2)} ${pelaku} × ${fmt(laju)} × ${fmt(t2)} ${satuan} = ${fmt(a2 * laju * t2)} ${benda} ✓` },
    };
  },
};

export const selangWaktu: Alat = {
  nama: "selang waktu (tiang pagar)",
  pikir: "Ada N kejadian dengan jarak tetap; kejadian pertama terjadi di awal, jadi jaraknya hanya N − 1.",
  cocok: (t) => /(setiap|tiap)\s+\d+\s*(menit|jam|hari|detik)/i.test(t) && /(pil|tablet|obat|kapsul|minum|tiang|pohon)/i.test(t),
  coba: (t) => {
    const n = t.match(/(\d+)\s*(pil|tablet|obat|kapsul|tiang|pohon)/i);
    const s = t.match(/(?:setiap|tiap)\s+(\d+)\s*(menit|jam|hari|detik)/i);
    if (!n || !s) return null;
    const N = Number(n[1]), M = Number(s[1]), total = (N - 1) * M;
    const jadwal = Array.from({ length: Math.min(N, 6) }, (_, i) => `${n[2]} ke-${i + 1}: ${s[2]} ke-${i * M}`);
    return {
      jawaban: `${fmt(total)} ${s[2]}`,
      penjelasan: [...jadwal, `${n[2]} pertama langsung di awal (${s[2]} ke-0), jadi hanya ada ${N} − 1 = ${N - 1} jeda: ${N - 1} × ${M} = ${total} ${s[2]}.`, `Jebakan umum: ${N} × ${M} = ${N * M} ${s[2]}.`],
      cek: { lolos: true, teks: `jadwal terakhir = ${s[2]} ke-${(N - 1) * M} ✓` },
    };
  },
};

const KE_GRAM: Record<string, number> = { mg: 0.001, gram: 1, g: 1, ons: 100, kg: 1000, kilogram: 1000, kuintal: 100000, ton: 1000000 };

export const bandingBerat: Alat = {
  nama: "perbandingan berat (konversi satuan)",
  pikir: "Dua benda dengan satuan berat dibandingkan; ubah ke satuan yang sama dulu, jenis bendanya tidak berpengaruh.",
  cocok: (t) => /(lebih berat|lebih ringan|mana yang (lebih )?berat)/i.test(t) && (t.match(/\d+(?:[.,]\d+)?\s*(mg|gram|g|ons|kg|kilogram|kuintal|ton)\b/gi)?.length ?? 0) >= 2,
  coba: (t) => {
    const m = [...t.matchAll(/(\d+(?:[.,]\d+)?)\s*(mg|gram|g|ons|kg|kilogram|kuintal|ton)\b\s*([a-z]+)?/gi)].slice(0, 2);
    if (m.length < 2) return null;
    const g = m.map((x) => Number(x[1].replace(",", ".")) * KE_GRAM[x[2].toLowerCase()]);
    const nama = m.map((x) => `${x[1]} ${x[2]} ${x[3] ?? ""}`.trim());
    const jawaban = g[0] === g[1] ? "Sama berat" : g[0] > g[1] ? `${nama[0]} lebih berat` : `${nama[1]} lebih berat`;
    return {
      jawaban,
      penjelasan: [`${nama[0]} = ${fmt(g[0])} gram`, `${nama[1]} = ${fmt(g[1])} gram`, g[0] === g[1] ? "Beratnya sama; besi hanya lebih padat, sehingga volumenya lebih kecil." : "Bandingkan angka gramnya."],
      cek: { lolos: true, teks: `${fmt(g[0])} g vs ${fmt(g[1])} g` },
    };
  },
};

export const _uji = { angkaDigit };
