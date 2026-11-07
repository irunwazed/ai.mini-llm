/**
 * KALKULATOR EKSPRESI dari nol (recursive descent parser), dengan langkah hitung.
 *
 * Tata bahasa (dari prioritas terendah ke tertinggi):
 *   ekspresi = suku (('+' | '−') suku)*
 *   suku     = pangkat (('×' | '÷' | perkalian-tersirat) pangkat)*     <- kiri ke kanan
 *   pangkat  = unari ('^' pangkat)?                                    <- kanan ke kiri
 *   unari    = '−' unari | dasar
 *   dasar    = angka | '(' ekspresi ')'
 * Perkalian tersirat "2(2+2)" diperlakukan sama dengan "2 × (2+2)", sehingga
 *   8 ÷ 2(2+2) = (8 ÷ 2) × 4 = 16     (aturan kiri-ke-kanan yang lazim)
 * Penulisan seperti itu AMBIGU: sebagian orang membaca 8 ÷ (2(2+2)) = 1. Keduanya dilaporkan.
 */
import { fmt } from "./numbers.js";

type Simpul = { angka: number } | { op: string; kiri: Simpul; kanan: Simpul; tersirat?: boolean } | { neg: Simpul };

function token(teks: string): string[] {
  const t = teks.replace(/[×xX*]/g, "*").replace(/[÷:/]/g, "/").replace(/[−–]/g, "-").replace(/(\d)\.(\d{3})\b/g, "$1$2").replace(/,/g, ".");
  const out = t.match(/\d+(?:\.\d+)?|[+\-*/^()]/g);
  if (!out) throw new Error("tidak ada ekspresi");
  return out;
}

export function urai(teks: string): Simpul {
  const tk = token(teks);
  let i = 0;
  const lihat = () => tk[i];
  const ambil = () => tk[i++];
  const ekspresi = (): Simpul => {
    let s = suku();
    while (lihat() === "+" || lihat() === "-") { const op = ambil(); s = { op, kiri: s, kanan: suku() }; }
    return s;
  };
  const suku = (): Simpul => {
    let s = pangkat();
    for (;;) {
      if (lihat() === "*" || lihat() === "/") { const op = ambil(); s = { op, kiri: s, kanan: pangkat() }; }
      else if (lihat() === "(" || (lihat() && /^\d/.test(lihat()))) s = { op: "*", kiri: s, kanan: pangkat(), tersirat: true };
      else return s;
    }
  };
  const pangkat = (): Simpul => {
    const b = unari();
    if (lihat() === "^") { ambil(); return { op: "^", kiri: b, kanan: pangkat() }; }
    return b;
  };
  const unari = (): Simpul => (lihat() === "-" ? (ambil(), { neg: unari() }) : dasar());
  const dasar = (): Simpul => {
    const t = ambil();
    if (t === "(") { const e = ekspresi(); if (ambil() !== ")") throw new Error("kurung tidak seimbang"); return e; }
    if (t && /^\d/.test(t)) return { angka: Number(t) };
    throw new Error(`tidak terduga: ${t ?? "akhir"}`);
  };
  const hasil = ekspresi();
  if (i < tk.length) throw new Error(`sisa token: ${tk.slice(i).join(" ")}`);
  return hasil;
}

const SIMBOL: Record<string, string> = { "+": "+", "-": "−", "*": "×", "/": "÷", "^": "^" };

/** Hitung sambil mencatat setiap operasi, urut sesuai yang benar-benar dikerjakan. */
export function hitung(s: Simpul, langkah: string[]): number {
  if ("angka" in s) return s.angka;
  if ("neg" in s) { const v = -hitung(s.neg, langkah); return v; }
  const a = hitung(s.kiri, langkah), b = hitung(s.kanan, langkah);
  const v = s.op === "+" ? a + b : s.op === "-" ? a - b : s.op === "*" ? a * b : s.op === "/" ? a / b : Math.pow(a, b);
  if (s.op === "/" && b === 0) throw new Error("pembagian dengan nol");
  langkah.push(`${fmt(a)} ${SIMBOL[s.op]} ${fmt(b)} = ${fmt(v)}${"tersirat" in s && s.tersirat ? "   (perkalian tersirat)" : ""}`);
  return v;
}

/** Ambil bagian ekspresi dari kalimat ("berapa hasil dari 8 ÷ 2(2 + 2)?" -> "8 ÷ 2(2 + 2)"). */
export function cariEkspresi(teks: string): string | null {
  const m = teks.match(/[-−(]?\s*\d[\d\s.,]*(?:\s*[+\-−×xX*÷:/^()]\s*[\d(][\d\s.,()+\-−×xX*÷:/^]*)+/);
  if (!m) return null;
  const e = m[0].trim().replace(/[.,]\s*$/, "");
  return /[+\-−×xX*÷:/^]/.test(e.replace(/^[-−]/, "")) ? e : null;
}

/** Bentuk a ÷ b(c) yang ambigu: hitung juga tafsiran "perkalian tersirat lebih dulu". */
export function tafsiranLain(ekspresi: string): { teks: string; nilai: number } | null {
  const m = ekspresi.replace(/\s+/g, "").match(/^(\d+(?:[.,]\d+)?)[÷/:](\d+(?:[.,]\d+)?)(\(.+\))$/);
  if (!m) return null;
  const teks = `${m[1]} ÷ (${m[2]}${m[3]})`;
  return { teks, nilai: hitung(urai(teks), []) };
}
