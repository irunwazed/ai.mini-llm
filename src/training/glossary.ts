/**
 * Belajar ISTILAH dan SINGKATAN otomatis dari dokumen.
 *
 * Hampir setiap peraturan punya "Ketentuan Umum" (biasanya Pasal 1):
 *   "1. Aparatur Sipil Negara yang selanjutnya disingkat ASN adalah profesi bagi ..."
 * Dari kalimat seperti ini model belajar dua hal tanpa diajari manual:
 *   - singkatan:  asn  <->  aparatur sipil negara       (perluasan pertanyaan dua arah)
 *   - definisi :  "ASN adalah profesi bagi ..."           (menjawab "apa itu ASN?")
 */
import type { Potongan } from "../types.js";

export interface Definisi { istilah: string[]; definisi: string; id: number }
export interface Istilah {
  singkatan: Record<string, string[]>; // kata/frasa (huruf kecil) -> padanan
  definisi: Definisi[];
}

const RE_ALIAS = /^(.{3,120}?)\s*,?\s+yang selanjutnya (?:disingkat|disebut)(?: dengan)?\s+([A-Za-z][\w/-]{1,30}(?:\s+[A-Z][\w-]*){0,3})/i;
/**
 * Butir Ketentuan Umum: "39. Penunjukan Langsung adalah ...".
 * Di PDF hasil OCR nomor butir sering berada di tengah baris, jadi teks diratakan
 * lalu dipecah sebelum setiap pola "angka. Huruf-kapital".
 */
function butir(teks: string): string[] {
  return teks
    .replace(/\s+/g, " ")
    .split(/\s(?=\d{1,3}\s?\.\s+[A-Z])/)
    .map((s) => s.match(/^\d{1,3}\s?\.\s+(.+)$/)?.[1]?.trim())
    .filter((s): s is string => !!s);
}

function tambahPadanan(m: Record<string, string[]>, a: string, b: string) {
  if (a === b) return;
  (m[a] ??= []).includes(b) || m[a].push(b);
}

export function pelajariIstilah(potongan: Potongan[]): Istilah {
  const singkatan: Record<string, string[]> = {};
  const definisi: Definisi[] = [];
  for (const p of potongan) {
    if (!/ketentuan umum/i.test(p.bab) && !/^Pasal 1( \(|$)/.test(p.pasal)) continue; // termasuk "Pasal 1 (bagian 2)"
    for (const item of butir(p.teks)) {
      const pisah = item.match(/^(.{2,160}?)\s+(?:adalah|ialah|merupakan|yaitu)\s+(.+)$/i);
      if (!pisah) continue;
      const kepala = pisah[1];
      const nama: string[] = [];
      const a = kepala.match(RE_ALIAS);
      if (a) {
        const panjang = a[1].toLowerCase().trim();
        const pendek = a[2].toLowerCase().trim();
        nama.push(pendek, panjang);
        tambahPadanan(singkatan, pendek, panjang);
        tambahPadanan(singkatan, panjang, pendek);
      } else nama.push(kepala.toLowerCase());
      definisi.push({
        istilah: nama,
        definisi: `${kepala.replace(/\s*,?\s+yang selanjutnya.*$/i, "")} ${pisah[0].slice(kepala.length).trim()}`.replace(/;$/, "."),
        id: p.id,
      });
    }
  }
  // Pola tanpa nomor butir (nomor sering hilang di PDF hasil OCR):
  //   "... Pegawai Negeri Sipil yang selanjutnya disingkat PNS adalah warga negara ...; "
  const RE_DEF_BEBAS =
    /((?:[A-Z][\w/-]*\s+){0,7}[A-Z][\w/-]*)\s*,?\s+yang selanjutnya (?:disingkat|disebut)(?: dengan)?\s+([A-Z][\w/-]*)\s+(?:adalah|ialah)\s+(.+?)(?=;|\.\s+(?:\d+\s*\.\s+)?[A-Z]|$)/g;
  const sudahAda = new Set(definisi.flatMap((d) => d.istilah));
  for (const p of potongan) {
    if (!/ketentuan umum/i.test(p.bab) && !/^Pasal 1( \(|$)/.test(p.pasal)) continue;
    for (const m of p.teks.replace(/\s+/g, " ").matchAll(RE_DEF_BEBAS)) {
      const panjang = m[1].toLowerCase().trim(), pendek = m[2].toLowerCase();
      if (sudahAda.has(pendek)) continue;
      sudahAda.add(pendek);
      tambahPadanan(singkatan, pendek, panjang);
      tambahPadanan(singkatan, panjang, pendek);
      definisi.push({ istilah: [pendek, panjang], definisi: `${m[1]} (${m[2]}) adalah ${m[3]}.`, id: p.id });
    }
  }

  // Pola tambahan di seluruh teks: "Aparatur Sipil Negara (ASN)"
  for (const p of potongan) {
    for (const m of p.teks.matchAll(/((?:[A-Z][a-z]+\s){1,6}[A-Z][a-z]+)\s+\(([A-Z]{2,10})\)/g)) {
      const panjang = m[1].toLowerCase();
      const pendek = m[2].toLowerCase();
      const inisial = panjang.split(" ").map((w) => w[0]).join("");
      if (!inisial.includes(pendek[0]) || pendek.length > inisial.length + 1) continue; // cek inisial kasar
      tambahPadanan(singkatan, pendek, panjang);
      tambahPadanan(singkatan, panjang, pendek);
    }
  }
  return { singkatan, definisi };
}
