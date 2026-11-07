/**
 * DATA LATIH "GPT PEMBACA" dibuat otomatis dari teks biasa, tanpa label manual.
 *
 * Idenya: banyak kalimat punya pola yang jawabannya jelas. Dari pola itu dibuat pertanyaan,
 * dan rentang jawabannya sudah diketahui letaknya di kalimat:
 *   definisi   "Virus adalah agen infeksi …"           -> "apa itu virus?"            jawaban: "agen infeksi …"
 *   besaran    "… cuti tahunan selama 12 hari kerja"   -> "berapa lama cuti tahunan?"  jawaban: "12 hari kerja"
 *   waktu      "… lahir pada 6 Juni 1901 …"            -> "kapan … lahir?"             jawaban: "6 Juni 1901"
 *   siapa      "… ditulis oleh Andrea Hirata"          -> "siapa yang menulis …?"      jawaban: "Andrea Hirata"
 *   tempat     "… didirikan di Bandung …"              -> "di mana … didirikan?"       jawaban: "Bandung"
 *
 * Konteks = kalimat sumber + kalimat tetangga + kalimat pengecoh dari artikel lain.
 * ±30% contoh NEGATIF: konteksnya tidak memuat jawaban -> model harus memilih "tidak ada jawaban".
 * Kata kunci pertanyaan diberi variasi (dibuang sebagian, diganti sinonim, ditulis santai, huruf kecil)
 * supaya model belajar MAKNA, bukan sekadar mencocokkan kata persis.
 */
import { besaran } from "../answer/question-type.js";
import { pecahKalimat } from "../answer/sentences.js";
import { padanan } from "../text/synonyms.js";
import { keTerm } from "../text/normalize.js";
import { Rng } from "./random.js";

export type Pola = "definisi" | "besaran" | "waktu" | "siapa" | "tempat";
export interface ContohPembaca {
  pertanyaan: string;
  konteks: string;
  awal: number;            // indeks karakter awal jawaban di konteks (-1 = tidak ada jawaban)
  jawaban: string;         // "" untuk contoh negatif
  pola: Pola;
  kelompok: string;        // artikel/dokumen asal (untuk membagi train/val tanpa bocor)
}
export interface Artikel { judul: string; teks: string; kelompok: string }

const BULAN = "Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember";
export const RE_TANGGAL = new RegExp(String.raw`\b(\d{1,2}\s+(?:${BULAN})\s+\d{3,4}|(?:${BULAN})\s+\d{4}|tahun\s+\d{3,4}|abad\s+ke-\d+)\b`, "u");
const NAMA = String.raw`(?:[A-Z][\p{L}.'-]*\s){0,3}[A-Z][\p{L}'-]+`;
const RE_OLEH = new RegExp(String.raw`\boleh\s+(${NAMA})`, "u");
const RE_TEMPAT = new RegExp(String.raw`\b(?:di|ke)\s+(${NAMA})`, "u");
const RE_DEFINISI = /^(.{3,60}?)\s+(adalah|merupakan|ialah)\s+(.{10,})$/;
const SANTAI: Record<string, string> = { berapa: "brp", bagaimana: "gimana", apa: "apa sih", tidak: "gak", tahu: "tau", sudah: "udah", saja: "aja", siapa: "siapa sih" };

/** Pecah artikel menjadi kalimat (memakai pemecah kalimat yang sama dengan penjawab). */
function kalimatArtikel(a: Artikel): string[] {
  return a.teks.split(/\n+/).flatMap((par) => pecahKalimat(par).map((k) => k.asli.trim())).filter((k) => k.length >= 30 && k.length <= 400);
}

export function buatDataPembaca(artikel: Artikel[], opsi: { seed?: number; maksPerPola?: number; negatif?: number } = {}): ContohPembaca[] {
  const rng = new Rng(opsi.seed ?? 1);
  const maks = opsi.maksPerPola ?? 6000, pNeg = opsi.negatif ?? 0.3;
  const semua = artikel.map((a) => ({ a, kal: kalimatArtikel(a) })).filter((x) => x.kal.length >= 2);

  // Frekuensi dokumen setiap kata (per kalimat) -> kata jarang lebih mungkin jadi kata kunci pertanyaan
  const df = new Map<string, number>();
  let N = 0;
  for (const { kal } of semua) for (const k of kal) { N++; for (const t of new Set(keTerm(k).filter((x) => !x.startsWith("~")))) df.set(t, (df.get(t) ?? 0) + 1); }
  const idf = (t: string) => Math.log(1 + N / ((df.get(t) ?? 0) + 1));

  /** Pilih kata kunci dari kalimat (urutan asli), tanpa kata jawaban; hanya kata yang DEKAT jawaban (±10 kata). */
  const kataKunci = (kal: string, jawaban: string, n: number, wajib: string[] = []): string[] => {
    const larang = new Set(keTerm(jawaban));
    const posJawab = kal.indexOf(jawaban);
    const dekat = (i: number) => posJawab < 0 || kal.slice(Math.min(i, posJawab), Math.max(i, posJawab)).split(" ").length <= 10;
    const kata = [...kal.matchAll(/[\p{L}\d-]+/gu)].filter((m) => dekat(m.index!)).map((m) => m[0]).filter((w) => {
      const t = keTerm(w);
      return t.length > 0 && !t.some((x) => larang.has(x)) && w.length > 2;
    });
    const unik = [...new Map(kata.map((w) => [w.toLowerCase(), w])).values()];
    const terpilih = new Set([...unik].sort((a, b) => idf(b.toLowerCase()) - idf(a.toLowerCase())).slice(0, n).map((w) => w.toLowerCase()));
    for (const w of wajib) terpilih.add(w.toLowerCase());
    return unik.filter((w) => terpilih.has(w.toLowerCase()));
  };

  /** Variasi: buang satu kata, ganti sinonim, huruf kecil, gaya santai. */
  const variasi = (q: string): string => {
    let kata = q.split(" ");
    if (kata.length > 4 && rng.next() < 0.2) kata.splice(1 + rng.int(kata.length - 2), 1);
    kata = kata.map((w) => {
      if (rng.next() < 0.1) { const p = padanan(w.toLowerCase().replace(/[?]/g, "")); if (p.length) return p[rng.int(p.length)] + (w.endsWith("?") ? "?" : ""); }
      if (rng.next() < 0.15 && SANTAI[w.toLowerCase()]) return SANTAI[w.toLowerCase()];
      return w;
    });
    const s = kata.join(" ");
    return rng.next() < 0.5 ? s.toLowerCase() : s;
  };
  const pilih = <T>(xs: T[]) => xs[rng.int(xs.length)];

  type Calon = { pola: Pola; pertanyaan: string; jawaban: string; awalDiKalimat: number };
  const calonDari = (kal: string, judul: string): Calon[] => {
    const hasil: Calon[] = [];
    const tanpaTitik = kal.replace(/[.]$/, "");
    // definisi
    const d = tanpaTitik.match(RE_DEFINISI);
    if (d && d[1].split(" ").length <= 6 && !/[,;]/.test(d[1])) {
      let y = d[3];
      const kataY = y.split(" ");
      if (kataY.length > 25) y = kataY.slice(0, 25).join(" ");
      const koma = y.indexOf(", ", 40);
      if (koma > 0) y = y.slice(0, koma);
      const x = d[1].replace(/^(Sebuah|Seorang|Para)\s+/i, "");
      const siapa = /^[A-Z]/.test(x) && /\b(seorang|tokoh|penulis|politikus|pemain|penyanyi|aktor|aktris|raja|ratu|presiden)\b/i.test(y);
      const q = siapa ? pilih([`siapa ${x}?`, `siapa itu ${x}?`, `${x} itu siapa?`]) : pilih([`apa itu ${x}?`, `apa yang dimaksud dengan ${x}?`, `${x} adalah?`, `apa pengertian ${x}?`, `jelaskan tentang ${x}`]);
      hasil.push({ pola: "definisi", pertanyaan: q, jawaban: y, awalDiKalimat: kal.indexOf(y, d[1].length) });
    }
    // besaran
    for (const b of besaran(kal).slice(0, 1)) {
      const kk = kataKunci(kal, b, 3);
      if (kk.length < 2) continue;
      const u = b.toLowerCase();
      const awalan = /rp|rupiah/.test(u) ? pilih(["berapa besar", "berapa biaya", "berapa"]) : /%|persen/.test(u) ? "berapa persen"
        : /hari|minggu|bulan|tahun|jam|menit/.test(u) ? pilih(["berapa lama", u.match(/hari|minggu|bulan|tahun|jam|menit/)![0] === "hari" ? "berapa hari" : "berapa lama"])
          : pilih(["berapa banyak", "berapa jumlah", "berapa"]);
      hasil.push({ pola: "besaran", pertanyaan: `${awalan} ${kk.join(" ")}?`, jawaban: b, awalDiKalimat: kal.indexOf(b) });
    }
    // waktu
    const t = kal.match(RE_TANGGAL);
    if (t) {
      const jaw = t[1].replace(/^tahun\s+/, "");
      const kk = kataKunci(kal, t[1], 3);
      if (kk.length >= 2) hasil.push({ pola: "waktu", pertanyaan: `${pilih(["kapan", "kapan", "tahun berapa"])} ${kk.join(" ")}?`, jawaban: jaw, awalDiKalimat: kal.indexOf(jaw, t.index) });
    }
    // siapa (… oleh Nama)
    const o = kal.match(RE_OLEH);
    if (o && o[1] !== judul) {
      const sebelum = kal.slice(0, o.index).trim().split(" ").at(-1) ?? "";
      const kk = kataKunci(kal, o[1], 3, /^di\w+$/.test(sebelum) ? [sebelum] : []);
      if (kk.length >= 2) hasil.push({ pola: "siapa", pertanyaan: `siapa yang ${kk.join(" ")}?`, jawaban: o[1], awalDiKalimat: kal.indexOf(o[1], o.index) });
    }
    // tempat (di/ke Nama)
    const p = kal.match(RE_TEMPAT);
    if (p && p.index! > 10) {
      const kk = kataKunci(kal, p[1], 3);
      if (kk.length >= 2) hasil.push({ pola: "tempat", pertanyaan: `di mana ${kk.join(" ")}?`, jawaban: p[1], awalDiKalimat: kal.indexOf(p[1], p.index) });
    }
    return hasil.filter((c) => c.awalDiKalimat >= 0 && c.jawaban.length >= 1);
  };

  const hitung: Record<Pola, number> = { definisi: 0, besaran: 0, waktu: 0, siapa: 0, tempat: 0 };
  const keluar: ContohPembaca[] = [];
  const urutan = semua.flatMap((x, ai) => x.kal.map((_, ki) => [ai, ki] as const));
  // acak urutan agar batas maksPerPola tidak hanya diisi artikel awal
  for (let i = urutan.length - 1; i > 0; i--) { const j = rng.int(i + 1); [urutan[i], urutan[j]] = [urutan[j], urutan[i]]; }

  for (const [ai, ki] of urutan) {
    const { a, kal } = semua[ai];
    for (const c of calonDari(kal[ki], a.judul)) {
      if (hitung[c.pola] >= maks) continue;
      // tetangga (0–2 di kiri, 0–2 di kanan) + pengecoh dari artikel lain
      const kiri = rng.int(3), kanan = rng.int(3);
      const pengecoh = Array.from({ length: rng.int(3) }, () => { const o = semua[rng.int(semua.length)]; return o === semua[ai] ? "" : pilih(o.kal); }).filter(Boolean);
      const negatif = rng.next() < pNeg;
      const tetangga = kal.slice(Math.max(0, ki - kiri), ki + kanan + 1);
      const isi = negatif ? tetangga.filter((s) => s !== kal[ki]) : tetangga;
      if (negatif && !isi.length) isi.push(...kal.filter((_, j) => j !== ki).slice(0, 2));
      const depan = rng.next() < 0.5;
      const bagian = depan ? [...pengecoh, ...isi] : [...isi, ...pengecoh];
      const konteks = bagian.join(" ");
      if (negatif) {
        if (konteks.toLowerCase().includes(c.jawaban.toLowerCase())) continue;
        keluar.push({ pertanyaan: variasi(c.pertanyaan), konteks, awal: -1, jawaban: "", pola: c.pola, kelompok: a.kelompok });
      } else {
        const posKal = konteks.indexOf(kal[ki]);
        if (posKal < 0) continue;
        const awal = posKal + c.awalDiKalimat;
        if (konteks.indexOf(c.jawaban) !== awal) continue; // jawaban muncul lebih dari sekali -> ambigu, lewati
        keluar.push({ pertanyaan: variasi(c.pertanyaan), konteks, awal, jawaban: c.jawaban, pola: c.pola, kelompok: a.kelompok });
      }
      hitung[c.pola]++;
    }
  }
  return keluar;
}

/** Bagi train/val per kelompok (artikel/dokumen), supaya val berisi artikel yang tidak pernah dilihat. */
export function bagiData(data: ContohPembaca[], porsiVal = 0.1): { train: ContohPembaca[]; val: ContohPembaca[] } {
  const hash = (s: string) => { let h = 2166136261; for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619); return (h >>> 0) / 2 ** 32; };
  const val = data.filter((d) => hash(d.kelompok) < porsiVal);
  return { train: data.filter((d) => hash(d.kelompok) >= porsiVal), val };
}
