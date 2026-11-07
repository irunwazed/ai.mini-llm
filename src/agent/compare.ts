/**
 * LANGKAH 3 AGENT: BANDINGKAN hasil alat dengan maksud pengguna.
 *
 * Skor 0–1 yang bisa dijelaskan + daftar KEKURANGAN. Kekurangan inilah yang menentukan langkah
 * berikutnya (lihat perbaikan di multi-mode.ts):
 *   kosong   tidak ada jawaban                         -> alat berikutnya
 *   bentuk   "berapa" tanpa angka/satuan, "kapan" tanpa tanggal, "siapa" tanpa nama
 *                                                       -> baca artikel lengkap / sumber lain
 *   kata     kata yang ditanya tidak ada di jawaban ("lahir")       -> kueri ditekankan + sinonim
 *   entitas  nama diri yang ditanya tidak disebut ("Soekarno")      -> cari artikel subjeknya
 *   sumber   sumber tidak sesuai (pertanyaan aturan dijawab Wikipedia, fakta dijawab GPT)
 *                                                       -> alat yang sesuai domain
 *   ambigu   pertanyaan cocok dengan beberapa hal berbeda           -> tanya balik
 * Skor jawaban pencarian = keyakinan × (0,2 + 0,8 × cakupan²) × faktor kekurangan.
 */
import { besaran } from "../answer/question-type.js";
import type { Respons } from "../chat-engine.js";
import { keTerm } from "../text/normalize.js";
import { obrolanJelas, type Niat } from "./intent.js";
import { kataTerpenuhi } from "./relations.js";

export type JenisKurang = "kosong" | "bentuk" | "kata" | "entitas" | "sumber" | "ambigu";
export interface Kekurangan { jenis: JenisKurang; rincian: string; kata?: string[] }
export interface Penilaian { skor: number; cocok: boolean; kurang: Kekurangan[]; alasan: string }

export const AMBANG_PUAS = 0.6;
/** Kekurangan yang membuat jawaban TIDAK boleh dianggap cocok walau skornya tinggi. */
const PENTING: JenisKurang[] = ["kosong", "bentuk", "entitas", "ambigu"];

/** Bagian jawaban sebelum daftar sumber, tanpa nomor sitasi "[1]" (itu bukan angka jawaban). */
export const inti = (t: string) => t.split(/\n\s*\n(?=Sumber|Dasar hukum|Keyakinan|Informasi pendukung|Jejak agent)/i)[0].replace(/\[\d+\]/g, "");

/**
 * Syarat sebuah KALIMAT bukti (dari pilihKalimatWeb) dianggap menjawab:
 *   kata yang ditanya: semua (≤ 3 kata) atau ≥ 70% (lebih banyak)
 *   subjek lanjutan  : salah satu katanya ada        nama diri: semua ada
 *   "berapa"/"kapan" : kalimat memuat angka
 */
export function kalimatCukup(niat: Niat) {
  const kata = niat.kataKunci, subjek = niat.kataSubjek;
  return (k: { teks: string; cakupan: number; cocok: string[] }) => {
    // kata relasi boleh terpenuhi oleh anggotanya ("bekerja" <- "menjabat", "dijabat", "Kepala …")
    const term = niat.relasi ? new Set(keTerm(k.teks)) : null;
    const n = kata.filter((w) => k.cocok.includes(w) || (term && kataTerpenuhi(w, term, niat.relasi, niat.kataPemicu))).length;
    const okKata = kata.length <= 3 ? n === kata.length : n / kata.length >= 0.7;
    const okSubjek = !subjek.length || subjek.some((w) => k.cocok.includes(w));
    const okNama = niat.entitas.every((e) => k.cocok.includes(e));
    const okBentuk = (niat.bentuk !== "angka" && niat.bentuk !== "waktu") || /\d/.test(k.teks);
    // (cakupan berbobot IDF sengaja tidak dipakai: kata langka yang TIDAK wajib, mis. "syarat" atau judul
    //  artikel "Pengantar …", bisa membuatnya rendah walau semua kata yang ditanya ada)
    return k.cocok.length > 0 && okKata && okSubjek && okNama && okBentuk;
  };
}

function kataHilang(niat: Niat, teks: string): string[] {
  const ada = new Set(keTerm(teks));
  return niat.kataKunci.filter((k) => !kataTerpenuhi(k, ada, niat.relasi, niat.kataPemicu));
}

export function bandingkan(niat: Niat, r: Respons): Penilaian {
  const kurang: Kekurangan[] = [];
  const hasil = (skor: number, alasan: string): Penilaian => {
    const cocok = skor >= AMBANG_PUAS && !kurang.some((k) => PENTING.includes(k.jenis));
    return { skor, cocok, kurang, alasan };
  };
  switch (r.jalur) {
    case "tidak-ditemukan":
      kurang.push({ jenis: "kosong", rincian: "alat tidak menemukan jawaban" });
      return hasil(0.3, "jawaban jujur \"tidak tahu\"");
    case "maksud": return hasil(1, "maksud dikenali dengan pola pasti");
    case "detail": return hasil(1, "permintaan detail sumber sebelumnya");
    case "penalaran": return hasil(0.95, "dihitung alat dan lolos cek ulang");
    case "memori": return hasil(0.85, "cocok dengan catatan memori Anda");
    case "klarifikasi":
      kurang.push({ jenis: "ambigu", rincian: "pertanyaan cocok dengan beberapa aturan berbeda" });
      return hasil(0.55, "pertanyaan ambigu -> tanya balik");
    case "obrolan": {
      if (niat.jenis === "obrolan" || niat.jenis === "diri") return hasil(0.8, "pesan berupa obrolan / ditujukan ke bot");
      if (niat.jenis === "tidak-jelas") { kurang.push({ jenis: "kosong", rincian: "niat obrolan tidak dikenali" }); return hasil(0.25, "balasan GPT bisa ngawur"); }
      kurang.push({ jenis: "sumber", rincian: "pertanyaan fakta tidak boleh dijawab karangan GPT" });
      return hasil(0.2, "GPT tidak boleh menjawab fakta");
    }
    case "aturan": case "wiki": case "internet": {
      const k = r.keyakinan === "tinggi" ? 0.9 : r.keyakinan === "sedang" ? 0.65 : 0.25;
      const isi = inti(r.teks);
      const hilang = kataHilang(niat, isi);
      // subjek lanjutan ("virus" untuk "bagaimana penyebarannya?"): cukup salah satu katanya muncul
      if (niat.kataSubjek.length && !niat.kataSubjek.some((w) => keTerm(w).some((t) => new Set(keTerm(isi)).has(t)))) hilang.push(...niat.kataSubjek.slice(0, 1));
      const c = niat.kataKunci.length ? Math.max(0, 1 - hilang.length / (niat.kataKunci.length + (niat.kataSubjek.length ? 1 : 0))) : 1;
      let skor = k * (0.2 + 0.8 * c * c);
      const alasan = [`keyakinan ${r.keyakinan ?? "-"} × cakupan kata ${Math.round(100 * c)}%`];
      if (hilang.length) kurang.push({ jenis: "kata", rincian: `kata yang ditanya tidak ada: ${hilang.join(", ")}`, kata: hilang });
      // bentuk jawaban
      if (niat.bentuk === "angka" && !/\d/.test(isi)) { skor *= 0.4; kurang.push({ jenis: "bentuk", rincian: `ditanya "berapa" tetapi tanpa angka` }); }
      else if (niat.bentuk === "angka" && niat.satuan && !besaran(isi).some((b) => niat.satuan!.test(b.toLowerCase())) && !new RegExp(String.raw`\d[\d.,]*\s?(?:${niat.satuan.source})`, "i").test(isi)) { skor *= 0.4; kurang.push({ jenis: "bentuk", rincian: `ditanya "berapa" tetapi tanpa satuan yang diminta` }); }
      if (niat.bentuk === "waktu" && !/\d/.test(isi)) { skor *= 0.5; kurang.push({ jenis: "bentuk", rincian: "ditanya waktu tetapi tanpa tanggal/angka" }); }
      if (niat.bentuk === "nama" && !/(?<=\s)[A-Z][\p{L}'-]+/u.test(isi)) { skor *= 0.6; kurang.push({ jenis: "bentuk", rincian: `ditanya "siapa" tetapi tidak ada nama` }); }
      // nama diri yang ditanya harus disebut
      const adaKata = new Set(keTerm(isi));
      const tanpaEntitas = niat.entitas.filter((e) => !adaKata.has(e));
      if (tanpaEntitas.length) { skor *= 0.5; kurang.push({ jenis: "entitas", rincian: `nama yang ditanya tidak disebut: ${tanpaEntitas.join(", ")}`, kata: tanpaEntitas }); }
      // sumber yang sesuai
      if (niat.jenis === "aturan" && r.jalur !== "aturan") { skor *= 0.8; kurang.push({ jenis: "sumber", rincian: "pertanyaan tentang aturan, tetapi jawaban bukan dari dokumen aturan" }); }
      if (obrolanJelas(niat.pesan) && niat.jenis === "obrolan") { skor *= 0.3; kurang.push({ jenis: "sumber", rincian: "pesan berupa obrolan, bukan pertanyaan" }); }
      for (const x of kurang.slice(hilang.length ? 1 : 0)) alasan.push(x.rincian);
      return hasil(skor, alasan.join("; "));
    }
  }
}
