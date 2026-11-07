/**
 * PENYUSUN JAWABAN: dari beberapa pasal menjadi satu jawaban yang menyimpulkan.
 *
 * Alur:
 *   1. cari pasal relevan (hibrida BM25 + embedding)
 *   2. pecah pasal teratas menjadi kalimat/ayat, beri skor tiap kalimat
 *   3. susun jawaban sesuai JENIS pertanyaan:
 *        jumlah   -> "12 hari kerja."             + kalimat dasarnya
 *        yaTidak  -> "Tidak." / "Ya, wajib."      + kalimat dasarnya
 *        definisi -> definisi dari Ketentuan Umum
 *        syarat/prosedur -> daftar ketentuan sesuai urutan pasal
 *        lainnya  -> kalimat paling relevan
 *   4. tambahkan rincian dari pasal/peraturan lain (sintesis lintas dokumen)
 *   5. dasar hukum: peraturan, pasal, ayat, file, halaman + peringatan jika aturan sudah diubah
 *
 * Setiap fakta di jawaban berasal dari kalimat asli dokumen dan diberi nomor sumber [n].
 * Penyusun tidak pernah menambah angka atau aturan yang tidak tertulis (tidak berhalusinasi).
 *
 * Skor kalimat s (BM25 mini, kalimat dianggap dokumen kecil, panjang rata-rata L = 20):
 *   skor(s) = Σ_q w_q · idf(q) · f(q,s)(k1+1) / ( f(q,s) + k1(1 - b + b·|s|/L) )  ×  1/(1 + 0.25·peringkat pasal)
 */
import { idf } from "../search/bm25.js";
import { cari, type Hasil } from "../search/search.js";
import type { Model } from "../model.js";
import { keTerm, tokenisasi } from "../text/normalize.js";
import type { Potongan } from "../types.js";
import { besaran, jenisPertanyaan, polaritas, satuanDiminta, type Jenis } from "./question-type.js";
import { mirip, pecahKalimat, rapikan, type Kalimat } from "./sentences.js";

export type Keyakinan = "tinggi" | "sedang" | "rendah";
export const AMBANG_TINGGI = 0.6;
export const AMBANG_SEDANG = 0.35;
/**
 * Kekuatan minimal (Σ idf kata cocok / ln(1+N)) agar tidak dianggap hanya cocok kata umum.
 * Ditentukan dari data uji: soal di luar dokumen <= 0.83, soal sah >= 0.87.
 * Setel ulang dengan `bun run eval` setelah dokumen Anda sendiri dimasukkan.
 */
export const AMBANG_KEKUATAN = 0.9;

/**
 * Pola kalimat yang biasanya menjadi jawaban untuk tiap jenis pertanyaan.
 * Kalimat yang cocok polanya mendapat skor x2.
 * Contoh: "apa sanksi ..." -> utamakan kalimat yang memuat "dikenakan sanksi",
 * bukan kalimat yang hanya menyebut pelanggarannya.
 */
const POLA_JENIS: Partial<Record<Jenis, RegExp>> = {
  sanksi: /\b(sanksi|dikenakan|denda|pidana|diberhentikan|daftar hitam|mengembalikan)\b/i,
  waktu: /\b(paling lambat|paling lama|paling cepat|sebelum|setelah|sejak|terhitung)\b/i,
  syarat: /\b(syarat|paling singkat|paling sedikit|paling kurang|memenuhi|telah bekerja|apabila)\b/i,
  yaTidak: /\b(dilarang|tidak boleh|tidak dapat|boleh|dapat|berhak|wajib|harus)\b/i,
  // "Lamanya cuti ... adalah 12 hari kerja" = pola baku hukum untuk MENETAPKAN besaran
  jumlah: /\b(lamanya|besarnya|jumlahnya|sebesar|paling (banyak|tinggi|lama)|adalah\s+(rp\s?)?\d)/i,
  // "Pengeboman Obersalzberg adalah sebuah serangan udara ..." = kalimat definisi
  definisi: /\b(adalah|merupakan|ialah|yaitu)\b/i,
};

export interface Sumber { potongan: Potongan; ayat: Set<string> }
export interface Jawaban {
  pertanyaan: string;
  jenis: Jenis;
  keyakinan: Keyakinan;
  utama: string[];     // kalimat kesimpulan
  rincian: string[];   // poin tambahan
  catatan: string[];
  sumber: Sumber[];
  hasil: Hasil[];
  jejak: Jejak;
}

/** Catatan cara kerja penjawab, untuk ditampilkan sebagai bahan belajar (tampilan web). */
export interface Jejak {
  terms: [string, number][];                               // term pencarian + bobotnya
  kelompok: { kata: string; bobot: number; terms: string[] }[]; // satu kelompok per kata pertanyaan
  kalimat: { teks: string; skor: number; cakupan: number; sumber: string }[]; // kalimat kandidat teratas
  catatan: string[];                                        // keputusan penting (mis. alasan keyakinan)
}

interface Bukti { k: Kalimat; p: Potongan; skor: number; peringkat: number; cak: number }

const KOPULA = /\b(adalah|merupakan|ialah|yaitu)\b/i;

/**
 * Untuk pertanyaan definisi ("apa itu virus?"), cari kalimat berbentuk
 *   "<subjek> adalah ..."   dengan semua kata subjek muncul SEBELUM kata adalah/merupakan,
 * dan paling banyak 10 kata sebelum kata itu. Di Wikipedia, kalimat seperti ini hampir
 * selalu ada di paragraf pembuka artikel, jadi paragraf pertama artikel ikut diperiksa.
 */
function kalimatDefinisi(model: Model, kueri: { kelompok: { terms: string[]; bobot: number }[] }, hasil: Hasil[]): { k: Kalimat; p: Potongan } | null {
  const subjek = kueri.kelompok.filter((g) => g.bobot > 0);
  if (!subjek.length) return null;
  const calon: Potongan[] = [];
  for (const h of hasil.slice(0, 4)) {
    if (model.jenis === "umum") {
      let id = h.potongan.id;
      while (id > 0 && model.potongan[id - 1].file === h.potongan.file) id--;
      if (!calon.includes(model.potongan[id])) calon.push(model.potongan[id]);
    }
    if (!calon.includes(h.potongan)) calon.push(h.potongan);
  }
  for (const p of calon) {
    for (const k of pecahKalimat(p.teks)) {
      const m = k.teks.match(KOPULA);
      if (!m || m.index === undefined) continue;
      const depan = k.teks.slice(0, m.index);
      if (depan.split(" ").length > 10) continue;
      const ada = new Set(keTerm(depan));
      if (subjek.every((g) => g.terms.some((t) => ada.has(t)))) return { k, p };
    }
  }
  return null;
}

/**
 * opsi.subjek: subjek dari pembicaraan sebelumnya ("kapan dia lahir?" + subjek "Hö'elün").
 * Subjek ikut dipakai untuk MENEMUKAN potongan, tetapi saat memilih KALIMAT bobotnya dikecilkan
 * (×0.3) dan cakupan kalimat dihitung dari kata yang benar-benar ditanya ("lahir"), karena
 * subjek muncul di hampir semua kalimat artikelnya.
 */
export function jawab(model: Model, pertanyaan: string, opsi: { k?: number; konteks?: string; jenis?: Jenis; subjek?: string } = {}): Jawaban {
  // Pertanyaan lanjutan tanpa bentuk jelas ("kalau cuti besar?") mewarisi jenis pertanyaan sebelumnya
  let jenis = jenisPertanyaan(pertanyaan);
  if (jenis === "umum" && opsi.jenis) jenis = opsi.jenis;
  if (jenis === "umum" && opsi.konteks) jenis = jenisPertanyaan(opsi.konteks);
  const termSubjek = new Set(opsi.subjek ? keTerm(opsi.subjek) : []);
  const { hasil, kueri } = cari(model, opsi.subjek ? `${pertanyaan} ${opsi.subjek}` : pertanyaan, { k: Math.max(opsi.k ?? 3, 5), konteks: opsi.konteks });
  const j: Jawaban = {
    pertanyaan, jenis, keyakinan: "rendah", utama: [], rincian: [], catatan: [], sumber: [], hasil,
    jejak: { terms: [...kueri.terms].sort((a, b) => b[1] - a[1]), kelompok: kueri.kelompok, kalimat: [], catatan: [] },
  };
  // Lolos jika kata yang cocok cukup kuat, ATAU satu topik spesifik cocok penuh
  // ("apa itu Obersalzberg?": hanya satu kata, tetapi langka dan cocok 100%).
  const lnN = Math.log(1 + model.indeks.N);
  const kataTerlangka = Math.max(0, ...kueri.kelompok.map((g) => g.bobot));
  const atas = hasil.slice(0, 3).filter((h) => h.kekuatan >= AMBANG_KEKUATAN || (h.cakupan >= 0.99 && kataTerlangka >= 0.6 * lnN));
  const c = Math.max(0, ...atas.map((h) => h.cakupan));
  j.keyakinan = c >= AMBANG_TINGGI ? "tinggi" : c >= AMBANG_SEDANG ? "sedang" : "rendah";
  const h0 = hasil[0];
  j.jejak.catatan.push(h0
    ? `Pasal/potongan teratas: cakupan ${(100 * h0.cakupan).toFixed(0)}%, kekuatan ${h0.kekuatan.toFixed(2)} (ambang: cakupan ≥ ${AMBANG_SEDANG * 100}% dan kekuatan ≥ ${AMBANG_KEKUATAN}, atau satu topik langka cocok penuh)`
    : "Tidak ada potongan yang memuat kata pertanyaan.");
  // Pertanyaan definisi: dari glosarium (Ketentuan Umum), atau kalimat "<subjek> adalah ..."
  const definisi = jenis === "definisi" ? cariDefinisi(model, pertanyaan) : null;
  const defKalimat = jenis === "definisi" && !definisi && j.keyakinan !== "rendah" ? kalimatDefinisi(model, kueri, hasil) : null;
  if (definisi || defKalimat) j.keyakinan = "tinggi";
  j.jejak.catatan.push(`Jenis pertanyaan: ${jenis}. Keyakinan: ${j.keyakinan}.`);
  if (j.keyakinan === "rendah") return j;

  // Nomor sumber [n] diberikan sesuai urutan pemakaian
  const ref = (p: Potongan, ayat: string | null) => {
    let i = j.sumber.findIndex((s) => s.potongan.id === p.id);
    if (i < 0) i = j.sumber.push({ potongan: p, ayat: new Set() }) - 1;
    if (ayat) j.sumber[i].ayat.add(ayat);
    return `[${i + 1}]`;
  };

  // 1. Skor setiap kalimat di 4 pasal teratas.
  //    Selain BM25 mini, kalimat yang mencakup LEBIH BANYAK kata yang ditanya diutamakan:
  //      skor = bm25(s) · (0.2 + cakupan(s))² / (1 + 0.25·peringkat)
  //    "batas pengadaan langsung BARANG" -> ayat yang memuat "barang" menang atas ayat "jasa konsultansi".
  const ind = model.indeks, L = 20;
  // Kelompok inti = kata yang benar-benar ditanya (tanpa subjek dari pembicaraan sebelumnya)
  const inti = kueri.kelompok.filter((g) => !g.terms.every((t) => termSubjek.has(t)) && g.bobot > 0);
  const kelompokInti = inti.length ? inti : kueri.kelompok;
  const bobotInti = kelompokInti.reduce((s, g) => s + g.bobot, 0) || 1;
  const bukti: Bukti[] = [];
  const pola = POLA_JENIS[jenis];
  const bahan = hasil.filter((h) => !/^(Pembukaan|Penutup)/.test(h.potongan.pasal));
  (bahan.length ? bahan : hasil).slice(0, 4).forEach((h, r) => {
    for (const k of pecahKalimat(h.potongan.teks)) {
      const tf = new Map<string, number>();
      const ts = keTerm(k.asli);
      for (const t of ts) tf.set(t, (tf.get(t) ?? 0) + 1);
      let s = 0;
      for (const [t, w0] of kueri.terms) {
        const f = tf.get(t);
        const w = termSubjek.has(t) ? 0.3 * w0 : w0;
        if (f) s += (w * idf(ind, t) * f * (ind.k1 + 1)) / (f + ind.k1 * (1 - ind.b + (ind.b * ts.length) / L));
      }
      const cak = kelompokInti.filter((g) => g.terms.some((t) => tf.has(t))).reduce((a, g) => a + g.bobot, 0) / bobotInti;
      // Bonus pola hanya jika kalimat juga memuat sebagian besar kata yang ditanya
      const cocokPola = pola && pola.test(k.teks) && cak >= 0.5 ? 2 : 1;
      if (s > 0) bukti.push({ k, p: h.potongan, skor: (cocokPola * s * (0.2 + cak) ** 2) / (1 + 0.25 * r), peringkat: r, cak });
    }
  });
  bukti.sort((a, b) => b.skor - a.skor);
  j.jejak.kalimat = bukti.slice(0, 6).map((b) => ({
    teks: b.k.teks, skor: b.skor, cakupan: b.cak, sumber: `${model.info[b.p.file]?.judulPendek ?? b.p.file}, ${b.p.pasal}`,
  }));
  const terpakai: string[] = [];
  let cakUtama: number | null = null; // cakupan kalimat utama pertama
  const pakai = (b: Bukti) => {
    terpakai.push(b.k.teks);
    if (cakUtama === null) cakUtama = b.cak;
    return `${b.k.teks} ${ref(b.p, b.k.ayat)}`;
  };

  // 2. Kalimat utama sesuai jenis pertanyaan
  if (definisi) {
    j.utama.push(`${rapikan(definisi.definisi)} ${ref(model.potongan[definisi.id], null)}`);
    terpakai.push(rapikan(definisi.definisi));
    j.jejak.catatan.push("Jawaban utama diambil dari glosarium (definisi di Ketentuan Umum).");
  } else if (defKalimat) {
    j.utama.push(`${defKalimat.k.teks} ${ref(defKalimat.p, defKalimat.k.ayat)}`);
    terpakai.push(defKalimat.k.teks);
    cakUtama = 1;
    j.jejak.catatan.push(`Jawaban utama = kalimat definisi "<subjek> adalah ..." dari ${defKalimat.p.pasal}.`);
  } else if (bukti.length === 0) {
    j.keyakinan = "rendah";
    return j;
  } else if (jenis === "jumlah") {
    const tanya = pertanyaan + " " + (jenisPertanyaan(pertanyaan) === "umum" ? opsi.konteks ?? "" : "");
    const satuan = satuanDiminta(tanya);
    const kandidat = bukti.filter((b) => besaran(b.k.teks).length > 0);
    const b = kandidat.find((x) => !satuan || besaran(x.k.teks).some((q) => satuan.test(q))) ?? kandidat[0] ?? bukti[0];
    const q = besaran(b.k.teks);
    // Pertanyaan DURASI ("berapa lama/hari/bulan"): utamakan besaran tepat setelah "lamanya / paling lama / selama"
    // ("bekerja paling singkat 5 tahun ... berhak atas cuti besar paling lama 3 bulan" -> 3 bulan)
    const setelahLama = /\bberapa\s+(lama|hari|minggu|bulan)\b/i.test(tanya)
      ? b.k.teks.match(/\b(?:lamanya|paling lama|selama)\b[^.;]{0,60}?(\d+\s(?:hari kerja|hari kalender|hari|minggu|bulan|tahun))/i)?.[1]
      : undefined;
    const fakta = setelahLama ?? (satuan ? q.find((x) => satuan.test(x)) : null) ?? q[0];
    if (fakta) j.utama.push(`${fakta.charAt(0).toUpperCase() + fakta.slice(1)}.`);
    j.utama.push(pakai(b));
  } else if (jenis === "yaTidak") {
    const b = bukti.find((x) => polaritas(x.k.teks)) ?? bukti[0];
    const pol = polaritas(b.k.teks);
    const tanyaLarang = /\b(dilarang|larangan)\b/i.test(pertanyaan);
    const tanyaWajib = /\b(wajib|harus|perlu)\w*\b/i.test(pertanyaan);
    const kata =
      pol === "larang" ? (tanyaLarang ? "Ya, dilarang." : "Tidak.")
      : pol === "wajib" ? "Ya, wajib."
      : pol === "boleh" ? (tanyaLarang ? "Tidak dilarang." : tanyaWajib ? "Boleh, tetapi tidak diwajibkan." : "Ya.")
      : null;
    if (kata) j.utama.push(kata);
    j.utama.push(pakai(b));
  } else if (jenis === "syarat" || jenis === "prosedur") {
    const pasal = bukti[0].p;
    const info = model.info[pasal.file];
    j.utama.push(`Menurut ${info?.judulPendek ?? pasal.file}, ${pasal.pasal}:`);
    for (const k of pecahKalimat(pasal.teks).slice(0, 8)) {
      j.utama.push(`• ${k.teks} ${ref(pasal, k.ayat)}`);
      terpakai.push(k.teks);
    }
  } else {
    j.utama.push(pakai(bukti[0]));
    // Jika kalimat terbaik tidak memuat pola jawaban (mis. menyebut pelanggaran tetapi
    // bukan sanksinya), tambahkan kalimat terbaik yang memuat pola itu.
    if (pola && !pola.test(bukti[0].k.teks)) {
      const b = bukti.find((x) => pola.test(x.k.teks) && x.peringkat <= 2);
      if (b) j.utama.push(pakai(b));
    }
  }

  // Kalimat utama hanya memuat sebagian kecil kata yang ditanya -> jawaban kemungkinan
  // tidak nyambung (mis. "tarif pajak penghasilan badan" dijawab kalimat tentang pajak abad ke-7).
  if (cakUtama !== null && cakUtama < 0.5) {
    j.keyakinan = j.keyakinan === "tinggi" ? "sedang" : "rendah";
    if (j.keyakinan === "rendah") { j.utama = []; j.sumber = []; return j; }
  }

  // 3. Rincian dari kalimat lain yang cukup relevan (boleh dari pasal / peraturan lain)
  //    (tidak untuk definisi: definisinya sendiri sudah jawaban lengkap)
  const terbaik = bukti[0]?.skor ?? 0;
  for (const b of definisi ? [] : bukti) {
    if (j.rincian.length >= 3) break;
    if (b.skor < 0.35 * terbaik) break;
    if (terpakai.some((t) => mirip(t, b.k.teks) > 0.6)) continue;
    // Penggalan daftar ("Cuti pegawai terdiri atas:", "b. cuti sakit;") tidak bermakna sendirian
    if (/:\s*$/.test(b.k.asli) || /^[a-z]\.\s/i.test(b.k.asli) || b.k.teks.split(" ").length < 6) continue;
    j.rincian.push(pakai(b));
  }

  // 4. Catatan perubahan peraturan
  for (const s of j.sumber) {
    const info = model.info[s.potongan.file];
    const pengubah = info?.kunci ? model.diubahOleh[info.kunci] : undefined;
    if (pengubah?.length) {
      const n = `${info.judulPendek} telah diubah oleh ${pengubah.join("; ")}. Pastikan memakai ketentuan terbaru.`;
      if (!j.catatan.includes(n)) j.catatan.push(n);
    }
  }
  return j;
}

/** Cari definisi dari istilah yang dipelajari: pilih istilah terpanjang yang muncul di pertanyaan. */
function cariDefinisi(model: Model, pertanyaan: string) {
  const q = " " + tokenisasi(pertanyaan).join(" ") + " ";
  let pilih: { definisi: string; id: number } | null = null;
  let panjang = 0;
  for (const d of model.istilah.definisi)
    for (const nama of d.istilah) {
      const n = " " + tokenisasi(nama).join(" ") + " ";
      if (n.trim() && q.includes(n) && n.length > panjang) { pilih = d; panjang = n.length; }
    }
  return pilih;
}

// ---------------------------------------------------------------- tampilan

export function labelSumber(model: Model, s: Sumber): string {
  const p = s.potongan;
  const info = model.info[p.file];
  const urutAyat = [...s.ayat].sort((a, b) => parseInt(a) - parseInt(b) || a.localeCompare(b));
  const ayat = urutAyat.length ? ` ayat ${urutAyat.map((a) => `(${a})`).join(", ")}` : "";
  if (model.jenis === "umum") return `${info?.judulPendek ?? p.file}, ${p.pasal}`;
  const lokasi = p.halaman !== null ? `hlm. ${p.halaman}` : `baris ${p.baris}`;
  return `${info?.judulPendek ?? p.file}, ${p.pasal}${ayat}\n      ${p.file}, ${lokasi}`;
}

export function formatJawaban(model: Model, j: Jawaban): string {
  const out: string[] = [];
  const umum = model.jenis === "umum";
  if (j.keyakinan === "rendah") {
    out.push(umum
      ? "Maaf, saya tidak menemukan informasi tentang hal ini di teks yang sudah saya baca."
      : "Maaf, saya tidak menemukan aturan yang membahas hal ini di dokumen yang sudah dilatih.");
    if (j.hasil.length) {
      out.push("", "Topik terdekat yang ada:");
      j.hasil.slice(0, 2).forEach((h) => out.push(`  - ${labelSumber(model, { potongan: h.potongan, ayat: new Set() }).split("\n")[0]}`));
    }
    return out.join("\n");
  }
  if (j.keyakinan === "sedang") out.push(`(Saya belum sepenuhnya yakin. Mohon periksa ${umum ? "sumbernya" : "pasal sumbernya"}.)`, "");
  out.push(...j.utama);
  if (j.rincian.length) out.push("", umum ? "Informasi terkait:" : "Ketentuan terkait:", ...j.rincian.map((r) => `• ${r}`));
  if (j.catatan.length) out.push("", ...j.catatan.map((c) => `Catatan: ${c}`));
  out.push("", umum ? "Sumber (Wikipedia bahasa Indonesia, CC BY-SA 4.0):" : "Dasar hukum:");
  j.sumber.forEach((s, i) => out.push(`  [${i + 1}] ${labelSumber(model, s)}`));
  return out.join("\n");
}
