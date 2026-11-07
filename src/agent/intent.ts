/**
 * LANGKAH 1 AGENT: PAHAMI MAKSUD PENGGUNA.
 *
 * Hasilnya "Niat": apa jenis pesannya, bentuk jawaban apa yang diharapkan, kata kunci & nama diri
 * yang harus muncul di jawaban, subjek dari percakapan sebelumnya, dan sumber yang paling mungkin.
 * Niat inilah yang nanti dipakai untuk MEMBANDINGKAN setiap hasil alat ("apakah ini menjawab?").
 */
import { jenisPertanyaan, satuanDiminta } from "../answer/question-type.js";
import { kataPertanyaan, namaDiri } from "../answer/web-extract.js";
import { KE_BOT, SAPAAN, TANYA_FAKTA, type MesinChat, type Topik } from "../chat-engine.js";
import { AMBANG_NIAT, niatPesan } from "../chat-intent.js";
import { pahamiMaksud } from "../intent.js";
import { PRIBADI } from "../memory.js";
import { adaRujukan, jenisRujukan, kataTopik, mungkinLanjutan, PEMBINGKAI, tulisUlang } from "../rag/dialog.js";
import { keTerm, tokenisasi } from "../text/normalize.js";
import { stem } from "../text/stemmer.js";
import { KATA_WAKTU, MINTA_KINI, relasiDari, type Relasi } from "./relations.js";
import { adaAlatPenalaran } from "../reasoning/agent.js";
import { AMBANG_KEKUATAN } from "../answer/compose.js";
import { cari } from "../search/search.js";
import { normalisasiSlang } from "../text/slang.js";

export type JenisNiat = "obrolan" | "diri" | "penalaran" | "aturan" | "fakta" | "pribadi" | "tidak-jelas";
export type Bentuk = "angka" | "waktu" | "nama" | "tempat" | "definisi" | "ya-tidak" | "daftar" | "bebas";

export interface Niat {
  pesan: string;                 // pesan setelah normalisasi bahasa santai
  kueri: string;                 // pesan yang sudah ditulis ulang dengan subjek percakapan (jika lanjutan)
  jenis: JenisNiat;
  bentuk: Bentuk;
  satuan: RegExp | null;
  kataKunci: string[];           // kata yang ditanya: harus ada di jawaban (tanpa kata pembingkai)
  kataSubjek: string[];          // kata subjek pertanyaan lanjutan: cukup salah satunya ada
  kataCari: string[];            // kata untuk mencari (subjek + kata kunci)
  entitas: string[];
  subjekLanjutan: string | null;
  domain: "aturan" | "umum" | "tidak-tahu";
  skorDomain: { aturan: number; umum: number };
  alasan: string[];
  jejak: JejakNiat;
  relasi: Relasi | null;         // hubungan yang ditanya ("pekerjaan", "kelahiran", …) atau null
  kataPemicu: string[];          // kata kunci yang boleh diganti anggota relasi ("bekerja" -> "menjabat", "Kepala …")
  mintaKini: boolean;            // ditanya keadaan SEKARANG ("sekarang", "saat ini", "terakhir")
}

export function obrolanJelas(pesan: string): boolean {
  const n = niatPesan(pesan);
  return SAPAAN.test(pesan) || (!!n && n.skor >= AMBANG_NIAT);
}
/** Pertanyaan fakta yang butuh sumber (bukan obrolan, bukan pertanyaan tentang bot). */
export function tanyaFakta(pesan: string): boolean {
  return TANYA_FAKTA.test(pesan) && !KE_BOT.test(pesan) && !obrolanJelas(pesan);
}

function bentukDari(pesan: string): Bentuk {
  if (/\bsiapa\b/i.test(pesan) && !/\bsiapa (itu|sih)\b/i.test(pesan)) return "nama";
  const j = jenisPertanyaan(pesan);
  if (j === "jumlah") return "angka";
  if (j === "waktu") return "waktu";
  if (j === "definisi") return "definisi";
  if (j === "yaTidak") return "ya-tidak";
  if (j === "syarat" || j === "prosedur" || j === "sanksi") return "daftar";
  if (/\bsiapa\b/i.test(pesan)) return "nama";
  if (/\b(di ?mana|ke ?mana|dari mana)\b/i.test(pesan)) return "tempat";
  return "bebas";
}

/** Jejak pemahaman maksud: setiap keputusan dicatat agar bisa ditampilkan di panel cara kerja. */
export interface JejakNiat {
  slang: [string, string][];                                   // kata santai -> baku
  topikSebelumnya: { pertanyaan: string; jalur: string; subjek: string } | null;
  lanjutan: string;                                            // mengapa (tidak) dianggap pertanyaan lanjutan
  kata: { kata: string; peran: string }[];                     // peran setiap kata pesan
  probe: { sumber: string; teratas: string | null; cakupan: number; kekuatan: number; skor: number }[];
  keputusan: { cek: string; hasil: boolean; dipilih: boolean }[];
  bentuk: string;
}

export function pahamiNiat(mesin: MesinChat, pesanAsli: string, topik: Topik | null): Niat {
  const slang = normalisasiSlang(pesanAsli);
  const pesan = slang.teks;
  const alasan: string[] = [];

  // 1. Subjek dari percakapan sebelumnya ("kapan DIA lahir?" setelah membahas Soekarno)
  let subjekLanjutan: string | null = null, kueri = pesan, lanjutan: string;
  const rujukan = jenisRujukan(pesan);
  if (!topik) lanjutan = "tidak ada topik sebelumnya -> pesan dibaca berdiri sendiri";
  else if (topik && (adaRujukan(pesan) || (topik.jalur === "aturan" && mungkinLanjutan(pesan)))) {
    subjekLanjutan = topik.jalur === "aturan" ? kataTopik(topik.pertanyaan) : topik.subjek;
    if (subjekLanjutan) {
      kueri = tulisUlang(pesan, subjekLanjutan);
      lanjutan = `${rujukan ?? "pertanyaan pendek"} -> subjek "${subjekLanjutan}" (${topik.jalur === "aturan" ? "kata topik pertanyaan aturan sebelumnya" : "judul sumber jawaban sebelumnya"}) -> "${kueri}"`;
      alasan.push(`lanjutan dari topik "${subjekLanjutan}" -> "${kueri}"`);
    } else lanjutan = `${rujukan}, tetapi topik sebelumnya tidak punya kata topik`;
  } else lanjutan = rujukan ? `${rujukan}, tetapi topik sebelumnya dari jalur ${topik.jalur} -> tidak ditulis ulang` : "pesan punya kata topik sendiri dan tidak memakai kata ganti -> topik baru";

  // 2. Sumber yang paling mungkin: seberapa kuat kata pertanyaan cocok dengan dokumen aturan vs Wikipedia lokal
  const probe: JejakNiat["probe"] = [];
  const kekuatan = (m: Parameters<typeof cari>[0] | null, sumber: string) => {
    if (!m) { probe.push({ sumber, teratas: null, cakupan: 0, kekuatan: 0, skor: 0 }); return 0; }
    const h = cari(m, kueri, { k: 3 }).hasil[0];
    const skor = h ? h.cakupan * Math.min(1, h.kekuatan / AMBANG_KEKUATAN) : 0;
    probe.push({ sumber, teratas: h ? `${m.info[h.potongan.file]?.judulPendek ?? h.potongan.file} · ${h.potongan.pasal}` : null, cakupan: h?.cakupan ?? 0, kekuatan: h?.kekuatan ?? 0, skor });
    return skor;
  };
  const skorDomain = { aturan: kekuatan(mesin.aturan, "dokumen aturan"), umum: kekuatan(mesin.wiki, "Wikipedia lokal") };
  const domain = skorDomain.aturan >= 0.6 && skorDomain.aturan >= skorDomain.umum - 0.1 ? "aturan" : skorDomain.umum >= 0.6 ? "umum" : "tidak-tahu";

  // 3. Jenis pesan: pemeriksaan berurutan, yang pertama benar yang dipilih
  const pemeriksaan: [string, () => boolean, JenisNiat, string][] = [
    ["pola perkenalan / pertanyaan tentang diri Anda", () => !!pahamiMaksud(pesan), "diri", "pola perkenalan / pertanyaan tentang diri Anda"],
    ["dikenali alat penalaran (soal hitung/logika)", () => adaAlatPenalaran(pesan), "penalaran", "dikenali alat penalaran (hitung/logika)"],
    ["sapaan / mirip contoh obrolan, dan bukan pertanyaan fakta", () => obrolanJelas(pesan) && !tanyaFakta(pesan), "obrolan", "sapaan / basa-basi"],
    ["bertanya tentang diri sendiri (saya/aku) + ada catatan memori", () => PRIBADI.test(pesan) && TANYA_FAKTA.test(pesan) && mesin.memori.semua().length > 0, "pribadi", "pertanyaan tentang diri Anda + ada catatan memori"],
    [`kata kunci cocok kuat dengan dokumen aturan (skor ${skorDomain.aturan.toFixed(2)} ≥ 0,60)`, () => domain === "aturan", "aturan", `kata kunci cocok kuat dengan dokumen aturan (${Math.round(100 * skorDomain.aturan)}%)`],
    ["pertanyaan fakta / bentuk jawaban jelas", () => tanyaFakta(pesan) || bentukDari(pesan) !== "bebas", "fakta", "pertanyaan fakta / pengetahuan umum"],
    ["pesan ditujukan kepada bot (kamu/-mu)", () => KE_BOT.test(pesan), "obrolan", "pesan ditujukan kepada bot"],
  ];
  let jenis: JenisNiat = "tidak-jelas";
  const keputusan: JejakNiat["keputusan"] = [];
  let sudah = false;
  for (const [cek, uji, j, a] of pemeriksaan) {
    const hasil = uji();
    keputusan.push({ cek, hasil, dipilih: hasil && !sudah });
    if (hasil && !sudah) { jenis = j; alasan.push(a); sudah = true; }
  }
  if (!sudah) alasan.push("bukan pertanyaan yang jelas maupun obrolan yang dikenali");

  // 4. Bentuk & kata kunci dari pertanyaan yang SUDAH ditulis ulang ("syaratnya?" -> "syarat cuti tahunan?");
  //    kata pembingkai ("syarat", "lama", "cara") tidak wajib muncul di jawaban
  const bentuk = bentukDari(kueri);
  const bukanBingkai = (k: string) => !PEMBINGKAI.has(stem(k.replace(/nya$/, "")));
  // Kata yang DITANYA diambil dari pesan pengguna sendiri ("bagaimana penyebarannya?" -> "penyebaran");
  // kata subjek lanjutan ("Pengantar tentang virus") hanya konteks: cukup salah satunya muncul.
  // Elipsis tanpa kata topik ("syaratnya?") memakai kata dari pertanyaan yang ditulis ulang.
  let kataKunci = kataPertanyaan(pesan.replace(/\b([a-z]{3,})nya\b/gi, "$1")).filter(bukanBingkai);
  let kataSubjek = subjekLanjutan ? kataPertanyaan(subjekLanjutan).filter((k) => !kataKunci.includes(k)) : [];
  if (!kataKunci.length) { kataKunci = kataPertanyaan(kueri).filter(bukanBingkai); kataSubjek = []; }
  if (!kataKunci.length) kataKunci = kataPertanyaan(kueri);
  // kata waktu ("sekarang", "kini") bukan kata yang wajib ada di kalimat bukti; itu petunjuk untuk penalaran waktu
  const relasi = relasiDari(kueri), mintaKini = MINTA_KINI.test(kueri);
  if (kataKunci.some((k) => KATA_WAKTU.has(k)) && kataKunci.some((k) => !KATA_WAKTU.has(k))) kataKunci = kataKunci.filter((k) => !KATA_WAKTU.has(k));
  const kataPemicu = relasi ? kataKunci.filter((k) => relasi.pemicu.test(k)) : [];
  // Nama diri: dari pesan, ditambah subjek lanjutan jika subjek itu sendiri berupa nama (semua kata berhuruf kapital)
  const entitas = namaDiri(pesan);
  // Nama yang diketik huruf kecil ("di mana zudan bekerja?"): kata yang TIDAK pernah muncul di indeks lokal
  // (aturan maupun Wikipedia) dan bukan kata relasi/waktu/pembingkai -> kemungkinan besar nama orang/tempat.
  const tidakDikenal = (w: string) => !mesin.aturan.indeks.kamus[w] && !mesin.aturan.indeks.kamus[`~${stem(w)}`]
    && (!mesin.wiki || (!mesin.wiki.indeks.kamus[w] && !mesin.wiki.indeks.kamus[`~${stem(w)}`]));
  // Hanya untuk pertanyaan tentang seseorang/sesuatu dengan relasi (pekerjaan, tempat tinggal, …): kosakata indeks
  // lokal terbatas, jadi kata biasa yang jarang ("mudik") tidak boleh dianggap nama di pertanyaan lain.
  if (!entitas.length && relasi) for (const w of kataPertanyaan(pesan)) {
    if (w.length >= 3 && !/^\d+$/.test(w) && !KATA_WAKTU.has(w) && bukanBingkai(w) && !(relasiDari(w)) && tidakDikenal(w)) entitas.push(w);
  }
  if (subjekLanjutan && /^(\p{Lu}[\p{L}'-]*\s?){1,4}$/u.test(subjekLanjutan.trim())) for (const e of namaDiri(` ${subjekLanjutan}`)) if (!entitas.includes(e)) entitas.push(e);

  // 5. Peran setiap kata (untuk panel cara kerja)
  const kata: JejakNiat["kata"] = [];
  for (const w of new Set(tokenisasi(kueri))) {
    const t = keTerm(w).find((x) => !x.startsWith("~"));
    const peran = !t ? "kata umum / kata tanya (dibuang)"
      : entitas.includes(t) ? "nama diri (wajib disebut di jawaban)"
        : kataKunci.includes(t) ? "kata yang ditanya (wajib ada di jawaban)"
          : kataSubjek.includes(t) ? "subjek dari topik sebelumnya (cukup salah satu)"
            : !bukanBingkai(t) ? "kata pembingkai: bentuk pertanyaan, tidak wajib ada"
              : KATA_WAKTU.has(t) ? "kata waktu: petunjuk KAPAN (mis. jabatan yang masih berjalan), tidak wajib ada" : "kata pencarian";
    kata.push({ kata: w, peran: t && kataPemicu.includes(t) ? `kata yang ditanya, relasi "${relasi!.nama}": boleh muncul sebagai ${relasi!.kata.slice(0, 8).join(", ")}, …` : peran });
  }
  const jp = jenisPertanyaan(kueri), satuan = satuanDiminta(kueri);
  const jejak: JejakNiat = {
    slang: slang.ganti, topikSebelumnya: topik ? { pertanyaan: topik.pertanyaan, jalur: topik.jalur, subjek: topik.subjek } : null,
    lanjutan, kata, probe, keputusan,
    bentuk: `jenis pertanyaan "${jp}"${/\bsiapa\b/i.test(kueri) ? ', ada kata "siapa"' : ""}${/\b(di ?mana|ke ?mana|dari mana)\b/i.test(kueri) ? ', ada kata "di mana"' : ""} -> bentuk ${bentuk}${satuan ? `, satuan yang diminta: ${satuan.source}` : ""}`,
  };
  return {
    pesan, kueri, jenis, bentuk, satuan, kataKunci, kataSubjek, kataCari: [...kataSubjek, ...kataKunci], entitas,
    subjekLanjutan, domain, skorDomain, alasan, jejak, relasi, kataPemicu, mintaKini,
  };
}
